#!/usr/bin/env python
"""Evaluate only unseen full-test rows and merge with a completed subset run."""

import argparse
import concurrent.futures
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import textgrad as tg

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.batch_trainer import HardCase
from adversarial_gv.engines import TokenUsageTracker, build_engine
from adversarial_gv.evaluation import is_correct, parse_verdict


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--experiment",
        default="runs/qwen4b_gen_qwen32b_oracle_routed_acc_iter3/experiment.json",
    )
    result.add_argument(
        "--full-data",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    result.add_argument(
        "--output",
        default=(
            "runs/qwen4b_gen_qwen32b_oracle_routed_acc_iter3/"
            "full_test_evaluation.json"
        ),
    )
    result.add_argument("--generator-vllm-base-url", default="http://127.0.0.1:8004/v1")
    result.add_argument("--verifier-vllm-base-url", default="http://127.0.0.1:8000/v1")
    result.add_argument("--vllm-api-key", default="EMPTY")
    result.add_argument("--workers", type=int, default=8)
    return result


def load_test_cases(path: Path) -> list[HardCase]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        HardCase(
            wrong_id=int(row["wrong_id"]),
            collection_split=row["collection_split"],
            source_split=row["split"],
            source_index=int(row["index"]),
            question=row["question"],
            ground_truth=row["ground_truth"],
        )
        for row in rows
        if row["collection_split"] == "test"
    ]


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    denominator = len(rows)

    def rate(predicate) -> float:
        if not denominator:
            return 0.0
        return sum(bool(predicate(row)) for row in rows) / denominator

    return {
        "accuracy": rate(lambda row: row["correct"]),
        "accept_rate": rate(lambda row: row["verdict"]["label"] == "ACCEPT"),
        "challenge_rate": rate(
            lambda row: row["verdict"]["label"] == "CHALLENGE"
        ),
        "reject_rate": rate(lambda row: row["verdict"]["label"] == "REJECT"),
        "invalid_rate": rate(lambda row: row["verdict"]["label"] == "INVALID"),
        "evaluated_count": denominator,
        "rows": list(rows),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    experiment_path = Path(args.experiment)
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not experiment.get("complete"):
        raise ValueError("source experiment is not complete")
    full_test = load_test_cases(Path(args.full_data))
    existing_rows = experiment["evaluations"][-1]["test"]["rows"]
    existing_by_id = {int(row["case"]["wrong_id"]): row for row in existing_rows}
    full_by_id = {case.wrong_id: case for case in full_test}
    if not set(existing_by_id) <= set(full_by_id):
        raise ValueError("subset test contains rows absent from the full test split")
    for wrong_id, row in existing_by_id.items():
        if row["case"]["question"] != full_by_id[wrong_id].question:
            raise ValueError(f"question mismatch for existing test id {wrong_id}")
    remaining = [case for case in full_test if case.wrong_id not in existing_by_id]

    tracker = TokenUsageTracker()
    enable_thinking = bool(
        experiment.get("config", {}).get("vllm_enable_thinking", False)
    )
    models = experiment["models"]
    generator = GeneratorAgent(
        build_engine(
            models["generator"],
            vllm_base_url=args.generator_vllm_base_url,
            vllm_api_key=args.vllm_api_key,
            vllm_enable_thinking=enable_thinking,
            token_usage_tracker=tracker,
            usage_role="generator",
        ),
        tg.Variable(
            experiment["current_generator_strategy_prompt"],
            requires_grad=False,
            role_description="frozen final Generator strategy",
        ),
        tg.Variable(
            experiment["generator_fixed_system_prompt"],
            requires_grad=False,
            role_description="frozen Generator system prompt",
        ),
    )
    verifier = VerifierAgent(
        build_engine(
            models["verifier"],
            vllm_base_url=args.verifier_vllm_base_url,
            vllm_api_key=args.vllm_api_key,
            vllm_enable_thinking=enable_thinking,
            token_usage_tracker=tracker,
            usage_role="verifier",
        ),
        tg.Variable(
            experiment["current_verifier_strategy_prompt"],
            requires_grad=False,
            role_description="frozen final Verifier strategy",
        ),
        tg.Variable(
            experiment["verifier_fixed_system_prompt"],
            requires_grad=False,
            role_description="frozen Verifier system prompt",
        ),
    )

    def evaluate_one(case: HardCase) -> Dict[str, Any]:
        question = tg.Variable(
            case.question,
            requires_grad=False,
            role_description="held-out full-test GSM8K question",
        )
        generated = generator.run(question).value
        verdict_text = verifier.run(
            question,
            tg.Variable(
                generated,
                requires_grad=False,
                role_description="detached full-test Generator trajectory",
            ),
        ).value
        return {
            "case": asdict(case),
            "generator_output": generated,
            "verifier_output": verdict_text,
            "verdict": asdict(parse_verdict(verdict_text)),
            "correct": is_correct(generated, case.ground_truth),
            "skipped": False,
            "error": None,
        }

    print(
        f"full_test={len(full_test)} existing={len(existing_rows)} "
        f"remaining={len(remaining)} workers={args.workers}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        new_rows = list(executor.map(evaluate_one, remaining))
    combined_by_id = {**existing_by_id, **{row["case"]["wrong_id"]: row for row in new_rows}}
    combined_rows = [combined_by_id[case.wrong_id] for case in full_test]
    if len(combined_rows) != len(full_test):
        raise AssertionError("combined full-test result is incomplete")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(experiment_path.resolve()),
        "source_experiment_completed_at_utc": experiment.get("completed_at_utc"),
        "models": {
            "generator": models["generator"],
            "verifier": models["verifier"],
        },
        "prompts": {
            "generator_strategy": experiment["current_generator_strategy_prompt"],
            "verifier_strategy": experiment["current_verifier_strategy_prompt"],
        },
        "existing_test_ids": sorted(existing_by_id),
        "new_test_ids": [case.wrong_id for case in remaining],
        "new_evaluation": summarize(new_rows),
        "combined_full_test": summarize(combined_rows),
        "new_call_token_usage": tracker.snapshot(),
        "complete": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        f"remaining_accuracy={result['new_evaluation']['accuracy']:.3f} "
        f"combined_full_test_accuracy="
        f"{result['combined_full_test']['accuracy']:.3f}",
        flush=True,
    )
    print(f"complete: {output}", flush=True)


if __name__ == "__main__":
    main()
