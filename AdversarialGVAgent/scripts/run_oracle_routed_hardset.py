#!/usr/bin/env python
"""Run GSM8K G/V prompt training with official-reasoning oracle routing."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence

import textgrad as tg
from dotenv import load_dotenv

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.engines import (
    TokenUsageTracker,
    build_engine,
    merge_token_usage,
    resolve_role_base_urls,
)
from adversarial_gv.gsm8k_compat import GSM8KDSPyCompat
from adversarial_gv.oracle_routing import OracleHardCase, OracleRoutedGVTrainer
from adversarial_gv.prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
    VERIFIER_FIXED_SYSTEM_PROMPT,
    VERIFIER_STRATEGY_PROMPT,
)
from adversarial_gv.recording import RecordingEngine
from adversarial_gv.wandb_monitoring import init_wandb_monitor


PROMPT_ARCHITECTURE = (
    "oracle-routed-fixed-system-plus-strategy-multiround-validation-v3"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Train G/V prompts with two isolated oracle judgments grounded in "
            "GSM8K's official reasoning."
        )
    )
    result.add_argument(
        "--data",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    result.add_argument("--output-dir", default="runs/qwen4b_qwen32b_oracle_routed")
    result.add_argument("--dataset-root", default="data/gsm8k")
    result.add_argument("--batch-size", type=int, default=6)
    result.add_argument(
        "--iterations",
        type=int,
        default=3,
        help=(
            "Number of consecutive oracle-routed G/V prompt-update rounds on "
            "each fixed batch."
        ),
    )
    result.add_argument(
        "--validation-mode",
        choices=("acc", "oracle"),
        default="acc",
        help=(
            "Strict candidate-prompt gate after every iteration: final-answer "
            "accuracy, or privileged G/V oracle scores."
        ),
    )
    result.add_argument("--generator-model", default="qwen4b-api")
    result.add_argument("--verifier-model", default="qwen32b-api")
    result.add_argument("--backward-model", default="qwen32b-api")
    result.add_argument(
        "--oracle-model",
        default=None,
        help=(
            "Independent oracle-judge model. Defaults to --backward-model; set "
            "this explicitly when a separate judge model is available."
        ),
    )
    result.add_argument(
        "--vllm-base-url",
        default=os.getenv("VLLM_BASE_URL"),
        help="Legacy shared vLLM endpoint used as a fallback for every role.",
    )
    result.add_argument(
        "--generator-vllm-base-url",
        default=os.getenv("GENERATOR_VLLM_BASE_URL"),
    )
    result.add_argument(
        "--verifier-vllm-base-url",
        default=os.getenv("VERIFIER_VLLM_BASE_URL"),
    )
    result.add_argument(
        "--backward-vllm-base-url",
        default=os.getenv("BACKWARD_VLLM_BASE_URL"),
    )
    result.add_argument(
        "--oracle-vllm-base-url",
        default=os.getenv("ORACLE_VLLM_BASE_URL"),
        help="Oracle endpoint; defaults to the resolved backward endpoint.",
    )
    result.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY"))
    result.add_argument("--vllm-enable-thinking", action="store_true")
    result.add_argument("--evaluation-workers", type=int, default=8)
    result.add_argument("--training-workers", type=int, default=1)
    result.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", "siyann"))
    result.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "GANAgent"))
    result.add_argument("--wandb-name", default=None)
    result.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.getenv("WANDB_MODE", "online"),
    )
    return result


def _normalized_question(value: str) -> str:
    value = value.strip()
    return value.removeprefix("Question:").strip()


def load_oracle_cases(path: Path, dataset_root: str | None = None):
    """Recover official reasoning using each CSV row's source split/index."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    datasets = {}
    cases = []
    for row in rows:
        source_split = row["split"]
        if source_split not in datasets:
            datasets[source_split] = GSM8KDSPyCompat(
                split=source_split,
                root=dataset_root,
            )
        source_index = int(row["index"])
        source_question, source_answer, gold_reasoning = datasets[source_split][
            source_index
        ]
        if _normalized_question(source_question) != _normalized_question(row["question"]):
            raise ValueError(
                "CSV question does not match GSM8K source row "
                f"{source_split}[{source_index}]"
            )
        if str(source_answer) != str(row["ground_truth"]):
            raise ValueError(
                "CSV answer does not match GSM8K source row "
                f"{source_split}[{source_index}]"
            )
        if not str(gold_reasoning).strip():
            raise ValueError(
                f"missing official reasoning for {source_split}[{source_index}]"
            )
        cases.append(
            OracleHardCase(
                wrong_id=int(row["wrong_id"]),
                collection_split=row["collection_split"],
                source_split=source_split,
                source_index=source_index,
                question=row["question"],
                ground_truth=row["ground_truth"],
                gold_reasoning=str(gold_reasoning),
            )
        )
    splits = tuple(
        [case for case in cases if case.collection_split == name]
        for name in ("train", "val", "test")
    )
    if any(not split for split in splits):
        raise ValueError(
            "expected non-empty train, val, and test collection splits, got "
            + ", ".join(str(len(split)) for split in splits)
        )
    return splits


def save_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def export_metrics(path: Path, experiment: Dict) -> None:
    rows = []
    for item in experiment["evaluations"]:
        row = {"stage": item["stage"], "batch_index": item["batch_index"]}
        for split in ("train", "val", "test"):
            metrics = item[split]
            for name in (
                "accuracy",
                "accept_rate",
                "challenge_rate",
                "reject_rate",
                "invalid_rate",
                "evaluated_count",
                "skipped_count",
            ):
                row[f"{split}_{name}"] = metrics.get(name, 0)
        rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_routes(path: Path, experiment: Dict) -> None:
    fields = [
        "batch_index",
        "iteration_index",
        "wrong_id",
        "route",
        "validation_status",
        "validation_accepted",
        "generator_label",
        "generator_correct",
        "generator_rationale",
        "verifier_correct",
        "verifier_rationale",
        "source_split",
        "source_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in experiment["batches"]:
            for iteration in batch["iterations"]:
                for item in iteration["interactions"]:
                    oracle = item["oracle"]
                    case = item["case"]
                    writer.writerow(
                        {
                            "batch_index": batch["batch_index"],
                            "iteration_index": iteration["iteration_index"],
                            "wrong_id": item["wrong_id"],
                            "route": item["route"],
                            "validation_status": iteration.get(
                                "validation_gate", {}
                            ).get("status"),
                            "validation_accepted": iteration.get(
                                "validation_gate", {}
                            ).get("accepted"),
                            "generator_label": oracle["generator_label"],
                            "generator_correct": oracle["generator_label"] == "ACCEPT",
                            "generator_rationale": oracle["generator_rationale"],
                            "verifier_correct": oracle["verifier_correct"],
                            "verifier_rationale": oracle["verifier_rationale"],
                            "source_split": case["source_split"],
                            "source_index": case["source_index"],
                        }
                    )


def export_gradients(path: Path, experiment: Dict) -> None:
    fields = [
        "batch_index",
        "iteration_index",
        "call_index",
        "kind",
        "system_prompt",
        "gradient_prompt",
        "response",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in experiment["batches"]:
            for iteration in batch["iterations"]:
                for index, call in enumerate(
                    iteration["gradient_trace"], start=1
                ):
                    writer.writerow(
                        {
                            "batch_index": batch["batch_index"],
                            "iteration_index": iteration["iteration_index"],
                            "call_index": index,
                            "kind": call["kind"],
                            "system_prompt": call["system_prompt"],
                            "gradient_prompt": call["prompt"],
                            "response": call["response"],
                        }
                    )


def summarize_oracle_routes(experiment: Dict) -> Dict:
    totals = {name: 0 for name in ("none", "generator", "verifier", "both")}
    generator_correct = 0
    verifier_correct = 0
    interactions = 0
    for batch in experiment["batches"]:
        for iteration in batch["iterations"]:
            for item in iteration["interactions"]:
                interactions += 1
                totals[item["route"]] += 1
                generator_correct += item["oracle"]["generator_label"] == "ACCEPT"
                verifier_correct += bool(item["oracle"]["verifier_correct"])
    return {
        "interaction_count": interactions,
        "route_counts": totals,
        "generator_oracle_accuracy": (
            generator_correct / interactions if interactions else 0.0
        ),
        "verifier_oracle_accuracy": (
            verifier_correct / interactions if interactions else 0.0
        ),
    }


def build_professor_diagnostics(experiment: Dict) -> Dict:
    """Summarize per-question correctness changes across batch-local rounds."""
    cases = []
    for batch in experiment["batches"]:
        history_by_id = {}
        for iteration in batch["iterations"]:
            round_index = iteration["iteration_index"]
            for item in iteration["interactions"]:
                oracle = item["oracle"]
                history_by_id.setdefault(item["wrong_id"], []).append(
                    {
                        "iteration_index": round_index,
                        "generator_label": oracle["generator_label"],
                        "generator_correct": oracle["generator_label"] == "ACCEPT",
                        "verifier_correct": bool(oracle["verifier_correct"]),
                        "route": item["route"],
                        "validation_status": iteration.get(
                            "validation_gate", {}
                        ).get("status"),
                        "validation_accepted": iteration.get(
                            "validation_gate", {}
                        ).get("accepted"),
                    }
                )
        for wrong_id in batch["case_ids"]:
            rounds = history_by_id.get(wrong_id, [])
            first_g = next(
                (
                    row["iteration_index"]
                    for row in rounds
                    if row["generator_correct"]
                ),
                None,
            )
            first_v = next(
                (
                    row["iteration_index"]
                    for row in rounds
                    if row["verifier_correct"]
                ),
                None,
            )
            g_regressions = []
            v_regressions = []
            for previous, current in zip(rounds, rounds[1:]):
                if previous["generator_correct"] and not current["generator_correct"]:
                    g_regressions.append(current["iteration_index"])
                if previous["verifier_correct"] and not current["verifier_correct"]:
                    v_regressions.append(current["iteration_index"])
            g_ever = first_g is not None
            v_ever = first_v is not None
            g_initial = rounds[0]["generator_correct"] if rounds else None
            if g_ever and v_ever:
                outcome_bucket = "generator_ever_correct_verifier_ever_correct"
            elif g_ever:
                outcome_bucket = "generator_ever_correct_verifier_never_correct"
            elif v_ever:
                outcome_bucket = "generator_never_correct_verifier_ever_correct"
            else:
                outcome_bucket = "generator_never_correct_verifier_never_correct"
            cases.append(
                {
                    "batch_index": batch["batch_index"],
                    "wrong_id": wrong_id,
                    "rounds_recorded": len(rounds),
                    "first_generator_correct_round": first_g,
                    "first_verifier_correct_round": first_v,
                    "generator_ever_correct": g_ever,
                    "verifier_ever_correct": v_ever,
                    "generator_initial_correct": g_initial,
                    "generator_final_correct": (
                        rounds[-1]["generator_correct"] if rounds else None
                    ),
                    "verifier_final_correct": (
                        rounds[-1]["verifier_correct"] if rounds else None
                    ),
                    "generator_regression_rounds": g_regressions,
                    "verifier_regression_rounds": v_regressions,
                    "generator_eventually_correct_verifier_never_correct": (
                        g_ever and not v_ever
                    ),
                    "initially_wrong_generator_eventually_correct_verifier_never_correct": (
                        g_initial is False and g_ever and not v_ever
                    ),
                    "outcome_bucket": outcome_bucket,
                    "rounds": rounds,
                }
            )
    completed = [case for case in cases if case["rounds_recorded"] > 0]
    target = [
        case
        for case in completed
        if case["generator_eventually_correct_verifier_never_correct"]
    ]
    professor_target = [
        case
        for case in completed
        if case[
            "initially_wrong_generator_eventually_correct_verifier_never_correct"
        ]
    ]
    return {
        "case_count": len(cases),
        "cases_with_observations": len(completed),
        "generator_eventually_correct_count": sum(
            case["generator_ever_correct"] for case in completed
        ),
        "verifier_never_correct_count": sum(
            not case["verifier_ever_correct"] for case in completed
        ),
        "generator_eventually_correct_verifier_never_correct_count": len(target),
        "generator_eventually_correct_verifier_never_correct_ids": [
            case["wrong_id"] for case in target
        ],
        "professor_target_count": len(professor_target),
        "professor_target_ids": [case["wrong_id"] for case in professor_target],
        "cases": cases,
    }


def _transition(previous: bool | None, current: bool) -> str:
    if previous is None:
        return "initial_correct" if current else "initial_wrong"
    if previous == current:
        return "stayed_correct" if current else "stayed_wrong"
    return "wrong_to_correct" if current else "correct_to_wrong"


def validation_scores(mode: str, snapshot: Dict) -> Dict[str, float]:
    if mode == "acc":
        return {"accuracy": snapshot["accuracy"]}
    if mode == "oracle":
        return {
            "generator_accuracy": snapshot["generator_accuracy"],
            "verifier_accuracy": snapshot["verifier_accuracy"],
        }
    raise ValueError(f"unsupported validation mode: {mode}")


def validation_candidate_is_better(
    mode: str,
    baseline: Dict,
    candidate: Dict,
) -> bool:
    """Strict TextGrad gate for acc; Pareto extension for compound G/V."""
    old = validation_scores(mode, baseline)
    new = validation_scores(mode, candidate)
    if mode == "acc":
        return new["accuracy"] > old["accuracy"]
    no_regression = all(new[name] >= old[name] for name in old)
    strict_gain = any(new[name] > old[name] for name in old)
    return no_regression and strict_gain


def export_round_transitions(path: Path, diagnostics: Dict) -> None:
    fields = [
        "batch_index",
        "wrong_id",
        "iteration_index",
        "generator_label",
        "generator_correct",
        "generator_transition",
        "verifier_correct",
        "verifier_transition",
        "route",
        "validation_status",
        "validation_accepted",
        "first_generator_correct_round",
        "first_verifier_correct_round",
        "generator_eventually_correct_verifier_never_correct",
        "initially_wrong_generator_eventually_correct_verifier_never_correct",
        "outcome_bucket",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in diagnostics["cases"]:
            previous_g = None
            previous_v = None
            for row in case["rounds"]:
                writer.writerow(
                    {
                        "batch_index": case["batch_index"],
                        "wrong_id": case["wrong_id"],
                        "iteration_index": row["iteration_index"],
                        "generator_label": row["generator_label"],
                        "generator_correct": row["generator_correct"],
                        "generator_transition": _transition(
                            previous_g, row["generator_correct"]
                        ),
                        "verifier_correct": row["verifier_correct"],
                        "verifier_transition": _transition(
                            previous_v, row["verifier_correct"]
                        ),
                        "route": row["route"],
                        "validation_status": row["validation_status"],
                        "validation_accepted": row["validation_accepted"],
                        "first_generator_correct_round": case[
                            "first_generator_correct_round"
                        ],
                        "first_verifier_correct_round": case[
                            "first_verifier_correct_round"
                        ],
                        "generator_eventually_correct_verifier_never_correct": case[
                            "generator_eventually_correct_verifier_never_correct"
                        ],
                        "initially_wrong_generator_eventually_correct_verifier_never_correct": case[
                            "initially_wrong_generator_eventually_correct_verifier_never_correct"
                        ],
                        "outcome_bucket": case["outcome_bucket"],
                    }
                )
                previous_g = row["generator_correct"]
                previous_v = row["verifier_correct"]


def _new_experiment(args, role_urls, oracle_model, oracle_url, sizes, total_batches):
    train_size, val_size, test_size = sizes
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_architecture": PROMPT_ARCHITECTURE,
        "models": {
            "generator": args.generator_model,
            "verifier": args.verifier_model,
            "backward": args.backward_model,
            "oracle": oracle_model,
        },
        "config": {
            "epochs": 1,
            "total_batches": total_batches,
            "batch_size": args.batch_size,
            "iterations_per_batch": args.iterations,
            "validation_mode": args.validation_mode,
            "validation_acceptance": "strict_improvement",
            "validation_rollback_scope": "joint_generator_and_verifier",
            "evaluation_workers": args.evaluation_workers,
            "training_workers": args.training_workers,
            "vllm_base_urls": {**role_urls, "oracle": oracle_url},
            "vllm_enable_thinking": args.vllm_enable_thinking,
            "oracle_calls_per_training_case": 2,
            "oracle_evaluation_scope": "training_interactions_only",
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
        },
        "generator_fixed_system_prompt": GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        "verifier_fixed_system_prompt": VERIFIER_FIXED_SYSTEM_PROMPT,
        "initial_generator_strategy_prompt": GSM8K_GENERATOR_STRATEGY_PROMPT,
        "initial_verifier_strategy_prompt": VERIFIER_STRATEGY_PROMPT,
        "current_generator_strategy_prompt": GSM8K_GENERATOR_STRATEGY_PROMPT,
        "current_verifier_strategy_prompt": VERIFIER_STRATEGY_PROMPT,
        "evaluations": [],
        "batches": [],
        "complete": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    args = parser().parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.evaluation_workers < 1 or args.training_workers < 1:
        raise ValueError("worker counts must be positive")

    role_urls = resolve_role_base_urls(
        shared=args.vllm_base_url,
        generator=args.generator_vllm_base_url,
        verifier=args.verifier_vllm_base_url,
        backward=args.backward_vllm_base_url,
    )
    oracle_model = args.oracle_model or args.backward_model
    oracle_url = (
        args.oracle_vllm_base_url.rstrip("/")
        if args.oracle_vllm_base_url
        else role_urls["backward"]
    )
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    train, val, test = load_oracle_cases(Path(args.data), str(dataset_root))
    batches = [
        train[index : index + args.batch_size]
        for index in range(0, len(train), args.batch_size)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "experiment.json"

    expected_models = {
        "generator": args.generator_model,
        "verifier": args.verifier_model,
        "backward": args.backward_model,
        "oracle": oracle_model,
    }
    expected_urls = {**role_urls, "oracle": oracle_url}
    if checkpoint.exists():
        experiment = json.loads(checkpoint.read_text(encoding="utf-8"))
        if experiment.get("prompt_architecture") != PROMPT_ARCHITECTURE:
            raise ValueError("checkpoint belongs to another prompt architecture")
        if experiment.get("models") != expected_models:
            raise ValueError("checkpoint model roles differ from this command")
        if experiment.get("config", {}).get("vllm_base_urls") != expected_urls:
            raise ValueError("checkpoint role endpoints differ from this command")
        if experiment.get("config", {}).get("batch_size") != args.batch_size:
            raise ValueError("checkpoint batch size differs from this command")
        if (
            experiment.get("config", {}).get("iterations_per_batch")
            != args.iterations
        ):
            raise ValueError("checkpoint iterations differ from this command")
        if (
            experiment.get("config", {}).get("validation_mode")
            != args.validation_mode
        ):
            raise ValueError("checkpoint validation mode differs from this command")
        if (
            experiment.get("generator_fixed_system_prompt")
            != GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT
            or experiment.get("verifier_fixed_system_prompt")
            != VERIFIER_FIXED_SYSTEM_PROMPT
        ):
            raise ValueError("checkpoint immutable system prompts differ")
        g_strategy_value = experiment["current_generator_strategy_prompt"]
        v_strategy_value = experiment["current_verifier_strategy_prompt"]
    else:
        experiment = _new_experiment(
            args,
            role_urls,
            oracle_model,
            oracle_url,
            (len(train), len(val), len(test)),
            len(batches),
        )
        g_strategy_value = GSM8K_GENERATOR_STRATEGY_PROMPT
        v_strategy_value = VERIFIER_STRATEGY_PROMPT

    g_fixed = tg.Variable(
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="immutable Generator role, rules, and output format",
    )
    v_fixed = tg.Variable(
        VERIFIER_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="immutable Verifier role, audit rules, and output format",
    )
    g_strategy = tg.Variable(
        g_strategy_value,
        requires_grad=True,
        role_description="trainable GSM8K Generator problem-solving strategy",
    )
    v_strategy = tg.Variable(
        v_strategy_value,
        requires_grad=True,
        role_description="trainable Verifier trajectory-audit strategy",
    )
    tracker = TokenUsageTracker()
    persisted_usage = experiment.get("token_usage", {})

    def engine(model, url, role):
        return build_engine(
            model,
            vllm_base_url=url,
            vllm_api_key=args.vllm_api_key,
            vllm_enable_thinking=args.vllm_enable_thinking,
            token_usage_tracker=tracker,
            usage_role=role,
        )

    backward = RecordingEngine(
        engine(args.backward_model, role_urls["backward"], "backward")
    )
    trainer = OracleRoutedGVTrainer(
        GeneratorAgent(
            engine(args.generator_model, role_urls["generator"], "generator"),
            g_strategy,
            g_fixed,
        ),
        VerifierAgent(
            engine(args.verifier_model, role_urls["verifier"], "verifier"),
            v_strategy,
            v_fixed,
        ),
        backward,
        oracle_engine=engine(oracle_model, oracle_url, "oracle"),
        evaluation_workers=args.evaluation_workers,
        training_workers=args.training_workers,
    )

    def token_usage():
        return merge_token_usage(persisted_usage, tracker.snapshot())

    existing_wandb = experiment.get("wandb", {})
    wandb_name = existing_wandb.get("name", args.wandb_name or output_dir.name)
    monitor = init_wandb_monitor(
        entity=existing_wandb.get("entity", args.wandb_entity),
        project=existing_wandb.get("project", args.wandb_project),
        name=wandb_name,
        mode=args.wandb_mode,
        output_dir=output_dir,
        run_id=existing_wandb.get("run_id"),
        config={
            "architecture": "oracle-routed GAN-like Generator-Verifier TextGrad",
            "dataset": "GSM8K balanced 1:1",
            "data_path": str(Path(args.data).resolve()),
            **experiment["models"],
            **experiment["config"],
        },
    )
    if monitor is not None and not existing_wandb.get("run_id"):
        experiment["wandb"] = {
            "run_id": monitor.id,
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": wandb_name,
            "url": monitor.url,
        }
        save_json(checkpoint, experiment)

    def evaluate(stage: str, batch_index: int):
        result = {
            "stage": stage,
            "batch_index": batch_index,
            "train": trainer.evaluate(train),
            "val": trainer.evaluate(val),
            "test": trainer.evaluate(test),
        }
        experiment["evaluations"].append(result)
        experiment["token_usage"] = token_usage()
        save_json(checkpoint, experiment)
        if monitor is not None:
            monitor.log_evaluation(
                result,
                experiment["token_usage"],
                len(g_strategy.value),
                len(v_strategy.value),
            )
        print(
            f"{stage} batch={batch_index} "
            f"train={result['train']['accuracy']:.3f} "
            f"val={result['val']['accuracy']:.3f} "
            f"test={result['test']['accuracy']:.3f}",
            flush=True,
        )

    if not experiment["evaluations"]:
        evaluate("initial", 0)

    def validation_snapshot():
        if args.validation_mode == "acc":
            return trainer.evaluate_generator_accuracy(val)
        return trainer.evaluate_oracle_validation(val)

    validation_gate = experiment.setdefault(
        "validation_gate",
        {
            "mode": args.validation_mode,
            "acceptance": (
                "strict_accuracy_improvement"
                if args.validation_mode == "acc"
                else "pareto_no_regression_and_one_strict_improvement"
            ),
            "rollback_scope": "joint_generator_and_verifier",
            "initial": None,
            "current": None,
        },
    )
    if validation_gate.get("current") is None:
        print(
            f"validation gate baseline start mode={args.validation_mode}",
            flush=True,
        )
        initial_validation = validation_snapshot()
        validation_gate["initial"] = initial_validation
        validation_gate["current"] = initial_validation
        experiment["token_usage"] = token_usage()
        save_json(checkpoint, experiment)
        print(
            "validation gate baseline scores="
            f"{validation_scores(args.validation_mode, initial_validation)}",
            flush=True,
        )

    def save_training_progress():
        diagnostics = build_professor_diagnostics(experiment)
        experiment["oracle_training_summary"] = summarize_oracle_routes(experiment)
        experiment["professor_diagnostics"] = diagnostics
        experiment["current_generator_strategy_prompt"] = g_strategy.value
        experiment["current_verifier_strategy_prompt"] = v_strategy.value
        experiment["token_usage"] = token_usage()
        save_json(checkpoint, experiment)
        export_routes(output_dir / "oracle_routes.csv", experiment)
        export_round_transitions(
            output_dir / "oracle_round_transitions.csv",
            diagnostics,
        )
        export_gradients(output_dir / "gradient_traces.csv", experiment)

    for batch_index, batch in enumerate(batches, start=1):
        if len(experiment["batches"]) < batch_index:
            experiment["batches"].append(
                {
                    "batch_index": batch_index,
                    "case_ids": [case.wrong_id for case in batch],
                    "iterations": [],
                }
            )
            save_training_progress()
        batch_record = experiment["batches"][batch_index - 1]
        if batch_record["case_ids"] != [case.wrong_id for case in batch]:
            raise ValueError(
                f"checkpoint cases differ for batch {batch_index}"
            )
        print(
            f"starting batch={batch_index}/{len(batches)} "
            f"cases={[case.wrong_id for case in batch]} "
            f"completed_iterations={len(batch_record['iterations'])}/"
            f"{args.iterations}",
            flush=True,
        )
        for iteration_index in range(
            len(batch_record["iterations"]) + 1,
            args.iterations + 1,
        ):
            print(
                f"starting batch={batch_index}/{len(batches)} "
                f"iteration={iteration_index}/{args.iterations}",
                flush=True,
            )
            record = trainer.train_batch(
                batch,
                batch_index,
                iteration_index=iteration_index,
            )
            print(
                f"validation gate candidate start mode={args.validation_mode}",
                flush=True,
            )
            candidate_validation = validation_snapshot()
            baseline_validation = validation_gate["current"]
            accepted = validation_candidate_is_better(
                args.validation_mode,
                baseline_validation,
                candidate_validation,
            )
            if accepted:
                validation_gate["current"] = candidate_validation
                gate_status = "accepted"
            else:
                g_strategy.set_value(record["generator_strategy_prompt_before"])
                v_strategy.set_value(record["verifier_strategy_prompt_before"])
                gate_status = "rejected_rolled_back"
            record["validation_gate"] = {
                "mode": args.validation_mode,
                "status": gate_status,
                "accepted": accepted,
                "baseline_scores": validation_scores(
                    args.validation_mode, baseline_validation
                ),
                "candidate_scores": validation_scores(
                    args.validation_mode, candidate_validation
                ),
                "candidate_evaluation": candidate_validation,
                "generator_strategy_prompt_after_gate": g_strategy.value,
                "verifier_strategy_prompt_after_gate": v_strategy.value,
            }
            batch_record["iterations"].append(record)
            save_training_progress()
            if monitor is not None:
                monitor.log_tokens(
                    batch_index,
                    f"training_iteration_{iteration_index}_complete",
                    experiment["token_usage"],
                )
            print(
                f"batch={batch_index} iteration={iteration_index} "
                f"routes={record['route_counts']} validation={gate_status} "
                f"scores={validation_scores(args.validation_mode, candidate_validation)}",
                flush=True,
            )
        evaluated_batches = {
            item["batch_index"]
            for item in experiment["evaluations"]
            if item["stage"] == "after_batch"
        }
        if batch_index not in evaluated_batches:
            evaluate("after_batch", batch_index)
        export_metrics(output_dir / "metrics.csv", experiment)

    experiment["complete"] = True
    experiment["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_training_progress()
    export_metrics(output_dir / "metrics.csv", experiment)
    if monitor is not None:
        monitor.finish()
    print(f"complete: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
