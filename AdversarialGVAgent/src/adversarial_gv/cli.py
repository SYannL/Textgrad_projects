"""Command-line entry point."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import textgrad as tg
from dotenv import load_dotenv

from .agents import GeneratorAgent, VerifierAgent
from .data import DATASET_CHOICES, case_from_dataset, load_textgrad_dataset
from .evaluation import is_correct
from .engines import build_engine, resolve_role_base_urls
from .gradient_reporting import append_gradient_csv
from .prompts import (
    GENERATOR_FIXED_SYSTEM_PROMPT,
    GENERATOR_STRATEGY_PROMPT,
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
    VERIFIER_FIXED_SYSTEM_PROMPT,
    VERIFIER_STRATEGY_PROMPT,
)
from .reporting import append_result_csv
from .recording import RecordingEngine
from .trainer import (
    GENERATOR_SUPERVISION_MODES,
    AdversarialGVTrainer,
    TrainingConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize Generator and Verifier prompts on one reasoning case."
    )
    parser.add_argument("--generator-model", default="gpt-4o-mini")
    parser.add_argument("--verifier-model", default="gpt-4o-mini")
    parser.add_argument("--backward-model", default="gpt-4o-mini")
    parser.add_argument(
        "--vllm-base-url",
        default=os.getenv("VLLM_BASE_URL"),
        help=(
            "OpenAI-compatible vLLM endpoint, for example http://localhost:8000/v1. "
            "Can also be set with VLLM_BASE_URL."
        ),
    )
    parser.add_argument(
        "--generator-vllm-base-url",
        default=os.getenv("GENERATOR_VLLM_BASE_URL"),
        help="Generator vLLM endpoint; overrides --vllm-base-url.",
    )
    parser.add_argument(
        "--verifier-vllm-base-url",
        default=os.getenv("VERIFIER_VLLM_BASE_URL"),
        help="Verifier vLLM endpoint; overrides --vllm-base-url.",
    )
    parser.add_argument(
        "--backward-vllm-base-url",
        default=os.getenv("BACKWARD_VLLM_BASE_URL"),
        help=(
            "TextGrad backward, CoT labeler, and optimizer endpoint; "
            "overrides --vllm-base-url."
        ),
    )
    parser.add_argument(
        "--vllm-api-key",
        default=os.getenv("VLLM_API_KEY"),
        help=(
            "API key for the vLLM endpoint; defaults to EMPTY when omitted. "
            "Can also be set with VLLM_API_KEY."
        ),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="bbh_object_counting")
    parser.add_argument(
        "--generator-supervision-mode",
        choices=GENERATOR_SUPERVISION_MODES,
        default="final_answer",
        help=(
            "Use only the final answer as Generator supervision, or include "
            "GSM8K gold reasoning as a training-only expected trajectory."
        ),
    )
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument(
        "--require-initial-wrong",
        action="store_true",
        help="Search forward until the initial Generator answer is incorrect.",
    )
    parser.add_argument(
        "--search-cases",
        type=int,
        default=20,
        help="Maximum candidates to probe when --require-initial-wrong is set.",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--run-check",
        action="store_true",
        help="Evaluate an untouched validation case before/after updates; no gradients.",
    )
    parser.add_argument("--check-case-index", type=int, default=0)
    parser.add_argument("--output-dir", default="runs")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    role_base_urls = resolve_role_base_urls(
        shared=args.vllm_base_url,
        generator=args.generator_vllm_base_url,
        verifier=args.verifier_vllm_base_url,
        backward=args.backward_vllm_base_url,
    )
    generator_engine = build_engine(
        args.generator_model, role_base_urls["generator"], args.vllm_api_key
    )
    verifier_engine = build_engine(
        args.verifier_model, role_base_urls["verifier"], args.vllm_api_key
    )
    backward_engine = RecordingEngine(
        build_engine(
            args.backward_model,
            role_base_urls["backward"],
            args.vllm_api_key,
        )
    )

    if args.search_cases < 1:
        parser.error("--search-cases must be at least 1")

    generator_fixed_text, generator_strategy_text = (
        (
            GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
            GSM8K_GENERATOR_STRATEGY_PROMPT,
        )
        if args.dataset == "gsm8k"
        else (GENERATOR_FIXED_SYSTEM_PROMPT, GENERATOR_STRATEGY_PROMPT)
    )
    generator_fixed = tg.Variable(
        generator_fixed_text,
        requires_grad=False,
        role_description="immutable Generator role, rules, and output format",
    )
    generator_strategy = tg.Variable(
        generator_strategy_text,
        requires_grad=True,
        role_description="trainable Generator problem-solving strategy",
    )
    verifier_fixed = tg.Variable(
        VERIFIER_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="immutable Verifier role, audit rules, and output format",
    )
    verifier_strategy = tg.Variable(
        VERIFIER_STRATEGY_PROMPT,
        requires_grad=True,
        role_description="trainable Verifier audit strategy",
    )
    trainer = AdversarialGVTrainer(
        GeneratorAgent(generator_engine, generator_strategy, generator_fixed),
        VerifierAgent(verifier_engine, verifier_strategy, verifier_fixed),
        backward_engine,
        TrainingConfig(
            iterations=args.iterations,
            run_check=args.run_check,
            generator_supervision_mode=args.generator_supervision_mode,
        ),
    )
    train_dataset = load_textgrad_dataset(
        args.dataset, split="train", root=args.dataset_root
    )
    selection = []
    train_case = None
    search_count = args.search_cases if args.require_initial_wrong else 1
    for index in range(args.case_index, args.case_index + search_count):
        candidate_case = case_from_dataset(train_dataset, args.dataset, index, "train")
        if args.require_initial_wrong:
            question = tg.Variable(
                candidate_case.question,
                requires_grad=False,
                role_description="candidate multi-step reasoning question",
            )
            initial_answer = trainer.generator.run(question).value
            correct = is_correct(initial_answer, candidate_case.answer)
            selection.append(
                {
                    "index": index,
                    "correct": correct,
                    "answer": initial_answer,
                    "ground_truth": candidate_case.answer,
                }
            )
            print(f"Probe case {index}: initial_correct={correct}")
            if correct:
                continue
        train_case = candidate_case
        break
    if train_case is None:
        raise RuntimeError(
            f"No initially incorrect case found in {args.search_cases} candidates "
            f"starting at index {args.case_index}. Increase --search-cases or change --case-index."
        )

    check_case = None
    if args.run_check:
        check_dataset = load_textgrad_dataset(
            args.dataset, split="val", root=args.dataset_root
        )
        check_case = case_from_dataset(
            check_dataset, args.dataset, args.check_case_index, "val"
        )
    result = trainer.train(train_case, check_case)
    result["selection"] = selection
    result["models"] = {
        "generator": args.generator_model,
        "verifier": args.verifier_model,
        "backward": args.backward_model,
    }
    result["backend"] = {
        "type": (
            "role-specific-vllm"
            if any(role_base_urls.values())
            else "textgrad-default"
        ),
        "base_urls": role_base_urls,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(timezone.utc)
    timestamp = recorded_at.strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"run_{timestamp}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    csv_path = output_dir / "results.csv"
    append_result_csv(
        result,
        csv_path,
        run_id=timestamp,
        recorded_at=recorded_at.isoformat(),
    )
    gradient_csv_path = output_dir / "gradient_traces.csv"
    append_gradient_csv(
        result,
        gradient_csv_path,
        run_id=timestamp,
        recorded_at=recorded_at.isoformat(),
    )

    final = result["final"]
    print(f"Run saved to: {output_path}")
    print(f"CSV appended to: {csv_path}")
    print(f"Gradient CSV appended to: {gradient_csv_path}")
    print(f"Final correct: {final['correct']}")
    print(f"Final verdict: {final['verdict']['label']}")
    print(f"Final answer:\n{final['answer']}")


if __name__ == "__main__":
    main()
