#!/usr/bin/env python
"""Label every project GSM8K sample with one fixed Generator answer."""

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
from typing import Dict, Sequence

import textgrad as tg
from dotenv import load_dotenv
from textgrad.tasks.big_bench_hard import parse_integer_answer

from adversarial_gv.agents import GeneratorAgent
from adversarial_gv.data import case_from_dataset, load_textgrad_dataset
from adversarial_gv.engines import build_engine
from adversarial_gv.evaluation import is_correct
from adversarial_gv.prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
)


DEFAULT_SPLITS = ("train", "val", "test")
CSV_FIELDS = [
    "sample_id", "label", "correct", "split", "index", "question",
    "ground_truth", "parsed_prediction", "model", "generator_prompt",
    "generator_output",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Traverse GSM8K and label every question correct/incorrect from one "
            "fixed Generator call."
        )
    )
    parser.add_argument("--model", default="qwen32b-api")
    parser.add_argument("--vllm-base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--vllm-api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument(
        "--vllm-enable-thinking", action="store_true",
        help="Enable Qwen3 thinking; disabled by default.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--splits", nargs="+", choices=DEFAULT_SPLITS,
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--dataset-root", default="data/gsm8k")
    parser.add_argument("--output-dir", default="runs/gsm8k_qwen32b_labels")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional per-invocation cap for a smoke test; omit for all samples.",
    )
    return parser


def _new_state(args) -> Dict:
    return {
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "vllm_enable_thinking": args.vllm_enable_thinking,
        "generator_prompt": GSM8K_GENERATOR_PROMPT,
        "splits": list(args.splits),
        "positions": {split: 0 for split in args.splits},
        "scanned_count": 0,
        "correct_count": 0,
        "incorrect_count": 0,
        "complete": False,
    }


def _load_state(path: Path, args) -> Dict:
    if not path.exists():
        return _new_state(args)
    state = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model": args.model,
        "vllm_base_url": args.vllm_base_url,
        "vllm_enable_thinking": args.vllm_enable_thinking,
        "splits": list(args.splits),
        "generator_prompt": GSM8K_GENERATOR_PROMPT,
    }
    mismatches = {
        key: (state.get(key), value) for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing state does not match this run: {mismatches}. "
            "Use a different --output-dir."
        )
    return state


def _save_state(path: Path, state: Dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _append_record(jsonl_path: Path, csv_path: Path, record: Dict) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record[field] for field in CSV_FIELDS})


def main(argv: Sequence[str] = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if not args.vllm_base_url:
        parser.error("--vllm-base-url or VLLM_BASE_URL is required")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    jsonl_path = output_dir / "labeled_samples.jsonl"
    csv_path = output_dir / "labeled_samples.csv"
    state = _load_state(state_path, args)
    if state["complete"]:
        print(f"Dataset already completely labeled: {csv_path}")
        return
    _save_state(state_path, state)

    engine = build_engine(
        args.model, args.vllm_base_url, args.vllm_api_key,
        args.vllm_enable_thinking,
    )
    fixed_prompt = tg.Variable(
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT, requires_grad=False,
        role_description="fixed GSM8K Generator role, rules, and output format",
    )
    strategy_prompt = tg.Variable(
        GSM8K_GENERATOR_STRATEGY_PROMPT, requires_grad=False,
        role_description="fixed initial GSM8K Generator strategy",
    )
    generator = GeneratorAgent(engine, strategy_prompt, fixed_prompt)

    def answer_one(dataset, split: str, index: int):
        case = case_from_dataset(dataset, "gsm8k", index, split)
        question = tg.Variable(
            case.question, requires_grad=False,
            role_description="GSM8K question for one-pass labeling",
        )
        output = generator.run(question).value
        return case, output, is_correct(output, case.answer)

    invocation_start = state["scanned_count"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split in args.splits:
            dataset = load_textgrad_dataset(
                "gsm8k", split=split, root=args.dataset_root
            )
            start = int(state["positions"].get(split, 0))
            for batch_start in range(start, len(dataset), args.workers):
                processed_now = state["scanned_count"] - invocation_start
                if args.max_samples is not None and processed_now >= args.max_samples:
                    _save_state(state_path, state)
                    print(f"Smoke-test limit reached: {csv_path}", flush=True)
                    return
                remaining = args.workers
                if args.max_samples is not None:
                    remaining = min(remaining, args.max_samples - processed_now)
                indices = list(
                    range(batch_start, min(batch_start + remaining, len(dataset)))
                )
                futures = [
                    executor.submit(answer_one, dataset, split, index)
                    for index in indices
                ]
                results = [future.result() for future in futures]
                for index, (case, output, correct) in zip(indices, results):
                    state["positions"][split] = index + 1
                    state["scanned_count"] += 1
                    state["correct_count" if correct else "incorrect_count"] += 1
                    record = {
                        "sample_id": state["scanned_count"],
                        "label": "correct" if correct else "incorrect",
                        "correct": correct,
                        "split": split,
                        "index": index,
                        "question": case.question,
                        "ground_truth": case.answer,
                        "parsed_prediction": parse_integer_answer(output),
                        "model": args.model,
                        "generator_prompt": GSM8K_GENERATOR_PROMPT,
                        "generator_output": output,
                    }
                    _append_record(jsonl_path, csv_path, record)
                    _save_state(state_path, state)
                    print(
                        f"split={split} index={index} label={record['label']} "
                        f"scanned={state['scanned_count']} "
                        f"correct={state['correct_count']} "
                        f"incorrect={state['incorrect_count']}", flush=True,
                    )

    state["complete"] = True
    _save_state(state_path, state)
    print(
        f"Labeling complete: total={state['scanned_count']} "
        f"correct={state['correct_count']} incorrect={state['incorrect_count']} "
        f"output={csv_path}", flush=True,
    )


if __name__ == "__main__":
    main()
