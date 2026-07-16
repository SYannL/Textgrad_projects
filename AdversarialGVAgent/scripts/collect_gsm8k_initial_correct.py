#!/usr/bin/env python
"""Collect GSM8K train examples that the initial Generator answers correctly.

This script performs one forward answer per sample. It does not instantiate a
Verifier, call backward(), run TGD, or update either prompt.
"""

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
from typing import Dict, Sequence

import textgrad as tg
from dotenv import load_dotenv
from textgrad.tasks.big_bench_hard import parse_integer_answer

from adversarial_gv.agents import GeneratorAgent
from adversarial_gv.data import case_from_dataset, load_textgrad_dataset
from adversarial_gv.evaluation import is_correct
from adversarial_gv.prompts import GSM8K_GENERATOR_PROMPT


MODEL = "gpt-4o-mini"
CSV_FIELDS = [
    "correct_id",
    "collection_split",
    "split",
    "index",
    "question",
    "ground_truth",
    "parsed_prediction",
    "generator_prompt",
    "generator_output",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect GSM8K train samples answered correctly by one initial "
            "gpt-4o-mini Generator call."
        )
    )
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--dataset-root", default="data/gsm8k")
    parser.add_argument("--output-dir", default="runs/gsm8k_initial_correct")
    return parser


def _load_state(path: Path, split: str) -> Dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("model") != MODEL:
            raise ValueError(f"state model must be {MODEL}")
        return state
    return {
        "model": MODEL,
        "generator_prompt": GSM8K_GENERATOR_PROMPT,
        "split": split,
        "position": 0,
        "scanned_count": 0,
        "correct_count": 0,
        "complete": False,
    }


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
    if args.target < 1:
        parser.error("--target must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    jsonl_path = output_dir / "correct_samples.jsonl"
    csv_path = output_dir / "correct_samples.csv"
    state = _load_state(state_path, args.split)
    state["target"] = args.target
    state["complete"] = state["correct_count"] >= args.target
    _save_state(state_path, state)
    if state["correct_count"] >= args.target:
        print(f"Already collected {state['correct_count']} correct samples: {csv_path}")
        return

    prompt = tg.Variable(
        GSM8K_GENERATOR_PROMPT,
        requires_grad=False,
        role_description="fixed initial GSM8K chain-of-thought Generator prompt",
    )
    generator = GeneratorAgent(tg.get_engine(MODEL), prompt)

    def answer_one(dataset, split: str, index: int):
        case = case_from_dataset(dataset, "gsm8k", index, split)
        question = tg.Variable(
            case.question,
            requires_grad=False,
            role_description="GSM8K question for one-pass correct collection",
        )
        output = generator.run(question).value
        return case, output, is_correct(output, case.answer)

    dataset = load_textgrad_dataset("gsm8k", split=args.split, root=args.dataset_root)
    start = int(state.get("position", 0))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for batch_start in range(start, len(dataset), args.workers):
            indices = list(
                range(batch_start, min(batch_start + args.workers, len(dataset)))
            )
            futures = [
                executor.submit(answer_one, dataset, args.split, index)
                for index in indices
            ]
            results = [future.result() for future in futures]
            for index, (case, output, correct) in zip(indices, results):
                state["position"] = index + 1
                state["scanned_count"] += 1

                if correct:
                    state["correct_count"] += 1
                    record = {
                        "correct_id": state["correct_count"],
                        "collection_split": "train",
                        "split": args.split,
                        "index": index,
                        "question": case.question,
                        "ground_truth": case.answer,
                        "parsed_prediction": parse_integer_answer(output),
                        "generator_prompt": GSM8K_GENERATOR_PROMPT,
                        "generator_output": output,
                    }
                    _append_record(jsonl_path, csv_path, record)

                _save_state(state_path, state)
                print(
                    f"split={args.split} index={index} correct={correct} "
                    f"scanned={state['scanned_count']} "
                    f"collected={state['correct_count']}/{args.target}",
                    flush=True,
                )
                if state["correct_count"] >= args.target:
                    state["complete"] = True
                    _save_state(state_path, state)
                    print(f"Collection complete: {csv_path}")
                    return

    state["complete"] = state["correct_count"] >= args.target
    _save_state(state_path, state)
    raise RuntimeError(
        f"Exhausted split {args.split} after {state['scanned_count']} samples; "
        f"collected only {state['correct_count']} of {args.target} correct samples."
    )


if __name__ == "__main__":
    main()
