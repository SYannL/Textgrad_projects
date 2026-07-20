#!/usr/bin/env python
"""Collect GSM8K examples that the initial gpt-4o-mini Generator gets wrong.

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
from adversarial_gv.prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
)


MODEL = "gpt-4o-mini"
DEFAULT_SPLITS = ("train", "val", "test")
CSV_FIELDS = [
    "wrong_id",
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
            "Collect GSM8K samples answered incorrectly by one initial "
            "gpt-4o-mini Generator call."
        )
    )
    parser.add_argument("--train-target", type=int, default=30)
    parser.add_argument("--val-target", type=int, default=15)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=DEFAULT_SPLITS,
        default=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--dataset-root", default="data/gsm8k")
    parser.add_argument("--output-dir", default="runs/gsm8k_initial_wrong")
    return parser


def _load_state(path: Path, splits: Sequence[str]) -> Dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("model") != MODEL:
            raise ValueError(f"state model must be {MODEL}")
        return state
    return {
        "model": MODEL,
        "generator_prompt": GSM8K_GENERATOR_PROMPT,
        "positions": {split: 0 for split in splits},
        "scanned_count": 0,
        "wrong_count": 0,
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


def _migrate_records(
    jsonl_path: Path, csv_path: Path, train_target: int
) -> None:
    if not jsonl_path.exists():
        return
    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        record["collection_split"] = (
            "train" if int(record["wrong_id"]) <= train_target else "val"
        )
    jsonl_tmp = jsonl_path.with_suffix(".jsonl.tmp")
    jsonl_tmp.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    jsonl_tmp.replace(jsonl_path)
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: record[field] for field in CSV_FIELDS} for record in records
        )
    csv_tmp.replace(csv_path)


def main(argv: Sequence[str] = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.train_target < 1 or args.val_target < 1:
        parser.error("--train-target and --val-target must both be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    jsonl_path = output_dir / "wrong_samples.jsonl"
    csv_path = output_dir / "wrong_samples.csv"
    state = _load_state(state_path, args.splits)
    target = args.train_target + args.val_target
    _migrate_records(jsonl_path, csv_path, args.train_target)
    state["train_target"] = args.train_target
    state["val_target"] = args.val_target
    state["complete"] = state["wrong_count"] >= target
    _save_state(state_path, state)
    if state["wrong_count"] >= target:
        print(f"Already collected {state['wrong_count']} wrong samples: {csv_path}")
        return

    fixed_prompt = tg.Variable(
        GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="fixed GSM8K Generator role, rules, and output format",
    )
    strategy_prompt = tg.Variable(
        GSM8K_GENERATOR_STRATEGY_PROMPT,
        requires_grad=False,
        role_description="fixed initial GSM8K Generator strategy",
    )
    generator = GeneratorAgent(tg.get_engine(MODEL), strategy_prompt, fixed_prompt)

    def answer_one(dataset, split: str, index: int):
        case = case_from_dataset(dataset, "gsm8k", index, split)
        question = tg.Variable(
            case.question,
            requires_grad=False,
            role_description="GSM8K question for one-pass collection",
        )
        output = generator.run(question).value
        return case, output, is_correct(output, case.answer)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for split in args.splits:
            dataset = load_textgrad_dataset(
                "gsm8k", split=split, root=args.dataset_root
            )
            start = int(state["positions"].get(split, 0))
            for batch_start in range(start, len(dataset), args.workers):
                indices = list(
                    range(batch_start, min(batch_start + args.workers, len(dataset)))
                )
                futures = [
                    executor.submit(answer_one, dataset, split, index)
                    for index in indices
                ]
                results = [future.result() for future in futures]
                for index, (case, output, correct) in zip(indices, results):
                    state["positions"][split] = index + 1
                    state["scanned_count"] += 1

                    if not correct:
                        state["wrong_count"] += 1
                        record = {
                            "wrong_id": state["wrong_count"],
                            "collection_split": (
                                "train"
                                if state["wrong_count"] <= args.train_target
                                else "val"
                            ),
                            "split": split,
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
                        f"split={split} index={index} correct={correct} "
                        f"scanned={state['scanned_count']} wrong={state['wrong_count']}/{target}",
                        flush=True,
                    )
                    if state["wrong_count"] >= target:
                        state["complete"] = True
                        _save_state(state_path, state)
                        print(f"Collection complete: {csv_path}")
                        return

    state["complete"] = state["wrong_count"] >= target
    _save_state(state_path, state)
    raise RuntimeError(
        f"Exhausted splits {args.splits} after {state['scanned_count']} samples; "
        f"collected only {state['wrong_count']} of {target} wrong samples."
    )


if __name__ == "__main__":
    main()
