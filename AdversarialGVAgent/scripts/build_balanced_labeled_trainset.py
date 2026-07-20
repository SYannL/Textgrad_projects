#!/usr/bin/env python
"""Build a deterministic 1:1 easy/hard G/V dataset from full labels."""

import argparse
import csv
import random
from pathlib import Path
from typing import Sequence


FIELDS = [
    "wrong_id",
    "collection_split",
    "difficulty",
    "initial_correct",
    "split",
    "index",
    "question",
    "ground_truth",
    "parsed_prediction",
    "label_model",
    "generator_prompt",
    "generator_output",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Balance full-model labels into interleaved easy/hard examples."
    )
    parser.add_argument(
        "--labels",
        default="runs/gsm8k_qwen4b_labels/labeled_samples.csv",
    )
    parser.add_argument(
        "--output",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=15,
        help="Number of easy and hard examples reserved for validation.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=15,
        help="Number of easy and hard examples reserved for test.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _interleave(easy, hard, collection_split):
    if len(easy) != len(hard):
        raise ValueError("easy and hard groups must have identical sizes")
    rows = []
    for easy_row, hard_row in zip(easy, hard):
        rows.extend(
            [
                _convert(easy_row, "easy", collection_split),
                _convert(hard_row, "hard", collection_split),
            ]
        )
    return rows


def _convert(row, difficulty, collection_split):
    return {
        "collection_split": collection_split,
        "difficulty": difficulty,
        "initial_correct": "true" if difficulty == "easy" else "false",
        "split": row["split"],
        "index": row["index"],
        "question": row["question"],
        "ground_truth": row["ground_truth"],
        "parsed_prediction": row["parsed_prediction"],
        "label_model": row["model"],
        "generator_prompt": row["generator_prompt"],
        "generator_output": row["generator_output"],
    }


def main(argv: Sequence[str] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.val_per_class < 1:
        raise ValueError("--val-per-class must be at least 1")
    if args.test_per_class < 1:
        raise ValueError("--test-per-class must be at least 1")
    with Path(args.labels).open(newline="", encoding="utf-8") as handle:
        labeled = list(csv.DictReader(handle))

    easy = [row for row in labeled if row["label"] == "correct"]
    hard = [row for row in labeled if row["label"] == "incorrect"]
    if not easy or not hard:
        raise ValueError(f"expected both labels, got easy={len(easy)} hard={len(hard)}")

    rng = random.Random(args.seed)
    rng.shuffle(easy)
    rng.shuffle(hard)
    class_size = min(len(easy), len(hard))
    reserved_per_class = args.val_per_class + args.test_per_class
    if reserved_per_class >= class_size:
        raise ValueError(
            "validation + test examples per class must be below "
            f"{class_size}, got {reserved_per_class}"
        )
    easy = easy[:class_size]
    hard = hard[:class_size]

    val_easy = easy[: args.val_per_class]
    val_hard = hard[: args.val_per_class]
    test_easy = easy[args.val_per_class : reserved_per_class]
    test_hard = hard[args.val_per_class : reserved_per_class]
    train_easy = easy[reserved_per_class:]
    train_hard = hard[reserved_per_class:]
    output_rows = _interleave(train_easy, train_hard, "train")
    output_rows.extend(_interleave(val_easy, val_hard, "val"))
    output_rows.extend(_interleave(test_easy, test_hard, "test"))
    for row_id, row in enumerate(output_rows, start=1):
        row["wrong_id"] = row_id

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"wrote {output}: train={len(train_easy)} easy + {len(train_hard)} hard; "
        f"val={len(val_easy)} easy + {len(val_hard)} hard; "
        f"test={len(test_easy)} easy + {len(test_hard)} hard; seed={args.seed}"
    )


if __name__ == "__main__":
    main()
