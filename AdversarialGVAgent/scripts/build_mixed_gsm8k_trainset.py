#!/usr/bin/env python
"""Build a mixed GSM8K split with wrong train, correct train, and val rows."""

import argparse
import csv
from pathlib import Path
from typing import Sequence


FIELDS = [
    "wrong_id",
    "collection_split",
    "split",
    "index",
    "question",
    "ground_truth",
    "parsed_prediction",
    "generator_prompt",
    "generator_output",
    "initial_correct",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--wrong-csv", default="runs/gsm8k_initial_wrong/wrong_samples.csv")
    result.add_argument("--correct-csv", default="runs/gsm8k_initial_correct/correct_samples.csv")
    result.add_argument("--output", default="runs/gsm8k_mixed_60train_15val/mixed_samples.csv")
    result.add_argument("--wrong-train", type=int, default=30)
    result.add_argument("--correct-train", type=int, default=30)
    result.add_argument("--val", type=int, default=15)
    return result


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] = None) -> None:
    args = parser().parse_args(argv)
    wrong_rows = read_rows(Path(args.wrong_csv))
    correct_rows = read_rows(Path(args.correct_csv))

    wrong_train = [
        row for row in wrong_rows if row["collection_split"] == "train"
    ][: args.wrong_train]
    val_rows = [row for row in wrong_rows if row["collection_split"] == "val"][: args.val]
    correct_train = correct_rows[: args.correct_train]

    if len(wrong_train) != args.wrong_train:
        raise ValueError(f"wrong train rows: expected {args.wrong_train}, got {len(wrong_train)}")
    if len(correct_train) != args.correct_train:
        raise ValueError(f"correct train rows: expected {args.correct_train}, got {len(correct_train)}")
    if len(val_rows) != args.val:
        raise ValueError(f"val rows: expected {args.val}, got {len(val_rows)}")

    output_rows = []
    next_id = 1
    seen = set()
    for source, initial_correct in (
        (wrong_train, "false"),
        (correct_train, "true"),
        (val_rows, "false"),
    ):
        for row in source:
            key = (row["split"], row["index"])
            if key in seen:
                raise ValueError(f"duplicate source sample: {key}")
            seen.add(key)
            collection_split = "val" if row["collection_split"] == "val" else "train"
            output_rows.append(
                {
                    "wrong_id": next_id,
                    "collection_split": collection_split,
                    "split": row["split"],
                    "index": row["index"],
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "parsed_prediction": row["parsed_prediction"],
                    "generator_prompt": row["generator_prompt"],
                    "generator_output": row["generator_output"],
                    "initial_correct": initial_correct,
                }
            )
            next_id += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"wrote {output}: train={len(wrong_train) + len(correct_train)} "
        f"val={len(val_rows)}"
    )


if __name__ == "__main__":
    main()
