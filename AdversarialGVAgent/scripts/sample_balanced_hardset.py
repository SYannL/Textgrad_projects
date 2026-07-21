#!/usr/bin/env python
"""Create a deterministic stratified subset of an existing hard-set CSV."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--input",
        default="runs/gsm8k_qwen4b_balanced_1to1/mixed_samples.csv",
    )
    result.add_argument(
        "--output",
        default="runs/gsm8k_qwen4b_balanced_8h/mixed_samples.csv",
    )
    result.add_argument("--train-per-class", type=int, default=30)
    result.add_argument("--val-per-class", type=int, default=5)
    result.add_argument("--test-per-class", type=int, default=5)
    result.add_argument("--seed", type=int, default=42)
    return result


def sample_rows(
    rows,
    *,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
):
    requested = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }
    if any(value < 1 for value in requested.values()):
        raise ValueError("every per-class split size must be positive")
    rng = random.Random(seed)
    selected = []
    for split in ("train", "val", "test"):
        groups = {
            difficulty: [
                row
                for row in rows
                if row["collection_split"] == split
                and row["difficulty"] == difficulty
            ]
            for difficulty in ("easy", "hard")
        }
        count = requested[split]
        for difficulty, group in groups.items():
            if len(group) < count:
                raise ValueError(
                    f"requested {count} {split}/{difficulty} rows, "
                    f"but only {len(group)} are available"
                )
            rng.shuffle(group)
        # Preserve the original experiment's alternating easy/hard layout.
        for easy, hard in zip(groups["easy"][:count], groups["hard"][:count]):
            selected.extend([easy, hard])
    return selected


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    source = Path(args.input)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields:
        raise ValueError(f"input CSV has no header: {source}")
    selected = sample_rows(
        rows,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    counts = Counter(
        (row["collection_split"], row["difficulty"])
        for row in selected
    )
    manifest = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "seed": args.seed,
        "preserved_original_wrong_ids": True,
        "counts": {
            split: {
                difficulty: counts[(split, difficulty)]
                for difficulty in ("easy", "hard")
            }
            for split in ("train", "val", "test")
        },
        "total": len(selected),
    }
    output.with_name("subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
