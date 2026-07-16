"""Dataset access through the unmodified TextGrad package."""

from dataclasses import dataclass
from typing import Optional

from textgrad.tasks.big_bench_hard import BigBenchHard

from .gsm8k_compat import GSM8KDSPyCompat


DATASET_CHOICES = ("bbh_object_counting", "gsm8k")


@dataclass(frozen=True)
class Case:
    question: str
    answer: str
    split: str
    index: int
    dataset: str = "bbh_object_counting"
    gold_reasoning: Optional[str] = None


def load_textgrad_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    root: Optional[str] = None,
):
    if dataset_name == "bbh_object_counting":
        return BigBenchHard("object_counting", split=split, root=root)
    if dataset_name == "gsm8k":
        return GSM8KDSPyCompat(split=split, root=root)
    raise ValueError(f"unsupported dataset: {dataset_name}")


def case_from_dataset(dataset, dataset_name: str, index: int, split: str) -> Case:
    if index < 0 or index >= len(dataset):
        raise IndexError(
            f"case index {index} is outside {split} split [0, {len(dataset) - 1}]"
        )
    row = dataset[index]
    if len(row) == 2:
        question, answer = row
        gold_reasoning = None
    elif len(row) == 3:
        question, answer, gold_reasoning = row
    else:
        raise ValueError(f"expected dataset row with 2 or 3 fields, got {len(row)}")
    return Case(
        str(question),
        str(answer),
        split,
        index,
        dataset_name,
        None if gold_reasoning is None else str(gold_reasoning),
    )


def load_case(
    dataset_name: str,
    index: int,
    *,
    split: str = "train",
    root: Optional[str] = None,
) -> Case:
    dataset = load_textgrad_dataset(dataset_name, split=split, root=root)
    return case_from_dataset(dataset, dataset_name, index, split)


def load_bbh_case(
    index: int,
    *,
    split: str = "train",
    root: Optional[str] = None,
) -> Case:
    return load_case(
        "bbh_object_counting", index, split=split, root=root
    )
