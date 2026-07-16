"""Deterministic metrics and robust verifier-output parsing."""

import re
from dataclasses import dataclass

from textgrad.tasks.big_bench_hard import parse_integer_answer


@dataclass(frozen=True)
class Verdict:
    label: str
    confidence: float | None
    critique: str


def is_correct(candidate: str, ground_truth: str) -> bool:
    return parse_integer_answer(candidate) == parse_integer_answer(ground_truth)


def parse_verdict(text: str) -> Verdict:
    label_match = re.search(
        r"<VERDICT>\s*(ACCEPT|REJECT)\s*</VERDICT>", text, re.IGNORECASE
    )
    confidence_match = re.search(
        r"<CONFIDENCE>\s*([01](?:\.\d+)?)\s*</CONFIDENCE>", text, re.IGNORECASE
    )
    critique_match = re.search(
        r"<CRITIQUE>\s*(.*?)\s*</CRITIQUE>", text, re.IGNORECASE | re.DOTALL
    )
    label = label_match.group(1).upper() if label_match else "INVALID"
    confidence = float(confidence_match.group(1)) if confidence_match else None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        confidence = None
    critique = critique_match.group(1).strip() if critique_match else ""
    return Verdict(label=label, confidence=confidence, critique=critique)

