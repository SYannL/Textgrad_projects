"""Deterministic metrics and robust verifier-output parsing."""

import re
from dataclasses import dataclass

from textgrad.tasks.big_bench_hard import parse_integer_answer


@dataclass(frozen=True)
class Verdict:
    label: str
    confidence: float | None
    critique: str
    trajectory_audit: str = ""
    first_error: str = ""
    final_answer_check: str = ""


@dataclass(frozen=True)
class TrajectoryLabel:
    """Training-only three-way supervision for a Generator trajectory."""

    label: str
    rationale: str


def is_correct(candidate: str, ground_truth: str) -> bool:
    return parse_integer_answer(candidate) == parse_integer_answer(ground_truth)


def reasoning_step_indices(text: str) -> list[int]:
    return [
        int(value)
        for value in re.findall(r"(?mi)^\s*Step\s+(\d+)\s*:", text)
    ]


def audited_step_indices(text: str) -> list[int]:
    return [
        int(value)
        for value in re.findall(
            r"<STEP_AUDIT\s+index=[\"']?(\d+)[\"']?\s+status=",
            text,
            re.IGNORECASE,
        )
    ]


def assert_complete_trajectory_audit(candidate: str, verifier_output: str) -> None:
    """Prevent TextGrad backward passes based on partial trajectory audits."""
    steps = reasoning_step_indices(candidate)
    audits = audited_step_indices(verifier_output)
    verdict = parse_verdict(verifier_output)
    if not steps:
        missing_cot_rejected = (
            verdict.label in {"CHALLENGE", "REJECT"}
            and bool(verdict.first_error)
            and verdict.first_error.strip().upper() != "NONE"
        )
        if not missing_cot_rejected:
            raise ValueError(
                "Generator omitted numbered Steps, but Verifier did not explicitly "
                "REJECT the missing CoT in FIRST_ERROR"
            )
        return
    expected = list(range(1, len(steps) + 1))
    if steps != expected:
        raise ValueError(
            f"Generator Steps must be consecutive starting at 1; got {steps}"
        )
    if audits != steps:
        raise ValueError(
            "Verifier must audit every Generator Step exactly once before backward: "
            f"generator_steps={steps}, audited_steps={audits}"
        )


def parse_verdict(text: str) -> Verdict:
    label_match = re.search(
        r"<VERDICT>\s*(ACCEPT|CHALLENGE|REJECT)\s*</VERDICT>",
        text,
        re.IGNORECASE,
    )
    confidence_match = re.search(
        r"<CONFIDENCE>\s*([01](?:\.\d+)?)\s*</CONFIDENCE>", text, re.IGNORECASE
    )
    critique_match = re.search(
        r"<CRITIQUE>\s*(.*?)\s*</CRITIQUE>", text, re.IGNORECASE | re.DOTALL
    )
    audit_match = re.search(
        r"<TRAJECTORY_AUDIT>\s*(.*?)\s*</TRAJECTORY_AUDIT>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    first_error_match = re.search(
        r"<FIRST_ERROR>\s*(.*?)\s*</FIRST_ERROR>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    final_check_match = re.search(
        r"<FINAL_ANSWER_CHECK>\s*(.*?)\s*</FINAL_ANSWER_CHECK>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    label = label_match.group(1).upper() if label_match else "INVALID"
    confidence = float(confidence_match.group(1)) if confidence_match else None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        confidence = None
    critique = critique_match.group(1).strip() if critique_match else ""
    trajectory_audit = audit_match.group(1).strip() if audit_match else ""
    first_error = first_error_match.group(1).strip() if first_error_match else ""
    final_answer_check = final_check_match.group(1).strip() if final_check_match else ""
    return Verdict(
        label=label,
        confidence=confidence,
        critique=critique,
        trajectory_audit=trajectory_audit,
        first_error=first_error,
        final_answer_check=final_answer_check,
    )


def parse_trajectory_label(text: str) -> TrajectoryLabel:
    """Parse the fixed training judge's non-differentiable supervision label."""
    label_match = re.search(
        r"<TRAJECTORY_LABEL>\s*(ACCEPT|CHALLENGE|REJECT)\s*"
        r"</TRAJECTORY_LABEL>",
        text,
        re.IGNORECASE,
    )
    rationale_match = re.search(
        r"<RATIONALE>\s*(.*?)\s*</RATIONALE>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if label_match is None:
        raise ValueError(
            "Training trajectory judge must return exactly one of ACCEPT, "
            "CHALLENGE, or REJECT inside <TRAJECTORY_LABEL> tags; "
            f"got: {text}"
        )
    return TrajectoryLabel(
        label=label_match.group(1).upper(),
        rationale=rationale_match.group(1).strip() if rationale_match else "",
    )
