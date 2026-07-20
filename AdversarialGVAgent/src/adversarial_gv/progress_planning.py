"""Measure JSONL progress totals by replaying the real graph with a fake engine."""

import contextlib
import io
import logging
from typing import Dict, List, Sequence, Tuple

import textgrad as tg
from textgrad.engine import EngineLM

from .agents import GeneratorAgent, VerifierAgent
from .batch_trainer import BatchAdversarialGVTrainer, HardCase
from .prompts import (
    GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
    GSM8K_GENERATOR_STRATEGY_PROMPT,
    VERIFIER_FIXED_SYSTEM_PROMPT,
    VERIFIER_STRATEGY_PROMPT,
)


class _CountingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


class _PlanningEngine(EngineLM):
    """Return format-valid fixed text without making any external model call."""

    model_string = "progress-planning-engine"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        prompt_text = str(prompt)
        system_text = str(system_prompt or "").lower()
        if "fixed training-only trajectory judge" in system_text:
            return (
                "<TRAJECTORY_LABEL>ACCEPT</TRAJECTORY_LABEL>"
                "<RATIONALE>NONE</RATIONALE>"
            )
        if "optimization system that improves text" in system_text:
            return "<IMPROVED_VARIABLE>general planning strategy</IMPROVED_VARIABLE>"
        if "candidate_answer" in prompt_text.lower():
            return (
                '<TRAJECTORY_AUDIT><STEP_AUDIT index="1" status="VALID">'
                "valid</STEP_AUDIT></TRAJECTORY_AUDIT>"
                "<FIRST_ERROR>NONE</FIRST_ERROR>"
                "<FINAL_ANSWER_CHECK>valid</FINAL_ANSWER_CHECK>"
                "<VERDICT>ACCEPT</VERDICT>"
                "<CONFIDENCE>1</CONFIDENCE>"
                "<CRITIQUE>valid</CRITIQUE>"
            )
        if (
            "gradient (feedback) engine" in system_text
            or "generator-verifier interaction" in system_text
            or "assess the verifier response" in system_text
            or "ground_truth" in prompt_text.lower()
        ):
            return "Concise actionable planning feedback."
        return "Step 1: Compute the requested quantity.\nAnswer: 2"


def _variable(value: str, requires_grad: bool, role: str) -> tg.Variable:
    return tg.Variable(
        value,
        requires_grad=requires_grad,
        role_description=role,
    )


def _planning_trainer() -> BatchAdversarialGVTrainer:
    engine = _PlanningEngine()
    return BatchAdversarialGVTrainer(
        GeneratorAgent(
            engine,
            _variable(
                GSM8K_GENERATOR_STRATEGY_PROMPT,
                True,
                "trainable GSM8K Generator problem-solving strategy",
            ),
            _variable(
                GSM8K_GENERATOR_FIXED_SYSTEM_PROMPT,
                False,
                "immutable Generator role, rules, and output format",
            ),
        ),
        VerifierAgent(
            engine,
            _variable(
                VERIFIER_STRATEGY_PROMPT,
                True,
                "trainable Verifier trajectory-audit strategy",
            ),
            _variable(
                VERIFIER_FIXED_SYSTEM_PROMPT,
                False,
                "immutable Verifier role, audit rules, and output format",
            ),
        ),
        engine,
        evaluation_workers=1,
        training_workers=1,
    )


def _count_textgrad_records(action) -> int:
    """Run an action while replacing TextGrad's outputs with one counter."""
    from textgrad import logger

    handler = _CountingHandler()
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    logger.handlers = [handler]
    logger.propagate = False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            action()
    finally:
        logger.handlers = original_handlers
        logger.propagate = original_propagate
    return handler.count


def measure_progress_line_totals(
    batches: Sequence[Sequence[HardCase]],
    evaluation_splits: Sequence[Sequence[HardCase]],
) -> Tuple[int, List[int]]:
    """Measure initial-eval and per-batch JSONL totals from actual control flow.

    A result is cached by actual batch length because batches of the same size
    instantiate the same TextGrad graph.  No event-count constants are used.
    """
    if not batches or any(not batch for batch in batches):
        raise ValueError("progress planning requires non-empty batches")
    if not evaluation_splits or any(not split for split in evaluation_splits):
        raise ValueError("progress planning requires non-empty evaluation splits")

    def evaluate_all(trainer: BatchAdversarialGVTrainer) -> None:
        for split in evaluation_splits:
            trainer.evaluate(split)

    initial_trainer = _planning_trainer()
    initial_total = _count_textgrad_records(
        lambda: evaluate_all(initial_trainer)
    )

    totals_by_size: Dict[int, int] = {}
    for batch in batches:
        batch_size = len(batch)
        if batch_size in totals_by_size:
            continue
        trainer = _planning_trainer()

        def run_batch_and_evaluation(
            trainer=trainer,
            batch=batch,
        ) -> None:
            trainer.train_batch(batch, 1)
            evaluate_all(trainer)

        totals_by_size[batch_size] = _count_textgrad_records(
            run_batch_and_evaluation
        )

    return initial_total, [totals_by_size[len(batch)] for batch in batches]
