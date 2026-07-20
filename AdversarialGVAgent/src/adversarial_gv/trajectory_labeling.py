"""Fixed, training-only CoT supervision for the three-way Verifier verdict."""

from typing import Tuple

import textgrad as tg
from textgrad.autograd import FormattedLLMCall
from textgrad.engine import EngineLM

from .evaluation import TrajectoryLabel, parse_trajectory_label
from .prompts import TRAJECTORY_LABELING_INSTRUCTION


def label_trajectory(
    engine: EngineLM,
    question: tg.Variable,
    candidate: tg.Variable,
    ground_truth: tg.Variable,
) -> Tuple[TrajectoryLabel, str]:
    """Label one detached trajectory without using the trainable Verifier prompt."""
    call = FormattedLLMCall(
        engine=engine,
        system_prompt=tg.Variable(
            TRAJECTORY_LABELING_INSTRUCTION,
            requires_grad=False,
            role_description="fixed training-only trajectory labeling policy",
        ),
        format_string=(
            "<QUESTION>{question}</QUESTION>\n"
            "<GROUND_TRUTH training_only='true'>{ground_truth}</GROUND_TRUTH>\n"
            "<GENERATOR_TRAJECTORY>{candidate}</GENERATOR_TRAJECTORY>"
        ),
        fields={"question": None, "ground_truth": None, "candidate": None},
    )
    response = call(
        inputs={
            "question": question,
            "ground_truth": ground_truth,
            "candidate": candidate,
        },
        response_role_description="fixed three-way trajectory supervision label",
    )
    return parse_trajectory_label(response.value), response.value
