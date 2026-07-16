"""Alternating TextGrad optimization for the Generator and Verifier prompts."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import textgrad as tg
from textgrad.autograd import FormattedLLMCall
from textgrad.engine import EngineLM
from textgrad.tasks.big_bench_hard import parse_integer_answer

from .agents import GeneratorAgent, VerifierAgent
from .data import Case
from .evaluation import is_correct, parse_verdict
from .prompts import GENERATOR_TRAINING_INSTRUCTION, VERIFIER_TRAINING_OBJECTIVE


GENERATOR_SUPERVISION_MODES = ("final_answer", "gold_reasoning")


@dataclass(frozen=True)
class TrainingConfig:
    iterations: int = 1
    run_check: bool = False
    generator_supervision_mode: str = "final_answer"

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.generator_supervision_mode not in GENERATOR_SUPERVISION_MODES:
            raise ValueError(
                "generator_supervision_mode must be one of "
                f"{GENERATOR_SUPERVISION_MODES}"
            )


def _variable(value: str, role: str) -> tg.Variable:
    return tg.Variable(value, requires_grad=False, role_description=role)


class AdversarialGVTrainer:
    """Optimizes V and G in alternating steps while retaining exact accuracy."""

    def __init__(
        self,
        generator: GeneratorAgent,
        verifier: VerifierAgent,
        backward_engine: EngineLM,
        config: TrainingConfig,
    ):
        self.generator = generator
        self.verifier = verifier
        self.backward_engine = backward_engine
        self.config = config
        tg.set_backward_engine(backward_engine, override=True)

        self.g_optimizer = tg.TextualGradientDescent(
            engine=backward_engine,
            parameters=generator.parameters(),
            constraints=[
                "Do not include a specific training question or its answer.",
                "Preserve the required final format: Answer: $VALUE.",
                "Optimize genuine task correctness, never verifier manipulation.",
            ],
        )
        self.v_optimizer = tg.TextualGradientDescent(
            engine=backward_engine,
            parameters=verifier.parameters(),
            constraints=[
                "Do not include a specific training question or its answer.",
                "Never claim access to ground truth at inference time.",
                "Preserve the VERDICT, CONFIDENCE, and CRITIQUE output schema.",
            ],
        )

    def _v_loss(
        self,
        question: tg.Variable,
        candidate: tg.Variable,
        ground_truth: tg.Variable,
        expected_label: str,
    ):
        verdict = self.verifier.run(question, candidate)
        objective = (
            VERIFIER_TRAINING_OBJECTIVE
            + f"\nGround-truth answer (training only): {ground_truth.value}"
            + f"\nExpected verdict: {expected_label}"
        )
        loss = tg.TextLoss(objective, engine=self.backward_engine)(verdict)
        return loss, {
            "candidate": candidate.value,
            "expected_label": expected_label,
            "verifier_output": verdict.value,
            "loss_output": loss.value,
        }

    def _update_verifier(
        self,
        question: tg.Variable,
        generated: tg.Variable,
        ground_truth: tg.Variable,
    ) -> Dict[str, Any]:
        trace_mark = (
            self.backward_engine.mark()
            if hasattr(self.backward_engine, "mark")
            else None
        )
        self.v_optimizer.zero_grad()
        reference = _variable(
            f"Answer: {ground_truth.value}",
            "known-correct reference answer used only for verifier training",
        )
        wrong_value = parse_integer_answer(ground_truth.value) + 1
        negative = _variable(
            (
                "I counted each relevant item exactly once and checked the total. "
                f"The total is {wrong_value}.\nAnswer: {wrong_value}"
            ),
            "plausible but numerically incorrect answer used only for verifier training",
        )
        generated_label = "ACCEPT" if is_correct(generated.value, ground_truth.value) else "REJECT"
        evaluated = [
            self._v_loss(question, reference, ground_truth, "ACCEPT"),
            self._v_loss(question, negative, ground_truth, "REJECT"),
            self._v_loss(question, generated, ground_truth, generated_label),
        ]
        losses = [loss for loss, _ in evaluated]
        tg.sum(losses).backward()
        self.v_optimizer.step()
        trace = (
            self.backward_engine.since(trace_mark)
            if trace_mark is not None
            else []
        )
        return {
            "generated_expected_label": generated_label,
            "examples": [record for _, record in evaluated],
            "gradient_trace": trace,
        }

    def _update_generator(
        self,
        question: tg.Variable,
        ground_truth: tg.Variable,
        gold_reasoning: Optional[tg.Variable] = None,
    ) -> Dict[str, Any]:
        trace_mark = (
            self.backward_engine.mark()
            if hasattr(self.backward_engine, "mark")
            else None
        )
        self.g_optimizer.zero_grad()
        generated = self.generator.run(question)
        verdict = self.verifier.run(question, generated)
        format_string = (
            "<QUESTION>{question}</QUESTION>\n"
            "<GROUND_TRUTH training_only='true'>{ground_truth}</GROUND_TRUTH>\n"
        )
        fields = {
            "question": None,
            "ground_truth": None,
            "candidate": None,
            "verdict": None,
        }
        inputs = {
            "question": question,
            "ground_truth": ground_truth,
            "candidate": generated,
            "verdict": verdict,
        }
        if self.config.generator_supervision_mode == "gold_reasoning":
            if gold_reasoning is None or not gold_reasoning.value.strip():
                raise ValueError(
                    "generator_supervision_mode='gold_reasoning' requires "
                    "Case.gold_reasoning"
                )
            format_string += (
                "<GOLD_REASONING training_only='true'>{gold_reasoning}</GOLD_REASONING>\n"
            )
            fields["gold_reasoning"] = None
            inputs["gold_reasoning"] = gold_reasoning
        format_string += (
            "<GENERATOR_ANSWER>{candidate}</GENERATOR_ANSWER>\n"
            "<VERIFIER_RESPONSE>{verdict}</VERIFIER_RESPONSE>"
        )
        loss_call = FormattedLLMCall(
            engine=self.backward_engine,
            system_prompt=_variable(
                GENERATOR_TRAINING_INSTRUCTION,
                "fixed evaluator instruction for generator training",
            ),
            format_string=format_string,
            fields=fields,
        )
        loss = loss_call(
            inputs=inputs,
            response_role_description="feedback on generator task correctness and honest verifier acceptance",
        )
        loss.backward()
        self.g_optimizer.step()
        trace = (
            self.backward_engine.since(trace_mark)
            if trace_mark is not None
            else []
        )
        return {
            "answer": generated.value,
            "verdict": verdict.value,
            "loss_output": loss.value,
            "supervision_mode": self.config.generator_supervision_mode,
            "gold_reasoning": gold_reasoning.value if gold_reasoning is not None else None,
            "gradient_trace": trace,
        }

    def evaluate(self, case: Case) -> Dict[str, Any]:
        question = _variable(case.question, "multi-step reasoning question")
        answer = self.generator.run(question)
        verdict_text = self.verifier.run(question, answer).value
        verdict = parse_verdict(verdict_text)
        return {
            "case": asdict(case),
            "answer": answer.value,
            "verdict_raw": verdict_text,
            "verdict": asdict(verdict),
            "correct": is_correct(answer.value, case.answer),
        }

    def train(self, train_case: Case, check_case: Optional[Case] = None) -> Dict[str, Any]:
        if self.config.run_check and check_case is None:
            raise ValueError("run_check=True requires a check case")

        result: Dict[str, Any] = {
            "config": asdict(self.config),
            "dataset": train_case.dataset,
            "train_case": asdict(train_case),
            "initial": self.evaluate(train_case),
            "check_initial": self.evaluate(check_case) if self.config.run_check else None,
            "steps": [],
        }
        question = _variable(train_case.question, "multi-step reasoning question")
        ground_truth = _variable(train_case.answer, "ground-truth numerical answer")
        gold_reasoning = (
            _variable(train_case.gold_reasoning, "gold reference reasoning trajectory")
            if train_case.gold_reasoning is not None
            else None
        )

        for iteration in range(self.config.iterations):
            generated_for_v = self.generator.run(question)
            v_prompt_before = self.verifier.system_prompt.value
            v_interaction = self._update_verifier(
                question, generated_for_v, ground_truth
            )
            v_prompt_after = self.verifier.system_prompt.value

            g_prompt_before = self.generator.system_prompt.value
            g_interaction = self._update_generator(
                question, ground_truth, gold_reasoning
            )
            g_prompt_after = self.generator.system_prompt.value

            result["steps"].append(
                {
                    "iteration": iteration + 1,
                    "v_training_answer": generated_for_v.value,
                    "v_training_interaction": v_interaction,
                    "v_prompt_before": v_prompt_before,
                    "v_prompt_after": v_prompt_after,
                    "g_training_interaction": g_interaction,
                    "g_prompt_before": g_prompt_before,
                    "g_prompt_after": g_prompt_after,
                    "evaluation": self.evaluate(train_case),
                    "check": self.evaluate(check_case) if self.config.run_check else None,
                }
            )

        result["final"] = self.evaluate(train_case)
        result["check_final"] = self.evaluate(check_case) if self.config.run_check else None
        return result
