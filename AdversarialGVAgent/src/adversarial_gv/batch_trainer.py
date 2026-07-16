"""Strictly alternating mini-batch G/V prompt training."""

import concurrent.futures
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import textgrad as tg
from textgrad.autograd import FormattedLLMCall
from textgrad.engine import EngineLM
from textgrad.tasks.big_bench_hard import parse_integer_answer

from .agents import GeneratorAgent, VerifierAgent
from .evaluation import is_correct, parse_verdict
from .prompts import GENERATOR_TRAINING_INSTRUCTION, VERIFIER_TRAINING_OBJECTIVE


@dataclass(frozen=True)
class HardCase:
    wrong_id: int
    collection_split: str
    source_split: str
    source_index: int
    question: str
    ground_truth: str


def variable(value: str, role: str, requires_grad: bool = False) -> tg.Variable:
    return tg.Variable(value, requires_grad=requires_grad, role_description=role)


class BatchAdversarialGVTrainer:
    def __init__(
        self,
        generator: GeneratorAgent,
        verifier: VerifierAgent,
        backward_engine: EngineLM,
        evaluation_workers: int = 8,
        training_workers: int = 1,
    ):
        self.generator = generator
        self.verifier = verifier
        self.backward_engine = backward_engine
        self.evaluation_workers = evaluation_workers
        self.training_workers = training_workers
        tg.set_backward_engine(backward_engine, override=True)
        self.g_optimizer = tg.TextualGradientDescent(
            engine=backward_engine,
            parameters=generator.parameters(),
            constraints=[
                "Remain general: never include a particular training question or answer.",
                "Require explicit multi-step mathematical reasoning and unit checks.",
                "Preserve the final format Answer: $VALUE.",
                "Improve genuine correctness, never manipulate the Verifier.",
            ],
        )
        self.v_optimizer = tg.TextualGradientDescent(
            engine=backward_engine,
            parameters=verifier.parameters(),
            constraints=[
                "Remain general: never include a particular training question or answer.",
                "Do not claim access to ground truth during inference.",
                "Preserve VERDICT, CONFIDENCE, and CRITIQUE tags.",
                "Act as a verifier/auditor of the candidate trajectory, not a second generator.",
                "Do not add instructions to solve every problem from scratch; check suspicious steps only.",
                "Tie critiques to the candidate's stated extraction, equations, arithmetic, and final answer.",
            ],
        )

    def _mark(self):
        return self.backward_engine.mark() if hasattr(self.backward_engine, "mark") else None

    def _trace(self, mark):
        return self.backward_engine.since(mark) if mark is not None else []

    def _v_loss(
        self,
        question: tg.Variable,
        candidate: tg.Variable,
        ground_truth: tg.Variable,
        expected_label: str,
        sample_kind: str,
    ):
        verdict = self.verifier.run(question, candidate)
        objective = (
            VERIFIER_TRAINING_OBJECTIVE
            + f"\nGround-truth final answer (training only): {ground_truth.value}"
            + f"\nExpected verdict: {expected_label}"
            + "\nJudge whether the critique correctly checks the full reasoning trajectory."
        )
        loss = tg.TextLoss(objective, engine=self.backward_engine)(verdict)
        return loss, {
            "sample_kind": sample_kind,
            "candidate": candidate.value,
            "expected_label": expected_label,
            "verifier_output": verdict.value,
            "loss_output": loss.value,
        }

    def update_verifier(self, cases: Sequence[HardCase]) -> Dict[str, Any]:
        """Detach G outputs, then update only the V prompt.

        Per-case G/V/loss-evaluation calls are run concurrently within the batch;
        backward and optimizer updates remain one synchronized batch operation.
        """
        mark = self._mark()
        self.v_optimizer.zero_grad()
        print(f"V-step start: {len(cases)} cases, workers={self.training_workers}", flush=True)

        def build_generated_example(case: HardCase):
            print(f"  V-step case {case.wrong_id}: start", flush=True)
            question = variable(case.question, "multi-step GSM8K question")
            generated_graph = self.generator.run(question)
            detached_generated = variable(
                generated_graph.value,
                "detached fixed Generator trajectory for Verifier training",
            )
            ground_truth = variable(case.ground_truth, "ground-truth final answer")
            label = "ACCEPT" if is_correct(generated_graph.value, case.ground_truth) else "REJECT"
            loss, record = self._v_loss(
                question,
                detached_generated,
                ground_truth,
                label,
                "generator",
            )
            generated_record = {
                "wrong_id": case.wrong_id,
                "generator_output": generated_graph.value,
                "expected_label": label,
            }
            print(f"  V-step case {case.wrong_id}: done label={label}", flush=True)
            return loss, record, generated_record

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.training_workers) as executor:
            generated_results = list(executor.map(build_generated_example, cases))

        evaluated = [(loss, record) for loss, record, _ in generated_results]
        generated_records = [record for _, _, record in generated_results]
        print("V-step generated/loss examples complete; running anchors", flush=True)

        anchor = cases[0]
        anchor_question = variable(anchor.question, "anchor GSM8K question")
        anchor_truth = variable(anchor.ground_truth, "anchor ground-truth final answer")
        correct_anchor = variable(
            f"The final result is {anchor.ground_truth}.\nAnswer: {anchor.ground_truth}",
            "fixed known-correct Verifier anchor",
        )
        wrong_value = parse_integer_answer(anchor.ground_truth) + 1
        negative_anchor = variable(
            f"The final result is {wrong_value}.\nAnswer: {wrong_value}",
            "fixed known-incorrect Verifier anchor",
        )
        evaluated.extend(
            [
                self._v_loss(
                    anchor_question,
                    correct_anchor,
                    anchor_truth,
                    "ACCEPT",
                    "positive_anchor",
                ),
                self._v_loss(
                    anchor_question,
                    negative_anchor,
                    anchor_truth,
                    "REJECT",
                    "negative_anchor",
                ),
            ]
        )
        print("V-step backward start", flush=True)
        tg.sum([loss for loss, _ in evaluated]).backward()
        print("V-step optimizer step start", flush=True)
        self.v_optimizer.step()
        print("V-step complete", flush=True)
        return {
            "generated": generated_records,
            "loss_examples": [record for _, record in evaluated],
            "gradient_trace": self._trace(mark),
        }

    def _g_loss(
        self,
        question: tg.Variable,
        ground_truth: tg.Variable,
        generated: tg.Variable,
        verdict: tg.Variable,
    ) -> tg.Variable:
        loss_call = FormattedLLMCall(
            engine=self.backward_engine,
            system_prompt=variable(
                GENERATOR_TRAINING_INSTRUCTION,
                "fixed Generator objective",
            ),
            format_string=(
                "<QUESTION>{question}</QUESTION>\n"
                "<GROUND_TRUTH training_only='true'>{ground_truth}</GROUND_TRUTH>\n"
                "<GENERATOR_TRAJECTORY>{candidate}</GENERATOR_TRAJECTORY>\n"
                "<VERIFIER_RESPONSE>{verdict}</VERIFIER_RESPONSE>"
            ),
            fields={
                "question": None,
                "ground_truth": None,
                "candidate": None,
                "verdict": None,
            },
        )
        return loss_call(
            inputs={
                "question": question,
                "ground_truth": ground_truth,
                "candidate": generated,
                "verdict": verdict,
            },
            response_role_description="Generator loss feedback on correctness, trajectory, and legitimate V acceptance",
        )

    def update_generator(self, cases: Sequence[HardCase]) -> Dict[str, Any]:
        """Freeze V prompt, retain V-output-to-G-output feedback, update only G."""
        mark = self._mark()
        self.g_optimizer.zero_grad()
        print(f"G-step start: {len(cases)} cases, workers={self.training_workers}", flush=True)
        original_v_requires_grad = self.verifier.system_prompt.requires_grad
        self.verifier.system_prompt.requires_grad = False
        interactions = []
        losses = []
        try:
            def build_interaction(case: HardCase):
                print(f"  G-step case {case.wrong_id}: start", flush=True)
                question = variable(case.question, "multi-step GSM8K question")
                ground_truth = variable(case.ground_truth, "ground-truth final answer")
                generated = self.generator.run(question)
                verdict = self.verifier.run(question, generated)
                loss = self._g_loss(question, ground_truth, generated, verdict)
                print(f"  G-step case {case.wrong_id}: done", flush=True)
                return loss, {
                    "wrong_id": case.wrong_id,
                    "generator_output": generated.value,
                    "verifier_output": verdict.value,
                    "loss_output": loss.value,
                }

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.training_workers) as executor:
                results = list(executor.map(build_interaction, cases))
            losses = [loss for loss, _ in results]
            interactions = [record for _, record in results]
            print("G-step backward start", flush=True)
            tg.sum(losses).backward()
            print("G-step optimizer step start", flush=True)
            self.g_optimizer.step()
            print("G-step complete", flush=True)
        finally:
            self.verifier.system_prompt.requires_grad = original_v_requires_grad
        return {
            "interactions": interactions,
            "gradient_trace": self._trace(mark),
        }

    def train_batch(self, cases: Sequence[HardCase], batch_index: int) -> Dict[str, Any]:
        g_before = self.generator.system_prompt.value
        v_before = self.verifier.system_prompt.value
        v_step = self.update_verifier(cases)
        v_after = self.verifier.system_prompt.value
        g_step = self.update_generator(cases)
        g_after = self.generator.system_prompt.value
        return {
            "batch_index": batch_index,
            "case_ids": [case.wrong_id for case in cases],
            "generator_prompt_before": g_before,
            "generator_prompt_after": g_after,
            "verifier_prompt_before": v_before,
            "verifier_prompt_after": v_after,
            "v_step": v_step,
            "g_step": g_step,
        }

    def evaluate_one(self, case: HardCase) -> Dict[str, Any]:
        question = variable(case.question, "held-out GSM8K question")
        generated = self.generator.run(question)
        verdict_text = self.verifier.run(question, generated).value
        verdict = parse_verdict(verdict_text)
        return {
            "case": asdict(case),
            "generator_output": generated.value,
            "verifier_output": verdict_text,
            "verdict": asdict(verdict),
            "correct": is_correct(generated.value, case.ground_truth),
        }

    def evaluate(self, cases: Sequence[HardCase]) -> Dict[str, Any]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.evaluation_workers
        ) as executor:
            rows = list(executor.map(self.evaluate_one, cases))
        return {
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "accept_rate": sum(row["verdict"]["label"] == "ACCEPT" for row in rows)
            / len(rows),
            "rows": rows,
        }
