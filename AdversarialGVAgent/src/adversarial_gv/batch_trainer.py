"""Strictly alternating mini-batch G/V prompt training."""

import concurrent.futures
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import textgrad as tg
from textgrad.autograd import FormattedLLMCall
from textgrad.engine import EngineLM

from .agents import GeneratorAgent, VerifierAgent
from .evaluation import assert_complete_trajectory_audit, is_correct, parse_verdict
from .prompts import GENERATOR_TRAINING_INSTRUCTION, VERIFIER_TRAINING_OBJECTIVE
from .trajectory_labeling import label_trajectory


GENERATOR_OPTIMIZER_CONSTRAINTS = [
    "Remain general: never include a particular training question or answer.",
    "Improve only the problem-solving strategy; do not restate or redefine fixed system rules.",
    "Support explicit multi-step mathematical reasoning and unit checks.",
    "Improve genuine correctness, never manipulate the Verifier.",
]


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
            constraints=GENERATOR_OPTIMIZER_CONSTRAINTS,
        )
        self.v_optimizer = tg.TextualGradientDescent(
            engine=backward_engine,
            parameters=verifier.parameters(),
            constraints=[
                "Remain general: never include a particular training question or answer.",
                "Do not claim access to ground truth during inference.",
                "Improve only the audit strategy; do not restate or redefine fixed system rules or tags.",
                "Audit every material candidate step and identify the earliest error precisely.",
                "Continue auditing every later step after finding the first error.",
                "Emit exactly one indexed STEP_AUDIT for every numbered Generator Step.",
                "Reject candidates without a meaningful auditable reasoning trajectory.",
                "Use ACCEPT only when both the final answer and every material reasoning step are correct.",
                "Use CHALLENGE when the final answer is correct but the trajectory is invalid, unsupported, incomplete, or missing.",
                "Use REJECT when the final answer is incorrect or missing.",
                "Act as a verifier/auditor of the candidate trajectory, not a second generator.",
                "Tie critiques to the candidate's stated extraction, equations, arithmetic, and final answer.",
            ],
        )

    def _mark(self):
        return self.backward_engine.mark() if hasattr(self.backward_engine, "mark") else None

    def _trace(self, mark):
        return self.backward_engine.since(mark) if mark is not None else []

    @staticmethod
    def _error_record(
        stage: str,
        exc: Exception,
        case: HardCase | None = None,
        **context: Any,
    ) -> Dict[str, Any]:
        record = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if case is not None:
            record["wrong_id"] = case.wrong_id
            record["case"] = asdict(case)
        record.update(context)
        return record

    @staticmethod
    def _print_skip(stage: str, error: Dict[str, Any]) -> None:
        case_text = (
            f" case {error['wrong_id']}" if "wrong_id" in error else ""
        )
        print(
            f"  {stage}{case_text}: skipped "
            f"{error['error_type']}: {error['error_message']}",
            flush=True,
        )

    def _v_loss(
        self,
        question: tg.Variable,
        candidate: tg.Variable,
        ground_truth: tg.Variable,
        expected_label: str,
        sample_kind: str,
    ):
        verdict = self.verifier.run(question, candidate)
        assert_complete_trajectory_audit(candidate.value, verdict.value)
        objective = (
            VERIFIER_TRAINING_OBJECTIVE
            + f"\nGround-truth final answer (training only): {ground_truth.value}"
            + f"\nExpected verdict: {expected_label}"
            + "\nJudge whether the step audit covers the full reasoning trajectory and "
            + "whether FIRST_ERROR pinpoints the earliest actual mistake."
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
            context: Dict[str, Any] = {}
            try:
                question = variable(case.question, "multi-step GSM8K question")
                generated_graph = self.generator.run(question)
                context["generator_output"] = generated_graph.value
                detached_generated = variable(
                    generated_graph.value,
                    "detached fixed Generator trajectory for Verifier training",
                )
                ground_truth = variable(
                    case.ground_truth,
                    "ground-truth final answer",
                )
                trajectory_label, labeler_output = label_trajectory(
                    self.backward_engine,
                    question,
                    detached_generated,
                    ground_truth,
                )
                label = trajectory_label.label
                context["expected_label"] = label
                context["labeler_output"] = labeler_output
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
                    "label_rationale": trajectory_label.rationale,
                    "labeler_output": labeler_output,
                }
                print(
                    f"  V-step case {case.wrong_id}: done label={label}",
                    flush=True,
                )
                return {
                    "loss": loss,
                    "loss_record": record,
                    "generated_record": generated_record,
                    "error": None,
                }
            except Exception as exc:
                error = self._error_record("V-step case", exc, case, **context)
                self._print_skip("V-step", error)
                return {"error": error}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.training_workers) as executor:
            generated_results = list(executor.map(build_generated_example, cases))

        successful = [item for item in generated_results if item["error"] is None]
        skipped_cases = [
            item["error"] for item in generated_results if item["error"] is not None
        ]
        losses = [item["loss"] for item in successful]
        print("V-step generated/loss examples complete", flush=True)
        update_status = "skipped_no_valid_cases"
        update_error = None
        prompt_before_update = self.verifier.system_prompt.value
        if losses:
            try:
                print("V-step backward start", flush=True)
                tg.sum(losses).backward()
                print("V-step optimizer step start", flush=True)
                self.v_optimizer.step()
                update_status = "updated"
            except Exception as exc:
                self.verifier.system_prompt.set_value(prompt_before_update)
                self.v_optimizer.zero_grad()
                update_error = self._error_record("V-step backward/optimizer", exc)
                self._print_skip("V-step backward/optimizer", update_error)
                update_status = "update_failed_rolled_back"
        print(f"V-step complete status={update_status}", flush=True)
        return {
            "generated": [item["generated_record"] for item in successful],
            "loss_examples": [item["loss_record"] for item in successful],
            "attempted_cases": len(cases),
            "successful_cases": len(successful),
            "skipped_cases": skipped_cases,
            "update_status": update_status,
            "update_error": update_error,
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
        assert_complete_trajectory_audit(generated.value, verdict.value)
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
                context: Dict[str, Any] = {}
                try:
                    question = variable(case.question, "multi-step GSM8K question")
                    ground_truth = variable(
                        case.ground_truth,
                        "ground-truth final answer",
                    )
                    generated = self.generator.run(question)
                    context["generator_output"] = generated.value
                    verdict = self.verifier.run(question, generated)
                    context["verifier_output"] = verdict.value
                    loss = self._g_loss(question, ground_truth, generated, verdict)
                    print(f"  G-step case {case.wrong_id}: done", flush=True)
                    return {
                        "loss": loss,
                        "interaction": {
                            "wrong_id": case.wrong_id,
                            "generator_output": generated.value,
                            "verifier_output": verdict.value,
                            "loss_output": loss.value,
                        },
                        "error": None,
                    }
                except Exception as exc:
                    error = self._error_record("G-step case", exc, case, **context)
                    self._print_skip("G-step", error)
                    return {"error": error}

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.training_workers) as executor:
                results = list(executor.map(build_interaction, cases))
            successful = [item for item in results if item["error"] is None]
            skipped_cases = [
                item["error"] for item in results if item["error"] is not None
            ]
            losses = [item["loss"] for item in successful]
            interactions = [item["interaction"] for item in successful]
            update_status = "skipped_no_valid_cases"
            update_error = None
            prompt_before_update = self.generator.system_prompt.value
            if losses:
                try:
                    print("G-step backward start", flush=True)
                    tg.sum(losses).backward()
                    print("G-step optimizer step start", flush=True)
                    self.g_optimizer.step()
                    update_status = "updated"
                except Exception as exc:
                    self.generator.system_prompt.set_value(prompt_before_update)
                    self.g_optimizer.zero_grad()
                    update_error = self._error_record(
                        "G-step backward/optimizer",
                        exc,
                    )
                    self._print_skip("G-step backward/optimizer", update_error)
                    update_status = "update_failed_rolled_back"
            print(f"G-step complete status={update_status}", flush=True)
        finally:
            self.verifier.system_prompt.requires_grad = original_v_requires_grad
        return {
            "interactions": interactions,
            "attempted_cases": len(cases),
            "successful_cases": len(interactions),
            "skipped_cases": skipped_cases,
            "update_status": update_status,
            "update_error": update_error,
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
            "generator_fixed_system_prompt": self.generator.fixed_system_prompt.value,
            "verifier_fixed_system_prompt": self.verifier.fixed_system_prompt.value,
            "generator_strategy_prompt_before": g_before,
            "generator_strategy_prompt_after": g_after,
            "verifier_strategy_prompt_before": v_before,
            "verifier_strategy_prompt_after": v_after,
            "generator_prompt_before": g_before,
            "generator_prompt_after": g_after,
            "verifier_prompt_before": v_before,
            "verifier_prompt_after": v_after,
            "v_step": v_step,
            "g_step": g_step,
        }

    def evaluate_one(self, case: HardCase) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        try:
            question = variable(case.question, "held-out GSM8K question")
            generated = self.generator.run(question)
            context["generator_output"] = generated.value
            verdict_text = self.verifier.run(question, generated).value
            context["verifier_output"] = verdict_text
            verdict = parse_verdict(verdict_text)
            return {
                "case": asdict(case),
                "generator_output": generated.value,
                "verifier_output": verdict_text,
                "verdict": asdict(verdict),
                "correct": is_correct(generated.value, case.ground_truth),
                "skipped": False,
                "error": None,
            }
        except Exception as exc:
            error = self._error_record("evaluation case", exc, case, **context)
            self._print_skip("evaluation", error)
            return {
                "case": asdict(case),
                "generator_output": context.get("generator_output", ""),
                "verifier_output": context.get("verifier_output", ""),
                "verdict": asdict(parse_verdict("")),
                "correct": None,
                "skipped": True,
                "error": error,
            }

    def evaluate(self, cases: Sequence[HardCase]) -> Dict[str, Any]:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.evaluation_workers
        ) as executor:
            rows = list(executor.map(self.evaluate_one, cases))
        evaluated_rows = [row for row in rows if not row["skipped"]]
        denominator = len(evaluated_rows)

        def rate(predicate) -> float:
            if denominator == 0:
                return 0.0
            return sum(predicate(row) for row in evaluated_rows) / denominator

        return {
            "accuracy": rate(lambda row: bool(row["correct"])),
            "accept_rate": rate(lambda row: row["verdict"]["label"] == "ACCEPT"),
            "challenge_rate": rate(
                lambda row: row["verdict"]["label"] == "CHALLENGE"
            ),
            "reject_rate": rate(lambda row: row["verdict"]["label"] == "REJECT"),
            "invalid_rate": rate(lambda row: row["verdict"]["label"] == "INVALID"),
            "attempted_count": len(rows),
            "evaluated_count": denominator,
            "skipped_count": len(rows) - denominator,
            "errors": [row["error"] for row in rows if row["error"] is not None],
            "rows": rows,
        }
