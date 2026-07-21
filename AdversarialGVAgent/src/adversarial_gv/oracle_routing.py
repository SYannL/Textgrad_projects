"""Oracle-routed mini-batch prompt training for GSM8K.

The train-time oracle is privileged: it sees GSM8K's official reference
reasoning and final answer.  Generator and Verifier forward calls never see
either field.
"""

import concurrent.futures
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import textgrad as tg
from textgrad.autograd import FormattedLLMCall
from textgrad.engine import EngineLM

from .agents import GeneratorAgent, VerifierAgent
from .batch_trainer import (
    GENERATOR_OPTIMIZER_CONSTRAINTS,
    BatchAdversarialGVTrainer,
    HardCase,
    variable,
)
from .evaluation import is_correct


GENERATOR_ORACLE_INSTRUCTION = """You are a fixed training-only GSM8K oracle.
You receive privileged official reference reasoning and the official final answer,
plus a Generator trajectory. Neither the Generator nor the Verifier saw the
privileged reference.

First assign the Generator exactly one label:
- ACCEPT: its final answer is correct and every material reasoning step is valid,
  supported, and sufficient.
- CHALLENGE: its final answer is correct, but some material reasoning is invalid,
  unsupported, missing, or not meaningfully auditable.
- REJECT: its final answer is incorrect or missing.

Carefully compare the mathematical substance of the generated process with the
question and official process. Do not require the Generator to copy the official
derivation: accept any valid alternative.

Return exactly:
<GENERATOR_LABEL>ACCEPT or CHALLENGE or REJECT</GENERATOR_LABEL>
<GENERATOR_RATIONALE>concise evidence, or NONE</GENERATOR_RATIONALE>"""


VERIFIER_ORACLE_INSTRUCTION = """You are a second fixed training-only GSM8K
oracle. An earlier oracle independently assessed the Generator without seeing the
Verifier response. You receive that assessment, the privileged official reasoning
and answer, the Generator trajectory, and the Verifier response.

Decide whether the Verifier's substantive mathematical evaluation is correct.
Carefully compare its verdict, claims about the generated reasoning, identified
first error, and final-answer analysis with the question, official process, and the
independent Generator assessment. Do not require wording or derivation to match the
official solution. Do not judge XML/tag formatting and do not require one audit item
per numbered Generator step; those are fixed interface requirements, not semantic
oracle criteria.

Return exactly:
<VERIFIER_CORRECT>YES or NO</VERIFIER_CORRECT>
<VERIFIER_RATIONALE>concise evidence</VERIFIER_RATIONALE>"""


ORACLE_VERIFIER_LOSS_INSTRUCTION = """Evaluate the Verifier response using the
privileged GSM8K oracle evidence. Improve only the Verifier's general audit
strategy. The official reasoning is a reference rather than the only permitted
derivation. Correct the verdict, inaccurate step audits, missed steps, false
allegations, FIRST_ERROR, and FINAL_ANSWER_CHECK. Do not rewrite the prompt and do
not copy details of this training question into the general strategy."""


ORACLE_GENERATOR_LOSS_INSTRUCTION = """Evaluate the Generator trajectory using
the privileged GSM8K oracle evidence. Improve only the Generator's general
problem-solving strategy. The official reasoning is a reference rather than the
only permitted derivation. Correct the earliest material extraction, equation,
arithmetic, unit, dependency, or completeness defect. The Verifier response may be
wrong; the oracle assessment has priority. Do not rewrite the prompt and do not
copy details of this training question into the general strategy."""


@dataclass(frozen=True)
class OracleHardCase(HardCase):
    gold_reasoning: str


@dataclass(frozen=True)
class OracleAssessment:
    generator_label: str
    generator_rationale: str
    verifier_correct: bool
    verifier_rationale: str

    @property
    def generator_correct(self) -> bool:
        return self.generator_label == "ACCEPT"


def _tag(text: str, name: str) -> str:
    match = re.search(
        rf"<{name}>\s*(.*?)\s*</{name}>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def parse_generator_oracle(text: str) -> tuple[str, str]:
    label = _tag(text, "GENERATOR_LABEL").upper()
    if label not in {"ACCEPT", "CHALLENGE", "REJECT"}:
        raise ValueError(
            "Generator oracle must return ACCEPT, CHALLENGE, or REJECT in "
            f"<GENERATOR_LABEL>; got: {text}"
        )
    return label, _tag(text, "GENERATOR_RATIONALE")


def parse_verifier_oracle(text: str) -> tuple[bool, str]:
    correct = _tag(text, "VERIFIER_CORRECT").upper()
    if correct not in {"YES", "NO"}:
        raise ValueError(
            "Verifier oracle must return YES or NO in <VERIFIER_CORRECT>; "
            f"got: {text}"
        )
    return correct == "YES", _tag(text, "VERIFIER_RATIONALE")


def route_for(assessment: OracleAssessment) -> str:
    """Return the exact optimizer route selected by the oracle."""
    if assessment.generator_correct and assessment.verifier_correct:
        return "none"
    if assessment.generator_correct:
        return "verifier"
    if assessment.verifier_correct:
        return "generator"
    return "both"


class OracleRoutedGVTrainer(BatchAdversarialGVTrainer):
    """Use one shared G/V interaction per case and route oracle-labelled losses."""

    def __init__(
        self,
        generator: GeneratorAgent,
        verifier: VerifierAgent,
        backward_engine: EngineLM,
        oracle_engine: EngineLM | None = None,
        evaluation_workers: int = 8,
        training_workers: int = 1,
    ):
        super().__init__(
            generator,
            verifier,
            backward_engine,
            evaluation_workers=evaluation_workers,
            training_workers=training_workers,
        )
        self.oracle_engine = oracle_engine or backward_engine
        # Keep these explicit here so the new experiment remains independent of
        # later changes to the legacy trainer's optimizer instances.
        self.g_optimizer = tg.TextualGradientDescent(
            engine=backward_engine,
            parameters=generator.parameters(),
            constraints=GENERATOR_OPTIMIZER_CONSTRAINTS,
        )

    def _oracle_assess(
        self,
        case: OracleHardCase,
        candidate: str,
        verifier_output: str,
    ) -> tuple[OracleAssessment, Dict[str, str]]:
        generator_prompt = (
            f"<QUESTION>{case.question}</QUESTION>\n"
            f"<OFFICIAL_REASONING training_only='true'>{case.gold_reasoning}"
            "</OFFICIAL_REASONING>\n"
            f"<OFFICIAL_ANSWER training_only='true'>{case.ground_truth}"
            "</OFFICIAL_ANSWER>\n"
            f"<GENERATOR_TRAJECTORY>{candidate}</GENERATOR_TRAJECTORY>"
        )
        generator_output = self.oracle_engine(
            generator_prompt,
            system_prompt=GENERATOR_ORACLE_INSTRUCTION,
        )
        generator_label, generator_rationale = parse_generator_oracle(
            generator_output
        )

        # Final-answer mismatch is deterministic in GSM8K and takes precedence
        # over a malformed oracle response that accidentally says ACCEPT.
        if not is_correct(candidate, case.ground_truth):
            generator_label = "REJECT"
            generator_rationale = (
                generator_rationale
                or "The parsed final answer differs from the official answer."
            )

        verifier_prompt = (
            f"<QUESTION>{case.question}</QUESTION>\n"
            f"<OFFICIAL_REASONING training_only='true'>{case.gold_reasoning}"
            "</OFFICIAL_REASONING>\n"
            f"<OFFICIAL_ANSWER training_only='true'>{case.ground_truth}"
            "</OFFICIAL_ANSWER>\n"
            f"<GENERATOR_TRAJECTORY>{candidate}</GENERATOR_TRAJECTORY>\n"
            f"<GENERATOR_ORACLE_LABEL>{generator_label}"
            "</GENERATOR_ORACLE_LABEL>\n"
            f"<GENERATOR_ORACLE_RATIONALE>{generator_rationale}"
            "</GENERATOR_ORACLE_RATIONALE>\n"
            f"<VERIFIER_RESPONSE>{verifier_output}</VERIFIER_RESPONSE>"
        )
        verifier_output_judgment = self.oracle_engine(
            verifier_prompt,
            system_prompt=VERIFIER_ORACLE_INSTRUCTION,
        )
        verifier_correct, verifier_rationale = parse_verifier_oracle(
            verifier_output_judgment
        )
        assessment = OracleAssessment(
            generator_label=generator_label,
            generator_rationale=generator_rationale,
            verifier_correct=verifier_correct,
            verifier_rationale=verifier_rationale,
        )
        return assessment, {
            "generator": generator_output,
            "verifier": verifier_output_judgment,
        }

    def _verifier_loss(self, item: Dict[str, Any]) -> tg.Variable:
        assessment: OracleAssessment = item["assessment"]
        objective = (
            ORACLE_VERIFIER_LOSS_INSTRUCTION
            + f"\nQuestion: {item['case'].question}"
            + f"\nOfficial reasoning (training only): {item['case'].gold_reasoning}"
            + f"\nOfficial answer (training only): {item['case'].ground_truth}"
            + f"\nGenerator trajectory: {item['generated'].value}"
            + f"\nOracle Generator label: {assessment.generator_label}"
            + f"\nOracle Generator rationale: {assessment.generator_rationale}"
            + f"\nOracle critique of Verifier: {assessment.verifier_rationale}"
        )
        return tg.TextLoss(objective, engine=self.backward_engine)(
            item["verdict"]
        )

    def _generator_loss(self, item: Dict[str, Any]) -> tg.Variable:
        assessment: OracleAssessment = item["assessment"]
        call = FormattedLLMCall(
            engine=self.backward_engine,
            system_prompt=variable(
                ORACLE_GENERATOR_LOSS_INSTRUCTION,
                "fixed oracle-routed Generator objective",
            ),
            format_string=(
                "<QUESTION>{question}</QUESTION>\n"
                "<OFFICIAL_REASONING training_only='true'>{gold_reasoning}"
                "</OFFICIAL_REASONING>\n"
                "<OFFICIAL_ANSWER training_only='true'>{ground_truth}"
                "</OFFICIAL_ANSWER>\n"
                "<GENERATOR_TRAJECTORY>{candidate}</GENERATOR_TRAJECTORY>\n"
                "<VERIFIER_RESPONSE reliability='{v_reliability}'>{verdict}"
                "</VERIFIER_RESPONSE>\n"
                "<ORACLE_LABEL>{oracle_label}</ORACLE_LABEL>\n"
                "<ORACLE_RATIONALE>{oracle_rationale}</ORACLE_RATIONALE>"
            ),
            fields={
                "question": None,
                "gold_reasoning": None,
                "ground_truth": None,
                "candidate": None,
                "verdict": None,
                "v_reliability": None,
                "oracle_label": None,
                "oracle_rationale": None,
            },
        )
        return call(
            inputs={
                "question": item["question"],
                "gold_reasoning": variable(
                    item["case"].gold_reasoning,
                    "privileged official GSM8K reasoning",
                ),
                "ground_truth": variable(
                    item["case"].ground_truth,
                    "privileged official GSM8K final answer",
                ),
                "candidate": item["generated"],
                "verdict": variable(
                    item["verdict"].value,
                    "detached Verifier response",
                ),
                "v_reliability": variable(
                    "correct" if assessment.verifier_correct else "incorrect",
                    "oracle Verifier reliability flag",
                ),
                "oracle_label": variable(
                    assessment.generator_label,
                    "oracle Generator label",
                ),
                "oracle_rationale": variable(
                    assessment.generator_rationale,
                    "oracle Generator rationale",
                ),
            },
            response_role_description="oracle-grounded Generator loss feedback",
        )

    def _apply_losses(
        self,
        losses: Sequence[tg.Variable],
        optimizer,
        parameter: tg.Variable,
        role: str,
    ) -> Dict[str, Any]:
        before = parameter.value
        if not losses:
            return {
                "update_status": "skipped_no_routed_cases",
                "routed_cases": 0,
                "prompt_before": before,
                "prompt_after": before,
                "update_error": None,
            }
        try:
            print(f"{role}-step backward start routed_cases={len(losses)}", flush=True)
            tg.sum(list(losses)).backward()
            print(f"{role}-step optimizer start", flush=True)
            optimizer.step()
            status = "updated"
            error = None
        except Exception as exc:
            parameter.set_value(before)
            optimizer.zero_grad()
            error = self._error_record(f"{role}-step backward/optimizer", exc)
            self._print_skip(f"{role}-step backward/optimizer", error)
            status = "update_failed_rolled_back"
        return {
            "update_status": status,
            "routed_cases": len(losses),
            "prompt_before": before,
            "prompt_after": parameter.value,
            "update_error": error,
        }

    def evaluate_generator_accuracy(
        self,
        cases: Sequence[OracleHardCase],
    ) -> Dict[str, Any]:
        """Evaluate only deterministic final-answer accuracy for acc gating."""
        def evaluate_one(case: OracleHardCase) -> Dict[str, Any]:
            try:
                question = variable(case.question, "validation GSM8K question")
                generated = self.generator.run(question).value
                return {
                    "wrong_id": case.wrong_id,
                    "generator_output": generated,
                    "correct": is_correct(generated, case.ground_truth),
                    "error": None,
                }
            except Exception as exc:
                return {
                    "wrong_id": case.wrong_id,
                    "generator_output": "",
                    "correct": False,
                    "error": self._error_record(
                        "validation acc case",
                        exc,
                        case,
                    ),
                }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.evaluation_workers
        ) as executor:
            rows = list(executor.map(evaluate_one, cases))
        denominator = len(rows)
        return {
            "mode": "acc",
            "accuracy": (
                sum(bool(row["correct"]) for row in rows) / denominator
                if denominator
                else 0.0
            ),
            "attempted_count": denominator,
            "error_count": sum(row["error"] is not None for row in rows),
            "rows": rows,
        }

    def evaluate_oracle_validation(
        self,
        cases: Sequence[OracleHardCase],
    ) -> Dict[str, Any]:
        """Evaluate G and V semantic correctness with the privileged oracle."""
        def evaluate_one(case: OracleHardCase) -> Dict[str, Any]:
            context: Dict[str, Any] = {}
            try:
                question = variable(case.question, "validation GSM8K question")
                generated = self.generator.run(question).value
                context["generator_output"] = generated
                verdict = self.verifier.run(
                    question,
                    variable(generated, "detached validation Generator trajectory"),
                ).value
                context["verifier_output"] = verdict
                assessment, oracle_output = self._oracle_assess(
                    case,
                    generated,
                    verdict,
                )
                return {
                    "wrong_id": case.wrong_id,
                    "generator_output": generated,
                    "verifier_output": verdict,
                    "generator_correct": assessment.generator_correct,
                    "verifier_correct": assessment.verifier_correct,
                    "oracle": asdict(assessment),
                    "oracle_output": oracle_output,
                    "error": None,
                }
            except Exception as exc:
                return {
                    "wrong_id": case.wrong_id,
                    "generator_output": context.get("generator_output", ""),
                    "verifier_output": context.get("verifier_output", ""),
                    "generator_correct": False,
                    "verifier_correct": False,
                    "oracle": None,
                    "oracle_output": None,
                    "error": self._error_record(
                        "validation oracle case",
                        exc,
                        case,
                        **context,
                    ),
                }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.evaluation_workers
        ) as executor:
            rows = list(executor.map(evaluate_one, cases))
        denominator = len(rows)

        def rate(field: str) -> float:
            if not denominator:
                return 0.0
            return sum(bool(row[field]) for row in rows) / denominator

        return {
            "mode": "oracle",
            "generator_accuracy": rate("generator_correct"),
            "verifier_accuracy": rate("verifier_correct"),
            "attempted_count": denominator,
            "error_count": sum(row["error"] is not None for row in rows),
            "rows": rows,
        }

    def train_batch(
        self,
        cases: Sequence[OracleHardCase],
        batch_index: int,
        iteration_index: int = 1,
    ) -> Dict[str, Any]:
        """Generate once per case, oracle-label, then update routed parameters."""
        if iteration_index < 1:
            raise ValueError("iteration_index must be positive")
        mark = self._mark()
        self.g_optimizer.zero_grad()
        self.v_optimizer.zero_grad()
        g_before = self.generator.system_prompt.value
        v_before = self.verifier.system_prompt.value
        print(
            f"oracle-routing batch={batch_index} iteration={iteration_index} "
            f"start: {len(cases)} cases, "
            f"workers={self.training_workers}",
            flush=True,
        )

        def build_interaction(case: OracleHardCase) -> Dict[str, Any]:
            context: Dict[str, Any] = {}
            try:
                print(f"  oracle case {case.wrong_id}: start", flush=True)
                question = variable(case.question, "multi-step GSM8K question")
                generated = self.generator.run(question)
                context["generator_output"] = generated.value
                detached_generated = variable(
                    generated.value,
                    "detached Generator trajectory for Verifier",
                )
                verdict = self.verifier.run(question, detached_generated)
                context["verifier_output"] = verdict.value
                assessment, oracle_output = self._oracle_assess(
                    case,
                    generated.value,
                    verdict.value,
                )
                route = route_for(assessment)
                print(
                    f"  oracle case {case.wrong_id}: G="
                    f"{assessment.generator_label} V="
                    f"{'correct' if assessment.verifier_correct else 'wrong'} "
                    f"route={route}",
                    flush=True,
                )
                return {
                    "case": case,
                    "question": question,
                    "generated": generated,
                    "verdict": verdict,
                    "assessment": assessment,
                    "oracle_output": oracle_output,
                    "route": route,
                    "error": None,
                }
            except Exception as exc:
                error = self._error_record(
                    "oracle-routing case",
                    exc,
                    case,
                    **context,
                )
                self._print_skip("oracle-routing", error)
                return {"case": case, "error": error}

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.training_workers
        ) as executor:
            results = list(executor.map(build_interaction, cases))

        successful = [item for item in results if item["error"] is None]
        skipped = [item["error"] for item in results if item["error"] is not None]
        v_items = [item for item in successful if item["route"] in {"verifier", "both"}]
        g_items = [item for item in successful if item["route"] in {"generator", "both"}]

        v_losses = [self._verifier_loss(item) for item in v_items]
        v_result = self._apply_losses(
            v_losses,
            self.v_optimizer,
            self.verifier.system_prompt,
            "V",
        )
        g_losses = [self._generator_loss(item) for item in g_items]
        g_result = self._apply_losses(
            g_losses,
            self.g_optimizer,
            self.generator.system_prompt,
            "G",
        )

        route_counts = {
            name: sum(item["route"] == name for item in successful)
            for name in ("none", "generator", "verifier", "both")
        }
        trace = self._trace(mark)
        records = []
        for item in successful:
            assessment = item["assessment"]
            records.append(
                {
                    "wrong_id": item["case"].wrong_id,
                    "case": asdict(item["case"]),
                    "generator_output": item["generated"].value,
                    "verifier_output": item["verdict"].value,
                    "oracle": asdict(assessment),
                    "oracle_output": item["oracle_output"],
                    "route": item["route"],
                }
            )
        return {
            "batch_index": batch_index,
            "iteration_index": iteration_index,
            "case_ids": [case.wrong_id for case in cases],
            "route_counts": route_counts,
            "attempted_cases": len(cases),
            "successful_cases": len(successful),
            "skipped_cases": skipped,
            "interactions": records,
            "generator_strategy_prompt_before": g_before,
            "generator_strategy_prompt_after": self.generator.system_prompt.value,
            "verifier_strategy_prompt_before": v_before,
            "verifier_strategy_prompt_after": self.verifier.system_prompt.value,
            "g_step": {**g_result, "routed_case_ids": [item["case"].wrong_id for item in g_items]},
            "v_step": {**v_result, "routed_case_ids": [item["case"].wrong_id for item in v_items]},
            "gradient_trace": trace,
        }
