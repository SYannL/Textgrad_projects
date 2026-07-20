import unittest

import textgrad as tg
from textgrad.engine import EngineLM

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.batch_trainer import BatchAdversarialGVTrainer, HardCase


class ToleranceEngine(EngineLM):
    model_string = "fake-tolerance"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        prompt_text = str(prompt)
        system_text = str(system_prompt or "")
        if "optimization system that improves text" in system_text:
            return "<IMPROVED_VARIABLE>general improved strategy</IMPROVED_VARIABLE>"
        if "CANDIDATE_ANSWER" in prompt_text:
            if "duplicate numbering" in prompt_text:
                audits = (
                    '<STEP_AUDIT index="1" status="VALID">ok</STEP_AUDIT>'
                    '<STEP_AUDIT index="2" status="VALID">ok</STEP_AUDIT>'
                    '<STEP_AUDIT index="1" status="VALID">duplicate</STEP_AUDIT>'
                    '<STEP_AUDIT index="2" status="VALID">duplicate</STEP_AUDIT>'
                )
            else:
                audits = '<STEP_AUDIT index="1" status="VALID">ok</STEP_AUDIT>'
            return (
                f"<TRAJECTORY_AUDIT>{audits}</TRAJECTORY_AUDIT>"
                "<FIRST_ERROR>NONE</FIRST_ERROR>"
                "<FINAL_ANSWER_CHECK>valid</FINAL_ANSWER_CHECK>"
                "<VERDICT>ACCEPT</VERDICT>"
                "<CONFIDENCE>1</CONFIDENCE>"
                "<CRITIQUE>valid</CRITIQUE>"
            )
        if "fixed training-only trajectory judge" in system_text:
            return (
                "<TRAJECTORY_LABEL>ACCEPT</TRAJECTORY_LABEL>"
                "<RATIONALE>NONE</RATIONALE>"
            )
        if (
            "Generator-Verifier interaction" in system_text
            or "gradient (feedback) engine" in system_text
            or "feedback to a variable" in prompt_text
        ):
            return "Use one consecutive top-level Step sequence."
        if "bad case" in prompt_text:
            return (
                "Step 1: Start duplicate numbering.\n"
                "Step 2: Continue.\n"
                "Step 1: Restart duplicate numbering.\n"
                "Step 2: Finish.\n"
                "Answer: 2"
            )
        return "Step 1: Compute the answer.\nAnswer: 2"


def case(wrong_id: int, question: str) -> HardCase:
    return HardCase(
        wrong_id=wrong_id,
        collection_split="train",
        source_split="train",
        source_index=wrong_id,
        question=question,
        ground_truth="2",
    )


class BatchToleranceTests(unittest.TestCase):
    def test_malformed_case_is_recorded_while_other_case_updates(self):
        engine = ToleranceEngine()
        g_strategy = tg.Variable(
            "generator strategy",
            requires_grad=True,
            role_description="generator strategy",
        )
        v_strategy = tg.Variable(
            "verifier strategy",
            requires_grad=True,
            role_description="verifier strategy",
        )
        trainer = BatchAdversarialGVTrainer(
            GeneratorAgent(engine, g_strategy),
            VerifierAgent(engine, v_strategy),
            engine,
            evaluation_workers=1,
            training_workers=1,
        )

        result = trainer.update_generator(
            [case(1, "bad case"), case(2, "good case")]
        )

        self.assertEqual(result["attempted_cases"], 2)
        self.assertEqual(result["successful_cases"], 1)
        self.assertEqual(len(result["skipped_cases"]), 1)
        self.assertEqual(result["skipped_cases"][0]["wrong_id"], 1)
        self.assertEqual(result["skipped_cases"][0]["error_type"], "ValueError")
        self.assertEqual(result["update_status"], "updated")
        self.assertEqual(g_strategy.value, "general improved strategy")


if __name__ == "__main__":
    unittest.main()
