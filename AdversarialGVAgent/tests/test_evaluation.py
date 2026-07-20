import unittest

from adversarial_gv.evaluation import (
    assert_complete_trajectory_audit,
    is_correct,
    parse_trajectory_label,
    parse_verdict,
)


class EvaluationTests(unittest.TestCase):
    def test_integer_correctness_matches_textgrad_task_metric(self):
        self.assertTrue(is_correct("Reasoning\nAnswer: 10", "10"))
        self.assertFalse(is_correct("Reasoning\nAnswer: 9", "10"))

    def test_parse_verdict(self):
        parsed = parse_verdict(
            "<TRAJECTORY_AUDIT>Step 1 arithmetic is valid.</TRAJECTORY_AUDIT>\n"
            "<FIRST_ERROR>NONE</FIRST_ERROR>\n"
            "<FINAL_ANSWER_CHECK>The answer follows.</FINAL_ANSWER_CHECK>\n"
            "<VERDICT>ACCEPT</VERDICT>\n"
            "<CONFIDENCE>0.9</CONFIDENCE>\n"
            "<CRITIQUE>The arithmetic checks out.</CRITIQUE>"
        )
        self.assertEqual(parsed.label, "ACCEPT")
        self.assertEqual(parsed.confidence, 0.9)
        self.assertEqual(parsed.critique, "The arithmetic checks out.")
        self.assertEqual(parsed.trajectory_audit, "Step 1 arithmetic is valid.")
        self.assertEqual(parsed.first_error, "NONE")
        self.assertEqual(parsed.final_answer_check, "The answer follows.")

    def test_malformed_verdict_is_invalid(self):
        parsed = parse_verdict("Looks good")
        self.assertEqual(parsed.label, "INVALID")
        self.assertIsNone(parsed.confidence)

    def test_parse_challenge_verdict(self):
        parsed = parse_verdict(
            "<VERDICT>CHALLENGE</VERDICT>"
            "<CONFIDENCE>0.8</CONFIDENCE>"
            "<CRITIQUE>The number is right but Step 2 is unsupported.</CRITIQUE>"
        )
        self.assertEqual(parsed.label, "CHALLENGE")

    def test_parse_training_trajectory_label(self):
        parsed = parse_trajectory_label(
            "<TRAJECTORY_LABEL>CHALLENGE</TRAJECTORY_LABEL>"
            "<RATIONALE>Step 2 is invalid.</RATIONALE>"
        )
        self.assertEqual(parsed.label, "CHALLENGE")
        self.assertEqual(parsed.rationale, "Step 2 is invalid.")

    def test_correct_final_without_cot_can_be_challenged(self):
        verifier_output = (
            "<TRAJECTORY_AUDIT></TRAJECTORY_AUDIT>"
            "<FIRST_ERROR>The required numbered CoT is missing.</FIRST_ERROR>"
            "<FINAL_ANSWER_CHECK>The final number is independently correct.</FINAL_ANSWER_CHECK>"
            "<VERDICT>CHALLENGE</VERDICT>"
            "<CONFIDENCE>1</CONFIDENCE>"
            "<CRITIQUE>Supply an auditable derivation.</CRITIQUE>"
        )
        assert_complete_trajectory_audit("Answer: 2", verifier_output)

    def test_complete_audit_must_cover_steps_after_first_error(self):
        candidate = "Step 1: valid\nStep 2: wrong\nStep 3: depends on step 2\nAnswer: 4"
        partial = (
            '<TRAJECTORY_AUDIT><STEP_AUDIT index="1" status="VALID">ok</STEP_AUDIT>'
            '<STEP_AUDIT index="2" status="INVALID">first error</STEP_AUDIT>'
            "</TRAJECTORY_AUDIT><FIRST_ERROR>Step 2</FIRST_ERROR>"
            "<FINAL_ANSWER_CHECK>invalid</FINAL_ANSWER_CHECK>"
            "<VERDICT>REJECT</VERDICT><CONFIDENCE>1</CONFIDENCE>"
            "<CRITIQUE>wrong</CRITIQUE>"
        )
        with self.assertRaises(ValueError):
            assert_complete_trajectory_audit(candidate, partial)

        complete = partial.replace(
            "</TRAJECTORY_AUDIT>",
            '<STEP_AUDIT index="3" status="INVALID">invalid dependency</STEP_AUDIT>'
            "</TRAJECTORY_AUDIT>",
        )
        assert_complete_trajectory_audit(candidate, complete)


if __name__ == "__main__":
    unittest.main()
