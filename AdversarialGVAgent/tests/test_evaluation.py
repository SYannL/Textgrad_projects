import unittest

from adversarial_gv.evaluation import is_correct, parse_verdict


class EvaluationTests(unittest.TestCase):
    def test_integer_correctness_matches_textgrad_task_metric(self):
        self.assertTrue(is_correct("Reasoning\nAnswer: 10", "10"))
        self.assertFalse(is_correct("Reasoning\nAnswer: 9", "10"))

    def test_parse_verdict(self):
        parsed = parse_verdict(
            "<VERDICT>ACCEPT</VERDICT>\n"
            "<CONFIDENCE>0.9</CONFIDENCE>\n"
            "<CRITIQUE>The arithmetic checks out.</CRITIQUE>"
        )
        self.assertEqual(parsed.label, "ACCEPT")
        self.assertEqual(parsed.confidence, 0.9)
        self.assertEqual(parsed.critique, "The arithmetic checks out.")

    def test_malformed_verdict_is_invalid(self):
        parsed = parse_verdict("Looks good")
        self.assertEqual(parsed.label, "INVALID")
        self.assertIsNone(parsed.confidence)


if __name__ == "__main__":
    unittest.main()

