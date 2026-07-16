import unittest

from adversarial_gv.gradient_reporting import gradient_rows


class GradientReportingTests(unittest.TestCase):
    def test_flattens_v_and_g_calls(self):
        call = {
            "kind": "backward_gradient",
            "system_prompt": "system",
            "prompt": "gradient prompt",
            "response": "feedback",
        }
        result = {
            "steps": [
                {
                    "iteration": 1,
                    "v_training_interaction": {"gradient_trace": [call]},
                    "g_training_interaction": {"gradient_trace": [call]},
                }
            ]
        }
        rows = gradient_rows(result, run_id="run", recorded_at="now")
        self.assertEqual([row["stage"] for row in rows], ["V", "G"])
        self.assertEqual(rows[0]["gradient_prompt"], "gradient prompt")


if __name__ == "__main__":
    unittest.main()
