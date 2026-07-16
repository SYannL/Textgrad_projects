import csv
import tempfile
import unittest
from pathlib import Path

from adversarial_gv.reporting import append_result_csv


def evaluation(label="ACCEPT"):
    return {
        "case": {"question": "q", "answer": "2", "split": "train", "index": 0},
        "answer": "Answer: 2",
        "verdict_raw": label,
        "verdict": {"label": label, "confidence": 1.0, "critique": "ok"},
        "correct": True,
    }


class ReportingTests(unittest.TestCase):
    def test_csv_contains_initial_iteration_and_final_rows(self):
        result = {
            "dataset": "BBH_object_counting",
            "models": {"generator": "g", "verifier": "v", "backward": "b"},
            "initial": evaluation("REJECT"),
            "check_initial": None,
            "steps": [
                {
                    "iteration": 1,
                    "g_prompt_before": "g0",
                    "g_prompt_after": "g1",
                    "v_prompt_before": "v0",
                    "v_prompt_after": "v1",
                    "evaluation": evaluation(),
                    "check": None,
                }
            ],
            "final": evaluation(),
            "check_final": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            append_result_csv(
                result, path, run_id="run-1", recorded_at="2026-01-01T00:00:00Z"
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([row["phase"] for row in rows], ["initial", "iteration", "final"])
        self.assertEqual(rows[-1]["verifier_label"], "ACCEPT")
        self.assertEqual(rows[-1]["generator_prompt"], "g1")


if __name__ == "__main__":
    unittest.main()
