import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textgrad.engine import EngineLM

from adversarial_gv.initial_baseline import run


class FakeEngine(EngineLM):
    model_string = "fake-model"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        answer = str(prompt).rsplit(" ", 1)[-1]
        return f"Step 1: Use the stated value.\nAnswer: {answer}"


class InitialBaselineTests(unittest.TestCase):
    def test_one_pass_baseline_writes_split_metrics_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "samples.csv"
            output = root / "output"
            with data.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "wrong_id",
                        "collection_split",
                        "difficulty",
                        "split",
                        "index",
                        "question",
                        "ground_truth",
                    ),
                )
                writer.writeheader()
                for index, split in enumerate(("train", "val", "test"), start=1):
                    writer.writerow(
                        {
                            "wrong_id": index,
                            "collection_split": split,
                            "difficulty": "easy",
                            "split": "test",
                            "index": index,
                            "question": f"Question: Return {index}",
                            "ground_truth": str(index),
                        }
                    )

            arguments = [
                "--data",
                str(data),
                "--output-dir",
                str(output),
                "--vllm-base-url",
                "http://127.0.0.1:9999/v1",
                "--workers",
                "2",
            ]
            with patch(
                "adversarial_gv.initial_baseline.build_engine",
                return_value=FakeEngine(),
            ) as mocked_engine:
                run(
                    arguments,
                    default_model="fake-model",
                    default_base_url="http://127.0.0.1:9999/v1",
                    default_output_dir=str(output),
                )
                run(
                    arguments,
                    default_model="fake-model",
                    default_base_url="http://127.0.0.1:9999/v1",
                    default_output_dir=str(output),
                )

            self.assertEqual(mocked_engine.call_count, 1)
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["metrics"]["overall"]["accuracy"], 1.0)
            self.assertEqual(summary["metrics"]["train"]["accuracy"], 1.0)
            self.assertEqual(summary["metrics"]["val"]["accuracy"], 1.0)
            self.assertEqual(summary["metrics"]["test"]["accuracy"], 1.0)
            self.assertEqual(
                len((output / "predictions.jsonl").read_text().splitlines()),
                3,
            )


if __name__ == "__main__":
    unittest.main()

