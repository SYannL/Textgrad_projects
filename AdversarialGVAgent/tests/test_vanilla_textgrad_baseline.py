import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from textgrad.engine import EngineLM

from adversarial_gv.vanilla_textgrad_baseline import _batch_schedule, run


class FakeForwardEngine(EngineLM):
    model_string = "fake-forward"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        answer = str(prompt).rsplit(" ", 1)[-1]
        return f"Step 1: Return the requested number.\nAnswer: {answer}"


class FakeBackwardEngine(EngineLM):
    model_string = "fake-backward"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        if "optimization system that improves text" in str(system_prompt):
            return "<IMPROVED_VARIABLE>Improved test prompt</IMPROVED_VARIABLE>"
        return "The current answer is correct; preserve the general strategy."


class VanillaTextGradBaselineTests(unittest.TestCase):
    def test_schedule_matches_gvgan_batches(self):
        schedule = _batch_schedule(list(range(208)), batch_size=6)
        self.assertEqual(len(schedule), 35)
        self.assertEqual(schedule[0], [0, 1, 2, 3, 4, 5])
        self.assertEqual(schedule[-1], [204, 205, 206, 207])

    def test_gvgan_aligned_strategy_update_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "samples.csv"
            output = root / "run"
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
                "--forward-vllm-base-url",
                "http://127.0.0.1:9998/v1",
                "--backward-vllm-base-url",
                "http://127.0.0.1:9999/v1",
                "--batch-size",
                "1",
                "--workers",
                "1",
            ]
            forward = FakeForwardEngine()
            backward = FakeBackwardEngine()
            with patch(
                "adversarial_gv.vanilla_textgrad_baseline.build_engine",
                side_effect=[forward, backward],
            ):
                run(arguments)

            state = json.loads((output / "state.json").read_text(encoding="utf-8"))
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["current_strategy"], "Improved test prompt")
            self.assertEqual(state["history"][0]["update_status"], "updated")
            self.assertEqual(len(state["evaluations"]), 2)
            self.assertEqual(summary["metrics"]["overall"]["accuracy"], 1.0)
            traces = [
                json.loads(line)
                for line in (output / "textgrad_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [record["kind"] for record in traces].count("optimizer_update"),
                1,
            )
            backward_calls = [
                record for record in traces if record["kind"] == "backward_gradient"
            ]
            self.assertEqual(len(backward_calls), 2)
            self.assertTrue(
                all(record["kwargs"]["max_tokens"] == 512 for record in backward_calls)
            )
            self.assertTrue(
                all("Return at most 500 tokens." in record["system_prompt"] for record in backward_calls)
            )

            with patch(
                "adversarial_gv.vanilla_textgrad_baseline.build_engine"
            ) as build:
                run(arguments)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
