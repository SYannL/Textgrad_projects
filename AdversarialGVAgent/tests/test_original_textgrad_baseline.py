import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import textgrad as tg
from textgrad.engine import EngineLM

from adversarial_gv.original_textgrad_baseline import (
    ORIGINAL_GSM8K_SYSTEM_PROMPT,
    original_batch_schedule,
    run,
)


class FakeForwardEngine(EngineLM):
    model_string = "fake-forward"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        answer = str(prompt).rsplit(" ", 1)[-1]
        return f"Reason through the request.\nAnswer: {answer}"


class FakeBackwardEngine(EngineLM):
    model_string = "fake-backward"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        if "optimization system that improves text" in str(system_prompt):
            return "<IMPROVED_VARIABLE>Original-protocol improved prompt</IMPROVED_VARIABLE>"
        return "The answer is correct; retain a general step-by-step strategy."


class OriginalTextGradBaselineTests(unittest.TestCase):
    def test_schedule_exactly_matches_original_dataloader(self):
        expected = original_batch_schedule(208)
        np.random.seed(42)
        loader = tg.tasks.DataLoader(list(range(208)), batch_size=3, shuffle=True)
        actual = []
        for _ in range(3):
            for step, batch in enumerate(loader):
                actual.append([int(value) for value in batch])
                if step == 3:
                    break
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 12)

    def test_original_prompt_no_constraints_no_backward_cap_and_resume(self):
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
                "--max-epochs",
                "1",
                "--steps-per-epoch",
                "1",
                "--workers",
                "1",
            ]
            with patch(
                "adversarial_gv.original_textgrad_baseline.build_engine",
                side_effect=[FakeForwardEngine(), FakeBackwardEngine()],
            ):
                run(arguments)

            config = json.loads(
                (output / "run_config.json").read_text(encoding="utf-8")
            )
            state = json.loads((output / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(config["starting_system_prompt"], ORIGINAL_GSM8K_SYSTEM_PROMPT)
            self.assertEqual(config["optimizer_constraints"], [])
            self.assertIsNone(config["backward_max_tokens_override"])
            self.assertEqual(state["status"], "complete")
            self.assertEqual(
                state["current_prompt"], "Original-protocol improved prompt"
            )
            self.assertEqual(state["history"][0]["decision"], "tied")

            traces = [
                json.loads(line)
                for line in (output / "textgrad_calls.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            optimizer_call = next(
                record for record in traces if record["kind"] == "optimizer_update"
            )
            self.assertNotIn("<CONSTRAINTS>", optimizer_call["prompt"])
            backward_calls = [
                record for record in traces if record["kind"] == "backward_gradient"
            ]
            self.assertEqual(len(backward_calls), 2)
            self.assertTrue(
                all("max_tokens" not in record["kwargs"] for record in backward_calls)
            )
            self.assertTrue(
                all(
                    "Return at most 500 tokens." not in record["system_prompt"]
                    for record in backward_calls
                )
            )

            with patch(
                "adversarial_gv.original_textgrad_baseline.build_engine"
            ) as build:
                run(arguments)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
