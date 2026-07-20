import unittest
from types import SimpleNamespace

from adversarial_gv.engines import TokenUsageTracker, merge_token_usage
from adversarial_gv.wandb_monitoring import WandbMonitor, token_metrics


class FakeRun:
    id = "run-123"
    url = "https://wandb.example/run-123"

    def __init__(self):
        self.rows = []
        self.finished = False

    def log(self, row):
        self.rows.append(row)

    def finish(self):
        self.finished = True


class WandbMonitoringTests(unittest.TestCase):
    def test_token_tracker_separates_roles_and_computes_overall(self):
        tracker = TokenUsageTracker()
        tracker.record(
            "generator",
            SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
            ),
        )
        tracker.record(
            "verifier",
            SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=6,
                total_tokens=26,
            ),
        )

        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["generator"]["total_tokens"], 14)
        self.assertEqual(snapshot["verifier"]["total_tokens"], 26)
        self.assertEqual(snapshot["all"]["prompt_tokens"], 30)
        self.assertEqual(snapshot["all"]["total_tokens"], 40)

    def test_persisted_and_current_token_usage_are_merged(self):
        merged = merge_token_usage(
            {
                "generator": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "all": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            {
                "generator": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
                "all": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
        )
        self.assertEqual(merged["generator"]["total_tokens"], 25)
        self.assertEqual(merged["all"]["total_tokens"], 25)

    def test_wandb_logs_train_and_val_accuracy_but_not_test(self):
        run = FakeRun()
        monitor = WandbMonitor(run)
        evaluation = {
            "stage": "after_batch",
            "batch_index": 2,
            "train": {
                "accuracy": 0.6,
                "accept_rate": 0.5,
                "challenge_rate": 0.2,
                "reject_rate": 0.3,
                "invalid_rate": 0.0,
            },
            "val": {
                "accuracy": 0.7,
                "accept_rate": 0.6,
                "challenge_rate": 0.1,
                "reject_rate": 0.3,
                "invalid_rate": 0.0,
            },
            "test": {
                "accuracy": 0.8,
                "accept_rate": 0.7,
                "challenge_rate": 0.1,
                "reject_rate": 0.2,
                "invalid_rate": 0.0,
            },
        }
        usage = {
            "all": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            }
        }

        monitor.log_evaluation(evaluation, usage, 200, 300)

        row = run.rows[-1]
        self.assertEqual(row["accuracy/train"], 0.6)
        self.assertEqual(row["accuracy/val"], 0.7)
        self.assertNotIn("accuracy/test", row)
        self.assertEqual(row["challenge_rate/train"], 0.2)
        self.assertEqual(row["reject_rate/val"], 0.3)
        self.assertEqual(row["tokens/all/total"], 120)
        self.assertEqual(row["prompt_chars/generator_strategy"], 200)
        self.assertEqual(row["prompt_chars/verifier_strategy"], 300)

    def test_token_metric_names_are_wandb_chart_friendly(self):
        metrics = token_metrics(
            {
                "backward": {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                }
            }
        )
        self.assertEqual(metrics["tokens/backward/prompt"], 9)
        self.assertEqual(metrics["tokens/backward/completion"], 4)
        self.assertEqual(metrics["tokens/backward/total"], 13)


if __name__ == "__main__":
    unittest.main()
