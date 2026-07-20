import unittest

from adversarial_gv.batch_trainer import HardCase
from adversarial_gv.progress_planning import measure_progress_line_totals


def case(index: int, collection_split: str = "train") -> HardCase:
    return HardCase(
        wrong_id=index,
        collection_split=collection_split,
        source_split="train",
        source_index=index,
        question="How many objects are there?",
        ground_truth="2",
    )


class ProgressPlanningTests(unittest.TestCase):
    def test_totals_are_measured_from_real_control_flow_for_each_batch_size(self):
        batches = ([case(1), case(2)], [case(3)])
        evaluation_splits = (
            [case(10, "train")],
            [case(11, "val")],
            [case(12, "test")],
        )

        initial_total, batch_totals = measure_progress_line_totals(
            batches,
            evaluation_splits,
        )

        self.assertEqual(initial_total, 6)
        # The fixed trajectory judge adds one measured forward event per
        # training case; totals still come from replaying the real graph.
        self.assertEqual(batch_totals, [54, 33])


if __name__ == "__main__":
    unittest.main()
