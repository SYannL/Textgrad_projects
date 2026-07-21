import unittest
from collections import Counter

from scripts.sample_balanced_hardset import sample_rows


class BalancedSubsetTests(unittest.TestCase):
    def test_sampling_is_balanced_and_deterministic(self):
        rows = []
        for split in ("train", "val", "test"):
            for difficulty in ("easy", "hard"):
                for index in range(6):
                    rows.append(
                        {
                            "wrong_id": f"{split}-{difficulty}-{index}",
                            "collection_split": split,
                            "difficulty": difficulty,
                        }
                    )
        kwargs = {
            "train_per_class": 3,
            "val_per_class": 2,
            "test_per_class": 1,
            "seed": 42,
        }
        first = sample_rows(rows, **kwargs)
        second = sample_rows(rows, **kwargs)
        self.assertEqual(first, second)
        counts = Counter(
            (row["collection_split"], row["difficulty"])
            for row in first
        )
        self.assertEqual(counts[("train", "easy")], 3)
        self.assertEqual(counts[("train", "hard")], 3)
        self.assertEqual(counts[("val", "easy")], 2)
        self.assertEqual(counts[("val", "hard")], 2)
        self.assertEqual(counts[("test", "easy")], 1)
        self.assertEqual(counts[("test", "hard")], 1)
        for offset in range(0, len(first), 2):
            self.assertEqual(first[offset]["difficulty"], "easy")
            self.assertEqual(first[offset + 1]["difficulty"], "hard")


if __name__ == "__main__":
    unittest.main()
