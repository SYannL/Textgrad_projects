import unittest

from adversarial_gv.data import case_from_dataset


class DataTests(unittest.TestCase):
    def test_case_from_two_field_dataset_has_no_gold_reasoning(self):
        case = case_from_dataset([("question", "2")], "bbh_object_counting", 0, "train")
        self.assertEqual(case.question, "question")
        self.assertEqual(case.answer, "2")
        self.assertIsNone(case.gold_reasoning)

    def test_case_from_three_field_dataset_keeps_gold_reasoning(self):
        case = case_from_dataset(
            [("question", "2", "one plus one equals two")],
            "gsm8k",
            0,
            "train",
        )
        self.assertEqual(case.answer, "2")
        self.assertEqual(case.gold_reasoning, "one plus one equals two")


if __name__ == "__main__":
    unittest.main()
