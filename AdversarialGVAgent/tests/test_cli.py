import unittest

from adversarial_gv.cli import build_parser
from adversarial_gv.trainer import TrainingConfig


class CliTests(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.generator_model, "gpt-4o-mini")
        self.assertEqual(args.verifier_model, "gpt-4o-mini")
        self.assertEqual(args.backward_model, "gpt-4o-mini")
        self.assertEqual(args.dataset, "bbh_object_counting")
        self.assertEqual(args.generator_supervision_mode, "final_answer")
        self.assertFalse(args.run_check)

    def test_check_is_parameter_driven(self):
        args = build_parser().parse_args(["--run-check", "--check-case-index", "3"])
        self.assertTrue(args.run_check)
        self.assertEqual(args.check_case_index, 3)

    def test_iterations_must_be_positive(self):
        with self.assertRaises(ValueError):
            TrainingConfig(iterations=0)

    def test_generator_supervision_mode_argument(self):
        args = build_parser().parse_args(
            ["--generator-supervision-mode", "gold_reasoning"]
        )
        self.assertEqual(args.generator_supervision_mode, "gold_reasoning")

    def test_generator_supervision_mode_must_be_supported(self):
        with self.assertRaises(ValueError):
            TrainingConfig(generator_supervision_mode="step_labels")

    def test_gsm8k_initial_wrong_search_arguments(self):
        args = build_parser().parse_args(
            ["--dataset", "gsm8k", "--require-initial-wrong", "--search-cases", "30"]
        )
        self.assertEqual(args.dataset, "gsm8k")
        self.assertTrue(args.require_initial_wrong)
        self.assertEqual(args.search_cases, 30)


if __name__ == "__main__":
    unittest.main()
