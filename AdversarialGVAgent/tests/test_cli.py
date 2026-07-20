import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from adversarial_gv.cli import build_parser
from adversarial_gv.engines import build_engine, resolve_role_base_urls
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
        self.assertIsNone(args.vllm_base_url)
        self.assertIsNone(args.vllm_api_key)

    def test_vllm_arguments(self):
        args = build_parser().parse_args(
            [
                "--generator-model",
                "local-model",
                "--vllm-base-url",
                "http://localhost:8000/v1",
                "--vllm-api-key",
                "secret",
            ]
        )
        self.assertEqual(args.generator_model, "local-model")
        self.assertEqual(args.vllm_base_url, "http://localhost:8000/v1")
        self.assertEqual(args.vllm_api_key, "secret")

    def test_role_specific_vllm_arguments(self):
        args = build_parser().parse_args(
            [
                "--vllm-base-url",
                "http://localhost:8999/v1",
                "--generator-vllm-base-url",
                "http://localhost:8004/v1/",
                "--verifier-vllm-base-url",
                "http://localhost:8000/v1",
            ]
        )
        urls = resolve_role_base_urls(
            shared=args.vllm_base_url,
            generator=args.generator_vllm_base_url,
            verifier=args.verifier_vllm_base_url,
            backward=args.backward_vllm_base_url,
        )
        self.assertEqual(urls["generator"], "http://localhost:8004/v1")
        self.assertEqual(urls["verifier"], "http://localhost:8000/v1")
        self.assertEqual(urls["backward"], "http://localhost:8999/v1")

    def test_vllm_engine_uses_external_openai_client(self):
        with TemporaryDirectory() as cache_home:
            with patch.dict("os.environ", {"XDG_CACHE_HOME": cache_home}):
                engine = build_engine(
                    "local-model", "http://localhost:8000/v1/", None
                )
                self.assertEqual(engine.model_string, "local-model")
                self.assertEqual(
                    str(engine.client.base_url), "http://localhost:8000/v1/"
                )
                self.assertEqual(engine.client.api_key, "EMPTY")

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
