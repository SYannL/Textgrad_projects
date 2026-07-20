import unittest

from scripts.run_hardset_main import parser


class HardsetCliTests(unittest.TestCase):
    def test_separate_generator_verifier_and_backward_endpoints(self):
        args = parser().parse_args(
            [
                "--generator-vllm-base-url",
                "http://127.0.0.1:8004/v1",
                "--verifier-vllm-base-url",
                "http://127.0.0.1:8000/v1",
                "--backward-vllm-base-url",
                "http://127.0.0.1:8000/v1",
            ]
        )
        self.assertEqual(
            args.generator_vllm_base_url,
            "http://127.0.0.1:8004/v1",
        )
        self.assertEqual(
            args.verifier_vllm_base_url,
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(
            args.backward_vllm_base_url,
            "http://127.0.0.1:8000/v1",
        )


if __name__ == "__main__":
    unittest.main()
