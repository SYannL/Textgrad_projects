import unittest

import textgrad as tg
from textgrad.engine import EngineLM

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.oracle_routing import (
    GENERATOR_ORACLE_INSTRUCTION,
    VERIFIER_ORACLE_INSTRUCTION,
    OracleAssessment,
    OracleHardCase,
    OracleRoutedGVTrainer,
    parse_generator_oracle,
    parse_verifier_oracle,
    route_for,
)
from scripts.run_oracle_routed_hardset import (
    build_professor_diagnostics,
    parser,
    validation_candidate_is_better,
)


class GeneratorEngine(EngineLM):
    model_string = "fake-generator"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        prompt = str(prompt)
        answer = "9" if "g-wrong" in prompt else "2"
        return f"Step 1: Compute the quantity.\nAnswer: {answer}"

    generate = __call__


class VerifierEngine(EngineLM):
    model_string = "fake-verifier"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        return "<VERDICT>ACCEPT</VERDICT><CRITIQUE>semantic judgment</CRITIQUE>"

    generate = __call__


class OracleEngine(EngineLM):
    model_string = "fake-oracle"

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, system_prompt=None, **kwargs):
        prompt = str(prompt)
        self.calls.append((prompt, str(system_prompt)))
        if system_prompt == GENERATOR_ORACLE_INSTRUCTION:
            label = "REJECT" if "g-wrong" in prompt else "ACCEPT"
            return (
                f"<GENERATOR_LABEL>{label}</GENERATOR_LABEL>"
                "<GENERATOR_RATIONALE>checked against reference</GENERATOR_RATIONALE>"
            )
        if system_prompt == VERIFIER_ORACLE_INSTRUCTION:
            correct = "NO" if "v-wrong" in prompt else "YES"
            return (
                f"<VERIFIER_CORRECT>{correct}</VERIFIER_CORRECT>"
                "<VERIFIER_RATIONALE>semantic comparison</VERIFIER_RATIONALE>"
            )
        raise AssertionError("unexpected oracle call")

    generate = __call__


class BackwardEngine(EngineLM):
    model_string = "fake-backward"

    def __call__(self, prompt, system_prompt=None, **kwargs):
        if system_prompt and "optimization system that improves text" in str(system_prompt):
            return "<IMPROVED_VARIABLE>general improved strategy</IMPROVED_VARIABLE>"
        return "Use the privileged evidence to correct the general strategy."

    generate = __call__


def make_case(wrong_id: int, markers: str) -> OracleHardCase:
    return OracleHardCase(
        wrong_id=wrong_id,
        collection_split="train",
        source_split="train",
        source_index=wrong_id,
        question=f"Question: {markers}",
        ground_truth="2",
        gold_reasoning="The official calculation gives 2.",
    )


class OracleRoutingTests(unittest.TestCase):
    def test_iterations_cli_default_and_override(self):
        self.assertEqual(parser().parse_args([]).iterations, 3)
        self.assertEqual(parser().parse_args(["--iterations", "5"]).iterations, 5)
        self.assertEqual(parser().parse_args([]).validation_mode, "acc")
        self.assertEqual(
            parser().parse_args(["--validation-mode", "oracle"]).validation_mode,
            "oracle",
        )

    def test_validation_modes_require_strict_improvement(self):
        self.assertTrue(
            validation_candidate_is_better(
                "acc", {"accuracy": 0.5}, {"accuracy": 0.6}
            )
        )
        self.assertFalse(
            validation_candidate_is_better(
                "acc", {"accuracy": 0.5}, {"accuracy": 0.5}
            )
        )
        self.assertTrue(
            validation_candidate_is_better(
                "oracle",
                {"generator_accuracy": 0.5, "verifier_accuracy": 0.4},
                {"generator_accuracy": 0.6, "verifier_accuracy": 0.4},
            )
        )
        self.assertFalse(
            validation_candidate_is_better(
                "oracle",
                {"generator_accuracy": 0.5, "verifier_accuracy": 0.4},
                {"generator_accuracy": 0.6, "verifier_accuracy": 0.3},
            )
        )

    def test_parsers_and_four_routes(self):
        self.assertEqual(
            parse_generator_oracle(
                "<GENERATOR_LABEL>accept</GENERATOR_LABEL>"
                "<GENERATOR_RATIONALE>valid</GENERATOR_RATIONALE>"
            ),
            ("ACCEPT", "valid"),
        )
        self.assertEqual(
            parse_verifier_oracle(
                "<VERIFIER_CORRECT>no</VERIFIER_CORRECT>"
                "<VERIFIER_RATIONALE>wrong claim</VERIFIER_RATIONALE>"
            ),
            (False, "wrong claim"),
        )
        self.assertEqual(route_for(OracleAssessment("ACCEPT", "", True, "")), "none")
        self.assertEqual(
            route_for(OracleAssessment("ACCEPT", "", False, "")),
            "verifier",
        )
        self.assertEqual(
            route_for(OracleAssessment("REJECT", "", True, "")),
            "generator",
        )
        self.assertEqual(route_for(OracleAssessment("REJECT", "", False, "")), "both")

    def test_two_isolated_oracle_calls_and_routed_batch(self):
        oracle = OracleEngine()
        backward = BackwardEngine()
        g_strategy = tg.Variable(
            "generator start",
            requires_grad=True,
            role_description="generator strategy",
        )
        v_strategy = tg.Variable(
            "verifier start",
            requires_grad=True,
            role_description="verifier strategy",
        )
        trainer = OracleRoutedGVTrainer(
            GeneratorAgent(GeneratorEngine(), g_strategy),
            VerifierAgent(VerifierEngine(), v_strategy),
            backward,
            oracle_engine=oracle,
            evaluation_workers=1,
            training_workers=1,
        )
        result = trainer.train_batch(
            [
                make_case(1, "both-correct"),
                make_case(2, "v-wrong"),
                make_case(3, "g-wrong"),
                make_case(4, "g-wrong v-wrong"),
            ],
            1,
        )

        self.assertEqual(
            result["route_counts"],
            {"none": 1, "generator": 1, "verifier": 1, "both": 1},
        )
        self.assertEqual(result["iteration_index"], 1)
        self.assertEqual(result["g_step"]["routed_case_ids"], [3, 4])
        self.assertEqual(result["v_step"]["routed_case_ids"], [2, 4])
        self.assertEqual(g_strategy.value, "general improved strategy")
        self.assertEqual(v_strategy.value, "general improved strategy")
        self.assertEqual(len(oracle.calls), 8)
        for index in range(0, len(oracle.calls), 2):
            first_prompt, first_system = oracle.calls[index]
            second_prompt, second_system = oracle.calls[index + 1]
            self.assertEqual(first_system, GENERATOR_ORACLE_INSTRUCTION)
            self.assertNotIn("<VERIFIER_RESPONSE>", first_prompt)
            self.assertEqual(second_system, VERIFIER_ORACLE_INSTRUCTION)
            self.assertIn("<VERIFIER_RESPONSE>", second_prompt)

        acc_validation = trainer.evaluate_generator_accuracy(
            [make_case(5, "correct"), make_case(6, "g-wrong")]
        )
        self.assertEqual(acc_validation["accuracy"], 0.5)
        oracle_validation = trainer.evaluate_oracle_validation(
            [make_case(7, "correct"), make_case(8, "g-wrong v-wrong")]
        )
        self.assertEqual(oracle_validation["generator_accuracy"], 0.5)
        self.assertEqual(oracle_validation["verifier_accuracy"], 0.5)

    def test_professor_diagnostics_records_first_success_and_regression(self):
        def interaction(wrong_id, g_correct, v_correct):
            return {
                "wrong_id": wrong_id,
                "route": "none",
                "oracle": {
                    "generator_label": "ACCEPT" if g_correct else "REJECT",
                    "verifier_correct": v_correct,
                },
            }

        experiment = {
            "batches": [
                {
                    "batch_index": 1,
                    "case_ids": [10, 11],
                    "iterations": [
                        {
                            "iteration_index": 1,
                            "interactions": [
                                interaction(10, False, False),
                                interaction(11, True, True),
                            ],
                        },
                        {
                            "iteration_index": 2,
                            "interactions": [
                                interaction(10, True, False),
                                interaction(11, False, False),
                            ],
                        },
                        {
                            "iteration_index": 3,
                            "interactions": [
                                interaction(10, True, False),
                                interaction(11, True, True),
                            ],
                        },
                    ],
                }
            ]
        }
        diagnostics = build_professor_diagnostics(experiment)
        by_id = {row["wrong_id"]: row for row in diagnostics["cases"]}
        self.assertEqual(by_id[10]["first_generator_correct_round"], 2)
        self.assertIsNone(by_id[10]["first_verifier_correct_round"])
        self.assertTrue(
            by_id[10]["generator_eventually_correct_verifier_never_correct"]
        )
        self.assertEqual(by_id[11]["generator_regression_rounds"], [2])
        self.assertEqual(by_id[11]["verifier_regression_rounds"], [2])
        self.assertEqual(diagnostics[
            "generator_eventually_correct_verifier_never_correct_ids"
        ], [10])
        self.assertEqual(diagnostics["professor_target_ids"], [10])


if __name__ == "__main__":
    unittest.main()
