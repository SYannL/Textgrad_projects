import unittest

import textgrad as tg
from textgrad.engine import EngineLM

from adversarial_gv.agents import GeneratorAgent, VerifierAgent
from adversarial_gv.data import Case
from adversarial_gv.trainer import AdversarialGVTrainer, TrainingConfig


class GeneratorEngine(EngineLM):
    model_string = "fake-generator"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        return "There are two objects.\nAnswer: 2"


class VerifierEngine(EngineLM):
    model_string = "fake-verifier"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        return (
            "<VERDICT>ACCEPT</VERDICT>"
            "<CONFIDENCE>1</CONFIDENCE>"
            "<CRITIQUE>The count is consistent.</CRITIQUE>"
        )


class BackwardEngine(EngineLM):
    model_string = "fake-backward"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        self.prompts.append(str(prompt))
        if system_prompt and "optimization system that improves text" in system_prompt:
            return "<IMPROVED_VARIABLE>general improved prompt</IMPROVED_VARIABLE>"
        return "Give more precise task-specific feedback and verify the arithmetic."


class TrainerTests(unittest.TestCase):
    def test_one_alternating_iteration_runs_without_network(self):
        g_prompt = tg.Variable(
            "generator", requires_grad=True, role_description="generator prompt"
        )
        v_prompt = tg.Variable(
            "verifier", requires_grad=True, role_description="verifier prompt"
        )
        trainer = AdversarialGVTrainer(
            GeneratorAgent(GeneratorEngine(), g_prompt),
            VerifierAgent(VerifierEngine(), v_prompt),
            BackwardEngine(),
            TrainingConfig(iterations=1),
        )
        result = trainer.train(
            Case("I have two apples. How many objects?", "2", "train", 0)
        )
        self.assertEqual(len(result["steps"]), 1)
        self.assertTrue(result["final"]["correct"])
        self.assertEqual(result["final"]["verdict"]["label"], "ACCEPT")
        self.assertEqual(g_prompt.value, "general improved prompt")
        self.assertEqual(v_prompt.value, "general improved prompt")

    def test_gold_reasoning_mode_adds_expected_generator_trajectory(self):
        g_prompt = tg.Variable(
            "generator", requires_grad=True, role_description="generator prompt"
        )
        v_prompt = tg.Variable(
            "verifier", requires_grad=True, role_description="verifier prompt"
        )
        backward = BackwardEngine()
        trainer = AdversarialGVTrainer(
            GeneratorAgent(GeneratorEngine(), g_prompt),
            VerifierAgent(VerifierEngine(), v_prompt),
            backward,
            TrainingConfig(
                iterations=1,
                generator_supervision_mode="gold_reasoning",
            ),
        )
        trainer.train(
            Case(
                "I have two apples. How many objects?",
                "2",
                "train",
                0,
                dataset="gsm8k",
                gold_reasoning="There are two apples, so the total is 2.",
            )
        )
        prompts = "\n".join(backward.prompts)
        self.assertIn("<GOLD_REASONING training_only='true'>", prompts)
        self.assertIn("There are two apples, so the total is 2.", prompts)


if __name__ == "__main__":
    unittest.main()
