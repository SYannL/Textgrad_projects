import unittest

import textgrad as tg
from textgrad.engine import EngineLM

from adversarial_gv.agents import GeneratorAgent, VerifierAgent


class FakeEngine(EngineLM):
    model_string = "fake"

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        if "CANDIDATE_ANSWER" in str(prompt):
            return (
                '<TRAJECTORY_AUDIT><STEP_AUDIT index="1" status="VALID">'
                "valid</STEP_AUDIT></TRAJECTORY_AUDIT>"
                "<FIRST_ERROR>NONE</FIRST_ERROR>"
                "<FINAL_ANSWER_CHECK>valid</FINAL_ANSWER_CHECK>"
                "<VERDICT>ACCEPT</VERDICT>"
                "<CONFIDENCE>1</CONFIDENCE>"
                "<CRITIQUE>valid</CRITIQUE>"
            )
        return "Step 1: There are two objects.\nAnswer: 2"


def ancestors(variable):
    found = set()
    pending = [variable]
    while pending:
        current = pending.pop()
        for predecessor in current.predecessors:
            if predecessor not in found:
                found.add(predecessor)
                pending.append(predecessor)
    return found


class AgentGraphTests(unittest.TestCase):
    def test_verdict_graph_connects_both_prompts(self):
        engine = FakeEngine()
        g_prompt = tg.Variable(
            "generator", requires_grad=True, role_description="generator prompt"
        )
        g_fixed = tg.Variable(
            "fixed generator\n", requires_grad=False, role_description="fixed generator"
        )
        v_prompt = tg.Variable(
            "verifier", requires_grad=True, role_description="verifier prompt"
        )
        v_fixed = tg.Variable(
            "fixed verifier\n", requires_grad=False, role_description="fixed verifier"
        )
        question = tg.Variable(
            "Count two objects.", requires_grad=False, role_description="question"
        )
        candidate = GeneratorAgent(engine, g_prompt, g_fixed).run(question)
        verdict = VerifierAgent(engine, v_prompt, v_fixed).run(question, candidate)
        graph = ancestors(verdict)
        self.assertIn(g_prompt, graph)
        self.assertIn(v_prompt, graph)
        self.assertNotIn(g_fixed, graph)
        self.assertNotIn(v_fixed, graph)

    def test_combined_prompt_passes_one_raw_gradient_per_example(self):
        engine = FakeEngine()
        strategy = tg.Variable(
            "strategy", requires_grad=True, role_description="trainable strategy"
        )
        fixed = tg.Variable(
            "fixed rules", requires_grad=False, role_description="fixed rules"
        )
        agent = VerifierAgent(engine, strategy, fixed)

        for index in range(6):
            combined = agent._combined_prompt()
            gradient = tg.Variable(
                f"feedback-{index}",
                requires_grad=False,
                role_description="raw feedback",
            )
            combined.gradients.add(gradient)
            combined.grad_fn(backward_engine=None)

        self.assertEqual(len(strategy.gradients), 6)
        self.assertEqual(
            {gradient.value for gradient in strategy.gradients},
            {f"feedback-{index}" for index in range(6)},
        )
        self.assertEqual(strategy.get_gradient_text().count("feedback-"), 6)

    def test_textgrad_idempotent_backward_adds_each_gradient_once(self):
        from textgrad.variable import _backward_idempotent

        target = tg.Variable(
            "target", requires_grad=True, role_description="target variable"
        )
        summation = tg.Variable(
            "sum", requires_grad=True, role_description="summation"
        )
        summation.gradients.add(
            tg.Variable(
                "one feedback",
                requires_grad=False,
                role_description="feedback",
            )
        )

        _backward_idempotent([target], summation, backward_engine=None)

        self.assertEqual(len(target.gradients), 1)
        self.assertEqual(target.get_gradient_text().count("one feedback"), 1)


if __name__ == "__main__":
    unittest.main()
