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
                "<VERDICT>ACCEPT</VERDICT>"
                "<CONFIDENCE>1</CONFIDENCE>"
                "<CRITIQUE>valid</CRITIQUE>"
            )
        return "Reasoning\nAnswer: 2"


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
        v_prompt = tg.Variable(
            "verifier", requires_grad=True, role_description="verifier prompt"
        )
        question = tg.Variable(
            "Count two objects.", requires_grad=False, role_description="question"
        )
        candidate = GeneratorAgent(engine, g_prompt).run(question)
        verdict = VerifierAgent(engine, v_prompt).run(question, candidate)
        graph = ancestors(verdict)
        self.assertIn(g_prompt, graph)
        self.assertIn(v_prompt, graph)


if __name__ == "__main__":
    unittest.main()
