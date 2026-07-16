"""Replaceable agent boundaries and their one-LLM implementations."""

from abc import ABC, abstractmethod
from typing import List

import textgrad as tg
from textgrad.autograd import FormattedLLMCall, LLMCall
from textgrad.engine import EngineLM


class LLMAgent(ABC):
    """Contract required by the trainer.

    Future agents may use tools or multiple internal calls, but must return a
    TextGrad Variable whose graph connects the result to trainable parameters.
    """

    @abstractmethod
    def parameters(self) -> List[tg.Variable]:
        raise NotImplementedError


class GeneratorAgent(LLMAgent):
    def __init__(self, engine: EngineLM, system_prompt: tg.Variable):
        self.system_prompt = system_prompt
        self._call = LLMCall(engine=engine, system_prompt=system_prompt)

    def parameters(self) -> List[tg.Variable]:
        return [self.system_prompt]

    def run(self, question: tg.Variable) -> tg.Variable:
        return self._call(
            question,
            response_role_description="generator reasoning and final answer",
        )


class VerifierAgent(LLMAgent):
    def __init__(self, engine: EngineLM, system_prompt: tg.Variable):
        self.system_prompt = system_prompt
        self._call = FormattedLLMCall(
            engine=engine,
            system_prompt=system_prompt,
            format_string=(
                "<QUESTION>\n{question}\n</QUESTION>\n"
                "<CANDIDATE_ANSWER>\n{candidate}\n</CANDIDATE_ANSWER>"
            ),
            fields={"question": None, "candidate": None},
        )

    def parameters(self) -> List[tg.Variable]:
        return [self.system_prompt]

    def run(self, question: tg.Variable, candidate: tg.Variable) -> tg.Variable:
        return self._call(
            inputs={"question": question, "candidate": candidate},
            response_role_description="verifier verdict, confidence, and critique",
        )

