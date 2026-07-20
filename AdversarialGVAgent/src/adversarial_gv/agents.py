"""Replaceable agent boundaries and their one-LLM implementations."""

from abc import ABC, abstractmethod
from typing import List

import textgrad as tg
from textgrad.autograd import FormattedLLMCall, LLMCall
from textgrad.engine import EngineLM


_STRATEGY_PREFIX = "\n\n<TRAINABLE_STRATEGY>\n"
_STRATEGY_SUFFIX = "\n</TRAINABLE_STRATEGY>"


def _backward_to_strategy_once(
    *,
    combined_prompt: tg.Variable,
    strategy_prompt: tg.Variable,
    backward_engine: EngineLM,
) -> None:
    """Pass each system-prompt gradient to the editable strategy exactly once.

    The immutable system text is useful forward context, but it is not an
    optimization parameter.  Copying only the raw feedback also avoids the
    recursive context wrapping performed by Variable.__add__ chains.
    """
    del backward_engine  # Gradient transfer itself does not require an LLM call.
    for gradient in combined_prompt.gradients:
        strategy_prompt.gradients.add(gradient)
        # The feedback was already computed with the complete fixed+strategy
        # system prompt in view.  Reattaching that conversation here would send
        # the same context to the optimizer a second time.
        strategy_prompt.gradients_context[gradient] = None
        if gradient._reduce_meta:
            strategy_prompt._reduce_meta.extend(gradient._reduce_meta)


def _combined_system_prompt(
    fixed_system_prompt: tg.Variable,
    strategy_prompt: tg.Variable,
    agent_name: str,
) -> tg.Variable:
    """Build one differentiable node whose only editable predecessor is strategy."""
    combined = tg.Variable(
        value=(
            fixed_system_prompt.value
            + _STRATEGY_PREFIX
            + strategy_prompt.value
            + _STRATEGY_SUFFIX
        ),
        predecessors=[strategy_prompt],
        requires_grad=strategy_prompt.requires_grad,
        role_description=(
            f"{agent_name} system prompt with immutable rules and one editable "
            "TRAINABLE_STRATEGY section; feedback must target only that section"
        ),
    )
    if combined.requires_grad:
        from functools import partial

        combined.set_grad_fn(
            partial(
                _backward_to_strategy_once,
                combined_prompt=combined,
                strategy_prompt=strategy_prompt,
            )
        )
    return combined


class LLMAgent(ABC):
    """Contract required by the trainer.

    Future agents may use tools or multiple internal calls, but must return a
    TextGrad Variable whose graph connects the result to trainable parameters.
    """

    @abstractmethod
    def parameters(self) -> List[tg.Variable]:
        raise NotImplementedError


class GeneratorAgent(LLMAgent):
    def __init__(
        self,
        engine: EngineLM,
        strategy_prompt: tg.Variable,
        fixed_system_prompt: tg.Variable | None = None,
    ):
        self.engine = engine
        self.strategy_prompt = strategy_prompt
        self.fixed_system_prompt = fixed_system_prompt or tg.Variable(
            "",
            requires_grad=False,
            role_description="empty fixed Generator system prompt",
        )

    @property
    def system_prompt(self) -> tg.Variable:
        """Backward-compatible name for the only trainable prompt variable."""
        return self.strategy_prompt

    def _combined_prompt(self) -> tg.Variable:
        return _combined_system_prompt(
            self.fixed_system_prompt,
            self.strategy_prompt,
            "Generator",
        )

    def parameters(self) -> List[tg.Variable]:
        return [self.strategy_prompt]

    def run(self, question: tg.Variable) -> tg.Variable:
        call = LLMCall(engine=self.engine, system_prompt=self._combined_prompt())
        return call(
            question,
            response_role_description="generator reasoning and final answer",
        )


class VerifierAgent(LLMAgent):
    def __init__(
        self,
        engine: EngineLM,
        strategy_prompt: tg.Variable,
        fixed_system_prompt: tg.Variable | None = None,
    ):
        self.engine = engine
        self.strategy_prompt = strategy_prompt
        self.fixed_system_prompt = fixed_system_prompt or tg.Variable(
            "",
            requires_grad=False,
            role_description="empty fixed Verifier system prompt",
        )

    @property
    def system_prompt(self) -> tg.Variable:
        """Backward-compatible name for the only trainable prompt variable."""
        return self.strategy_prompt

    def _combined_prompt(self) -> tg.Variable:
        return _combined_system_prompt(
            self.fixed_system_prompt,
            self.strategy_prompt,
            "Verifier",
        )

    def _call(self) -> FormattedLLMCall:
        return FormattedLLMCall(
            engine=self.engine,
            system_prompt=self._combined_prompt(),
            format_string=(
                "<QUESTION>\n{question}\n</QUESTION>\n"
                "<CANDIDATE_ANSWER>\n{candidate}\n</CANDIDATE_ANSWER>"
            ),
            fields={"question": None, "candidate": None},
        )

    def parameters(self) -> List[tg.Variable]:
        return [self.strategy_prompt]

    def run(self, question: tg.Variable, candidate: tg.Variable) -> tg.Variable:
        return self._call()(
            inputs={"question": question, "candidate": candidate},
            response_role_description="verifier verdict, confidence, and critique",
        )
