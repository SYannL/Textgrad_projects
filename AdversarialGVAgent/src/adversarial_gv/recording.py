"""Record TextGrad evaluator, backward, and optimizer calls for auditing."""

from typing import Any, Dict, List

from textgrad.engine import EngineLM


BACKWARD_MAX_TOKENS = 512
BACKWARD_LENGTH_INSTRUCTION = "Return at most 500 tokens."


class RecordingEngine(EngineLM):
    def __init__(self, engine: EngineLM):
        self.engine = engine
        self.model_string = getattr(engine, "model_string", type(engine).__name__)
        self.records: List[Dict[str, Any]] = []

    @staticmethod
    def _kind(system_prompt: Any, prompt: Any) -> str:
        system_text = str(system_prompt or "")
        prompt_text = str(prompt)
        if "optimization system that improves text" in system_text:
            return "optimizer_update"
        if (
            "gradient (feedback) engine" in system_text.lower()
            or "feedback to a variable" in prompt_text.lower()
        ):
            return "backward_gradient"
        if "reduce" in system_text.lower() and "gradient" in system_text.lower():
            return "gradient_reduction"
        return "loss_evaluation"

    def mark(self) -> int:
        return len(self.records)

    def since(self, mark: int) -> List[Dict[str, Any]]:
        return self.records[mark:]

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        kind = self._kind(system_prompt, prompt)
        if kind == "backward_gradient":
            kwargs.setdefault("max_tokens", BACKWARD_MAX_TOKENS)
            system_prompt = (
                f"{str(system_prompt).rstrip()}\n\n{BACKWARD_LENGTH_INSTRUCTION}"
                if system_prompt
                else BACKWARD_LENGTH_INSTRUCTION
            )
        call_index = len(self.records) + 1
        print(f"    engine call {call_index} start kind={kind}", flush=True)
        response = self.engine(prompt, system_prompt=system_prompt, **kwargs)
        print(f"    engine call {call_index} done kind={kind}", flush=True)
        self.records.append(
            {
                "kind": kind,
                "system_prompt": system_prompt,
                "prompt": prompt,
                "response": response,
            }
        )
        return response
