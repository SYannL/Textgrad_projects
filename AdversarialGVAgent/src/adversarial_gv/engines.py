"""Language-model engine construction for hosted and OpenAI-compatible APIs."""

import threading
from collections import defaultdict
from typing import List, Union

import textgrad as tg
from openai import OpenAI
from textgrad.engine import EngineLM


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def resolve_role_base_urls(
    shared: str | None = None,
    generator: str | None = None,
    verifier: str | None = None,
    backward: str | None = None,
) -> dict[str, str | None]:
    """Resolve role-specific vLLM endpoints with a legacy shared fallback."""

    def normalize(value: str | None) -> str | None:
        return value.rstrip("/") if value else None

    shared = normalize(shared)
    return {
        "generator": normalize(generator) or shared,
        "verifier": normalize(verifier) or shared,
        "backward": normalize(backward) or shared,
    }


class TokenUsageTracker:
    """Thread-safe token accounting for concurrent OpenAI-compatible calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals = defaultdict(
            lambda: {field: 0 for field in TOKEN_FIELDS}
        )

    def record(self, role: str, usage) -> None:
        if usage is None:
            return
        values = {
            field: int(getattr(usage, field, 0) or 0)
            for field in TOKEN_FIELDS
        }
        with self._lock:
            target = self._totals[role]
            for field, value in values.items():
                target[field] += value

    def snapshot(self):
        with self._lock:
            result = {
                role: dict(values)
                for role, values in self._totals.items()
            }
        overall = {field: 0 for field in TOKEN_FIELDS}
        for values in result.values():
            for field in TOKEN_FIELDS:
                overall[field] += values[field]
        result["all"] = overall
        return result


def merge_token_usage(*snapshots):
    """Add persisted and current-process token snapshots."""
    merged = defaultdict(lambda: {field: 0 for field in TOKEN_FIELDS})
    for snapshot in snapshots:
        for role, values in (snapshot or {}).items():
            if role == "all":
                continue
            for field in TOKEN_FIELDS:
                merged[role][field] += int(values.get(field, 0) or 0)
    result = {role: dict(values) for role, values in merged.items()}
    overall = {field: 0 for field in TOKEN_FIELDS}
    for values in result.values():
        for field in TOKEN_FIELDS:
            overall[field] += values[field]
    result["all"] = overall
    return result


class OpenAICompatibleEngine(EngineLM):
    """TextGrad engine for vLLM's OpenAI-compatible chat API."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str,
        base_url: str,
        api_key: str = "EMPTY",
        enable_thinking: bool = False,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        token_usage_tracker: TokenUsageTracker | None = None,
        usage_role: str = "model",
    ) -> None:
        self.model_string = model_string
        self.system_prompt = system_prompt
        self.enable_thinking = enable_thinking
        self.token_usage_tracker = token_usage_tracker
        self.usage_role = usage_role
        self.client = OpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key or "EMPTY",
        )

    def generate(
        self,
        content: Union[str, List[Union[str, bytes]]],
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 0.99,
        **kwargs,
    ) -> str:
        if not isinstance(content, str):
            raise TypeError("The current vLLM engine supports text input only")
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
            **kwargs,
        )
        if self.token_usage_tracker is not None:
            self.token_usage_tracker.record(self.usage_role, response.usage)
        return response.choices[0].message.content

    def __call__(self, content, **kwargs) -> str:
        return self.generate(content, **kwargs)


def build_engine(
    model: str,
    vllm_base_url: str | None = None,
    vllm_api_key: str | None = None,
    vllm_enable_thinking: bool = False,
    token_usage_tracker: TokenUsageTracker | None = None,
    usage_role: str = "model",
) -> EngineLM:
    if not vllm_base_url:
        return tg.get_engine(model)
    return OpenAICompatibleEngine(
        model_string=model,
        base_url=vllm_base_url,
        api_key=vllm_api_key or "EMPTY",
        enable_thinking=vllm_enable_thinking,
        token_usage_tracker=token_usage_tracker,
        usage_role=usage_role,
    )
