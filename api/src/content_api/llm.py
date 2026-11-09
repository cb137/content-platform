"""One interface for LLM providers, with a Claude and an OpenAI version."""

import time
from dataclasses import dataclass
from typing import Protocol

import anthropic
import openai
from anthropic.types import TextBlock
from pydantic import SecretStr

from content_api.pricing import price_for
from content_api.settings import Settings

DEFAULT_MAX_TOKENS = 16000


@dataclass(frozen=True)
class LLMResult:
    """What every provider returns for one call."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


class LLMRefusalError(RuntimeError):
    """The model declined to answer."""


class LLMProvider(Protocol):
    """Anything with these attributes and this method counts as a provider."""

    name: str
    model: str

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMResult: ...


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMResult:
        start = time.perf_counter()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        if response.stop_reason == "refusal":
            raise LLMRefusalError(f"{self.model} declined to answer")
        text = "".join(b.text for b in response.content if isinstance(b, TextBlock))
        return LLMResult(
            text=text,
            provider=self.name,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_seconds=latency,
        )


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self._client = openai.OpenAI(api_key=api_key)

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> LLMResult:
        start = time.perf_counter()
        response = self._client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        usage = response.usage
        return LLMResult(
            text=response.output_text,
            provider=self.name,
            model=response.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_seconds=latency,
        )


def _secret(value: SecretStr | None) -> str | None:
    """The real key, or None if it's missing or blank."""
    return value.get_secret_value() or None if value else None


def get_provider(settings: Settings) -> LLMProvider:
    """Build the provider LLM_PROVIDER names.

    The model's price is checked first, so a typo fails here, before any paid call.
    """
    if settings.llm_provider == "openai":
        price_for(settings.openai_model)
        return OpenAIProvider(settings.openai_model, _secret(settings.openai_api_key))
    price_for(settings.claude_model)
    return ClaudeProvider(settings.claude_model, _secret(settings.anthropic_api_key))
