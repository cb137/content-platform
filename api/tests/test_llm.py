"""AI layer tests. Nothing here talks to a real AI service or costs money."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, Usage
from pydantic import ValidationError

from content_api.drafts import (
    FEATURE,
    PROMPT_VERSION,
    DraftError,
    DraftParseError,
    generate_draft,
    parse_draft,
)
from content_api.llm import (
    ClaudeProvider,
    LLMRefusalError,
    LLMResult,
    OpenAIProvider,
    get_provider,
)
from content_api.pricing import UnknownModelError, cost_usd
from content_api.settings import Settings
from content_api.usage_log import log_call, summarize

ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "LLM_PROVIDER",
    "CLAUDE_MODEL",
    "OPENAI_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide any real settings or keys on this computer from every test.

    Settings reads .env from the current folder, so each test runs in an
    empty temporary folder where there is no .env to read.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def make_settings() -> Settings:
    """Settings from environment variables only (there's no .env in tmp_path)."""
    return Settings()


# ---------- settings and provider choice ----------


def test_default_provider_is_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = get_provider(make_settings())
    assert isinstance(provider, ClaudeProvider)
    assert provider.model == "claude-sonnet-5-5"


def test_llm_provider_openai_switches_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = get_provider(make_settings())
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-6.1-sol"


def test_model_names_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-haiku-5-5")
    assert get_provider(make_settings()).model == "claude-haiku-5-5"


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValidationError, match="llm_provider"):
        make_settings()


def test_blank_values_mean_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # What you get if you copy .env.example without filling everything in.
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("CLAUDE_MODEL", "")
    settings = make_settings()
    assert settings.llm_provider == "claude"
    assert settings.claude_model == "claude-sonnet-5-5"


def test_unknown_model_fails_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_MODEL", "claude-made-up-9")
    with pytest.raises(UnknownModelError, match="'claude-made-up-9'"):
        get_provider(make_settings())


def test_api_key_is_never_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-very-secret")
    settings = make_settings()
    assert "sk-very-secret" not in str(settings)
    assert "sk-very-secret" not in repr(settings)


# ---------- providers, with fake SDK clients (no network) ----------


def test_claude_provider_returns_text_tokens_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ClaudeProvider("claude-sonnet-5-5", api_key="test-key")
    sent: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> Message:
        sent.update(kwargs)
        return Message.model_construct(
            model="claude-sonnet-5-5",
            stop_reason="end_turn",
            content=[TextBlock.model_construct(type="text", text="Hello")],
            usage=Usage.model_construct(input_tokens=12, output_tokens=3),
        )

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(provider, "_client", fake_client)

    result = provider.complete("Be brief.", "Say hello")
    assert (result.text, result.provider, result.model) == (
        "Hello",
        "claude",
        "claude-sonnet-5-5",
    )
    assert (result.input_tokens, result.output_tokens) == (12, 3)
    assert result.latency_seconds >= 0
    assert sent["system"] == "Be brief."
    assert sent["messages"] == [{"role": "user", "content": "Say hello"}]


def test_claude_refusal_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ClaudeProvider("claude-sonnet-5-5", api_key="test-key")

    def fake_create(**kwargs: Any) -> Message:
        return Message.model_construct(
            model="claude-sonnet-5-5",
            stop_reason="refusal",
            content=[],
            usage=Usage.model_construct(input_tokens=5, output_tokens=0),
        )

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(provider, "_client", fake_client)
    with pytest.raises(LLMRefusalError):
        provider.complete("system", "prompt")


def test_openai_provider_returns_text_tokens_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAIProvider("gpt-6.1-sol", api_key="test-key")
    sent: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        sent.update(kwargs)
        return SimpleNamespace(
            output_text="Hi",
            model="gpt-6.1-sol",
            usage=SimpleNamespace(input_tokens=8, output_tokens=2),
        )

    fake_client = SimpleNamespace(responses=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(provider, "_client", fake_client)

    result = provider.complete("Be brief.", "Say hi")
    assert (result.text, result.provider, result.model) == (
        "Hi",
        "openai",
        "gpt-6.1-sol",
    )
    assert (result.input_tokens, result.output_tokens) == (8, 2)
    assert sent["instructions"] == "Be brief."
    assert sent["input"] == "Say hi"


# ---------- cost ----------


def test_exact_cost_you_can_check_by_hand() -> None:
    # Sonnet 5.5: $2 per million input, $10 per million output.
    # 1,000 x $2 / 1,000,000 = $0.002 and 500 x $10 / 1,000,000 = $0.005
    assert cost_usd("claude-sonnet-5-5", 1000, 500) == 0.007


def test_exact_cost_for_haiku() -> None:
    # 2,000 x $0.10 / 1M = $0.0002 and 1,000 x $0.50 / 1M = $0.0005
    assert cost_usd("claude-haiku-5-5", 2000, 1000) == 0.0007


def test_dated_model_name_uses_base_price() -> None:
    assert cost_usd("gpt-6.1-sol-2026-10-01", 1000, 500) == 0.007


def test_unknown_model_gives_clear_error() -> None:
    with pytest.raises(UnknownModelError) as exc_info:
        cost_usd("gpt-imaginary", 1, 1)
    message = str(exc_info.value)
    assert "no price for model 'gpt-imaginary'" in message
    assert "claude-sonnet-5-5" in message  # lists the known models


# ---------- usage log ----------


def make_result(
    text: str = "", input_tokens: int = 1000, output_tokens: int = 500
) -> LLMResult:
    return LLMResult(
        text=text,
        provider="claude",
        model="claude-sonnet-5-5",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_seconds=1.23456,
    )


def read_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_call_appends_one_json_line_per_call(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    log_call(make_result(), feature="draft", prompt_version="v1", path=log)
    log_call(make_result(), feature="other", prompt_version="v2", path=log)

    first, second = read_log(log)
    assert set(first) == {
        "time",
        "provider",
        "model",
        "prompt_version",
        "feature",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_seconds",
    }
    assert first["feature"] == "draft"
    assert first["prompt_version"] == "v1"
    assert first["cost_usd"] == 0.007
    assert first["latency_seconds"] == 1.235
    assert second["feature"] == "other"


def test_summarize_adds_up_the_log(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    log_call(make_result(), feature="draft", prompt_version="v1", path=log)
    log_call(
        make_result(input_tokens=2000), feature="draft", prompt_version="v1", path=log
    )

    summary = summarize(log)
    assert summary.calls == 2
    assert summary.input_tokens == 3000
    assert summary.output_tokens == 1000
    assert summary.total_tokens == 4000
    assert summary.cost_usd == 0.016  # 0.007 + (0.004 + 0.005)


def test_summarize_missing_log_is_zero(tmp_path: Path) -> None:
    summary = summarize(tmp_path / "nothing.jsonl")
    assert (summary.calls, summary.total_tokens, summary.cost_usd) == (0, 0, 0.0)


def test_summarize_names_a_broken_line(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    log_call(make_result(), feature="draft", prompt_version="v1", path=log)
    with log.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    with pytest.raises(ValueError, match="line 2"):
        summarize(log)


# ---------- drafts ----------

GOOD_DRAFT = {
    "hook": "Most RAG demos fail in production.",
    "pain_points": ["Wrong answers", "Slow retrieval"],
    "body": "Test the retrieval step before you trust the model.",
    "cta": "Comment RAG for my checklist",
    "hashtags": ["#RAG", "#LLM", "#AI"],
}


class FakeProvider:
    """Returns the replies it was given, in order, and remembers each prompt."""

    name = "claude"
    model = "claude-sonnet-5-5"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 16000
    ) -> LLMResult:
        self.prompts.append(prompt)
        return make_result(text=self.replies.pop(0))


def test_good_draft(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    provider = FakeProvider(json.dumps(GOOD_DRAFT))

    draft = generate_draft("RAG", "ai-engineering", provider, log_path=log)

    assert draft.post.hook == GOOD_DRAFT["hook"]
    assert draft.post.niche == "ai-engineering"
    assert draft.post.platform == "linkedin"
    assert len(draft.calls) == 1
    assert draft.cost_usd == 0.007
    [entry] = read_log(log)
    assert entry["feature"] == FEATURE == "draft"
    assert entry["prompt_version"] == PROMPT_VERSION


def test_draft_wrapped_in_code_fence_is_accepted(tmp_path: Path) -> None:
    reply = "```json\n" + json.dumps(GOOD_DRAFT) + "\n```"
    draft = generate_draft(
        "RAG", "ai-engineering", FakeProvider(reply), log_path=tmp_path / "log.jsonl"
    )
    assert draft.post.cta == GOOD_DRAFT["cta"]


def test_bad_draft_then_good_one(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    too_long_hook = {**GOOD_DRAFT, "hook": " ".join(["word"] * 20)}
    provider = FakeProvider(json.dumps(too_long_hook), json.dumps(GOOD_DRAFT))

    draft = generate_draft("RAG", "ai-engineering", provider, log_path=log)

    assert draft.post.hook == GOOD_DRAFT["hook"]
    assert len(draft.calls) == 2
    assert draft.cost_usd == 0.014
    # The retry tells the model exactly what was wrong.
    assert "hook: has 20 words, but must be 15 words or fewer" in provider.prompts[1]
    assert len(read_log(log)) == 2  # both calls cost money, so both are logged


def test_two_bad_drafts_raise_draft_error(tmp_path: Path) -> None:
    log = tmp_path / "llm_calls.jsonl"
    seven_tags = {**GOOD_DRAFT, "hashtags": [f"#t{n}" for n in range(7)]}
    provider = FakeProvider("Sure! Here is your post.", json.dumps(seven_tags))

    with pytest.raises(DraftError) as exc_info:
        generate_draft("RAG", "ai-engineering", provider, log_path=log)

    error = exc_info.value
    assert len(error.reasons) == 2
    assert error.reasons[0].startswith("reply is not valid JSON")
    assert error.reasons[1] == "hashtags: has 7 hashtags, but must have 3 to 5"
    assert str(error).startswith("2 bad drafts: 1) reply is not valid JSON")
    assert error.cost_usd == 0.014
    assert len(provider.prompts) == 2  # gave up after one retry
    assert len(read_log(log)) == 2


def test_json_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(DraftParseError, match="JSON list, not an object"):
        parse_draft("[1, 2, 3]", "career")
