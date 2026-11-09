"""Calls the REAL AI API with your key and costs a few cents.

Normal test runs skip this file. Run it on purpose from api/ with:

    uv run pytest -m integration
"""

import pytest

from content_api.drafts import generate_draft
from content_api.llm import get_provider
from content_api.settings import Settings

pytestmark = pytest.mark.integration


def test_real_draft_from_the_configured_provider() -> None:
    settings = Settings()
    key = (
        settings.openai_api_key
        if settings.llm_provider == "openai"
        else settings.anthropic_api_key
    )
    if key is None or not key.get_secret_value():
        pytest.skip(f"no API key set for LLM_PROVIDER={settings.llm_provider}")

    provider = get_provider(settings)
    # Logged to the normal llm_calls.jsonl, so cost_report.py includes it.
    draft = generate_draft("why tests should never sleep", "ai-engineering", provider)

    assert draft.post.hook
    assert draft.cost_usd > 0
    assert all(call.input_tokens > 0 and call.output_tokens > 0 for call in draft.calls)
