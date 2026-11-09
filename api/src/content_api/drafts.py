"""Ask a model for a LinkedIn post draft and check it with the Post model."""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from content_api.llm import LLMProvider, LLMResult
from content_api.models import Post, format_errors
from content_api.usage_log import DEFAULT_LOG, log_call

FEATURE = "draft"
PROMPT_VERSION = "draft-v1"  # change this whenever SYSTEM_PROMPT changes
MAX_ATTEMPTS = 2  # the first try plus one retry

SYSTEM_PROMPT = """\
You write LinkedIn posts. Reply with only a JSON object and no other text.
The object must have exactly these keys:
- "hook": the first line, 15 words or fewer
- "pain_points": a list of 1 to 5 short reader problems
- "body": the main text, under 2,500 characters
- "cta": one call to action
- "hashtags": a list of 3 to 5 hashtags, each starting with #"""


class DraftParseError(ValueError):
    """The model's reply isn't a valid post."""


@dataclass
class DraftResult:
    post: Post
    calls: list[LLMResult]
    cost_usd: float


class DraftError(Exception):
    """Every attempt gave a bad draft. Holds each reason and what they cost."""

    def __init__(
        self, reasons: list[str], cost_usd: float, calls: list[LLMResult]
    ) -> None:
        numbered = "; ".join(f"{n}) {r}" for n, r in enumerate(reasons, start=1))
        super().__init__(f"{len(reasons)} bad drafts: {numbered}")
        self.reasons = reasons
        self.cost_usd = cost_usd
        self.calls = calls


def build_prompt(topic: str, niche: str) -> str:
    return f"Write a LinkedIn post about {topic!r} for the {niche!r} niche."


def build_retry_prompt(prompt: str, bad_reply: str, reason: str) -> str:
    return (
        f"{prompt}\n\nYour previous reply was:\n{bad_reply}\n\n"
        f"It was rejected because: {reason}\n"
        "Reply again with a corrected JSON object only."
    )


def parse_draft(text: str, niche: str) -> Post:
    """Turn the model's reply into a Post, or raise DraftParseError saying why not."""
    cleaned = text.strip()
    # Models sometimes wrap JSON in a ```json ... ``` block; remove it.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        cleaned = cleaned.removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise DraftParseError(f"reply is not valid JSON ({error.msg})") from error
    if not isinstance(data, dict):
        raise DraftParseError(f"reply is a JSON {type(data).__name__}, not an object")
    try:
        return Post.model_validate({**data, "niche": niche, "platform": "linkedin"})
    except ValidationError as error:
        raise DraftParseError(format_errors(error)) from error


def generate_draft(
    topic: str,
    niche: str,
    provider: LLMProvider,
    *,
    log_path: str | Path = DEFAULT_LOG,
) -> DraftResult:
    """Get a valid post from the model, retrying once with the error if needed."""
    prompt = build_prompt(topic, niche)
    calls: list[LLMResult] = []
    reasons: list[str] = []
    cost = 0.0

    for _ in range(MAX_ATTEMPTS):
        result = provider.complete(SYSTEM_PROMPT, prompt)
        entry = log_call(
            result, feature=FEATURE, prompt_version=PROMPT_VERSION, path=log_path
        )
        calls.append(result)
        cost += entry["cost_usd"]
        try:
            post = parse_draft(result.text, niche)
        except DraftParseError as error:
            reasons.append(str(error))
            prompt = build_retry_prompt(
                build_prompt(topic, niche), result.text, str(error)
            )
            continue
        return DraftResult(post=post, calls=calls, cost_usd=cost)

    raise DraftError(reasons=reasons, cost_usd=cost, calls=calls)
