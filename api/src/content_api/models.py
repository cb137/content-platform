"""The Post model: one planned social media post, with validation rules."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ValidationError, ValidationInfo, field_validator

Platform = Literal["linkedin", "youtube", "instagram", "tiktok"]
PLATFORMS: tuple[str, ...] = ("linkedin", "youtube", "instagram", "tiktok")

MAX_HOOK_WORDS = 15
MIN_PAIN_POINTS = 1
MAX_PAIN_POINTS = 5
LINKEDIN_BODY_LIMIT = 3000
MIN_HASHTAGS = 3
MAX_HASHTAGS = 5


class Post(BaseModel):
    """A validated post. Creating one with bad data raises ValidationError."""

    niche: str
    platform: Platform
    hook: str
    pain_points: list[str]
    body: str
    cta: str
    hashtags: list[str]

    @field_validator("niche", "hook", "body", "cta")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only text, and trim spaces at the ends."""
        if not value.strip():
            raise ValueError("is required and cannot be blank")
        return value.strip()

    @field_validator("platform", mode="before")
    @classmethod
    def known_platform(cls, value: object) -> object:
        """Accept any capitalisation (LinkedIn -> linkedin), reject others."""
        if isinstance(value, str):
            value = value.strip().lower()
        if value not in PLATFORMS:
            raise ValueError(
                f"must be one of linkedin, youtube, instagram or tiktok, "
                f"but got {value!r}"
            )
        return value

    @field_validator("hook")
    @classmethod
    def short_hook(cls, value: str) -> str:
        """The hook must be 15 words or fewer."""
        word_count = len(value.split())
        if word_count > MAX_HOOK_WORDS:
            raise ValueError(
                f"has {word_count} words, but must be {MAX_HOOK_WORDS} words or fewer"
            )
        return value

    @field_validator("pain_points")
    @classmethod
    def pain_point_count(cls, value: list[str]) -> list[str]:
        """There must be 1 to 5 pain points, and none may be blank."""
        if not MIN_PAIN_POINTS <= len(value) <= MAX_PAIN_POINTS:
            raise ValueError(
                f"has {len(value)} pain points, but must have "
                f"{MIN_PAIN_POINTS} to {MAX_PAIN_POINTS}"
            )
        for number, point in enumerate(value, start=1):
            if not point.strip():
                raise ValueError(f"pain point {number} is blank")
        return [point.strip() for point in value]

    @field_validator("body")
    @classmethod
    def linkedin_body_length(cls, value: str, info: ValidationInfo) -> str:
        """On LinkedIn the body must be under 3,000 characters."""
        # info.data holds the fields already validated; platform comes before body.
        if (
            info.data.get("platform") == "linkedin"
            and len(value) >= LINKEDIN_BODY_LIMIT
        ):
            raise ValueError(
                f"is {len(value):,} characters, but LinkedIn posts must be "
                f"under {LINKEDIN_BODY_LIMIT:,}"
            )
        return value

    @field_validator("hashtags")
    @classmethod
    def valid_hashtags(cls, value: list[str]) -> list[str]:
        """There must be 3 to 5 hashtags, and each must start with #."""
        if not MIN_HASHTAGS <= len(value) <= MAX_HASHTAGS:
            raise ValueError(
                f"has {len(value)} hashtags, but must have "
                f"{MIN_HASHTAGS} to {MAX_HASHTAGS}"
            )
        for tag in value:
            if not tag.startswith("#"):
                raise ValueError(f"{tag!r} does not start with #")
        return value


def format_errors(error: ValidationError) -> str:
    """Turn pydantic's errors into one plain-English line, e.g.
    "hook: has 20 words, but must be 15 words or fewer; cta: is required ...".
    """
    messages = []
    for detail in error.errors():
        field = ".".join(str(part) for part in detail["loc"]) or "post"
        messages.append(f"{field}: {describe_error(detail)}")
    return "; ".join(messages)


def describe_error(detail: Mapping[str, object]) -> str:
    """Say what is wrong in one pydantic error, without the field name."""
    context = detail.get("ctx")
    if detail.get("type") == "value_error" and isinstance(context, dict):
        # Our own ValueError text, without pydantic's "Value error, " prefix.
        return str(context["error"])
    if detail.get("type") == "missing":
        return "is required"
    return str(detail.get("msg", "is not valid"))
