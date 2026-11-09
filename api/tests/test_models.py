from typing import Any

import pytest
from pydantic import ValidationError

from content_api.models import Post, format_errors


def valid_data(**changes: Any) -> dict[str, Any]:
    """A valid post as a dict; pass keyword arguments to change single fields."""
    data: dict[str, Any] = {
        "niche": "ai-engineering",
        "platform": "linkedin",
        "hook": "Most RAG demos fail in production.",
        "pain_points": ["Wrong answers", "Slow retrieval"],
        "body": "Test the retrieval step before you trust the model.",
        "cta": "Comment RAG for my checklist",
        "hashtags": ["#RAG", "#LLM", "#AI"],
    }
    data.update(changes)
    return data


def error_for(**changes: Any) -> str:
    """Build a post that should fail and return its plain-English error."""
    with pytest.raises(ValidationError) as exc_info:
        Post.model_validate(valid_data(**changes))
    return format_errors(exc_info.value)


def test_valid_post() -> None:
    post = Post.model_validate(valid_data())
    assert post.platform == "linkedin"


# niche: required, not blank
def test_niche_passes() -> None:
    assert Post.model_validate(valid_data(niche="career")).niche == "career"


def test_niche_blank_fails() -> None:
    assert error_for(niche="   ") == "niche: is required and cannot be blank"


# platform: linkedin, youtube, instagram or tiktok
@pytest.mark.parametrize("platform", ["linkedin", "youtube", "instagram", "tiktok"])
def test_platform_passes(platform: str) -> None:
    assert Post.model_validate(valid_data(platform=platform)).platform == platform


def test_platform_capitals_pass() -> None:
    assert Post.model_validate(valid_data(platform="LinkedIn")).platform == "linkedin"


def test_platform_unknown_fails() -> None:
    message = error_for(platform="facebook")
    assert message.startswith("platform: must be one of")
    assert "'facebook'" in message


# hook: 15 words or fewer
def test_hook_of_15_words_passes() -> None:
    hook = " ".join(["word"] * 15)
    assert Post.model_validate(valid_data(hook=hook)).hook == hook


def test_hook_of_16_words_fails() -> None:
    message = error_for(hook=" ".join(["word"] * 16))
    assert message == "hook: has 16 words, but must be 15 words or fewer"


def test_hook_blank_fails() -> None:
    assert error_for(hook="") == "hook: is required and cannot be blank"


# pain_points: 1 to 5
@pytest.mark.parametrize("count", [1, 5])
def test_pain_points_1_to_5_pass(count: int) -> None:
    points = [f"Pain {n}" for n in range(count)]
    assert len(Post.model_validate(valid_data(pain_points=points)).pain_points) == count


@pytest.mark.parametrize("count", [0, 6])
def test_pain_points_outside_1_to_5_fail(count: int) -> None:
    points = [f"Pain {n}" for n in range(count)]
    message = error_for(pain_points=points)
    assert message == f"pain_points: has {count} pain points, but must have 1 to 5"


def test_blank_pain_point_fails() -> None:
    message = error_for(pain_points=["Real one", " "])
    assert message == "pain_points: pain point 2 is blank"


# body: under 3,000 characters when the platform is linkedin
def test_linkedin_body_under_3000_passes() -> None:
    body = "x" * 2999
    assert Post.model_validate(valid_data(body=body)).body == body


def test_linkedin_body_of_3000_fails() -> None:
    message = error_for(body="x" * 3000)
    assert message == (
        "body: is 3,000 characters, but LinkedIn posts must be under 3,000"
    )


def test_long_body_passes_on_other_platforms() -> None:
    post = Post.model_validate(valid_data(platform="youtube", body="x" * 5000))
    assert len(post.body) == 5000


def test_body_blank_fails() -> None:
    assert error_for(body=" ") == "body: is required and cannot be blank"


# cta: required
def test_cta_passes() -> None:
    assert Post.model_validate(valid_data(cta="Follow me")).cta == "Follow me"


def test_cta_blank_fails() -> None:
    assert error_for(cta="") == "cta: is required and cannot be blank"


# hashtags: 3 to 5, each starting with #
@pytest.mark.parametrize("count", [3, 5])
def test_hashtags_3_to_5_pass(count: int) -> None:
    tags = [f"#tag{n}" for n in range(count)]
    assert Post.model_validate(valid_data(hashtags=tags)).hashtags == tags


@pytest.mark.parametrize("count", [2, 7])
def test_hashtags_outside_3_to_5_fail(count: int) -> None:
    tags = [f"#tag{n}" for n in range(count)]
    message = error_for(hashtags=tags)
    assert message == f"hashtags: has {count} hashtags, but must have 3 to 5"


def test_hashtag_without_hash_fails() -> None:
    message = error_for(hashtags=["#AI", "Python", "#LLM"])
    assert message == "hashtags: 'Python' does not start with #"


# format_errors
def test_several_errors_are_all_named() -> None:
    message = error_for(niche="", cta="")
    assert message == (
        "niche: is required and cannot be blank; cta: is required and cannot be blank"
    )


def test_missing_field_is_named() -> None:
    data = valid_data()
    del data["cta"]
    with pytest.raises(ValidationError) as exc_info:
        Post.model_validate(data)
    assert format_errors(exc_info.value) == "cta: is required"
