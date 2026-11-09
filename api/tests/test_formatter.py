from typing import Any

import pytest

from content_api.formatter import PostTooLongError, format_linkedin_post
from content_api.models import Post


def make_post(**changes: Any) -> Post:
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
    return Post.model_validate(data)


def test_normal_post() -> None:
    assert format_linkedin_post(make_post()) == (
        "Most RAG demos fail in production.\n"
        "\n"
        "Wrong answers\n"
        "Slow retrieval\n"
        "Test the retrieval step before you trust the model.\n"
        "\n"
        "Comment RAG for my checklist\n"
        "#RAG #LLM #AI"
    )


def test_emoji_stay_in_place() -> None:
    post = make_post(
        hook="Big news 🚀 today",
        pain_points=["No time ⏰"],
        body="Step 1 ✅ done.\nStep 2 👉 next.",
    )
    lines = format_linkedin_post(post).split("\n")
    assert lines[0] == "Big news 🚀 today"
    assert lines[2] == "No time ⏰"
    assert lines[3:5] == ["Step 1 ✅ done.", "Step 2 👉 next."]


def post_with_total_length(total: int) -> Post:
    """Build a post whose formatted text is exactly `total` characters long."""
    base = make_post(body="x")
    # Everything except the body: hook, pain points, cta, hashtags, line breaks.
    other_chars = len(format_linkedin_post(base)) - 1
    return make_post(body="x" * (total - other_chars))


def test_exactly_3000_characters_passes() -> None:
    text = format_linkedin_post(post_with_total_length(3000))
    assert len(text) == 3000


def test_3001_characters_fails_with_clear_error() -> None:
    with pytest.raises(PostTooLongError) as exc_info:
        format_linkedin_post(post_with_total_length(3001))
    message = str(exc_info.value)
    assert "3,001 characters" in message
    assert "3,000" in message


def test_duplicate_hashtags_appear_once() -> None:
    post = make_post(hashtags=["#AI", "#LLM", "#ai", "#AI"])
    last_line = format_linkedin_post(post).split("\n")[-1]
    assert last_line == "#AI #LLM"  # first spelling kept, in original order


def test_trailing_spaces_are_stripped() -> None:
    post = make_post(body="Line one   \nLine two\t ", pain_points=["Pain   "])
    for line in format_linkedin_post(post).split("\n"):
        assert line == line.rstrip()


def test_3_or_more_blank_lines_collapse_to_one() -> None:
    post = make_post(body="Part one\n\n\n\n\nPart two")  # 4 blank lines
    assert "Part one\n\nPart two" in format_linkedin_post(post)


def test_lines_of_only_spaces_count_as_blank() -> None:
    post = make_post(body="Part one\n  \n \n\t\nPart two")  # 3 "blank" lines
    assert "Part one\n\nPart two" in format_linkedin_post(post)


def test_2_blank_lines_are_kept() -> None:
    post = make_post(body="Part one\n\n\nPart two")  # 2 blank lines
    assert "Part one\n\n\nPart two" in format_linkedin_post(post)
