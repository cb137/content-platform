"""Turn a Post into the exact text to paste into LinkedIn."""

import re

from content_api.models import Post

LINKEDIN_CHAR_LIMIT = 3000

# 3 or more blank lines in a row = 4 or more line breaks in a row.
_THREE_OR_MORE_BLANK_LINES = re.compile(r"\n{4,}")


class PostTooLongError(ValueError):
    """The formatted post is over LinkedIn's character limit."""


def format_linkedin_post(post: Post) -> str:
    """Lay the post out for LinkedIn: hook, blank line, pain points, body,
    blank line, call to action, hashtags.

    Raises PostTooLongError if the result is over 3,000 characters.
    """
    lines = [
        post.hook,
        "",
        *post.pain_points,
        post.body,
        "",
        post.cta,
        " ".join(unique_hashtags(post.hashtags)),
    ]
    # Join, then split again so lines inside the body are cleaned too.
    text = "\n".join(line.rstrip() for line in "\n".join(lines).split("\n"))
    text = _THREE_OR_MORE_BLANK_LINES.sub("\n\n", text)

    if len(text) > LINKEDIN_CHAR_LIMIT:
        extra = len(text) - LINKEDIN_CHAR_LIMIT
        raise PostTooLongError(
            f"LinkedIn post is {len(text):,} characters, but LinkedIn allows at "
            f"most {LINKEDIN_CHAR_LIMIT:,}. Remove at least {extra:,} characters."
        )
    return text


def unique_hashtags(hashtags: list[str]) -> list[str]:
    """Drop repeated hashtags (ignoring capitals), keeping the first spelling."""
    seen: set[str] = set()
    unique = []
    for tag in hashtags:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique.append(tag)
    return unique
