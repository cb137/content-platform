from pathlib import Path

import pytest
from content_api.cli import hook, main, post_stats


def make_row(text: str) -> dict[str, str]:
    return {
        "id": "1",
        "niche": "test",
        "text": text,
        "posted_at": "2026-11-05",
        "likes": "0",
        "comments": "0",
    }


def test_post_with_no_line_breaks() -> None:
    text = "One tip for learning to code: write the test first."
    assert hook(text) == text
    stats = post_stats(make_row(text))
    assert stats["word_count"] == 10
    assert stats["hook_words"] == 10


def test_post_with_emoji() -> None:
    text = "Shipped it 🚀\nMore soon"
    stats = post_stats(make_row(text))
    assert stats["word_count"] == 5
    assert stats["hook_words"] == 3
    assert stats["char_count"] == 22


def test_empty_post() -> None:
    assert hook("") == ""
    assert post_stats(make_row("")) == {
        "word_count": 0,
        "hook_words": 0,
        "char_count": 0,
        "hashtags": 0,
    }


def test_post_with_zero_hashtags() -> None:
    assert post_stats(make_row("No tags here at all"))["hashtags"] == 0


def test_post_with_three_hashtags() -> None:
    text = "Ship it.\nLearn in public. #AI #Python #BuildInPublic"
    assert post_stats(make_row(text))["hashtags"] == 3


def test_main_prints_one_line_per_post(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_file = tmp_path / "posts.csv"
    csv_file.write_text(
        "id,niche,text,posted_at,likes,comments\n"
        '7,career,"Hello world\nSecond line #Jobs",2026-11-05,1,0\n',
        encoding="utf-8",
    )
    main([str(csv_file)])
    output = capsys.readouterr().out
    assert output == "id=7   words=5    hook_words=2   chars=29    hashtags=1\n"
