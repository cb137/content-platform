import csv
import json
import logging
from pathlib import Path

import pytest
from content_api.loader import (
    EmptyCSVError,
    JsonFormatter,
    WrongHeadersError,
    load_posts,
)

DATA_DIR = Path(__file__).parent.parent / "data"
HEADER = "niche,platform,hook,pain_points,body,cta,hashtags\n"
GOOD_ROW = 'career,linkedin,Ship one real thing,No projects|No time,"Build it, ship it.",Follow me,#Career #Jobs #Learning\n'
EMOJI_ROW = "marketing,tiktok,Big news 🚀,Low budget,Try this ✅,Save this,#a #b #c\n"
BAD_HOOK_ROW = (
    "career,youtube," + " ".join(["word"] * 20) + ",Pain,Body,Follow,#a #b #c\n"
)


def write_csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "posts.csv"
    path.write_text(text, encoding="utf-8", newline="")
    return path


def read_rejects(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_good_rows_become_posts(tmp_path: Path) -> None:
    result = load_posts(write_csv(tmp_path, HEADER + GOOD_ROW + EMOJI_ROW))
    assert len(result.posts) == 2
    assert result.rejects == []
    first = result.posts[0]
    assert first.pain_points == ["No projects", "No time"]
    assert first.hashtags == ["#Career", "#Jobs", "#Learning"]
    assert first.body == "Build it, ship it."
    assert result.posts[1].hook == "Big news 🚀"


def test_bad_row_is_rejected_with_row_number(tmp_path: Path) -> None:
    result = load_posts(write_csv(tmp_path, HEADER + GOOD_ROW + BAD_HOOK_ROW))
    assert len(result.posts) == 1
    [reject] = result.rejects
    assert reject.row == 3  # header is row 1, so the second post is row 3
    assert reject.reason == "hook: has 20 words, but must be 15 words or fewer"


def test_row_with_too_few_values_is_rejected(tmp_path: Path) -> None:
    result = load_posts(write_csv(tmp_path, HEADER + "career,linkedin,Hi\n"))
    [reject] = result.rejects
    assert reject.reason == "has 3 values, but expected 7"
    assert reject.raw == "career,linkedin,Hi"


def test_rejects_csv_is_written(tmp_path: Path) -> None:
    load_posts(write_csv(tmp_path, HEADER + GOOD_ROW + BAD_HOOK_ROW))
    rows = read_rejects(tmp_path / "rejects.csv")
    assert len(rows) == 1
    assert rows[0]["row"] == "3"
    assert rows[0]["reason"].startswith("hook:")
    assert rows[0]["raw"] == BAD_HOOK_ROW.rstrip("\n")


def test_rejects_path_can_be_chosen(tmp_path: Path) -> None:
    rejects = tmp_path / "out" / "bad.csv"
    rejects.parent.mkdir()
    load_posts(write_csv(tmp_path, HEADER + BAD_HOOK_ROW), rejects)
    assert len(read_rejects(rejects)) == 1


def test_empty_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(EmptyCSVError, match="posts.csv is empty"):
        load_posts(write_csv(tmp_path, ""))


def test_wrong_headers_give_clear_error(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "niche,platform,title,pain_points,body,cta,tags\n")
    with pytest.raises(WrongHeadersError) as exc_info:
        load_posts(path)
    message = str(exc_info.value)
    assert "missing columns: hook, hashtags" in message
    assert "unexpected columns: title, tags" in message


def test_errors_are_logged_with_row_number(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="content_api"):
        load_posts(write_csv(tmp_path, HEADER + GOOD_ROW + BAD_HOOK_ROW))
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].__dict__["row"] == 3


def test_header_errors_are_logged_with_row_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        caplog.at_level(logging.ERROR, logger="content_api"),
        pytest.raises(EmptyCSVError),
    ):
        load_posts(write_csv(tmp_path, ""))
    assert caplog.records[0].__dict__["row"] == 1


def test_json_formatter_writes_one_json_object() -> None:
    record = logging.makeLogRecord(
        {"name": "content_api.loader", "levelname": "ERROR", "msg": "row rejected"}
    )
    record.row = 5
    line = JsonFormatter().format(record)
    assert "\n" not in line
    entry = json.loads(line)
    assert entry["level"] == "ERROR"
    assert entry["message"] == "row rejected"
    assert entry["row"] == 5


def test_1000_row_file(tmp_path: Path) -> None:
    result = load_posts(DATA_DIR / "posts_1000.csv", tmp_path / "rejects.csv")
    assert len(result.posts) == 970
    assert len(result.rejects) == 30
