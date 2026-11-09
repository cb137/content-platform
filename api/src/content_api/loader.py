"""Load posts from a CSV file, setting bad rows aside instead of crashing."""

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from content_api.models import Post, format_errors

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    "niche",
    "platform",
    "hook",
    "pain_points",
    "body",
    "cta",
    "hashtags",
)
REJECTS_COLUMNS = ("row", "reason", "raw")
HEADER_ROW = 1  # Row numbers match a spreadsheet: the header is row 1.


class CSVFormatError(ValueError):
    """The file as a whole can't be read as a posts CSV."""


class EmptyCSVError(CSVFormatError):
    """The file has no header row."""


class WrongHeadersError(CSVFormatError):
    """The header row isn't the 7 expected columns."""


@dataclass(frozen=True)
class Reject:
    """A row that was set aside, and why."""

    row: int
    reason: str
    raw: str


@dataclass
class LoadResult:
    """What load_posts found: the good posts and the rejected rows."""

    posts: list[Post] = field(default_factory=list)
    rejects: list[Reject] = field(default_factory=list)


def load_posts(
    csv_path: str | Path, rejects_path: str | Path | None = None
) -> LoadResult:
    """Read a posts CSV. Good rows become Posts; bad rows become Rejects.

    Rejects are saved to rejects_path (default: rejects.csv next to the input).
    Raises EmptyCSVError or WrongHeadersError if the whole file is unusable.
    """
    path = Path(csv_path)
    rejects_file = Path(rejects_path) if rejects_path else path.with_name("rejects.csv")
    result = LoadResult()

    # utf-8-sig also handles the invisible "BOM" that Excel adds on Windows.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = _check_header(next(reader, None), path)

        for row_number, values in enumerate(reader, start=HEADER_ROW + 1):
            if not values:  # A completely blank line: nothing to load.
                continue
            post_or_reason = _parse_row(header, values)
            if isinstance(post_or_reason, Post):
                result.posts.append(post_or_reason)
            else:
                reject = Reject(row_number, post_or_reason, _to_csv_line(values))
                result.rejects.append(reject)
                logger.error(
                    "row rejected",
                    extra={
                        "row": row_number,
                        "reason": post_or_reason,
                        "file": str(path),
                    },
                )

    write_rejects(result.rejects, rejects_file)
    logger.info(
        "finished loading",
        extra={
            "file": str(path),
            "posts": len(result.posts),
            "rejects": len(result.rejects),
            "rejects_file": str(rejects_file),
        },
    )
    return result


def write_rejects(rejects: list[Reject], path: str | Path) -> None:
    """Save rejects to a CSV with columns row, reason, raw.

    The file is written even when there are no rejects, so an old one never lingers.
    """
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REJECTS_COLUMNS)
        for reject in rejects:
            writer.writerow([reject.row, reject.reason, reject.raw])


def _check_header(header: list[str] | None, path: Path) -> list[str]:
    """Return the cleaned header, or raise if the file is empty or headers are wrong."""
    if header is None or not any(name.strip() for name in header):
        message = (
            f"{path.name} is empty: expected a header row with the columns "
            f"{', '.join(EXPECTED_COLUMNS)}"
        )
        logger.error(message, extra={"row": HEADER_ROW, "file": str(path)})
        raise EmptyCSVError(message)

    names = [name.strip() for name in header]
    missing = [name for name in EXPECTED_COLUMNS if name not in names]
    unexpected = [name for name in names if name not in EXPECTED_COLUMNS]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if missing or unexpected or duplicated:
        problems = []
        if missing:
            problems.append(f"missing columns: {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected columns: {', '.join(unexpected)}")
        if duplicated:
            problems.append(f"repeated columns: {', '.join(duplicated)}")
        message = (
            f"{path.name} has the wrong headers ({'; '.join(problems)}). "
            f"Expected: {', '.join(EXPECTED_COLUMNS)}"
        )
        logger.error(message, extra={"row": HEADER_ROW, "file": str(path)})
        raise WrongHeadersError(message)
    return names


def _parse_row(header: list[str], values: list[str]) -> Post | str:
    """Return a Post for a good row, or a plain-English reason for a bad one."""
    if len(values) != len(header):
        return f"has {len(values)} values, but expected {len(header)}"
    data = dict(zip(header, values, strict=True))
    pain_points = data["pain_points"]
    try:
        return Post.model_validate(
            {
                **data,
                # "a|b" -> ["a", "b"]; an empty cell -> [] (0 pain points).
                "pain_points": pain_points.split("|") if pain_points.strip() else [],
                # Hashtags are separated by spaces.
                "hashtags": data["hashtags"].split(),
            }
        )
    except ValidationError as error:
        return format_errors(error)


def _to_csv_line(values: list[str]) -> str:
    """Turn a row back into one line of CSV text, for the raw column."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(values)
    return buffer.getvalue()


# Attributes every LogRecord has; anything else came from extra={...}.
_STANDARD_LOG_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Format each log record as one JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_LOG_ATTRS:
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Send content_api logs to the terminal (stderr) as JSON lines."""
    package_logger = logging.getLogger("content_api")
    package_logger.setLevel(level)
    if not any(isinstance(h.formatter, JsonFormatter) for h in package_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        package_logger.addHandler(handler)
