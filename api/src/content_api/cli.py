"""post-stats: print simple stats for each LinkedIn post in a CSV file."""

import argparse
import csv

DEFAULT_CSV = "data/sample_posts.csv"


def hook(text: str) -> str:
    """Return the first line of a post (its hook), or "" for an empty post."""
    lines = text.splitlines()
    return lines[0] if lines else ""


def post_stats(row: dict[str, str]) -> dict[str, int]:
    """Count words, hook words, characters and hashtags for one CSV row."""
    # Windows line endings (\r\n) would add an extra character per line break.
    text = row["text"].replace("\r\n", "\n")
    words = text.split()
    return {
        "word_count": len(words),
        "hook_words": len(hook(text).split()),
        "char_count": len(text),
        "hashtags": sum(1 for word in words if word.startswith("#")),
    }


def main(argv: list[str] | None = None) -> None:
    """Read the CSV and print one stats line per post."""
    parser = argparse.ArgumentParser(
        prog="post-stats", description="Print stats for each post in a CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"path to the posts CSV (default: {DEFAULT_CSV})",
    )
    args = parser.parse_args(argv)

    with open(args.csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stats = post_stats(row)
            print(
                f"id={row['id']:<4}"
                f"words={stats['word_count']:<5}"
                f"hook_words={stats['hook_words']:<4}"
                f"chars={stats['char_count']:<6}"
                f"hashtags={stats['hashtags']}"
            )
