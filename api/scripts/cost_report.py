"""Print how many LLM calls were logged, and their total tokens and cost.

Run from api/ (no AI calls, free):

    uv run python scripts/cost_report.py
    uv run python scripts/cost_report.py path\\to\\llm_calls.jsonl
"""

import argparse

from content_api.usage_log import DEFAULT_LOG, summarize


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarise the LLM usage log.")
    parser.add_argument(
        "log", nargs="?", default=str(DEFAULT_LOG), help=f"default: {DEFAULT_LOG}"
    )
    args = parser.parse_args(argv)

    summary = summarize(args.log)
    print(f"Calls:         {summary.calls:,}")
    print(f"Input tokens:  {summary.input_tokens:,}")
    print(f"Output tokens: {summary.output_tokens:,}")
    print(f"Total tokens:  {summary.total_tokens:,}")
    print(f"Total cost:    ${summary.cost_usd:.4f}")


if __name__ == "__main__":
    main()
