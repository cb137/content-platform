"""Generate LinkedIn post drafts with the configured LLM and print their cost.

This calls the real AI API and costs money. Run from api/:

    uv run python scripts/generate_drafts.py "rate limits" ai-engineering
    uv run python scripts/generate_drafts.py "rate limits" ai-engineering --count 3
"""

import argparse
from decimal import Decimal

from content_api.drafts import DraftError, generate_draft
from content_api.llm import get_provider
from content_api.settings import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LinkedIn post drafts.")
    parser.add_argument("topic", help='what the post is about, e.g. "rate limits"')
    parser.add_argument("niche", help="the niche, e.g. ai-engineering")
    parser.add_argument("--count", type=int, default=10, help="how many (default 10)")
    args = parser.parse_args(argv)

    provider = get_provider(Settings())
    print(f"Using {provider.name} / {provider.model}\n")

    total = Decimal(0)
    failures = 0
    for number in range(1, args.count + 1):
        try:
            draft = generate_draft(args.topic, args.niche, provider)
        except DraftError as error:
            failures += 1
            total += Decimal(str(error.cost_usd))
            print(f"{number:>2}. ${error.cost_usd:.4f}  FAILED: {error}")
            continue
        total += Decimal(str(draft.cost_usd))
        retried = " (after 1 retry)" if len(draft.calls) > 1 else ""
        print(f"{number:>2}. ${draft.cost_usd:.4f}  {draft.post.hook}{retried}")

    print(f"\nTotal cost: ${total:.4f}   Failed drafts: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
