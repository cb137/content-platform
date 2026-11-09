"""What each model costs, and the cost of one call.

Prices are US dollars per million tokens, standard (not batch, not cached) rates.

Sources, both read on 2026-11-09:
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
  (Claude Haiku 5.5 costs more for prompts over 100,000 tokens: $0.50 / $2.50.
  Our prompts are far smaller, so only the lower price is listed.)
- OpenAI: https://developers.openai.com/api/docs/pricing
  (Read through a summarising tool, so double-check these against the page.)
"""

import re
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: Decimal
    output_per_million: Decimal


PRICES: dict[str, ModelPrice] = {
    # Anthropic
    "claude-opus-5-5": ModelPrice(Decimal("4.00"), Decimal("20.00")),
    "claude-sonnet-5-5": ModelPrice(Decimal("2.00"), Decimal("10.00")),
    "claude-haiku-5-5": ModelPrice(Decimal("0.10"), Decimal("0.50")),
    # OpenAI
    "gpt-6.1-sol": ModelPrice(Decimal("2.00"), Decimal("10.00")),
    "gpt-6.1-terra": ModelPrice(Decimal("1.00"), Decimal("5.00")),
    "gpt-6-luna": ModelPrice(Decimal("0.10"), Decimal("0.50")),
}

# Some APIs answer with a dated name, e.g. "gpt-6.1-sol-2026-10-01".
_DATE_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class UnknownModelError(ValueError):
    """The model isn't in the price table, so its cost can't be worked out."""


def price_for(model: str) -> ModelPrice:
    """Look up a model's price. Raises UnknownModelError if it isn't listed."""
    price = PRICES.get(model) or PRICES.get(_DATE_SUFFIX.sub("", model))
    if price is None:
        raise UnknownModelError(
            f"no price for model {model!r}. Known models: {', '.join(sorted(PRICES))}. "
            "Add it to PRICES in pricing.py with the price from the provider's page."
        )
    return price


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call: tokens x price per million / 1,000,000."""
    price = price_for(model)
    # Decimal does exact decimal maths, so 0.002 + 0.005 is exactly 0.007.
    total = (
        input_tokens * price.input_per_million
        + output_tokens * price.output_per_million
    ) / 1_000_000
    return float(total)
