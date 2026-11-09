"""Log every LLM call as one JSON line, and add the log up."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from content_api.llm import LLMResult
from content_api.pricing import cost_usd

DEFAULT_LOG = Path("llm_calls.jsonl")  # relative to the folder you run from (api/)


def log_call(
    result: LLMResult,
    *,
    feature: str,
    prompt_version: str,
    path: str | Path = DEFAULT_LOG,
) -> dict[str, Any]:
    """Append one line describing this call to the log, and return it."""
    entry: dict[str, Any] = {
        "time": datetime.now(UTC).isoformat(),
        "provider": result.provider,
        "model": result.model,
        "prompt_version": prompt_version,
        "feature": feature,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": cost_usd(result.model, result.input_tokens, result.output_tokens),
        "latency_seconds": round(result.latency_seconds, 3),
    }
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


@dataclass(frozen=True)
class UsageSummary:
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def summarize(path: str | Path = DEFAULT_LOG) -> UsageSummary:
    """Add up every call in the log. A missing log means no calls yet."""
    log = Path(path)
    if not log.exists():
        return UsageSummary(0, 0, 0, 0.0)

    calls = input_tokens = output_tokens = 0
    cost = Decimal(0)
    with log.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{log.name} line {line_number} is not valid JSON"
                ) from error
            calls += 1
            input_tokens += entry["input_tokens"]
            output_tokens += entry["output_tokens"]
            cost += Decimal(str(entry["cost_usd"]))  # exact sum, no float drift
    return UsageSummary(calls, input_tokens, output_tokens, float(cost))
