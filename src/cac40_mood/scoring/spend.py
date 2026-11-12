"""Prices, the spend log and the hard spending cap (CLAUDE.md, "LLM cost control").

Every call appends one line to `data/cache/llm/spend.jsonl`. Before a request is sent, its
worst-case cost is added to the total spent so far; if that would pass `LLM_SPEND_CAP_USD`
the request is refused. The cap can therefore never be exceeded, only approached.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

# USD per million tokens (input, output), standard price for prompts up to 100k tokens.
# Source: Anthropic pricing page, checked 2026-11-11. The Batches API costs half.
# A model that is not listed here cannot be used: without a price there is no cost bound.
PRICES_PER_MILLION: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
BATCH_DISCOUNT: float = 0.5

# Used only for the worst-case bound. Real filings measured 2.0-2.75 characters per token
# (pilot step 1), so assuming 1.5 always overstates the input size.
WORST_CASE_CHARS_PER_TOKEN: float = 1.5


class SpendCapExceededError(RuntimeError):
    """Sending this request could push total spend above the cap, so it was not sent."""


def price_per_million(model: str) -> tuple[float, float]:
    """(input, output) price for a model id. A dated id such as 'claude-haiku-5-5-20260101'
    is priced like its base id; an unknown model raises, because it has no cost bound."""
    for known, prices in PRICES_PER_MILLION.items():
        if model == known or model.startswith(known + "-"):
            return prices
    raise KeyError(f"No price recorded for model {model!r}; add it to PRICES_PER_MILLION")


def cost_usd(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
    """Cost of one call in USD. Thinking tokens are billed as output tokens."""
    input_price, output_price = price_per_million(model)
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


def worst_case_cost_usd(model: str, prompt_chars: int, max_output_tokens: int) -> float:
    """Upper bound for one request: pessimistic input size and the full output allowance."""
    input_tokens = int(prompt_chars / WORST_CASE_CHARS_PER_TOKEN) + 1
    return cost_usd(model, input_tokens, max_output_tokens)


class SpendLog:
    """Append-only log of paid calls with a running total and the cap check."""

    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = path
        self.cap_usd = cap_usd

    def total_spent(self) -> float:
        """Sum of all logged costs (0.0 when nothing has been spent yet)."""
        if not self.path.exists():
            return 0.0
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        return sum(json.loads(line)["cost_usd"] for line in lines)

    def check_can_spend(self, worst_case_usd: float) -> None:
        """Raise unless spent-so-far plus this request's worst case stays within the cap."""
        spent = self.total_spent()
        if spent + worst_case_usd > self.cap_usd:
            raise SpendCapExceededError(
                f"spent ${spent:.4f} + worst case ${worst_case_usd:.4f} > cap ${self.cap_usd:.2f}"
            )

    def record(
        self,
        model: str,
        version_tag: str,
        doc_id: str,
        input_tokens: int,
        output_tokens: int,
        batch: bool = False,
    ) -> float:
        """Log one paid call (at batch prices if it came through a batch) and return its cost."""
        cost = cost_usd(model, input_tokens, output_tokens, batch)
        entry = {
            "at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": model,
            "version_tag": version_tag,
            "doc_id": doc_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batch": batch,
            "cost_usd": cost,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        return cost
