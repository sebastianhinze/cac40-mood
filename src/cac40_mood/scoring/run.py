"""Phase 2 production runs (through the Batches API).

    uv run python -m cac40_mood.scoring.run revenue          # firm size per fiscal year
    uv run python -m cac40_mood.scoring.run bulk --dry-run   # cost estimate only
    uv run python -m cac40_mood.scoring.run bulk             # score every document

Run one step at a time. Every answer is cached and every paid call is logged; an interrupted
run can simply be started again and picks up its submitted batches.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

from cac40_mood.config import CACHE_DIR, PROCESSED_DIR, load_settings
from cac40_mood.scoring import revenue
from cac40_mood.scoring.samples import (
    CHARS_PER_TOKEN,
    chars_for_token_cap,
    documents_with_text,
    scoreable_documents,
)
from cac40_mood.scoring.scorer import (
    BULK_MAX_OUTPUT_TOKENS,
    BULK_THINKING,
    ScoringConfig,
    score_documents_in_batch,
)
from cac40_mood.scoring.spend import BATCH_DISCOUNT, SpendLog, cost_usd, worst_case_cost_usd

LLM_CACHE_DIR: Path = CACHE_DIR / "llm"
SPEND_LOG_PATH: Path = LLM_CACHE_DIR / "spend.jsonl"
BATCH_STATE_PATH: Path = LLM_CACHE_DIR / "batches.jsonl"

SCORES_PATH: Path = PROCESSED_DIR / "scores.parquet"

# Measured in the v2 pilot: tokens of instructions, schema and message header per request, and
# the typical length of an answer. Only used for the dry-run estimate.
OVERHEAD_INPUT_TOKENS: int = 2_170
TYPICAL_OUTPUT_TOKENS: int = 125


def revenue_step(client: anthropic.Anthropic, spend_log: SpendLog) -> pd.DataFrame:
    """Extract full-year revenue from train/validation results releases; save facts and table."""
    settings = load_settings()
    candidates = revenue.candidate_documents(scoreable_documents())
    facts = revenue.extract_facts(
        client, settings.llm_model, candidates, LLM_CACHE_DIR, spend_log, BATCH_STATE_PATH
    )
    table = revenue.fiscal_year_table(facts)
    facts.to_parquet(revenue.FACTS_PATH, index=False)
    table.to_parquet(revenue.TABLE_PATH, index=False)
    print(f"revenue: {len(candidates)} candidates, {len(facts)} answers, {len(table)} firm-years")
    return table


def bulk_config() -> ScoringConfig:
    settings = load_settings()
    return ScoringConfig(
        settings.llm_model,
        BULK_THINKING,
        chars_for_token_cap(settings.llm_max_doc_tokens),
        BULK_MAX_OUTPUT_TOKENS,
    )


def bulk_documents() -> pd.DataFrame:
    """Every document with usable text (train, validation and test), with the issuer's size."""
    documents = documents_with_text(include_test=True)
    return revenue.with_size(documents)


def dry_run_estimate(documents: pd.DataFrame, config: ScoringConfig) -> dict[str, float]:
    """Expected and worst-case batch cost of scoring `documents` (nothing is sent)."""
    doc_chars = documents["text"].str.len().clip(upper=config.max_doc_chars).to_numpy()
    input_tokens = doc_chars / CHARS_PER_TOKEN + OVERHEAD_INPUT_TOKENS
    expected = sum(
        cost_usd(config.model, int(tokens), TYPICAL_OUTPUT_TOKENS, batch=True)
        for tokens in input_tokens
    )
    # 6,000 characters stand in for instructions and header in the pessimistic bound.
    worst = BATCH_DISCOUNT * sum(
        worst_case_cost_usd(config.model, int(chars) + 6_000, config.max_output_tokens)
        for chars in doc_chars
    )
    return {
        "documents": float(len(documents)),
        "expected_input_tokens_million": float(np.sum(input_tokens) / 1e6),
        "expected_usd": float(expected),
        "worst_case_usd": float(worst),
    }


def bulk_step(client: anthropic.Anthropic, spend_log: SpendLog, dry_run: bool) -> pd.DataFrame:
    """Print the cost estimate; unless `dry_run`, score every document and save the scores."""
    config = bulk_config()
    documents = bulk_documents()
    estimate = dry_run_estimate(documents, config)
    print(
        f"dry run: {estimate['documents']:.0f} documents, "
        f"{estimate['expected_input_tokens_million']:.1f}M input tokens, "
        f"expected ${estimate['expected_usd']:.2f}, worst case ${estimate['worst_case_usd']:.2f}; "
        f"spent so far ${spend_log.total_spent():.2f} of ${spend_log.cap_usd:.2f}",
        flush=True,
    )
    if dry_run:
        return pd.DataFrame()
    columns = ["doc_id", "issuer", "size", "title", "text"]
    scores = score_documents_in_batch(
        client,
        config,
        documents[columns].to_dict("records"),
        LLM_CACHE_DIR,
        spend_log,
        BATCH_STATE_PATH,
    )
    scores = scores.merge(documents[["doc_id", "size", "size_fy", "size_backfilled"]], on="doc_id")
    scores.to_parquet(SCORES_PATH, index=False)
    problems = int((scores["problem"] != "").sum())
    print(
        f"bulk: {len(scores)} documents, {problems} problems; spent ${spend_log.total_spent():.2f}"
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 production runs.")
    parser.add_argument("step", choices=["revenue", "bulk"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    client = anthropic.Anthropic()
    spend_log = SpendLog(SPEND_LOG_PATH, settings.llm_spend_cap_usd)
    if args.step == "revenue":
        revenue_step(client, spend_log)
    else:
        bulk_step(client, spend_log, args.dry_run)


if __name__ == "__main__":
    main()
