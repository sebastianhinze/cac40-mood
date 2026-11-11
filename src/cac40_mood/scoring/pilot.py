"""Phase 2 pilot: small paid experiments that settle how the bulk run is configured.

    uv run python -m cac40_mood.scoring.pilot count-tokens   # real token counts (free endpoint)
    uv run python -m cac40_mood.scoring.pilot pilot          # 20 documents, scored twice
    uv run python -m cac40_mood.scoring.pilot reference      # 50 documents, cheap vs reference

Rules that hold for every step:
- documents come from the train and validation periods only (never the test period);
- every answer is cached on disk and every paid call is logged, so re-running a step is free;
- the spend cap is checked before each request;
- run one step at a time: two jobs running together can both miss the cache for the same
  document and pay for it twice.
Results are written to `data/processed/pilot/`, one table per step and prompt version.

Earlier pilot experiments (thinking off vs low, 4k vs 8k vs 12k token caps, a start-plus-outlook
cut) were run with prompt v1; their code is in git history (commit 4dea98e) and their outcomes
are DECISIONS #31 and #33.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anthropic
import pandas as pd

from cac40_mood.config import CACHE_DIR, PROCESSED_DIR, load_settings
from cac40_mood.scoring.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from cac40_mood.scoring.revenue import with_size
from cac40_mood.scoring.samples import chars_for_token_cap, pilot_sample, reference_sample
from cac40_mood.scoring.scorer import (
    BULK_MAX_OUTPUT_TOKENS,
    BULK_THINKING,
    ScoringConfig,
    score_document,
)
from cac40_mood.scoring.spend import SpendLog

LLM_CACHE_DIR: Path = CACHE_DIR / "llm"
SPEND_LOG_PATH: Path = LLM_CACHE_DIR / "spend.jsonl"
PILOT_DIR: Path = PROCESSED_DIR / "pilot"


def score_many(
    client: anthropic.Anthropic, config: ScoringConfig, documents: pd.DataFrame, spend_log: SpendLog
) -> pd.DataFrame:
    """Score each row of `documents` with one configuration; return one result row per doc."""
    columns = ["doc_id", "issuer", "size", "title", "text"]
    records = [
        score_document(client, config, document, LLM_CACHE_DIR, spend_log)
        for document in documents[columns].to_dict("records")
    ]
    return pd.DataFrame(records)


def count_tokens_step(client: anthropic.Anthropic) -> pd.DataFrame:
    """Count real tokens of the pilot documents with the token-counting endpoint.

    The endpoint is free and generates nothing. Each document is counted twice: uncut (to
    learn the characters-per-token ratio and how many documents exceed the cap) and cut at
    the configured cap exactly as it would be sent (to learn the real input size).
    """
    settings = load_settings()
    documents = with_size(pilot_sample(settings.llm_pilot_size))
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    rows = []
    for row in documents.itertuples():
        full_message, _ = build_user_message(
            row.issuer, row.size, row.title, row.text, max_chars=10**9
        )
        cut_message, was_cut = build_user_message(
            row.issuer, row.size, row.title, row.text, max_chars=cap_chars
        )
        document_only = client.messages.count_tokens(
            model=settings.llm_model, messages=[{"role": "user", "content": full_message}]
        ).input_tokens
        as_sent = client.messages.count_tokens(
            model=settings.llm_model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": cut_message}],
        ).input_tokens
        rows.append(
            {
                "doc_id": row.doc_id,
                "lang": row.lang,
                "chars": len(full_message),
                "tokens_uncut": document_only,
                "chars_per_token": len(full_message) / document_only,
                "cut_at_cap": was_cut,
                "tokens_as_sent": as_sent,
            }
        )
    result = pd.DataFrame(rows)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PILOT_DIR / f"token_counts_{PROMPT_VERSION}.parquet", index=False)
    return result


def pilot_step(client: anthropic.Anthropic, spend_log: SpendLog) -> pd.DataFrame:
    """The pilot documents scored twice with the bulk-run configuration.

    The second pass is not a duplicate by accident: comparing the two passes shows how much
    scores move between identical requests, which is the yardstick for every other comparison.
    """
    settings = load_settings()
    documents = with_size(pilot_sample(settings.llm_pilot_size))
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    tables = []
    for repeat in (1, 2):
        config = ScoringConfig(
            settings.llm_model, BULK_THINKING, cap_chars, BULK_MAX_OUTPUT_TOKENS, repeat
        )
        tables.append(score_many(client, config, documents, spend_log).assign(repeat=repeat))
    result = pd.concat(tables, ignore_index=True)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PILOT_DIR / f"pilot_{PROMPT_VERSION}.parquet", index=False)
    return result


def reference_step(client: anthropic.Anthropic, spend_log: SpendLog) -> pd.DataFrame:
    """50 documents scored by the cheap model and by the reference model, same prompt and cut.

    The reference model reasons at medium effort. No automatic fallback to another model is
    configured: a silent switch would make "agreement with the reference" meaningless, so a
    refusal is simply recorded as a problem (DECISIONS #35).
    """
    settings = load_settings()
    documents = with_size(reference_sample())
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    cheap = ScoringConfig(settings.llm_model, BULK_THINKING, cap_chars, BULK_MAX_OUTPUT_TOKENS)
    reference = ScoringConfig(
        settings.llm_reference_model, "medium", cap_chars, BULK_MAX_OUTPUT_TOKENS
    )
    result = pd.concat(
        [
            score_many(client, cheap, documents, spend_log).assign(role="cheap"),
            score_many(client, reference, documents, spend_log).assign(role="reference"),
        ],
        ignore_index=True,
    )
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PILOT_DIR / f"reference_{PROMPT_VERSION}.parquet", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 pilot experiments.")
    parser.add_argument("step", choices=["count-tokens", "pilot", "reference"])
    args = parser.parse_args()
    settings = load_settings()
    client = anthropic.Anthropic()
    spend_log = SpendLog(SPEND_LOG_PATH, settings.llm_spend_cap_usd)
    if args.step == "count-tokens":
        result = count_tokens_step(client)
    elif args.step == "pilot":
        result = pilot_step(client, spend_log)
    else:
        result = reference_step(client, spend_log)
    print(f"{args.step}: {len(result)} rows; total spent so far ${spend_log.total_spent():.4f}")


if __name__ == "__main__":
    main()
