"""Phase 2 pilot: small paid experiments that settle how the bulk run is configured.

    uv run python -m cac40_mood.scoring.pilot count-tokens   # step 1: real token counts (free)
    uv run python -m cac40_mood.scoring.pilot thinking       # step 3: thinking off vs low, twice
    uv run python -m cac40_mood.scoring.pilot truncation     # step 2: short cap vs long cap
    uv run python -m cac40_mood.scoring.pilot reference      # step 4: cheap model vs reference

Rules that hold for every step:
- documents come from the train and validation periods only (never the test period);
- every answer is cached on disk and every paid call is logged, so re-running a step is free;
- the spend cap is checked before each request.
Results are written to `data/processed/pilot/` as one table per step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anthropic
import pandas as pd

from cac40_mood.config import CACHE_DIR, PROCESSED_DIR, load_settings
from cac40_mood.data.build import FILINGS_PATH, TEXTS_PATH
from cac40_mood.scoring.prompt import SYSTEM_PROMPT, build_user_message
from cac40_mood.scoring.scorer import ScoringConfig, score_document
from cac40_mood.scoring.spend import SpendLog

LLM_CACHE_DIR: Path = CACHE_DIR / "llm"
SPEND_LOG_PATH: Path = LLM_CACHE_DIR / "spend.jsonl"
PILOT_DIR: Path = PROCESSED_DIR / "pilot"

# Seeds are fixed so every step sees the same documents on every run.
PILOT_SEED: int = 20
REFERENCE_SEED: int = 50
REFERENCE_SIZE: int = 50

# Measured in step 1 on the 20 pilot documents (DECISIONS #31): French filing text has a median
# of 2.37 characters per token for this model family (range 2.0-2.75). Used to turn the token
# cap into a character cap, so that documents can be cut offline and deterministically.
CHARS_PER_TOKEN: float = 2.37

LONG_CAP_TOKENS: int = 12_000

# Room for the JSON answer (about 150 tokens) plus, when thinking is on, the reasoning.
MAX_OUTPUT_TOKENS_NO_THINKING: int = 600
MAX_OUTPUT_TOKENS_THINKING: int = 4_000


def scoreable_documents() -> pd.DataFrame:
    """Documents to score with usable text, joined to their text, train and validation only."""
    filings = pd.read_parquet(FILINGS_PATH)
    texts = pd.read_parquet(TEXTS_PATH)
    documents = filings[~filings["is_admin"]].merge(texts, on="doc_id")
    usable = documents["text_quality"] == "ok"
    in_train_or_val = documents["split"].isin(["train", "val"])
    return documents[usable & in_train_or_val].reset_index(drop=True)


def pilot_sample(size: int) -> pd.DataFrame:
    """The pilot documents: a fixed random draw from the train period."""
    documents = scoreable_documents()
    return documents[documents["split"] == "train"].sample(size, random_state=PILOT_SEED)


def reference_sample() -> pd.DataFrame:
    """A fixed random draw from train and validation for the reference-model comparison."""
    return scoreable_documents().sample(REFERENCE_SIZE, random_state=REFERENCE_SEED)


def chars_for_token_cap(token_cap: int) -> int:
    return int(token_cap * CHARS_PER_TOKEN)


def score_many(
    client: anthropic.Anthropic, config: ScoringConfig, documents: pd.DataFrame, spend_log: SpendLog
) -> pd.DataFrame:
    """Score each row of `documents` with one configuration; return one result row per doc."""
    records = [
        score_document(client, config, row.doc_id, row.title, row.text, LLM_CACHE_DIR, spend_log)
        for row in documents.itertuples()
    ]
    return pd.DataFrame(records)


def count_tokens_step(client: anthropic.Anthropic) -> pd.DataFrame:
    """Step 1: count real tokens of the pilot documents with the token-counting endpoint.

    The endpoint is free and generates nothing. Each document is counted twice: uncut (to
    learn the characters-per-token ratio and how many documents exceed the cap) and cut at
    the configured cap exactly as it would be sent (to learn the real input size).
    """
    settings = load_settings()
    documents = pilot_sample(settings.llm_pilot_size)
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    rows = []
    for row in documents.itertuples():
        full_message, _ = build_user_message(row.title, row.text, max_chars=10**9)
        cut_message, was_cut = build_user_message(row.title, row.text, max_chars=cap_chars)
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
    result.to_parquet(PILOT_DIR / "token_counts.parquet", index=False)
    return result


def thinking_step(client: anthropic.Anthropic, spend_log: SpendLog) -> pd.DataFrame:
    """Step 3: the pilot documents with thinking off and at low effort, each scored twice."""
    settings = load_settings()
    documents = pilot_sample(settings.llm_pilot_size)
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    tables = []
    for thinking, max_output in [
        ("off", MAX_OUTPUT_TOKENS_NO_THINKING),
        ("low", MAX_OUTPUT_TOKENS_THINKING),
    ]:
        for repeat in (1, 2):
            config = ScoringConfig(settings.llm_model, thinking, cap_chars, max_output, repeat)
            scores = score_many(client, config, documents, spend_log)
            tables.append(scores.assign(thinking=thinking, repeat=repeat))
    result = pd.concat(tables, ignore_index=True)
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PILOT_DIR / "thinking.parquet", index=False)
    return result


def truncation_step(
    client: anthropic.Anthropic, spend_log: SpendLog, thinking: str
) -> pd.DataFrame:
    """Step 2: documents longer than the cap, scored under four ways of shortening them.

    Uses every document from the pilot and reference samples that is longer than the cap.
    The 12,000-token version is the yardstick: the closer a cheaper variant's scores are to
    it, the less is lost by not reading further.
    """
    settings = load_settings()
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    candidates = pd.concat([pilot_sample(settings.llm_pilot_size), reference_sample()])
    candidates = candidates.drop_duplicates("doc_id")
    long_documents = candidates[candidates["text"].str.len() > cap_chars]
    max_output = MAX_OUTPUT_TOKENS_NO_THINKING if thinking == "off" else MAX_OUTPUT_TOKENS_THINKING
    # (label, characters kept from the start, characters of the outlook section added)
    variants = [
        ("cap_4k", cap_chars, 0),
        ("cap_8k", chars_for_token_cap(8_000), 0),
        ("start_4k_plus_outlook_2k", cap_chars, chars_for_token_cap(2_000)),
        ("cap_12k", chars_for_token_cap(LONG_CAP_TOKENS), 0),
    ]
    tables = []
    for label, chars, outlook_chars in variants:
        config = ScoringConfig(
            settings.llm_model, thinking, chars, max_output, outlook_chars=outlook_chars
        )
        scores = score_many(client, config, long_documents, spend_log)
        tables.append(scores.assign(cap=label))
    result = pd.concat(tables, ignore_index=True)
    result.to_parquet(PILOT_DIR / "truncation.parquet", index=False)
    return result


def reference_step(client: anthropic.Anthropic, spend_log: SpendLog, thinking: str) -> pd.DataFrame:
    """Step 4: 50 documents scored by the cheap model and by the reference model.

    The reference model reasons at medium effort. No automatic fallback to another model is
    configured: a silent switch would make "agreement with the reference" meaningless, so a
    refusal is simply recorded as a problem.
    """
    settings = load_settings()
    documents = reference_sample()
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    max_output = MAX_OUTPUT_TOKENS_NO_THINKING if thinking == "off" else MAX_OUTPUT_TOKENS_THINKING
    cheap = ScoringConfig(settings.llm_model, thinking, cap_chars, max_output)
    reference = ScoringConfig(
        settings.llm_reference_model, "medium", cap_chars, MAX_OUTPUT_TOKENS_THINKING
    )
    result = pd.concat(
        [
            score_many(client, cheap, documents, spend_log).assign(role="cheap"),
            score_many(client, reference, documents, spend_log).assign(role="reference"),
        ],
        ignore_index=True,
    )
    result.to_parquet(PILOT_DIR / "reference.parquet", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 pilot experiments.")
    parser.add_argument("step", choices=["count-tokens", "thinking", "truncation", "reference"])
    parser.add_argument("--thinking", default="off", help="setting used in steps 2 and 4")
    args = parser.parse_args()
    settings = load_settings()
    client = anthropic.Anthropic()
    spend_log = SpendLog(SPEND_LOG_PATH, settings.llm_spend_cap_usd)
    if args.step == "count-tokens":
        result = count_tokens_step(client)
    elif args.step == "thinking":
        result = thinking_step(client, spend_log)
    elif args.step == "truncation":
        result = truncation_step(client, spend_log, args.thinking)
    else:
        result = reference_step(client, spend_log, args.thinking)
    print(f"{args.step}: {len(result)} rows; total spent so far ${spend_log.total_spent():.4f}")


if __name__ == "__main__":
    main()
