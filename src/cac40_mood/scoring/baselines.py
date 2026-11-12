"""Run the two baselines over every document with usable text.

    uv run python -m cac40_mood.scoring.baselines lexicon
    uv run python -m cac40_mood.scoring.baselines local-model

Both read exactly the text the LLM reads (the first 12,000 tokens' worth of characters), so the
three scorers are compared on the same material. Like the LLM run, test-period documents are
scored but nothing is analysed on them here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cac40_mood.config import PROCESSED_DIR, load_settings
from cac40_mood.scoring.lexicon import lexicon_scores
from cac40_mood.scoring.prompt import truncate_text
from cac40_mood.scoring.samples import chars_for_token_cap, documents_with_text

LEXICON_PATH: Path = PROCESSED_DIR / "baseline_lexicon.parquet"
LOCAL_MODEL_PATH: Path = PROCESSED_DIR / "baseline_local_model.parquet"
SAVE_EVERY: int = 500


def documents_as_read_by_the_llm() -> pd.DataFrame:
    """doc_id, lang and the text cut where the LLM stops reading."""
    cap_chars = chars_for_token_cap(load_settings().llm_max_doc_tokens)
    documents = documents_with_text(include_test=True)[["doc_id", "lang", "text"]].copy()
    documents["text"] = [truncate_text(text, cap_chars)[0] for text in documents["text"]]
    return documents


def run_lexicon() -> pd.DataFrame:
    documents = documents_as_read_by_the_llm()
    rows = [
        {"doc_id": row.doc_id, **lexicon_scores(row.text, row.lang)}
        for row in documents.itertuples()
    ]
    scores = pd.DataFrame(rows)
    scores.to_parquet(LEXICON_PATH, index=False)
    return scores


def run_local_model() -> pd.DataFrame:
    """Score all documents; finished documents are saved every 500 and skipped on a restart."""
    from cac40_mood.scoring import local_model  # imported here: needs torch and transformers

    documents = documents_as_read_by_the_llm()
    done = pd.read_parquet(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else pd.DataFrame()
    finished = set(done["doc_id"]) if len(done) else set()
    device = local_model.pick_device()
    rows = done.to_dict("records")
    for language in ("fr", "en"):
        todo = documents[(documents["lang"] == language) & ~documents["doc_id"].isin(finished)]
        if todo.empty:
            continue
        tokenizer, model, labels = local_model.load_classifier(language, device)
        for count, row in enumerate(todo.itertuples(), start=1):
            sentences = local_model.split_sentences(row.text)
            probabilities = local_model.classify(tokenizer, model, sentences, device)
            rows.append(
                {
                    "doc_id": row.doc_id,
                    "lm_model": local_model.MODELS[language][0],
                    **local_model.document_scores(probabilities, labels),
                }
            )
            if count % SAVE_EVERY == 0:
                pd.DataFrame(rows).to_parquet(LOCAL_MODEL_PATH, index=False)
                print(f"{language}: {count} of {len(todo)}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_parquet(LOCAL_MODEL_PATH, index=False)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline scorers.")
    parser.add_argument("step", choices=["lexicon", "local-model"])
    args = parser.parse_args()
    scores = run_lexicon() if args.step == "lexicon" else run_local_model()
    print(f"{args.step}: {len(scores)} documents scored")


if __name__ == "__main__":
    main()
