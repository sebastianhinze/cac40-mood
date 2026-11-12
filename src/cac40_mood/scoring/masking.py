"""Memory check: does the LLM score differently when it cannot tell who published the text?

    uv run python -m cac40_mood.scoring.masking pilot     # first 20 documents
    uv run python -m cac40_mood.scoring.masking full      # all of the sample (about 100)

Worry (LIMITATIONS #10, #20): the scoring model may remember what happened to a company after
a release. If so, its scores would contain hindsight, and "surprise explains returns" would
partly be the model reciting history.

The check: take a sample of documents, hide who and when, score them again with the same
prompt, and compare with the original scores.

Masking is done in two steps so that every replacement is visible in code:
1. the model **lists** the identifying strings it finds in the text (companies, brands,
   people, places, tickers); it does not rewrite anything;
2. code replaces each listed string with a neutral placeholder, and every year with its
   distance to the publication year ("[year 0]", "[year -1]"), so "2021 against 2020" stays
   readable as "this year against last year".
Numbers are left alone: the model needs them to judge the news.

A third call asks the model to guess company and year from the masked text. That measures how
well the masking worked, instead of assuming it did.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd

from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import PROCESSED_DIR, load_settings
from cac40_mood.scoring import llm
from cac40_mood.scoring.pilot import LLM_CACHE_DIR, SPEND_LOG_PATH
from cac40_mood.scoring.prompt import truncate_text
from cac40_mood.scoring.revenue import with_size
from cac40_mood.scoring.samples import chars_for_token_cap, scoreable_documents
from cac40_mood.scoring.scorer import (
    BULK_MAX_OUTPUT_TOKENS,
    BULK_THINKING,
    ScoringConfig,
    score_document,
)
from cac40_mood.scoring.spend import SpendLog

MASK_PROMPT_VERSION: str = "mask-v1"
GUESS_PROMPT_VERSION: str = "guess-v1"
SAMPLE_SEED: int = 100
# Documents per surprise group. Non-zero surprise is over-sampled on purpose: the question is
# about surprise, and 79% of all documents have surprise 0.
SAMPLE_QUOTAS: dict[str, int] = {"negative": 35, "positive": 35, "zero": 30}
PILOT_SIZE: int = 20
ISSUER_PLACEHOLDER: str = "the Company"
MIN_NAME_LENGTH: int = 3

RESULT_PATH: Path = PROCESSED_DIR / "memory_check" / "masked_scores.parquet"

KINDS: dict[str, str] = {
    "company": "Company",
    "brand": "Brand",
    "person": "Person",
    "place": "Place",
    "other": "Name",
}

LIST_SYSTEM_PROMPT: str = """\
You prepare company announcements for an anonymised reading test. The announcement was \
published by the company named as "Issuer". List every string in the title and text that \
could tell a reader which company this is. Copy each string exactly as it is written.

Include:
- the issuer's own name in every form used (full name, short name, group name, abbreviations);
  give these the kind "issuer";
- other companies: subsidiaries, partners, customers, competitors, targets, banks, auditors
  (kind "company");
- brand, product, drug, project and site names (kind "brand");
- people's names (kind "person");
- cities, streets, addresses, stock exchanges and index names (kind "place");
- stock tickers, ISIN codes, web addresses, e-mail addresses, phone numbers (kind "other").

Do not include: countries and continents, currencies, amounts, dates, ordinary words, \
accounting terms, or names of regulators and laws.
"""

LIST_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["issuer", *KINDS]},
                },
                "required": ["text", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["names"],
    "additionalProperties": False,
}

GUESS_SYSTEM_PROMPT: str = """\
Below is a company announcement in which names and years were replaced by placeholders. \
Guess which listed company published it and in which calendar year. Always give your best \
guess, even if unsure.
"""

GUESS_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer"},
        "sure": {"type": "boolean"},
    },
    "required": ["company", "year", "sure"],
    "additionalProperties": False,
}

YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


# ----------------------------------------------------------------- sample


def surprise_group(surprise: pd.Series) -> pd.Series:
    """'negative', 'zero' or 'positive'."""
    return surprise.map(
        lambda value: "zero" if value == 0 else ("positive" if value > 0 else "negative")
    )


def memory_sample(documents: pd.DataFrame, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """A fixed draw from the analysis documents: a quota per surprise group, then shuffled."""
    groups = surprise_group(documents["surprise"])
    chosen = [
        documents[groups == name].sample(quota, random_state=seed)
        for name, quota in SAMPLE_QUOTAS.items()
    ]
    return pd.concat(chosen).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- masking


def mask_years(text: str, publication_year: int) -> str:
    """Replace every four-digit year by its distance to the publication year."""

    def relative(match: re.Match[str]) -> str:
        return f"[year {int(match.group()) - publication_year:+d}]".replace("+0]", "0]")

    return YEAR.sub(relative, text)


def placeholders_for(names: list[dict[str, str]], always_issuer: list[str]) -> dict[str, str]:
    """Map each identifying string to its placeholder.

    The issuer's names all become "the Company". Other names are numbered per kind in the
    order they were listed ("Company 1", "Person 2"); the same string, ignoring case, always
    gets the same placeholder. Strings shorter than three characters are skipped: replacing
    "SA" or "AG" everywhere would damage ordinary words.
    """
    mapping = {name: ISSUER_PLACEHOLDER for name in always_issuer if len(name) >= MIN_NAME_LENGTH}
    seen = {name.lower() for name in mapping}
    counters = dict.fromkeys(KINDS, 0)
    for entry in names:
        text = entry["text"].strip()
        if len(text) < MIN_NAME_LENGTH or text.lower() in seen:
            continue
        seen.add(text.lower())
        if entry["kind"] == "issuer":
            mapping[text] = ISSUER_PLACEHOLDER
        else:
            counters[entry["kind"]] += 1
            mapping[text] = f"{KINDS[entry['kind']]} {counters[entry['kind']]}"
    return mapping


def apply_mask(text: str, mapping: dict[str, str], publication_year: int) -> str:
    """Replace the listed strings (longest first, whole words, any case), then the years."""
    for name in sorted(mapping, key=len, reverse=True):
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", flags=re.IGNORECASE)
        text = pattern.sub(mapping[name], text)
    return mask_years(text, publication_year)


# -------------------------------------------------------------- LLM calls


def list_request(model: str, issuer: str, title: str, text: str) -> dict[str, Any]:
    message = f"Issuer: {issuer}\nTitle: {title.strip()}\n\n<document>\n{text}\n</document>"
    return {
        "model": model,
        "max_tokens": 4_000,
        "system": LIST_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": message}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": LIST_JSON_SCHEMA},
        },
    }


def guess_request(model: str, title: str, text: str) -> dict[str, Any]:
    message = f"Title: {title.strip()}\n\n<document>\n{text}\n</document>"
    return {
        "model": model,
        "max_tokens": 600,
        "system": GUESS_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": message}],
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": GUESS_JSON_SCHEMA},
        },
    }


def json_answer(answer: dict[str, Any]) -> dict[str, Any] | None:
    """The parsed JSON of a finished answer, or None if the call did not end normally."""
    if answer["stop_reason"] != "end_turn":
        return None
    try:
        return json.loads(answer["text"])
    except json.JSONDecodeError:
        return None


def plain(text: str) -> str:
    """Lower case without accents or punctuation, for comparing company names."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    letters = "".join(c for c in decomposed if c.isalnum() or c.isspace())
    return " ".join(letters.split())


def guess_is_right(guess: str, issuer: str) -> bool:
    """True if the guessed name contains the issuer's name or the other way round."""
    guessed, actual = plain(guess), plain(issuer)
    return bool(guessed) and (actual in guessed or guessed in actual)


def check_document(
    client: anthropic.Anthropic,
    document: dict[str, Any],
    config: ScoringConfig,
    spend_log: SpendLog,
) -> dict[str, Any]:
    """List names, mask, re-score and ask for a guess: one result row for one document."""
    model = config.model
    doc_id = document["doc_id"]
    shown, _ = truncate_text(document["text"], config.max_doc_chars)
    year = int(document["ts_paris"].year)

    request = list_request(model, document["issuer"], document["title"], shown)
    key = llm.cache_key(request, MASK_PROMPT_VERSION)
    listed = json_answer(
        llm.call_once(client, request, key, MASK_PROMPT_VERSION, doc_id, LLM_CACHE_DIR, spend_log)
    )
    if listed is None:
        return {"doc_id": doc_id, "problem": "name list failed"}
    # The name as we know it and as the feed writes it. Not the ticker: "VIE" and "CAP" are
    # also ordinary French words.
    always = [document["issuer"], document["issuer_name"]]
    mapping = placeholders_for(listed["names"], always)
    masked_text = apply_mask(shown, mapping, year)
    masked_title = apply_mask(document["title"], mapping, year)

    masked_document = {
        "doc_id": doc_id,
        "issuer": ISSUER_PLACEHOLDER,
        "size": document["size"],
        "title": masked_title,
        "text": masked_text,
    }
    score = score_document(client, config, masked_document, LLM_CACHE_DIR, spend_log)

    request = guess_request(model, masked_title, masked_text)
    key = llm.cache_key(request, GUESS_PROMPT_VERSION)
    guess = json_answer(
        llm.call_once(client, request, key, GUESS_PROMPT_VERSION, doc_id, LLM_CACHE_DIR, spend_log)
    ) or {"company": "", "year": 0, "sure": False}
    return {
        "doc_id": doc_id,
        "problem": score["problem"],
        "names_masked": len(mapping),
        "masked_title": masked_title,
        "masked_start": masked_text[:400],
        "masked_tone": score.get("tone"),
        "masked_surprise": score.get("surprise"),
        "masked_uncertainty": score.get("uncertainty"),
        "masked_topic": score.get("topic"),
        "guess_company": guess["company"],
        "guess_year": guess["year"],
        "guess_sure": guess["sure"],
        "guessed_company": guess_is_right(guess["company"], document["issuer"]),
        "guessed_year": guess["year"] == year,
    }


def sample_with_text() -> pd.DataFrame:
    """The memory-check sample with text, issuer names and the size given in the bulk run."""
    documents, _ = load_documents()
    sample = memory_sample(documents)
    texts = with_size(scoreable_documents())
    columns = ["doc_id", "issuer", "issuer_name", "title", "text", "size"]
    return sample.merge(texts[columns], on="doc_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked re-scoring (memory check).")
    parser.add_argument("step", choices=["pilot", "full"])
    args = parser.parse_args()
    settings = load_settings()
    client = anthropic.Anthropic()
    spend_log = SpendLog(SPEND_LOG_PATH, settings.llm_spend_cap_usd)
    config = ScoringConfig(
        settings.llm_model,
        BULK_THINKING,
        chars_for_token_cap(settings.llm_max_doc_tokens),
        BULK_MAX_OUTPUT_TOKENS,
    )
    sample = sample_with_text()
    if args.step == "pilot":
        sample = sample.head(PILOT_SIZE)
    rows = [
        check_document(client, document, config, spend_log)
        for document in sample.to_dict("records")
    ]
    result = sample.drop(columns=["text"]).merge(
        pd.DataFrame(rows), on="doc_id", suffixes=("", "_masked")
    )
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(RESULT_PATH.with_name(f"masked_scores_{args.step}.parquet"), index=False)
    print(f"{args.step}: {len(result)} documents; total spent ${spend_log.total_spent():.4f}")


if __name__ == "__main__":
    main()
