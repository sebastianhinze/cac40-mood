"""Baseline 3: the Loughran-McDonald finance word lists, on the official English versions.

    uv run python -m cac40_mood.data.build fetch-pdfs-en   # English twins of French filings
    uv run python -m cac40_mood.data.build texts-en
    uv run python -m cac40_mood.scoring.loughran

Loughran and McDonald (2011, Journal of Finance) built word lists from US annual reports:
words that are negative, positive or uncertain *in financial text* ("liability" is not
negative there, "impairment" is). It is the standard dictionary baseline in finance research.

The lists are English only, so this baseline covers the documents that have an official
English version: English originals, plus French filings whose English twin is in the feed.
French-only documents get no score here (they are not machine-translated: a translation
would add its own errors to a baseline whose point is to be simple).

Licence (DECISIONS #59, docs/DATA_SOURCES.md): the dictionary is "free for use in academic
research"; commercial use needs a licence from the authors. The file is downloaded to
`data/raw/lexicon/` and is never committed or redistributed.

Scoring, the same formulas as the hand-written lexicon so the two can be compared:
    tone        = (positive - negative) / (positive + negative)     (0 if neither occurs)
    uncertainty = uncertainty words per 100 words
A positive word does not count when one of the three words before it is a negation
("not improved"), the rule Loughran and McDonald suggest for positive words.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cac40_mood.config import PROCESSED_DIR, RAW_DIR, load_settings
from cac40_mood.data.build import FILINGS_PATH, TEXTS_EN_PATH, TEXTS_PATH
from cac40_mood.scoring.prompt import truncate_text
from cac40_mood.scoring.samples import chars_for_token_cap

DICTIONARY_PATH: Path = RAW_DIR / "lexicon" / "download" / "lm_master_dictionary.csv"
SCORES_PATH: Path = PROCESSED_DIR / "baseline_loughran.parquet"

WORD = re.compile(r"[a-z]+")
NEGATION_WINDOW: int = 3
NEGATIONS: frozenset[str] = frozenset(
    {"no", "not", "none", "neither", "never", "nobody", "nor", "without", "cannot"}
)
CATEGORIES: dict[str, str] = {
    "positive": "Positive",
    "negative": "Negative",
    "uncertainty": "Uncertainty",
}


def load_word_lists(path: Path = DICTIONARY_PATH) -> dict[str, frozenset[str]]:
    """{"positive": words, "negative": words, "uncertainty": words}, in lower case.

    In the master dictionary a category column holds 0 for "not in this list" and otherwise
    the year the word was added to it.
    """
    dictionary = pd.read_csv(path, keep_default_na=False)  # "NULL" and "NA" are real words
    words = dictionary["Word"].str.lower()
    return {
        name: frozenset(words[dictionary[column].astype(int) != 0])
        for name, column in CATEGORIES.items()
    }


def loughran_scores(text: str, lists: dict[str, frozenset[str]]) -> dict[str, float]:
    """Counts and scores for one English text."""
    words = WORD.findall(text.lower())
    positive = negative = uncertain = 0
    for position, word in enumerate(words):
        if word in lists["negative"]:
            negative += 1
        elif word in lists["positive"]:
            previous = words[max(0, position - NEGATION_WINDOW) : position]
            if not any(earlier in NEGATIONS for earlier in previous):
                positive += 1
        if word in lists["uncertainty"]:
            uncertain += 1
    polar = positive + negative
    return {
        "lmd_words": len(words),
        "lmd_positive": positive,
        "lmd_negative": negative,
        "lmd_uncertain": uncertain,
        "lmd_tone": (positive - negative) / polar if polar else 0.0,
        "lmd_uncertainty": 100.0 * uncertain / len(words) if words else 0.0,
    }


def english_texts(
    filings: pd.DataFrame, texts: pd.DataFrame, english_twins: pd.DataFrame
) -> pd.DataFrame:
    """One English text per document where one exists: doc_id, text, english_source.

    `english_source` is "original" for a document filed in English and "twin" for the official
    English version of a French document. Only texts of usable quality are kept.
    """
    originals = filings.loc[filings["lang"] == "en", ["doc_id"]].merge(texts, on="doc_id")
    originals = originals.loc[originals["text_quality"] == "ok", ["doc_id", "text"]]
    twins = english_twins.loc[english_twins["text_quality"] == "ok", ["doc_id", "text"]]
    return pd.concat(
        [originals.assign(english_source="original"), twins.assign(english_source="twin")],
        ignore_index=True,
    ).drop_duplicates("doc_id")


def score_all(
    english: pd.DataFrame, lists: dict[str, frozenset[str]], cap_chars: int
) -> pd.DataFrame:
    """Score every English text, cut where the LLM stops reading."""
    rows = [
        {
            "doc_id": doc_id,
            "english_source": source,
            **loughran_scores(truncate_text(text, cap_chars)[0], lists),
        }
        for doc_id, text, source in zip(
            english["doc_id"], english["text"], english["english_source"], strict=True
        )
    ]
    return pd.DataFrame(rows)


def main() -> None:
    filings = pd.read_parquet(FILINGS_PATH)
    in_sample = filings[~filings["is_admin"] & filings["split"].isin(["train", "val"])]
    english = english_texts(in_sample, pd.read_parquet(TEXTS_PATH), pd.read_parquet(TEXTS_EN_PATH))
    cap_chars = chars_for_token_cap(load_settings().llm_max_doc_tokens)
    scores = score_all(english, load_word_lists(), cap_chars)
    scores.to_parquet(SCORES_PATH, index=False)
    print(
        f"Loughran-McDonald: {len(scores)} documents scored "
        f"({scores['english_source'].value_counts().to_dict()}); "
        f"mean tone {scores['lmd_tone'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
