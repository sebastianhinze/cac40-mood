"""Which documents the pilot, the reference check and the gold set use.

This module only reads the filings and their texts. It never touches model answers, so the
labelling tool can import it without any risk of showing a score to the person labelling.
"""

from __future__ import annotations

import pandas as pd

from cac40_mood.data.build import FILINGS_PATH, TEXTS_PATH
from cac40_mood.data.universe import load_universe

# Seeds are fixed so every step sees the same documents on every run.
PILOT_SEED: int = 20
REFERENCE_SEED: int = 50
REFERENCE_SIZE: int = 50

# Measured on the 20 pilot documents (DECISIONS #32): French filing text has a median of 2.37
# characters per token for this model family (range 2.0-2.75). Used to turn the token cap
# into a character cap, so that documents can be cut offline and deterministically.
CHARS_PER_TOKEN: float = 2.37


def chars_for_token_cap(token_cap: int) -> int:
    return int(token_cap * CHARS_PER_TOKEN)


def documents_with_text(include_test: bool) -> pd.DataFrame:
    """Documents to score that have usable text, with the issuer's name.

    `include_test=True` is only for the bulk scoring run: test-period documents must be scored
    so that features exist for the final evaluation, but nothing is analysed on them before.
    """
    filings = pd.read_parquet(FILINGS_PATH)
    texts = pd.read_parquet(TEXTS_PATH)
    firms = load_universe().rename(columns={"name": "issuer"})
    documents = filings[~filings["is_admin"]].merge(texts, on="doc_id")
    documents = documents.merge(firms[["firm_id", "issuer"]], on="firm_id")
    keep = documents["text_quality"] == "ok"
    if not include_test:
        keep &= documents["split"].isin(["train", "val"])
    return documents[keep].reset_index(drop=True)


def scoreable_documents() -> pd.DataFrame:
    """Usable documents from the train and validation periods (everything except the bulk run)."""
    return documents_with_text(include_test=False)


def pilot_sample(size: int) -> pd.DataFrame:
    """The pilot documents: a fixed random draw from the train period."""
    documents = scoreable_documents()
    return documents[documents["split"] == "train"].sample(size, random_state=PILOT_SEED)


def reference_sample() -> pd.DataFrame:
    """A fixed random draw from train and validation for the reference-model comparison."""
    return scoreable_documents().sample(REFERENCE_SIZE, random_state=REFERENCE_SEED)
