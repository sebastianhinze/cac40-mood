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


def scoreable_documents() -> pd.DataFrame:
    """Documents to score with usable text, with issuer name and size, train and validation only."""
    filings = pd.read_parquet(FILINGS_PATH)
    texts = pd.read_parquet(TEXTS_PATH)
    firms = load_universe().rename(
        columns={"name": "issuer", "revenue_2018_eur_bn": "revenue_eur_bn"}
    )
    documents = filings[~filings["is_admin"]].merge(texts, on="doc_id")
    documents = documents.merge(firms[["firm_id", "issuer", "revenue_eur_bn"]], on="firm_id")
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
