"""The hand-labelled gold set: which documents to label, and where the labels are stored.

Purpose (DECISIONS #17): the only direct check that the model's scores measure tone is to
compare them with a human reading of the same documents.

Design choices
- **Blind:** this module and the labelling page read filings and texts only. Nothing here opens
  the model cache or any score table, so no model score can be shown while labelling.
- **Unseen documents:** documents used in the pilot or the Sonnet comparison are excluded.
  The prompt was tuned while looking at those, so agreement on them would be flattering.
- **Stratified:** the model's topics do not exist yet for unseen documents, so the strata use
  the AMF type code of the filing, which is a rough stand-in for topic (results releases,
  operations, other press releases, quarterly information, reports, governance, takeovers).
- **Long documents included on purpose:** about a third of each stratum is drawn from documents
  longer than 4,000 tokens, where the question "did the model read enough?" matters most.
- **Same text as the model:** the page shows the text cut at the model's cap, so human and
  model judge the same material.
- Train and validation periods only, like every other tuning step.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from cac40_mood.config import PROCESSED_DIR, load_settings
from cac40_mood.scoring.prompt import truncate_text
from cac40_mood.scoring.samples import (
    chars_for_token_cap,
    pilot_sample,
    reference_sample,
    scoreable_documents,
)
from cac40_mood.scoring.schema import TOPICS

GOLD_DIR: Path = PROCESSED_DIR / "gold"
SAMPLE_PATH: Path = GOLD_DIR / "sample.parquet"
LABELS_PATH: Path = GOLD_DIR / "labels.csv"
GOLD_SEED: int = 40

# Tone scale for the human label. The model's -1..1 tone is compared with it by rank.
TONE_LABELS: dict[int, str] = {
    -2: "-2  clearly bad",
    -1: "-1  somewhat bad",
    0: "0  neutral / routine",
    1: "+1  somewhat good",
    2: "+2  clearly good",
}

# Documents per stratum (40 in total). Strata follow the AMF type code of the filing.
STRATUM_QUOTAS: dict[str, int] = {
    "results_release": 8,  # 001007, 001008
    "operations": 6,  # 001002: acquisitions, disposals, partnerships
    "other_press_release": 12,  # 001006, 015000: the large mixed bag
    "quarterly_information": 5,  # 005xxx
    "periodic_report": 4,  # 003000, 004000, 300005
    "governance": 2,  # 008000
    "takeover": 2,  # 009xxx
    "other": 1,
}
LONG_DOCUMENT_TOKENS: int = 4_000
LONG_SHARE: float = 1 / 3

LABEL_COLUMNS: list[str] = ["doc_id", "tone", "topic", "comment", "labelled_at_utc"]


def stratum_of(type_code: str) -> str:
    """Map an AMF type code to its gold-set stratum."""
    code = str(type_code)
    if code in {"001007", "001008"}:
        return "results_release"
    if code == "001002":
        return "operations"
    if code in {"001006", "015000"}:
        return "other_press_release"
    by_family = {
        "005": "quarterly_information",
        "003": "periodic_report",
        "004": "periodic_report",
        "300": "periodic_report",
        "008": "governance",
        "009": "takeover",
    }
    return by_family.get(code[:3], "other")


def select_gold_sample(
    documents: pd.DataFrame, exclude_doc_ids: set[str], long_chars: int, seed: int
) -> pd.DataFrame:
    """Draw the gold sample: fixed quota per stratum, about a third of it long documents.

    If a stratum has too few long (or short) documents, the other kind fills the gap, so the
    quota is always met when the stratum has enough documents at all.
    """
    pool = documents[~documents["doc_id"].isin(exclude_doc_ids)].copy()
    pool["stratum"] = pool["type_code"].map(stratum_of)
    pool["is_long"] = pool["text"].str.len() > long_chars
    chosen = []
    for stratum, quota in STRATUM_QUOTAS.items():
        members = pool[pool["stratum"] == stratum]
        long_members, short_members = members[members["is_long"]], members[~members["is_long"]]
        n_long = min(math.ceil(quota * LONG_SHARE), len(long_members))
        n_short = min(quota - n_long, len(short_members))
        n_long = min(quota - n_short, len(long_members))  # top up with long ones if needed
        chosen.append(long_members.sample(n_long, random_state=seed))
        chosen.append(short_members.sample(n_short, random_state=seed))
    sample = pd.concat(chosen)
    # Shuffle so the labeller does not see all results releases in a row.
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_or_create_sample() -> pd.DataFrame:
    """Return the gold sample, creating and saving it on first use so it never changes."""
    if SAMPLE_PATH.exists():
        return pd.read_parquet(SAMPLE_PATH)
    settings = load_settings()
    used_for_tuning = set(pilot_sample(settings.llm_pilot_size)["doc_id"])
    used_for_tuning |= set(reference_sample()["doc_id"])
    sample = select_gold_sample(
        scoreable_documents(),
        used_for_tuning,
        chars_for_token_cap(LONG_DOCUMENT_TOKENS),
        GOLD_SEED,
    )
    cap_chars = chars_for_token_cap(settings.llm_max_doc_tokens)
    shown = [truncate_text(text, cap_chars) for text in sample["text"]]
    sample["shown_text"] = [text for text, _ in shown]
    sample["shown_text_was_cut"] = [was_cut for _, was_cut in shown]
    columns = [
        "doc_id",
        "issuer",
        "title",
        "lang",
        "stratum",
        "is_long",
        "pdf_path",
        "shown_text",
        "shown_text_was_cut",
    ]
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    sample[columns].to_parquet(SAMPLE_PATH, index=False)
    return sample[columns]


def load_labels(path: Path = LABELS_PATH) -> pd.DataFrame:
    """All labels saved so far (one row per document; empty table if none)."""
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(path, dtype={"doc_id": "string", "topic": "string", "comment": "string"})


def save_label(
    doc_id: str, tone: int, topic: str, comment: str = "", path: Path = LABELS_PATH
) -> pd.DataFrame:
    """Store one label (replacing an earlier label for the same document) and return all labels."""
    if tone not in TONE_LABELS:
        raise ValueError(f"tone must be one of {sorted(TONE_LABELS)}, got {tone!r}")
    if topic not in TOPICS:
        raise ValueError(f"unknown topic {topic!r}")
    labels = load_labels(path)
    labels = labels[labels["doc_id"] != doc_id]
    new_row = pd.DataFrame(
        [
            {
                "doc_id": doc_id,
                "tone": tone,
                "topic": topic,
                "comment": comment.strip(),
                "labelled_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        ]
    )
    labels = pd.concat([labels, new_row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    return labels


def first_unlabelled(sample: pd.DataFrame, labels: pd.DataFrame) -> int:
    """Position of the first document without a label (the last document if all are done)."""
    done = set(labels["doc_id"])
    for position, doc_id in enumerate(sample["doc_id"]):
        if doc_id not in done:
            return position
    return len(sample) - 1
