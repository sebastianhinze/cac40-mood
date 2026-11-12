"""Gold-set agreement report: do the scorers agree with my hand labels?

    uv run python -m cac40_mood.scoring.gold_report

This is the only direct check that a score measures tone (DECISIONS #17). It compares my
labels with three scorers on the same 40 documents: the LLM, the local sentiment model and the
word-list (lexicon) scorer.

Why this is a separate module from gold.py: the labelling page imports gold.py, and that
module must never open a score table (labelling is blind). Everything that reads scores lives
here. The command also refuses to run while labels are missing, so no agreement number can be
seen in the middle of labelling.

How tone is compared (DECISIONS #51)
- By rank (Spearman): my scale is -2..+2, the scorers' is -1..1, so only the order is
  comparable.
- By direction: negative / neutral / positive. My label 0 is neutral; a scorer's tone within
  +/- 0.15 of zero is neutral (the same band as in the pilot comparisons).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cac40_mood.config import PROCESSED_DIR, PROJECT_ROOT
from cac40_mood.scoring import gold
from cac40_mood.scoring.agreement import cohens_kappa, direction

REPORT_PATH: Path = PROJECT_ROOT / "reports" / "02_gold_agreement.md"

# Scorer name -> column that holds its tone in the combined table.
TONE_COLUMNS: dict[str, str] = {
    "LLM": "llm_tone",
    "Local model": "lm_tone",
    "Lexicon": "lex_tone",
}


class LabelsIncompleteError(RuntimeError):
    """Raised when the report is asked for before every gold document has a label."""


def check_labels_complete(sample: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Stop unless every gold document is labelled.

    Seeing agreement numbers half-way through could change how the remaining documents are
    labelled, which would make the gold set worthless as an independent check.
    """
    missing = set(sample["doc_id"]) - set(labels["doc_id"])
    if missing:
        raise LabelsIncompleteError(
            f"{len(missing)} of {len(sample)} gold documents have no label yet; "
            "finish labelling before running the report."
        )


def combine(
    labels: pd.DataFrame, scores: pd.DataFrame, lexicon: pd.DataFrame, local_model: pd.DataFrame
) -> pd.DataFrame:
    """One row per labelled document: my label next to each scorer's tone and the LLM's topic."""
    human = labels[["doc_id", "tone", "topic"]].rename(
        columns={"tone": "human_tone", "topic": "human_topic"}
    )
    valid_scores = scores.loc[scores["problem"] == "", ["doc_id", "tone", "topic"]]
    llm = valid_scores.rename(columns={"tone": "llm_tone", "topic": "llm_topic"})
    table = human.merge(llm, on="doc_id", how="left")
    table = table.merge(local_model[["doc_id", "lm_tone"]], on="doc_id", how="left")
    return table.merge(lexicon[["doc_id", "lex_tone"]], on="doc_id", how="left")


def tone_agreement(human_tone: pd.Series, scorer_tone: pd.Series) -> dict[str, float]:
    """Rank correlation and direction agreement between my labels and one scorer.

    Documents the scorer has no tone for are left out and counted in `n`.
    """
    has_score = scorer_tone.notna()
    human, scorer = human_tone[has_score].astype(float), scorer_tone[has_score].astype(float)
    constant = human.nunique() < 2 or scorer.nunique() < 2
    human_direction = np.sign(human)
    scorer_direction = direction(scorer)
    return {
        "n": int(has_score.sum()),
        "spearman": np.nan if constant else round(float(human.corr(scorer, method="spearman")), 3),
        "same_direction": round(float((human_direction == scorer_direction).mean()), 3),
        "opposite_sign": round(float((human_direction * scorer_direction < 0).mean()), 3),
    }


def tone_agreement_table(table: pd.DataFrame) -> pd.DataFrame:
    """`tone_agreement` for each of the three scorers, one row per scorer."""
    rows = {
        name: tone_agreement(table["human_tone"], table[column])
        for name, column in TONE_COLUMNS.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def topic_agreement(table: pd.DataFrame) -> dict[str, float]:
    """Share of documents where the LLM picked my topic, and the chance-corrected kappa."""
    both = table.dropna(subset=["llm_topic"])
    return {
        "n": len(both),
        "same_topic": round(float((both["human_topic"] == both["llm_topic"]).mean()), 3),
        "topic_kappa": round(cohens_kappa(both["human_topic"], both["llm_topic"]), 3),
    }


def direction_crosstab(table: pd.DataFrame) -> pd.DataFrame:
    """Counts of my direction (rows) against the LLM's direction (columns)."""
    both = table.dropna(subset=["llm_tone"])
    names = {-1.0: "negative", 0.0: "neutral", 1.0: "positive"}
    human = np.sign(both["human_tone"].astype(float)).map(names)
    llm = direction(both["llm_tone"]).map(names)
    counts = pd.crosstab(human, llm)
    order = list(names.values())
    return counts.reindex(index=order, columns=order, fill_value=0)


def opposite_sign_documents(table: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Documents where the LLM and I point in opposite directions: the ones worth re-reading."""
    both = table.dropna(subset=["llm_tone"])
    opposite = both[np.sign(both["human_tone"].astype(float)) * direction(both["llm_tone"]) < 0]
    columns = ["doc_id", "issuer", "title"]
    listed = opposite.merge(sample[columns], on="doc_id", how="left")
    return listed[["issuer", "title", "human_tone", "llm_tone", "human_topic", "llm_topic"]]


def markdown_table(frame: pd.DataFrame, index_label: str = "") -> str:
    """Render a small table as Markdown (written out by hand to avoid an extra dependency)."""
    header = [index_label, *[str(column) for column in frame.columns]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, row in frame.iterrows():
        cells = [str(label), *[str(value) for value in row]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report_text(table: pd.DataFrame, sample: pd.DataFrame) -> str:
    """The full report as Markdown."""
    topics = topic_agreement(table)
    opposite = opposite_sign_documents(table, sample)
    parts = [
        "# Gold set: do the scorers agree with the hand labels?",
        "",
        f"{len(table)} documents from train and validation, labelled blind (tone -2..+2 and "
        "topic). None of them was used to tune the prompt. With this few documents every number "
        "below is rough: a rank correlation of 0.6 on 40 documents has a 95% interval of about "
        "0.35 to 0.77.",
        "",
        "## Tone",
        "",
        "`spearman`: rank correlation with my label. `same_direction`: share with the same "
        "negative / neutral / positive class. `opposite_sign`: share where one says good and "
        "the other bad.",
        "",
        markdown_table(tone_agreement_table(table), "scorer"),
        "",
        "## My direction (rows) against the LLM's (columns)",
        "",
        markdown_table(direction_crosstab(table), "mine \\ LLM"),
        "",
        "## Topic (LLM only)",
        "",
        f"Same topic on {topics['same_topic']:.0%} of {topics['n']} documents "
        f"(kappa {topics['topic_kappa']}).",
        "",
        "## Documents where the LLM and I point in opposite directions",
        "",
        markdown_table(opposite.set_index("issuer"), "issuer") if len(opposite) else "None.",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    sample = gold.load_or_create_sample()
    labels = gold.load_labels()
    check_labels_complete(sample, labels)
    table = combine(
        labels,
        pd.read_parquet(PROCESSED_DIR / "scores.parquet"),
        pd.read_parquet(PROCESSED_DIR / "baseline_lexicon.parquet"),
        pd.read_parquet(PROCESSED_DIR / "baseline_local_model.parquet"),
    )
    REPORT_PATH.write_text(report_text(table, sample), encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
