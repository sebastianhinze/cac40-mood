"""Gold-set agreement report, tested on made-up labels and scores (never on the real ones)."""

import pandas as pd
import pytest

from cac40_mood.scoring import gold_report


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Five documents: the LLM follows the labels, the local model reverses them."""
    doc_ids = ["a", "b", "c", "d", "e"]
    sample = pd.DataFrame(
        {"doc_id": doc_ids, "issuer": ["Firm"] * 5, "title": [f"Title {d}" for d in doc_ids]}
    )
    labels = pd.DataFrame(
        {
            "doc_id": doc_ids,
            "tone": [-2, -1, 0, 1, 2],
            "topic": ["results", "results", "other", "guidance", "results"],
        }
    )
    scores = pd.DataFrame(
        {
            "doc_id": doc_ids,
            "problem": [""] * 5,
            "tone": [-0.8, -0.3, 0.05, 0.4, 0.9],
            "topic": ["results", "results", "other", "results", "results"],
        }
    )
    local_model = pd.DataFrame({"doc_id": doc_ids, "lm_tone": [0.9, 0.4, 0.0, -0.3, -0.8]})
    lexicon = pd.DataFrame({"doc_id": doc_ids, "lex_tone": [0.6, 0.6, 0.6, 0.6, 0.6]})
    return sample, labels, scores, lexicon, local_model


def test_report_refuses_to_run_while_labels_are_missing() -> None:
    sample, labels, *_ = make_inputs()
    with pytest.raises(gold_report.LabelsIncompleteError, match="2 of 5"):
        gold_report.check_labels_complete(sample, labels.iloc[:3])
    gold_report.check_labels_complete(sample, labels)  # complete: no error


def test_tone_agreement_for_a_perfect_a_reversed_and_a_constant_scorer() -> None:
    _sample, labels, scores, lexicon, local_model = make_inputs()
    table = gold_report.combine(labels, scores, lexicon, local_model)
    result = gold_report.tone_agreement_table(table)

    assert result.loc["LLM", "spearman"] == 1.0
    assert result.loc["LLM", "same_direction"] == 1.0
    assert result.loc["LLM", "opposite_sign"] == 0.0

    assert result.loc["Local model", "spearman"] == -1.0
    assert result.loc["Local model", "opposite_sign"] == 0.8  # 4 of 5; the neutral one is not

    # A scorer that always says the same has no rank correlation, and is right only on the
    # two documents I labelled positive.
    assert pd.isna(result.loc["Lexicon", "spearman"])
    assert result.loc["Lexicon", "same_direction"] == 0.4


def test_documents_without_a_valid_score_are_left_out_not_counted_as_neutral() -> None:
    _sample, labels, scores, lexicon, local_model = make_inputs()
    scores.loc[0, "problem"] = "stop_reason=max_tokens"
    table = gold_report.combine(labels, scores, lexicon, local_model)
    assert gold_report.tone_agreement_table(table).loc["LLM", "n"] == 4
    assert gold_report.topic_agreement(table)["n"] == 4


def test_topic_agreement_and_direction_crosstab() -> None:
    _sample, labels, scores, lexicon, local_model = make_inputs()
    table = gold_report.combine(labels, scores, lexicon, local_model)
    assert gold_report.topic_agreement(table)["same_topic"] == 0.8  # "guidance" was missed
    counts = gold_report.direction_crosstab(table)
    assert counts.loc["negative", "negative"] == 2
    assert counts.loc["neutral", "neutral"] == 1
    assert counts.loc["positive", "positive"] == 2
    assert counts.to_numpy().sum() == 5


def test_opposite_sign_documents_and_report_text() -> None:
    sample, labels, scores, lexicon, local_model = make_inputs()
    scores.loc[4, "tone"] = -0.5  # I said +2, the model says bad
    table = gold_report.combine(labels, scores, lexicon, local_model)
    opposite = gold_report.opposite_sign_documents(table, sample)
    assert list(opposite["title"]) == ["Title e"]
    text = gold_report.report_text(table, sample)
    assert "Title e" in text
    assert "| LLM |" in text
