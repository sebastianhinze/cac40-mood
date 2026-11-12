"""Gold set: stratified sample, label storage, and the guarantee that labelling is blind."""

from pathlib import Path

import pandas as pd
import pytest

from cac40_mood.scoring import gold

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_documents(per_code: int = 30) -> pd.DataFrame:
    """A synthetic pool: `per_code` documents for each type code, every third one long."""
    codes = ["001007", "001002", "001006", "005001", "004000", "008000", "009007", "014000"]
    rows = []
    for code in codes:
        for number in range(per_code):
            rows.append(
                {
                    "doc_id": f"{code}_{number}",
                    "type_code": code,
                    "text": "x" * (500 if number % 3 == 0 else 50),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("type_code", "stratum"),
    [
        ("001007", "results_release"),
        ("001008", "results_release"),
        ("001002", "operations"),
        ("001006", "other_press_release"),
        ("015000", "other_press_release"),
        ("005002", "quarterly_information"),
        ("300005", "periodic_report"),
        ("008000", "governance"),
        ("009005", "takeover"),
        ("014000", "other"),
    ],
)
def test_type_codes_map_to_strata(type_code, stratum) -> None:
    assert gold.stratum_of(type_code) == stratum


def test_sample_meets_every_quota_and_includes_long_documents() -> None:
    sample = gold.select_gold_sample(make_documents(), set(), long_chars=100, seed=1)
    assert len(sample) == sum(gold.STRATUM_QUOTAS.values()) == 40
    assert sample["stratum"].value_counts().to_dict() == gold.STRATUM_QUOTAS
    assert sample["doc_id"].is_unique
    long_per_stratum = sample.groupby("stratum")["is_long"].sum()
    assert (long_per_stratum >= 1).all()
    assert 12 <= sample["is_long"].sum() <= 20  # about a third


def test_sample_is_reproducible_and_skips_excluded_documents() -> None:
    documents = make_documents()
    first = gold.select_gold_sample(documents, set(), long_chars=100, seed=1)
    again = gold.select_gold_sample(documents, set(), long_chars=100, seed=1)
    assert first["doc_id"].tolist() == again["doc_id"].tolist()
    excluded = set(first["doc_id"])
    other = gold.select_gold_sample(documents, excluded, long_chars=100, seed=1)
    assert excluded.isdisjoint(other["doc_id"])


def test_a_stratum_without_long_documents_is_filled_with_short_ones() -> None:
    documents = make_documents()
    documents.loc[documents["type_code"] == "008000", "text"] = "short"
    sample = gold.select_gold_sample(documents, set(), long_chars=100, seed=1)
    governance = sample[sample["stratum"] == "governance"]
    assert len(governance) == gold.STRATUM_QUOTAS["governance"] and not governance["is_long"].any()


def test_labels_are_saved_replaced_and_read_back(tmp_path) -> None:
    path = tmp_path / "gold" / "labels.csv"
    assert gold.load_labels(path).empty
    gold.save_label("doc1", 1, "results", "first read", path)
    gold.save_label("doc2", -2, "guidance", "", path)
    labels = gold.save_label("doc1", 2, "results", "changed my mind", path)
    assert len(labels) == 2
    stored = gold.load_labels(path).set_index("doc_id")
    assert stored.loc["doc1", "tone"] == 2 and stored.loc["doc1", "comment"] == "changed my mind"
    assert stored.loc["doc2", "tone"] == -2


@pytest.mark.parametrize(("tone", "topic"), [(3, "results"), (0, "weather")])
def test_labels_outside_the_scale_or_topic_list_are_refused(tmp_path, tone, topic) -> None:
    with pytest.raises(ValueError, match=r"tone|topic"):
        gold.save_label("doc1", tone, topic, path=tmp_path / "labels.csv")


def test_first_unlabelled_document_is_found() -> None:
    sample = pd.DataFrame({"doc_id": ["a", "b", "c"]})
    assert gold.first_unlabelled(sample, pd.DataFrame({"doc_id": ["a"]})) == 1
    assert gold.first_unlabelled(sample, pd.DataFrame({"doc_id": ["a", "b", "c"]})) == 2


def test_labelling_code_cannot_reach_model_scores() -> None:
    """Blind by construction: neither the page nor the modules it loads import the scorer,
    the pilot results or the model cache."""
    sources = [
        REPO_ROOT / "app" / "label_gold.py",
        REPO_ROOT / "src" / "cac40_mood" / "scoring" / "gold.py",
        REPO_ROOT / "src" / "cac40_mood" / "scoring" / "samples.py",
    ]
    forbidden = ["scoring.scorer", "scoring.pilot", "scoring import pilot", "CACHE_DIR", "spend"]
    for source in sources:
        text = source.read_text()
        import_lines = [line for line in text.splitlines() if "import" in line]
        for word in forbidden:
            assert not any(word in line for line in import_lines), f"{source.name} imports {word}"
