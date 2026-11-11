"""Scoring core: schema, prompt building, cache, spend cap, and the agreement measures.

No test here talks to the API: a small fake client returns canned answers.
"""

import json
from types import SimpleNamespace
from typing import get_args

import pandas as pd
import pytest
from pydantic import ValidationError

from cac40_mood.scoring import agreement as ag
from cac40_mood.scoring import prompt, scorer, spend
from cac40_mood.scoring.schema import SCORE_JSON_SCHEMA, TOPICS, DocumentScore, Topic

GOOD_ANSWER = {
    "tone": 0.4,
    "surprise": 0.1,
    "uncertainty": 0.2,
    "topic": "results",
    "confidence": 0.8,
    "rationale": "Revenue up 8%, guidance confirmed.",
}


# ------------------------------------------------------------------- schema


def test_valid_answer_is_accepted() -> None:
    score = DocumentScore.model_validate(GOOD_ANSWER)
    assert score.tone == 0.4 and score.topic == "results"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("tone", 1.5),
        ("surprise", -1.2),
        ("uncertainty", -0.1),
        ("confidence", 1.01),
        ("topic", "weather"),
    ],
)
def test_out_of_range_or_unknown_values_are_rejected(field, bad_value) -> None:
    with pytest.raises(ValidationError):
        DocumentScore.model_validate({**GOOD_ANSWER, field: bad_value})


def test_api_schema_and_pydantic_model_describe_the_same_fields_and_topics() -> None:
    assert set(SCORE_JSON_SCHEMA["properties"]) == set(DocumentScore.model_fields)
    assert set(SCORE_JSON_SCHEMA["required"]) == set(DocumentScore.model_fields)
    assert SCORE_JSON_SCHEMA["properties"]["topic"]["enum"] == list(TOPICS)
    assert get_args(Topic) == TOPICS


# ------------------------------------------------------------------- prompt


def test_short_text_is_sent_whole() -> None:
    message, was_cut = prompt.build_user_message("Résultats 2023", "Chiffre d'affaires +8 %.", 100)
    assert not was_cut
    assert message.startswith("Title: Résultats 2023")
    assert "Chiffre d'affaires +8 %." in message
    assert prompt.TRUNCATION_NOTE not in message


def test_long_text_is_cut_and_the_model_is_told() -> None:
    text = "a" * 96 + "\n" + "b" * 50
    cut, was_cut = prompt.truncate_text(text, 100)
    assert was_cut and cut == "a" * 96  # steps back to the line break just before the limit
    message, _ = prompt.build_user_message("T", text, 100)
    assert prompt.TRUNCATION_NOTE in message and "bbb" not in message


def test_cut_does_not_step_back_to_a_far_away_line_break() -> None:
    text = "a" * 10 + "\n" + "b" * 200
    cut, _ = prompt.truncate_text(text, 100)
    assert len(cut) == 100


def test_outlook_section_after_the_cut_can_be_appended() -> None:
    text = (
        "Résultats annuels\n" + "x" * 300 + "\nPerspectives 2024\nLe Groupe vise +5 %.\n" + "y" * 80
    )
    message, was_cut = prompt.build_user_message("T", text, 100, outlook_chars=40)
    assert was_cut
    assert prompt.OMISSION_NOTE in message
    assert "Perspectives 2024\nLe Groupe vise +5 %." in message
    without, _ = prompt.build_user_message("T", text, 100)
    assert "Perspectives 2024" not in without


def test_outlook_heading_must_be_a_short_line() -> None:
    assert prompt.OUTLOOK_HEADING.search("Objectifs 2025 confirmés")
    assert prompt.OUTLOOK_HEADING.search("2024 Outlook")
    long_sentence = "x" * 60 + " perspectives " + "y" * 60
    assert prompt.OUTLOOK_HEADING.search(long_sentence) is None


def test_message_carries_no_date_or_firm_identifier() -> None:
    """Only the title and the text go to the model: a date would invite hindsight."""
    message, _ = prompt.build_user_message("Communiqué", "Texte.", 100)
    assert message == "Title: Communiqué\n\n<document>\nTexte.\n</document>"


# -------------------------------------------------------------------- spend


def test_cost_uses_per_million_prices_and_batch_is_half() -> None:
    assert spend.cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert spend.cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)
    assert spend.cost_usd("claude-haiku-5-5", 4000, 125) == pytest.approx(0.0004625)
    assert spend.cost_usd("claude-haiku-5-5", 4000, 125, batch=True) == pytest.approx(0.00023125)


def test_model_without_a_price_cannot_be_used() -> None:
    with pytest.raises(KeyError):
        spend.cost_usd("some-unknown-model", 1, 1)


def test_worst_case_is_never_below_a_realistic_cost() -> None:
    realistic = spend.cost_usd("claude-haiku-5-5", int(9480 / 2.37), 125)
    assert spend.worst_case_cost_usd("claude-haiku-5-5", 9480, 600) > realistic


def test_spend_log_totals_and_refuses_past_the_cap(tmp_path) -> None:
    log = spend.SpendLog(tmp_path / "llm" / "spend.jsonl", cap_usd=0.001)
    assert log.total_spent() == 0.0
    log.record("claude-haiku-5-5", "v1", "doc1", 4000, 125)
    assert log.total_spent() == pytest.approx(0.0004625)
    log.check_can_spend(0.0005)  # 0.0009625 <= 0.001: allowed
    with pytest.raises(spend.SpendCapExceededError):
        log.check_can_spend(0.0006)


# ------------------------------------------------------------------- scorer


class FakeClient:
    """Stands in for anthropic.Anthropic: returns one canned answer and counts the calls."""

    def __init__(self, answer_text: str, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        self._response = SimpleNamespace(
            stop_reason=stop_reason,
            content=[SimpleNamespace(type="text", text=answer_text)],
            usage=SimpleNamespace(input_tokens=4000, output_tokens=125),
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._response


CONFIG = scorer.ScoringConfig("claude-haiku-5-5", "low", max_doc_chars=9480, max_output_tokens=600)


def run(client: FakeClient, tmp_path, cap_usd: float = 30.0, config=CONFIG) -> dict:
    log = spend.SpendLog(tmp_path / "spend.jsonl", cap_usd)
    return scorer.score_document(
        client, config, "doc1", "Résultats", "Chiffre d'affaires +8 %.", tmp_path / "cache", log
    )


def test_first_call_is_paid_and_second_comes_from_the_cache(tmp_path) -> None:
    client = FakeClient(json.dumps(GOOD_ANSWER))
    first = run(client, tmp_path)
    second = run(client, tmp_path)
    assert len(client.calls) == 1
    assert (first["from_cache"], second["from_cache"]) == (False, True)
    assert second["tone"] == 0.4 and second["problem"] == ""
    assert spend.SpendLog(tmp_path / "spend.jsonl", 30).total_spent() == pytest.approx(0.0004625)


def test_a_different_setting_is_a_different_cache_entry(tmp_path) -> None:
    client = FakeClient(json.dumps(GOOD_ANSWER))
    run(client, tmp_path)
    repeat = scorer.ScoringConfig("claude-haiku-5-5", "low", 9480, 600, repeat=2)
    run(client, tmp_path, config=repeat)
    assert len(client.calls) == 2


def test_request_is_not_sent_when_it_could_break_the_cap(tmp_path) -> None:
    client = FakeClient(json.dumps(GOOD_ANSWER))
    with pytest.raises(spend.SpendCapExceededError):
        run(client, tmp_path, cap_usd=0.0001)
    assert client.calls == []


@pytest.mark.parametrize(
    ("answer", "stop_reason", "expected_problem"),
    [
        (json.dumps({**GOOD_ANSWER, "tone": 3.0}), "end_turn", "invalid answer"),
        ("not json", "end_turn", "invalid answer"),
        (json.dumps(GOOD_ANSWER), "max_tokens", "stop_reason=max_tokens"),
        (json.dumps(GOOD_ANSWER), "refusal", "stop_reason=refusal"),
    ],
)
def test_bad_answers_are_recorded_as_problems_not_scores(
    tmp_path, answer, stop_reason, expected_problem
) -> None:
    result = run(FakeClient(answer, stop_reason), tmp_path)
    assert result["problem"].startswith(expected_problem)
    assert "tone" not in result
    assert result["cost_usd"] > 0  # the call was still paid for and logged


def test_thinking_setting_maps_to_the_api_parameters() -> None:
    off = scorer.ScoringConfig("claude-haiku-5-5", "off", 9480, 600)
    arguments = scorer.request_arguments(off, "message")
    assert arguments["thinking"] == {"type": "disabled"}
    assert "effort" not in arguments["output_config"]
    arguments = scorer.request_arguments(CONFIG, "message")
    assert arguments["thinking"] == {"type": "adaptive"}
    assert arguments["output_config"]["effort"] == "low"
    assert arguments["output_config"]["format"]["schema"] == SCORE_JSON_SCHEMA
    assert "temperature" not in arguments  # this model family rejects sampling settings


def test_cache_key_changes_with_model_version_tag_and_text() -> None:
    base = scorer.cache_key("model-a", "v1", "text")
    assert base == scorer.cache_key("model-a", "v1", "text")
    assert (
        len(
            {
                base,
                *(
                    scorer.cache_key(*args)
                    for args in [
                        ("model-b", "v1", "text"),
                        ("model-a", "v2", "text"),
                        ("model-a", "v1", "text!"),
                    ]
                ),
            }
        )
        == 4
    )


def test_version_tag_spells_out_every_setting() -> None:
    assert CONFIG.version_tag == "v1|thinking=low|chars=9480|rep=1"
    with_outlook = scorer.ScoringConfig("m", "low", 9480, 600, outlook_chars=4740)
    assert with_outlook.version_tag == "v1|thinking=low|chars=9480|outlook=4740|rep=1"


# ---------------------------------------------------------------- agreement


def scores(
    tones: list[float], topics: list[str], problems: list[str] | None = None
) -> pd.DataFrame:
    count = len(tones)
    return pd.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(count)],
            "tone": tones,
            "surprise": [0.0] * count,
            "uncertainty": [0.2] * count,
            "confidence": [0.8] * count,
            "topic": topics,
            "problem": problems or [""] * count,
        }
    )


def test_pairing_drops_documents_with_a_problem_on_either_side() -> None:
    left = scores([0.5, 0.1, -0.4], ["results"] * 3, ["", "stop_reason=refusal", ""])
    right = scores([0.4, 0.2, -0.5], ["results"] * 3)
    assert ag.paired(left, right)["doc_id"].tolist() == ["d0", "d2"]


def test_direction_uses_a_neutral_band() -> None:
    values = pd.Series([0.5, 0.15, 0.1, 0.0, -0.1, -0.16])
    assert ag.direction(values).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_agreement_table_on_known_numbers() -> None:
    pairs = ag.paired(
        scores([0.5, 0.0, -0.5, 0.2], ["results"] * 4),
        scores([0.4, 0.1, -0.6, 0.2], ["results"] * 4),
    )
    table = ag.agreement_table(pairs)
    assert table.loc["tone", "mean_abs_diff"] == pytest.approx(0.075)
    assert table.loc["tone", "mean_b_minus_a"] == pytest.approx(-0.025)
    assert table.loc["tone", "pearson"] > 0.95
    assert pd.isna(table.loc["surprise", "pearson"])  # constant column: correlation undefined
    assert table.loc["surprise", "mean_abs_diff"] == 0.0


def test_cohens_kappa_matches_a_hand_calculation() -> None:
    """4 documents, 3 agreements. Shares: A picks x 50% / y 50%, B picks x 75% / y 25%.

    observed = 0.75; expected = 0.5*0.75 + 0.5*0.25 = 0.5; kappa = (0.75-0.5)/(1-0.5) = 0.5.
    """
    first = pd.Series(["x", "x", "y", "y"])
    second = pd.Series(["x", "x", "x", "y"])
    assert ag.cohens_kappa(first, second) == pytest.approx(0.5)
    assert ag.cohens_kappa(first, first) == pytest.approx(1.0)


def test_categorical_agreement_counts_directions_and_topics() -> None:
    pairs = ag.paired(
        scores([0.5, 0.05, -0.5, 0.3], ["results", "m_and_a", "results", "other"]),
        scores([0.4, 0.10, 0.4, 0.3], ["results", "m_and_a", "guidance", "other"]),
    )
    summary = ag.categorical_agreement(pairs)
    assert summary["same_tone_direction"] == 0.75
    assert summary["opposite_tone_sign"] == 0.25
    assert summary["same_topic"] == 0.75
