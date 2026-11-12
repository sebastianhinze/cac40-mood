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
from cac40_mood.scoring import llm, prompt, scorer, spend
from cac40_mood.scoring.schema import (
    SCORE_JSON_SCHEMA,
    TOPICS,
    DocumentScore,
    Topic,
    clean_rationale,
)

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


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("Revenue up 8%, guidance confirmed.}", "Revenue up 8%, guidance confirmed."),
        ("The gain is tempered.</reasoning_effort>", "The gain is tempered."),
        ('Margin fell 110bp."} ', "Margin fell 110bp."),
        ("Nothing to remove (see table).", "Nothing to remove (see table)."),
    ],
)
def test_stray_characters_are_removed_from_the_end_of_the_rationale(raw, cleaned) -> None:
    assert clean_rationale(raw) == cleaned
    assert DocumentScore.model_validate({**GOOD_ANSWER, "rationale": raw}).rationale == cleaned


def test_api_schema_and_pydantic_model_describe_the_same_fields_and_topics() -> None:
    assert set(SCORE_JSON_SCHEMA["properties"]) == set(DocumentScore.model_fields)
    assert set(SCORE_JSON_SCHEMA["required"]) == set(DocumentScore.model_fields)
    assert SCORE_JSON_SCHEMA["properties"]["topic"]["enum"] == list(TOPICS)
    assert get_args(Topic) == TOPICS


# ------------------------------------------------------------------- prompt


def message(text: str, max_chars: int = 100) -> tuple[str, bool]:
    return prompt.build_user_message(
        "Veolia", "about EUR 26 billion", "Résultats 2023", text, max_chars
    )


def test_short_text_is_sent_whole_with_issuer_and_size() -> None:
    built, was_cut = message("Chiffre d'affaires +8 %.")
    assert not was_cut
    assert built == (
        "Issuer: Veolia\n"
        "Issuer's approximate annual revenue: about EUR 26 billion\n"
        "Title: Résultats 2023\n"
        "\n<document>\nChiffre d'affaires +8 %.\n</document>"
    )


def test_long_text_is_cut_and_the_model_is_told() -> None:
    text = "a" * 96 + "\n" + "b" * 50
    cut, was_cut = prompt.truncate_text(text, 100)
    assert was_cut and cut == "a" * 96  # steps back to the line break just before the limit
    built, _ = message(text)
    assert prompt.TRUNCATION_NOTE in built and "bbb" not in built


def test_cut_does_not_step_back_to_a_far_away_line_break() -> None:
    cut, _ = prompt.truncate_text("a" * 10 + "\n" + "b" * 200, 100)
    assert len(cut) == 100


@pytest.mark.parametrize(
    ("revenue_bn", "expected"),
    [
        (177, "about EUR 177 billion"),
        (4.8, "about EUR 4.8 billion"),
        (0.6, "about EUR 600 million"),
    ],
)
def test_size_is_described_in_round_numbers(revenue_bn, expected) -> None:
    assert prompt.describe_size(revenue_bn) == expected
    assert prompt.describe_size(9.66, "USD") == "about USD 9.7 billion"
    assert prompt.describe_size(184.0, "other") == "about 184 billion"


def test_message_carries_no_date() -> None:
    """Issuer, size, title and text go to the model; a date would invite hindsight."""
    built, _ = message("Texte.")
    assert "20" not in built.replace("Résultats 2023", "").replace("EUR 26", "")


def test_prompt_names_every_topic_and_the_issuer_perspective() -> None:
    for topic in TOPICS:
        assert f"  {topic}:" in prompt.SYSTEM_PROMPT
    assert "issuer's own shareholders" in prompt.SYSTEM_PROMPT
    assert "no basis to judge" in prompt.SYSTEM_PROMPT


# -------------------------------------------------------------------- spend


def test_cost_uses_per_million_prices_and_batch_is_half() -> None:
    assert spend.cost_usd("claude-haiku-5-5", 1_000_000, 0) == pytest.approx(0.10)
    assert spend.cost_usd("claude-haiku-5-5", 0, 1_000_000) == pytest.approx(0.50)
    assert spend.cost_usd("claude-haiku-5-5", 4000, 125) == pytest.approx(0.0004625)
    assert spend.cost_usd("claude-haiku-5-5", 4000, 125, batch=True) == pytest.approx(0.00023125)


def test_model_without_a_price_cannot_be_used() -> None:
    with pytest.raises(KeyError):
        spend.cost_usd("some-unknown-model", 1, 1)


def test_dated_model_id_is_priced_like_its_base_id() -> None:
    assert spend.price_per_million("claude-haiku-5-5-20260301") == (0.10, 0.50)
    with pytest.raises(KeyError):
        spend.price_per_million("claude-haiku-5-50")  # a different model, not a dated id


def test_batch_calls_are_logged_at_half_price(tmp_path) -> None:
    log = spend.SpendLog(tmp_path / "spend.jsonl", cap_usd=1.0)
    assert log.record("claude-haiku-5-5", "v2", "doc1", 4000, 125, batch=True) == pytest.approx(
        0.00023125
    )


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
            model="claude-haiku-5-5",
        )
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._response


CONFIG = scorer.ScoringConfig("claude-haiku-5-5", "low", max_doc_chars=9480, max_output_tokens=600)
DOCUMENT = {
    "doc_id": "doc1",
    "issuer": "Veolia",
    "size": "about EUR 26 billion",
    "title": "Résultats",
    "text": "Chiffre d'affaires +8 %.",
}


def run(client: FakeClient, tmp_path, cap_usd: float = 30.0, config=CONFIG) -> dict:
    log = spend.SpendLog(tmp_path / "spend.jsonl", cap_usd)
    return scorer.score_document(client, config, DOCUMENT, tmp_path / "cache", log)


def test_first_call_is_paid_and_second_comes_from_the_cache(tmp_path) -> None:
    client = FakeClient(json.dumps(GOOD_ANSWER))
    first = run(client, tmp_path)
    second = run(client, tmp_path)
    assert len(client.calls) == 1
    assert (first["from_cache"], second["from_cache"]) == (False, True)
    assert second["tone"] == 0.4 and second["problem"] == ""
    assert spend.SpendLog(tmp_path / "spend.jsonl", 30).total_spent() == pytest.approx(0.0004625)


def test_a_repeat_run_is_a_different_cache_entry(tmp_path) -> None:
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
    # the call was still paid for and logged
    assert spend.SpendLog(tmp_path / "spend.jsonl", 30).total_spent() > 0


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


def base_request() -> dict:
    return scorer.request_arguments(CONFIG, "Issuer: Veolia\nTitle: T\n<document>x</document>")


def test_cache_key_is_stable_for_the_same_request() -> None:
    assert llm.cache_key(base_request(), "v2", 1) == llm.cache_key(base_request(), "v2", 1)


def changed(path: list, value: object) -> dict:
    """A copy of the base request with one nested value replaced."""
    request = json.loads(json.dumps(base_request()))
    target = request
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return request


@pytest.mark.parametrize(
    ("what", "request_or_args"),
    [
        ("model", (changed(["model"], "claude-sonnet-5-5"), "v2", 1)),
        ("instructions", (changed(["system"], "A different prompt."), "v2", 1)),
        ("document text", (changed(["messages", 0, "content"], "other text"), "v2", 1)),
        ("thinking mode", (changed(["thinking"], {"type": "disabled"}), "v2", 1)),
        ("effort", (changed(["output_config", "effort"], "medium"), "v2", 1)),
        ("output limit", (changed(["max_tokens"], 4000), "v2", 1)),
        (
            "schema",
            (changed(["output_config", "format", "schema", "required"], ["tone"]), "v2", 1),
        ),
        ("prompt version label", (base_request(), "v3", 1)),
        ("repeat number", (base_request(), "v2", 2)),
    ],
)
def test_any_change_to_the_request_gives_a_new_cache_key(what, request_or_args) -> None:
    """An answer produced under one prompt, model or setting must never be reused for another."""
    assert llm.cache_key(*request_or_args) != llm.cache_key(base_request(), "v2", 1), what


def test_version_tag_spells_out_the_main_settings() -> None:
    assert CONFIG.version_tag == f"{prompt.PROMPT_VERSION}|thinking=low|chars=9480|rep=1"


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
