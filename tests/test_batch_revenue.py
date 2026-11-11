"""The Batches runner and the firm-size (revenue) rules. No test here talks to the API."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from cac40_mood.scoring import batch, llm, revenue, scorer, spend

ANSWER = {
    "tone": 0.2,
    "surprise": 0.0,
    "uncertainty": 0.1,
    "topic": "results",
    "confidence": 0.8,
    "rationale": "Revenue up 3%, in line.",
}


def make_request(text: str) -> dict:
    return {
        "model": "claude-haiku-5-5",
        "max_tokens": 1000,
        "system": "Rate the text.",
        "messages": [{"role": "user", "content": text}],
    }


def make_items(count: int) -> list[batch.BatchItem]:
    requests = [make_request(f"document {i}") for i in range(count)]
    return [batch.BatchItem(f"doc{i}", r, llm.cache_key(r, "v2")) for i, r in enumerate(requests)]


def message(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=4000, output_tokens=125),
        model="claude-haiku-5-5",
    )


class FakeBatches:
    """Stands in for client.messages.batches: every batch 'ends' after `polls_until_done` polls."""

    def __init__(self, failing_ids: set[str] | None = None, polls_until_done: int = 1) -> None:
        self.created: list[list[dict]] = []
        self.failing_ids = failing_ids or set()
        self.polls_until_done = polls_until_done
        self.polls: dict[str, int] = {}

    def create(self, requests: list[dict]) -> SimpleNamespace:
        self.created.append(requests)
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.polls[batch_id] = self.polls.get(batch_id, 0) + 1
        done = self.polls[batch_id] >= self.polls_until_done
        return SimpleNamespace(processing_status="ended" if done else "in_progress")

    def results(self, batch_id: str) -> list[SimpleNamespace]:
        index = int(batch_id.split("_")[1]) - 1
        results = []
        for request in self.created[index]:
            if request["custom_id"] in self.failing_ids:
                result = SimpleNamespace(type="errored")
            else:
                result = SimpleNamespace(type="succeeded", message=message(json.dumps(ANSWER)))
            results.append(SimpleNamespace(custom_id=request["custom_id"], result=result))
        return results


def fake_client(batches: FakeBatches) -> SimpleNamespace:
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


def run(batches: FakeBatches, items: list[batch.BatchItem], tmp_path, cap: float = 30.0) -> dict:
    log = spend.SpendLog(tmp_path / "spend.jsonl", cap)
    return batch.run_batch(
        fake_client(batches), items, "v2", tmp_path / "cache", log, tmp_path / "state.jsonl", 0.0
    )


# ------------------------------------------------------------------ batches


def test_batch_answers_are_stored_logged_at_half_price_and_returned(tmp_path) -> None:
    batches = FakeBatches(polls_until_done=3)
    answers = run(batches, make_items(3), tmp_path)
    assert len(batches.created) == 1 and len(batches.created[0]) == 3
    assert batches.polls["batch_1"] == 3
    assert all(
        answer["batch"] and answer["stop_reason"] == "end_turn" for answer in answers.values()
    )
    total = spend.SpendLog(tmp_path / "spend.jsonl", 30).total_spent()
    assert total == pytest.approx(3 * 0.00023125)


def test_second_run_sends_nothing(tmp_path) -> None:
    items = make_items(3)
    run(FakeBatches(), items, tmp_path)
    second = FakeBatches()
    answers = run(second, items, tmp_path)
    assert second.created == [] and len(answers) == 3


def test_only_uncached_requests_are_sent(tmp_path) -> None:
    items = make_items(4)
    run(FakeBatches(), items[:2], tmp_path)
    second = FakeBatches()
    run(second, items, tmp_path)
    assert [request["custom_id"] for request in second.created[0]] == ["doc2", "doc3"]


def test_failed_requests_return_none_and_are_retried_next_time(tmp_path) -> None:
    items = make_items(3)
    answers = run(FakeBatches(failing_ids={"doc1"}), items, tmp_path)
    assert answers["doc1"] is None and answers["doc0"] is not None
    retry = FakeBatches()
    answers = run(retry, items, tmp_path)
    assert [request["custom_id"] for request in retry.created[0]] == ["doc1"]
    assert answers["doc1"] is not None


def test_interrupted_run_collects_its_batch_instead_of_resubmitting(tmp_path) -> None:
    """A batch was submitted, then the process died before results were stored."""
    items = make_items(2)
    batches = FakeBatches()
    batches.create([{"custom_id": item.custom_id, "params": item.request} for item in items])
    state = tmp_path / "state.jsonl"
    keys = {item.custom_id: item.key for item in items}
    state.write_text(json.dumps({"event": "submitted", "batch_id": "batch_1", "keys": keys}) + "\n")
    answers = run(batches, items, tmp_path)
    assert len(batches.created) == 1  # nothing new was submitted
    assert all(answer is not None for answer in answers.values())
    assert batch.read_open_batches(state) == {}


def test_batch_is_not_submitted_if_it_could_break_the_cap(tmp_path) -> None:
    batches = FakeBatches()
    with pytest.raises(spend.SpendCapExceededError):
        run(batches, make_items(3), tmp_path, cap=0.0001)
    assert batches.created == []


def test_chunks_respect_the_request_limit(monkeypatch) -> None:
    monkeypatch.setattr(batch, "MAX_REQUESTS_PER_CHUNK", 2)
    chunks = batch.split_into_chunks(make_items(5))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]


def test_chunks_respect_the_size_limit(monkeypatch) -> None:
    monkeypatch.setattr(batch, "MAX_BYTES_PER_CHUNK", 400)
    chunks = batch.split_into_chunks(make_items(4))
    assert len(chunks) > 1 and sum(len(chunk) for chunk in chunks) == 4


def test_documents_are_scored_through_a_batch(tmp_path) -> None:
    config = scorer.ScoringConfig("claude-haiku-5-5", "low", 9480, 1000)
    documents = [
        {
            "doc_id": f"d{i}",
            "issuer": "Veolia",
            "size": "about EUR 26 billion",
            "title": "T",
            "text": f"Texte {i}.",
        }
        for i in range(3)
    ]
    log = spend.SpendLog(tmp_path / "spend.jsonl", 30.0)
    scores = scorer.score_documents_in_batch(
        fake_client(FakeBatches()), config, documents, tmp_path / "cache", log, tmp_path / "s.jsonl"
    )
    assert scores["doc_id"].tolist() == ["d0", "d1", "d2"]
    assert (scores["problem"] == "").all() and (scores["tone"] == 0.2).all()


# ------------------------------------------------------------------ revenue


def facts_frame(rows: list[tuple]) -> pd.DataFrame:
    columns = [
        "doc_id",
        "firm_id",
        "published",
        "period",
        "fiscal_year_end",
        "revenue_millions",
        "prior_year_revenue_millions",
        "currency",
        "quote",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["published"] = pd.to_datetime(frame["published"]).dt.tz_localize("Europe/Paris")
    return frame


FACTS = facts_frame(
    [
        (
            "r18",
            "MC",
            "2019-01-29",
            "full_year",
            "2018-12",
            46826.0,
            42636.0,
            "EUR",
            "ventes de 46,8 Md",
        ),
        (
            "r19",
            "MC",
            "2020-01-28",
            "full_year",
            "2019-12",
            53670.0,
            46826.0,
            "EUR",
            "ventes de 53,7 Md",
        ),
        ("h19", "MC", "2019-07-24", "half_year", "", 25082.0, None, "EUR", "premier semestre"),
        ("x20", "MC", "2021-01-26", "full_year", "2020-12", None, 53670.0, "EUR", ""),
        (
            "s18",
            "STMPA",
            "2019-01-24",
            "full_year",
            "2018-12",
            9664.0,
            8347.0,
            "USD",
            "net revenues $9.66B",
        ),
    ]
)


def test_fiscal_year_table_prefers_the_reported_figure_and_backfills_from_comparatives() -> None:
    table = revenue.fiscal_year_table(FACTS).set_index(["firm_id", "fy"])
    assert table.loc[("MC", 2018), ["revenue", "basis", "doc_id"]].tolist() == [
        46826.0,
        "reported",
        "r18",
    ]
    assert table.loc[("MC", 2017), ["revenue", "basis", "doc_id"]].tolist() == [
        42636.0,
        "comparative",
        "r18",
    ]
    assert table.loc[("MC", 2019), "basis"] == "reported"
    assert ("MC", 2020) not in table.index  # the 2020 release stated no revenue for 2020
    assert table.loc[("STMPA", 2018), "currency"] == "USD"
    assert not table["needs_check"].any()


def test_half_year_figures_never_enter_the_table() -> None:
    table = revenue.fiscal_year_table(FACTS)
    assert 25082.0 not in table["revenue"].tolist()


def test_large_disagreement_between_two_sources_is_flagged() -> None:
    restated = facts_frame(
        [
            ("a", "EL", "2019-03-01", "full_year", "2018-12", 10800.0, None, "EUR", ""),
            ("b", "EL", "2020-03-01", "full_year", "2019-12", 17390.0, 16160.0, "EUR", ""),
        ]
    )
    table = revenue.fiscal_year_table(restated).set_index("fy")
    assert bool(table.loc[2018, "needs_check"]) and table.loc[2018, "revenue"] == 10800.0


@pytest.mark.parametrize(
    ("document_date", "fiscal_year"),
    [
        ("2019-01-15", 2017),
        ("2019-06-30", 2017),
        ("2019-07-01", 2018),
        ("2020-02-20", 2018),  # the 2019 results are published about now: not used yet
        ("2020-07-01", 2019),
        ("2024-06-28", 2022),
    ],
)
def test_one_july_rule(document_date, fiscal_year) -> None:
    assert revenue.fiscal_year_for(pd.Series([document_date])).iloc[0] == fiscal_year


def test_documents_get_the_size_allowed_by_the_rule() -> None:
    table = revenue.fiscal_year_table(FACTS)
    documents = pd.DataFrame(
        {
            "firm_id": ["MC", "MC", "MC", "MC", "STMPA"],
            "ts_paris": pd.to_datetime(
                ["2019-02-01", "2019-09-01", "2020-09-01", "2025-09-01", "2019-09-01"]
            ).tz_localize("Europe/Paris"),
        }
    )
    sized = revenue.attach_size(documents, table)
    assert sized["size_fy"].tolist() == [2017, 2018, 2019, 2019, 2018]
    assert sized["size"].tolist() == [
        "about EUR 43 billion",
        "about EUR 47 billion",
        "about EUR 54 billion",
        "about EUR 54 billion",  # latest year available, never a later one than allowed
        "about USD 9.7 billion",
    ]
    assert not sized["size_backfilled"].any()


def test_size_is_backfilled_only_when_no_earlier_year_exists() -> None:
    table = revenue.fiscal_year_table(FACTS[FACTS["doc_id"] == "r19"])  # years 2018 and 2019 only
    documents = pd.DataFrame(
        {"firm_id": ["MC"], "ts_paris": pd.to_datetime(["2019-02-01"]).tz_localize("Europe/Paris")}
    )
    sized = revenue.attach_size(documents, table)
    assert sized["size_fy"].tolist() == [2018] and sized["size_backfilled"].tolist() == [True]


def test_candidates_are_early_year_documents_plus_results_coded_ones() -> None:
    documents = pd.DataFrame(
        {
            "doc_id": ["jan_any", "sep_results", "sep_other", "apr_any"],
            "type_code": ["001006", "001007", "001006", "005001"],
            "ts_paris": pd.to_datetime(
                ["2021-01-20", "2021-09-01", "2021-09-01", "2021-04-30"]
            ).tz_localize("Europe/Paris"),
        }
    )
    assert revenue.candidate_documents(documents)["doc_id"].tolist() == [
        "jan_any",
        "sep_results",
        "apr_any",
    ]


def test_revenue_request_asks_for_the_schema_and_cuts_the_text() -> None:
    request = revenue.build_request("claude-haiku-5-5", "LVMH", "Résultats", "x" * 50_000)
    assert request["output_config"]["format"]["schema"] == revenue.REVENUE_JSON_SCHEMA
    assert len(request["messages"][0]["content"]) < revenue.MAX_CHARS + 200
    assert set(revenue.REVENUE_JSON_SCHEMA["required"]) == set(revenue.RevenueFact.model_fields)
