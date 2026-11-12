"""The README figures and the dashboard read numbers out of the committed reports.

These tests check the Markdown table reader on a small made-up file, and that the real reports
still contain the rows the figures need (a renamed column would otherwise break the figures
silently or pick the wrong table).
"""

from pathlib import Path

import pytest

from cac40_mood import presentation

SAMPLE = """# Title

Some text | with a pipe that is not a table.

| model | horizon | qlike |
|---|---|---|
| B1 | 5 | 0.16964 |
| M | 5 | 1e-05 |

| outcome | mean |
|---|---|
| CAR[0] (RQ1) | -0.0012 |
"""


def test_reads_tables_and_numbers(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text(SAMPLE, encoding="utf-8")
    tables = presentation.read_markdown_tables(path)
    assert len(tables) == 2
    assert list(tables[0].columns) == ["model", "horizon", "qlike"]
    assert tables[0].loc[1, "qlike"] == pytest.approx(1e-05)
    assert tables[1].loc[0, "outcome"] == "CAR[0] (RQ1)"


def test_find_table_by_header(tmp_path: Path):
    path = tmp_path / "report.md"
    path.write_text(SAMPLE, encoding="utf-8")
    assert presentation.find_table(path, ["outcome", "mean"]).shape == (1, 2)
    with pytest.raises(ValueError):
        presentation.find_table(path, ["outcome", "mean"], occurrence=1)


def test_real_reports_give_every_figure_its_rows():
    terciles = presentation.day0_terciles()
    assert len(terciles) == 6  # three thirds, two periods
    scorers = presentation.scorer_slopes()
    assert set(scorers["scorer"]) == {"LLM", "Local model", "Lexicon", "Loughran-McDonald"}
    volatility = presentation.volatility_validation_vs_test()
    assert len(volatility) == 2 * len(presentation.VOL_COMPARISONS)


def test_volatility_numbers_match_the_test_report():
    """The confirmatory test row must be the one in reports/06_test_results.md (M vs B1)."""
    volatility = presentation.volatility_validation_vs_test()
    row = volatility[
        (volatility["comparison"].str.startswith("RQ3"))
        & (volatility["period"] == presentation.TEST)
    ].iloc[0]
    assert row["dm_p_value"] == pytest.approx(0.55039)
    assert row["change_pct"] == pytest.approx(-0.00007 / 0.21419 * 100, rel=1e-3)
