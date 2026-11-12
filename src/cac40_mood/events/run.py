"""Event study with the scores: RQ1 (day-0 reaction), RQ2 (drift on days +1..+5), RQ5.

    uv run python -m cac40_mood.events.run

Train and validation periods only: documents come from `analysis_documents` (which drops the
test period first) and the return panel is cut at the end of validation, so an event whose
window would need a later day gets no CAR.

Everything here is exploratory in the sense of the spec (the primary tests are the volatility
ones), so p-values are also shown with the Benjamini-Hochberg adjustment.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cac40_mood import plotting
from cac40_mood.analysis_documents import load_documents
from cac40_mood.config import FIGURES_DIR, PROCESSED_DIR, PROJECT_ROOT
from cac40_mood.data.build import FILINGS_PATH, PRICES_PATH
from cac40_mood.events import market_model, study
from cac40_mood.multiple_tests import benjamini_hochberg
from cac40_mood.scoring import gold
from cac40_mood.splits import ROBUSTNESS_SAMPLE_START, VAL_END

REPORT_PATH: Path = PROJECT_ROOT / "reports" / "03_event_study.md"
FIGURE_PATH: Path = FIGURES_DIR / "03_car_paths_by_tone.png"
EVENTS_PATH: Path = PROCESSED_DIR / "events" / "events_train_val.parquet"

OUTCOMES: dict[str, str] = {
    "car_0": "CAR[0] (RQ1)",
    "car_1_5": "CAR[+1,+5] (RQ2)",
    "car_placebo": "CAR[-5,-1] (placebo)",
}
SCORES: list[str] = ["tone", "surprise", "uncertainty"]
CONTROLS: list[str] = ["intraday_flag", "residual_sd", "log_revenue"]
SCORERS: dict[str, str] = {
    "LLM": "tone",
    "Local model": "lm_tone",
    "Lexicon": "lex_tone",
    "Loughran-McDonald": "lmd_tone",
}


def main_topic(documents: pd.DataFrame) -> pd.DataFrame:
    """Topic of each firm-day: that of the day's document the LLM was most confident about."""
    ordered = documents.sort_values("confidence", ascending=False)
    first = ordered.drop_duplicates(["firm_id", "event_session"])
    return first[["firm_id", "event_session", "topic"]]


def build_events(
    documents: pd.DataFrame,
    prices: pd.DataFrame,
    last_day: date = VAL_END,
    splits: tuple[str, ...] = ("train", "val"),
) -> pd.DataFrame:
    """One row per firm-day event with scores, controls and abnormal returns.

    Returns after `last_day` are never loaded (default: end of validation).
    """
    documents = documents.assign(log_revenue=np.log(documents["revenue"]))
    columns = [*SCORES, "lm_tone", "lex_tone", "lmd_tone", "log_revenue"]
    events = study.prepare_events(documents, columns, splits)
    events = events.merge(main_topic(documents), on=["firm_id", "event_session"], how="left")
    panel = market_model.build_return_panel(prices, "FCHI", pd.Timestamp(last_day))
    events = market_model.abnormal_returns(panel, events)
    events["intraday_flag"] = events["intraday"].astype(float)
    return events


def with_topic_dummies(events: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add one 0/1 column per topic except the most frequent one (the comparison group)."""
    dummies = pd.get_dummies(events["topic"], prefix="topic", dtype=float)
    most_frequent = "topic_" + events["topic"].value_counts().index[0]
    dummies = dummies.drop(columns=most_frequent)
    return pd.concat([events, dummies], axis=1), list(dummies.columns)


def score_regressions(events: pd.DataFrame) -> pd.DataFrame:
    """For each outcome: tone alone, and all three scores with controls and topic dummies."""
    table, topics = with_topic_dummies(events)
    rows = []
    for outcome, label in OUTCOMES.items():
        alone = study.clustered_regression(table, outcome, ["tone"])
        rows.append(alone.loc[["tone"]].assign(outcome=label, specification="tone only"))
        full = study.clustered_regression(table, outcome, [*SCORES, *CONTROLS, *topics])
        rows.append(full.loc[SCORES].assign(outcome=label, specification="scores + controls"))
    result = pd.concat(rows).reset_index(names="regressor")
    result["bh_p_value"] = benjamini_hochberg(result["p_value"].tolist())
    columns = ["outcome", "specification", "regressor", "coefficient", "ci_low", "ci_high", "t"]
    return result[[*columns, "p_value", "bh_p_value", "n"]]


def robustness_regressions(events: pd.DataFrame) -> pd.DataFrame:
    """CAR[0] and CAR[+1,+5] on tone alone in sub-samples."""
    late = events["event_session"] >= pd.Timestamp(ROBUSTNESS_SAMPLE_START)
    train = events["event_session"] <= pd.Timestamp("2022-12-31")
    samples = {
        "all events": events,
        "not during trading hours": events[~events["intraday"]],
        f"from {ROBUSTNESS_SAMPLE_START}": events[late],
        "train period": events[train],
        "validation period": events[~train],
    }
    rows = []
    for name, sample in samples.items():
        for outcome in ("car_0", "car_1_5"):
            fit = study.clustered_regression(sample, outcome, ["tone"]).loc[["tone"]]
            rows.append(fit.assign(sample=name, outcome=OUTCOMES[outcome]))
    result = pd.concat(rows)
    return result[["sample", "outcome", "coefficient", "ci_low", "ci_high", "t", "p_value", "n"]]


def scorer_comparison(events: pd.DataFrame) -> pd.DataFrame:
    """CAR[0] on each scorer's tone, in standard-deviation units so the slopes are comparable.

    Slope = change in the day-0 abnormal return for a one-standard-deviation higher tone.
    """
    complete = events.dropna(subset=["car_0", *SCORERS.values()])
    rows = []
    for name, column in SCORERS.items():
        standardised = (complete[column] - complete[column].mean()) / complete[column].std()
        fit = study.clustered_regression(complete.assign(z=standardised), "car_0", ["z"])
        r_squared = float(np.corrcoef(standardised, complete["car_0"])[0, 1] ** 2)
        rows.append(fit.loc[["z"]].assign(scorer=name, r_squared=r_squared))
    result = pd.concat(rows)
    return result[["scorer", "coefficient", "ci_low", "ci_high", "t", "p_value", "r_squared", "n"]]


def tercile_means(events: pd.DataFrame) -> pd.DataFrame:
    """Average CAR per tone tercile, with date-clustered standard errors."""
    table = events.assign(tone_tercile=study.terciles(events["tone"]))
    pieces = []
    for outcome, label in OUTCOMES.items():
        means = study.mean_car_by_group(table, "tone_tercile", outcome)
        pieces.append(means.assign(outcome=label))
    result = pd.concat(pieces)
    return result[["outcome", "tone_tercile", "events", "mean", "std_error", "t"]]


def gold_check(
    events: pd.DataFrame, labels: pd.DataFrame, filings: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    """On the gold documents: does my label or the LLM's tone go with the day-0 return?

    Each gold document is matched to its firm-day event. Both tones are put in standard-
    deviation units. About 40 events: indicative only.
    """
    human = labels[["doc_id", "tone"]].rename(columns={"tone": "human_tone"})
    table = human.merge(filings[["doc_id", "firm_id", "event_session"]], on="doc_id")
    table = table.merge(scores[["doc_id", "tone"]], on="doc_id")
    returns = events[["firm_id", "event_session", "car_0", "car_1_5", "n_documents"]]
    table = table.merge(returns, on=["firm_id", "event_session"]).dropna(subset=["car_0"])
    rows = []
    for name, column in {"My label": "human_tone", "LLM tone": "tone"}.items():
        values = table[column].astype(float)
        standardised = (values - values.mean()) / values.std()
        fit = study.clustered_regression(table.assign(z=standardised), "car_0", ["z"]).loc[["z"]]
        rank = float(values.corr(table["car_0"], method="spearman"))
        rows.append(fit.assign(scorer=name, spearman_with_car_0=rank))
    result = pd.concat(rows)
    columns = ["scorer", "coefficient", "ci_low", "ci_high", "t", "p_value"]
    return result[[*columns, "spearman_with_car_0", "n"]]


def plot_paths(events: pd.DataFrame) -> Path:
    """Average cumulative abnormal return from day -5 to +5, one line per tone tercile."""
    plotting.apply_style()
    table = events.assign(tone_tercile=study.terciles(events["tone"]))
    paths = study.cumulative_paths(table, "tone_tercile") * 100.0
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    names = {"low": "Lowest tone third", "middle": "Middle third", "high": "Highest tone third"}
    for colour, group in zip(plotting.SERIES, ["high", "middle", "low"], strict=False):
        axes.plot(paths.index, paths[group], color=colour, marker="o", markersize=4)
        axes.annotate(
            names[group],
            xy=(paths.index[-1], paths[group].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=plotting.INK_SECONDARY,
            fontsize=9,
            va="center",
        )
    axes.axvline(0, color=plotting.AXIS, linewidth=0.8)
    axes.axhline(0, color=plotting.AXIS, linewidth=0.8)
    axes.set_xlabel("Trading days relative to the announcement")
    axes.set_ylabel("Cumulative abnormal return, %")
    axes.set_xticks(list(paths.index))
    plotting.titled(
        axes,
        "Prices move on the announcement day, in the direction of the tone",
        "Average cumulative abnormal return by tone third; train and validation events",
    )
    figure.subplots_adjust(right=0.8)
    path = plotting.save(figure, FIGURE_PATH)
    plt.close(figure)
    return path


def markdown(table: pd.DataFrame, digits: int = 4) -> str:
    shown = table.round(digits).astype(object).where(table.notna(), "")
    lines = ["| " + " | ".join(str(c) for c in shown.columns) + " |", "|" + "---|" * shown.shape[1]]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in shown.to_numpy()]
    return "\n".join(lines)


def write_report(events: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    with_car = int(events["car_0"].notna().sum())
    lines = [
        "# Event study with the scores (train and validation)",
        "",
        f"{len(events):,} firm-day events of 36 firms, {with_car:,} with a complete window.",
        "Abnormal return = return minus the market-model prediction (estimated on days -250 to",
        "-30). Returns are in log points: 0.01 is about 1%. Standard errors are clustered by",
        "event date. All of this is exploratory; `bh_p_value` is the Benjamini-Hochberg",
        "adjustment across the rows of the first table.",
        "",
        "## Scores and abnormal returns",
        "",
        "Coefficient = change in the CAR for a tone (or surprise, uncertainty) higher by 1.",
        "Controls: released during trading hours, volatility before the event, log revenue,",
        "topic dummies.",
        "",
        markdown(tables["scores"]),
        "",
        "## Average CAR by tone third",
        "",
        markdown(tables["terciles"], 5),
        "",
        "![CAR paths](figures/03_car_paths_by_tone.png)",
        "",
        "## Tone alone, in sub-samples",
        "",
        markdown(tables["robustness"]),
        "",
        "## Which scorer's tone goes with the day-0 return? (RQ5)",
        "",
        "Slope = change in CAR[0] for a tone one standard deviation higher. Only events that",
        "every scorer has a tone for (Loughran-McDonald needs an official English version).",
        "",
        markdown(tables["scorers"], 5),
        "",
        "## Gold documents: my labels against the LLM (indicative only)",
        "",
        "The 40 hand-labelled documents, matched to their firm-day. With about 40 events",
        "neither row can be told apart from zero or from the other; read it as a hint.",
        "",
        markdown(tables["gold"], 5),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    documents, _ = load_documents()
    events = build_events(documents, pd.read_parquet(PRICES_PATH))
    filings = pd.read_parquet(FILINGS_PATH)
    scores = pd.read_parquet(PROCESSED_DIR / "scores.parquet")
    tables = {
        "scores": score_regressions(events),
        "terciles": tercile_means(events),
        "robustness": robustness_regressions(events),
        "scorers": scorer_comparison(events),
        "gold": gold_check(events, gold.load_labels(), filings, scores),
    }
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS_PATH, index=False)
    plot_paths(events)
    write_report(events, tables)
    for name, table in tables.items():
        print(f"\n== {name}\n{table.round(5).to_string(index=False)}")


if __name__ == "__main__":
    main()
