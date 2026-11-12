"""CAC 40 Mood dashboard: saved results only, no API calls, nothing estimated.

    uv run streamlit run app/main.py

Four tabs:
- Company mood: the filtered mood of one firm over time, its 95% band, and its announcements;
- Announcements: every scored filing with its scores, the model's one-line rationale and a link
  to the official PDF (plus a short excerpt when the filing texts exist on this machine);
- Event study: abnormal returns around announcements, by tone third;
- Validation vs test: what the validation period suggested and what the frozen test showed.

Data: the committed public bundle in `public_data/` and the reports in `reports/`, so the app
runs the same on a fresh clone, locally and on Streamlit Community Cloud.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Streamlit Community Cloud installs the dependencies but not this project itself, so the
# package is imported from src/ directly when it is not installed.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from cac40_mood import dashboard_data as data  # noqa: E402
from cac40_mood import plotting  # noqa: E402
from cac40_mood.presentation import REPORTS_DIR, volatility_validation_vs_test  # noqa: E402
from cac40_mood.splits import VAL_END  # noqa: E402

FIGURES = REPORTS_DIR / "figures"
FEED_GAP = ("2020-03-05", "2020-06-15")
TEST_START = (pd.Timestamp(VAL_END) + pd.Timedelta(days=1)).date()
TOPICS_ALL = "all topics"
COMPANIES_ALL = "all companies"
REPOSITORY_URL = "https://github.com/sebastianhinze/cac40-mood"

st.set_page_config(page_title="CAC 40 Mood", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner="Reading saved announcements …")
def announcements() -> pd.DataFrame:
    return data.load_announcements()


@st.cache_data(show_spinner="Reading saved mood …")
def mood() -> pd.DataFrame:
    return data.load_mood()


@st.cache_data
def car_paths() -> pd.DataFrame:
    return data.car_paths_by_tone()


def chart_layout(figure: go.Figure, y_title: str, height: int = 430) -> go.Figure:
    """The project's chart look (see `cac40_mood.plotting`), applied to a Plotly figure."""
    figure.update_layout(
        height=height,
        margin={"l": 70, "r": 10, "t": 40, "b": 40},
        plot_bgcolor=plotting.SURFACE,
        paper_bgcolor=plotting.SURFACE,
        font={"color": plotting.INK_SECONDARY},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="closest",
    )
    figure.update_xaxes(showgrid=False, linecolor=plotting.AXIS)
    figure.update_yaxes(
        title=y_title, gridcolor=plotting.GRID, zerolinecolor=plotting.AXIS, linecolor=plotting.AXIS
    )
    return figure


def mood_figure(company: str, firm_mood: pd.DataFrame, documents: pd.DataFrame) -> go.Figure:
    """Mood line with its 95% band, the market mood, and the firm's announcements as dots."""
    centre = data.tone_centre()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=pd.concat([firm_mood["date"], firm_mood["date"][::-1]]),
            y=pd.concat([firm_mood["mood_high"], firm_mood["mood_low"][::-1]]),
            fill="toself",
            fillcolor="rgba(42,120,214,0.15)",
            line={"width": 0},
            hoverinfo="skip",
            name="95% band",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=firm_mood["date"],
            y=firm_mood["mood_total"],
            line={"color": plotting.SERIES[0], "width": 2},
            name=f"{company} mood (market + firm)",
            hovertemplate="%{x|%d %b %Y}<br>mood %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=firm_mood["date"],
            y=firm_mood["mood_market"],
            line={"color": plotting.SERIES[1], "width": 1.5, "dash": "dot"},
            name="market mood (all firms)",
            hovertemplate="%{x|%d %b %Y}<br>market %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=documents["event_session"],
            y=documents["tone"] - centre,
            mode="markers",
            marker={
                "color": plotting.INK_MUTED,
                "size": 7,
                "line": {"color": plotting.SURFACE, "width": 1},
            },
            name="announcement (tone minus its usual level)",
            customdata=documents[["title", "tone", "topic"]],
            hovertemplate=(
                "%{x|%d %b %Y}<br>%{customdata[0]}<br>"
                "tone %{customdata[1]:.2f} · %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.add_vrect(
        x0=TEST_START.isoformat(),
        x1="2025-12-31",
        fillcolor=plotting.GRID,
        opacity=0.35,
        line_width=0,
        annotation_text="test period: mood for display only",
        annotation_position="top left",
    )
    figure.add_vrect(
        x0=FEED_GAP[0],
        x1=FEED_GAP[1],
        fillcolor=plotting.GRID,
        opacity=0.6,
        line_width=0,
        annotation_text="feed gap",
        annotation_position="bottom left",
    )
    figure.update_xaxes(range=["2018-12-15", "2026-01-15"])
    return chart_layout(figure, "tone units, centred")


def show_document(document: pd.Series) -> None:
    """Scores, rationale and link of one announcement (excerpt only where texts exist locally)."""
    st.markdown(f"#### {document['company']}")
    st.markdown(f"**{html.escape(document['title'][:160])}**")
    st.caption(
        f"Published {document['published_paris']} Paris · counts from the session of "
        f"{document['event_session']:%d %b %Y}"
        + (" (released during trading hours)" if document["intraday"] else "")
        + f" · period: {document['period']} · language: {document['lang']}"
    )
    first, second = st.columns(2)
    first.metric("Tone (-1…1)", f"{document['tone']:.2f}")
    second.metric("Surprise (-1…1)", f"{document['surprise']:.2f}")
    first.metric("Uncertainty (0…1)", f"{document['uncertainty']:.2f}")
    second.metric("Model confidence", f"{document['confidence']:.2f}")
    st.markdown(f"**Topic:** `{document['topic']}`")
    st.markdown(f"**Model's rationale:** {html.escape(document['rationale'])}")
    excerpt_text = document.get("excerpt")
    if isinstance(excerpt_text, str):
        # Escaped HTML, not Markdown: the issuer's text may contain *, # or _.
        st.html(
            f"<div style='font-size:0.85rem;color:{plotting.INK_SECONDARY};"
            f"border-left:3px solid {plotting.GRID};padding-left:0.6rem'>"
            f"<b>Start of the document</b> (short excerpt, the issuer's text, local only): "
            f"{html.escape(excerpt_text)}</div>"
        )
    st.link_button("Open the official PDF (DILA)", document["pdf_url"])


# ---------------------------------------------------------------------------------------------
st.title("CAC 40 Mood")
st.markdown(
    "**Does the tone of official CAC 40 announcements predict returns and volatility?** "
    "Prices react to the LLM-scored tone on the announcement day and do not drift afterwards. "
    "For volatility, the gains seen on validation did not survive the frozen test."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- 36 CAC 40 firms, 2019-2025, official AMF filings\n"
        "- 6,996 filings scored by Claude Haiku (tone, surprise, uncertainty, topic)\n"
        "- Train 2019-2022 · validation 2023 - mid-2024 · test mid-2024 - 2025\n"
        "- Everything shown here is read from saved results; no API calls\n"
        "- Filing metadata: DILA info-financière (AMF), Licence Ouverte"
    )
    st.markdown(f"[Code and write-up on GitHub]({REPOSITORY_URL})")

mood_tab, viewer_tab, event_tab, test_tab = st.tabs(
    ["Company mood", "Announcements", "Event study", "Validation vs test"]
)

# --- Company mood ----------------------------------------------------------------------------
with mood_tab:
    firms = announcements()[["company", "firm_id"]].drop_duplicates().sort_values("company")
    by_name = dict(zip(firms["company"], firms["firm_id"], strict=True))
    names = list(by_name)
    company = st.selectbox("Company", names, index=names.index("LVMH") if "LVMH" in names else 0)
    firm = by_name[company]
    firm_mood = mood()[mood()["firm_id"] == firm]
    firm_documents = announcements()[announcements()["firm_id"] == firm]
    st.plotly_chart(mood_figure(company, firm_mood, firm_documents), width="stretch", theme=None)
    st.caption(
        "The mood is a hidden state filtered from the announcement tones with a particle filter "
        "(market part shared by all firms, plus a firm part). Between announcements it decays "
        "towards zero and its band widens. Parameters were fitted on 2019-2022 only, by Kalman "
        "maximum likelihood. From July 2024 (the test period) the same filter with the same "
        "frozen parameters is run forward for display only; no test result uses it. Spring 2020 "
        "is missing from the official feed and is treated as missing, not as 'no news'."
    )

# --- Announcement viewer ---------------------------------------------------------------------
with viewer_tab:
    table = announcements()
    left, middle, right = st.columns([2, 2, 3])
    company = left.selectbox(
        "Company", [COMPANIES_ALL, *sorted(table["company"].unique())], key="viewer_company"
    )
    topic = middle.selectbox("Topic", [TOPICS_ALL, *sorted(table["topic"].unique())])
    tone_range = right.slider("Tone", -1.0, 1.0, (-1.0, 1.0), step=0.05)

    shown = table[table["tone"].between(*tone_range)]
    if company != COMPANIES_ALL:
        shown = shown[shown["company"] == company]
    if topic != TOPICS_ALL:
        shown = shown[shown["topic"] == topic]

    st.caption(f"{len(shown):,} announcements. Click a row to see its scores and rationale.")
    list_column, detail_column = st.columns([3, 2], gap="large")
    listing = shown[["published_paris", "company", "title", "tone", "topic", "period"]]
    selection = list_column.dataframe(
        listing.assign(published_paris=listing["published_paris"].str[:10]),
        hide_index=True,
        width="stretch",
        height=520,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "published_paris": "date",
            "title": st.column_config.TextColumn(width="medium"),
            "tone": st.column_config.NumberColumn(format="%.2f", width="small"),
        },
    )
    with detail_column:
        rows = selection.selection.rows
        if rows:
            show_document(shown.iloc[rows[0]])
        else:
            st.info("Select a row on the left.")

# --- Event study -----------------------------------------------------------------------------
with event_tab:
    chart_column, image_column = st.columns(2, gap="large")
    with chart_column:
        st.markdown("**Cumulative abnormal return around the announcement, by tone third**")
        paths = car_paths() * 100
        figure = go.Figure()
        for colour, tercile in zip(plotting.SERIES, ["high", "middle", "low"], strict=False):
            figure.add_trace(
                go.Scatter(
                    x=paths.index,
                    y=paths[tercile],
                    mode="lines+markers",
                    line={"color": colour},
                    name=f"{tercile} tone",
                    hovertemplate="day %{x}<br>CAR %{y:.2f}%<extra></extra>",
                )
            )
        figure.add_vline(x=-0.5, line_dash="dot", line_color=plotting.AXIS)
        figure.update_xaxes(
            title="trading day relative to the announcement", dtick=1, zeroline=False
        )
        st.plotly_chart(
            chart_layout(figure, "cumulative abnormal return (%)", height=380),
            width="stretch",
            theme=None,
        )
        st.caption(
            "Train and validation events (4,369). Market model on days -250 to -30. High and "
            "low tone separate on day 0 and stay apart; regressions find no drift by tone over "
            "days +1 to +5 (reports/03_event_study.md)."
        )
    with image_column:
        st.markdown("**Day 0 in every period, including the frozen test**")
        st.image(str(FIGURES / "07_day0_by_tone.png"), width="stretch")
    st.image(str(FIGURES / "07_scorers_day0.png"), width=720)

# --- Validation vs test ----------------------------------------------------------------------
with test_tab:
    st.markdown(
        "**The pre-registered question:** do the scores, and the mood built from them, improve "
        "5-day volatility forecasts beyond a HAR model with announcement dummies? Validation "
        "said yes, by a little. The test period, run once on a frozen and git-tagged "
        "configuration, said no."
    )
    st.image(str(FIGURES / "07_vol_validation_vs_test.png"), width=820)
    numbers = volatility_validation_vs_test()
    st.dataframe(
        numbers.rename(
            columns={
                "change_pct": "QLIKE change (% of baseline)",
                "ci_low_pct": "95% low",
                "ci_high_pct": "95% high",
                "dm_p_value": "DM p-value",
            }
        )
        .round(3)
        .assign(**{"DM p-value": numbers["dm_p_value"].round(4)}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        "- **Why it vanished:** gains of 0.1-0.8% in loss are small. Several mood variants were "
        "compared on validation and the best one was picked, so its validation p-value was too "
        "optimistic (`docs/LIMITATIONS.md` #22).\n"
        "- **Holm-adjusted test p-values:** 1.0 for both pre-registered questions.\n"
        "- Read from `reports/04_*`, `reports/05_*` and `reports/06_test_results.md`."
    )
