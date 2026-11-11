"""Gold-set labelling page.

    uv run streamlit run app/label_gold.py

Shows one document at a time (company, title, text) and stores a tone from -2 to +2 and a
topic. It never loads model scores: everything it shows comes from `cac40_mood.scoring.gold`,
which reads filings and texts only. Labels go to `data/processed/gold/labels.csv`.
"""

from __future__ import annotations

import streamlit as st

from cac40_mood.data.schema import PDF_BASE_URL
from cac40_mood.scoring import gold
from cac40_mood.scoring.schema import TOPICS

TOPIC_HELP: dict[str, str] = {
    "results": "financial results with profit or margin figures",
    "sales_update": "revenue, orders, traffic or volumes without profit figures",
    "guidance": "new or changed targets or outlook outside a results release",
    "m_and_a": "the issuer buys, sells or merges businesses; takeover bids",
    "financing": "bonds, loans, credit lines, capital increases, ratings",
    "shareholder_returns": "dividends, share buyback programmes",
    "contract_win": "a specific order, contract or commercial partnership",
    "product_pipeline": "trial results, product approvals, product launches",
    "strategy_operations": "strategy plans, investor days, restructuring, sites, investments",
    "governance": "board and management changes, shareholder meetings, pay",
    "legal_regulatory": "litigation, fines, investigations, regulatory decisions",
    "esg": "sustainability, climate and social commitments",
    "other": "none of the above, including administrative notices",
}

st.set_page_config(page_title="Gold set labelling", layout="wide")

sample = gold.load_or_create_sample()
labels = gold.load_labels()
labelled_ids = set(labels["doc_id"])

if "position" not in st.session_state:
    st.session_state.position = gold.first_unlabelled(sample, labels)


def go_to(position: int) -> None:
    st.session_state.position = max(0, min(position, len(sample) - 1))


with st.sidebar:
    st.header("Gold set")
    st.progress(
        len(labelled_ids) / len(sample), text=f"{len(labelled_ids)} of {len(sample)} labelled"
    )
    st.markdown(
        "**Tone:** how good or bad is this news for the shareholders of the company that "
        "published it, given the company's size? Judge the substance, not the wording.\n\n"
        "**Topic:** the single best category for the main subject.\n\n"
        "No model scores are shown anywhere on this page."
    )
    choice = st.selectbox(
        "Jump to document",
        options=list(range(len(sample))),
        index=st.session_state.position,
        format_func=lambda i: (
            f"{i + 1:02d} {'done' if sample.loc[i, 'doc_id'] in labelled_ids else 'open'}"
            f" · {sample.loc[i, 'issuer']}"
        ),
    )
    if choice != st.session_state.position:
        go_to(choice)
        st.rerun()
    with st.expander("Topic definitions"):
        for topic in TOPICS:
            st.markdown(f"**{topic}**: {TOPIC_HELP[topic]}")

document = sample.loc[st.session_state.position]
previous = labels[labels["doc_id"] == document["doc_id"]]
previous_tone = int(previous["tone"].iloc[0]) if len(previous) else 0
previous_topic = previous["topic"].iloc[0] if len(previous) else "other"
previous_comment = (
    "" if not len(previous) or previous["comment"].isna().iloc[0] else previous["comment"].iloc[0]
)

text_column, form_column = st.columns([2, 1], gap="medium")

with text_column:
    st.caption(f"Document {st.session_state.position + 1} of {len(sample)}")
    st.subheader(document["issuer"])
    st.markdown(f"**{document['title']}**")
    st.markdown(f"[Open the original PDF]({PDF_BASE_URL}{document['pdf_path']})")
    with st.container(height=640, border=True):
        st.text(document["shown_text"])
        if document["shown_text_was_cut"]:
            st.info("The text is cut here, at the same point where the model stops reading.")

with form_column:
    with st.form(key=f"label_{document['doc_id']}"):
        tone = st.radio(
            "Tone",
            options=list(gold.TONE_LABELS),
            index=list(gold.TONE_LABELS).index(previous_tone),
            format_func=gold.TONE_LABELS.get,
        )
        topic = st.selectbox(
            "Topic",
            options=list(TOPICS),
            index=list(TOPICS).index(previous_topic),
            help="Definitions are in the sidebar under Topic definitions.",
        )
        comment = st.text_input("Comment (optional)", value=previous_comment)
        saved = st.form_submit_button("Save and next", type="primary", use_container_width=True)
    if saved:
        gold.save_label(document["doc_id"], int(tone), topic, comment)
        go_to(st.session_state.position + 1)
        st.rerun()
    left, right = st.columns(2)
    if left.button("Previous", use_container_width=True):
        go_to(st.session_state.position - 1)
        st.rerun()
    if right.button("Skip", use_container_width=True):
        go_to(st.session_state.position + 1)
        st.rerun()
    if len(previous):
        st.success("This document already has a label; saving replaces it.")
