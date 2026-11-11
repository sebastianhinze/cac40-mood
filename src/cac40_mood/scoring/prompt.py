"""The scoring prompt and the rule for cutting long documents.

`PROMPT_VERSION` labels the prompt in logs and result files. The cache does not rely on the
label: its key is a hash of the complete request (see scorer.py), so any edit to the text
below automatically counts as a different prompt.

Version history
- v1 (pilot only): first prompt.
- v2 (DECISIONS #36): rates strictly from the issuer's point of view, is told the issuer's
  approximate size and asked to judge materiality against it, defines surprise against what
  investors likely expected, adds the product_pipeline topic and sharper topic definitions,
  and asks for a plain-text rationale.
"""

from __future__ import annotations

PROMPT_VERSION: str = "v2"

SYSTEM_PROMPT: str = """\
You read one official announcement published by a listed French company (regulated information \
filed with the market regulator AMF) and rate it for a research project on how markets react \
to such announcements. The company that published the announcement is called the issuer; its \
name and approximate size are given before the text. The text is in French or English. It may \
be cut off after the first pages; a note at the end says so.

Rate only what this text says. Do not use anything you may know about what happened to the \
issuer, its results or its share price afterwards.

Perspective. Rate everything from the point of view of the issuer's own shareholders and \
nobody else's. When the issuer buys another company, a high price or premium is good news for \
the seller's shareholders, not for the issuer's: judge what the deal means for the issuer \
(strategic fit, price paid, how it is financed) and stay near 0 when the text does not let \
you tell.

Materiality. Judge how much the news matters for a company of the issuer's size. A contract, \
investment, disposal or fine worth well under 1% of annual revenue is minor however \
enthusiastic the wording, and belongs close to 0 (within about 0.2 either side). Keep strong \
ratings (beyond 0.6 either side) for news that clearly changes the picture for the whole \
group: results far above or below the previous year or the targets, guidance raised or cut, a \
transforming deal, a dividend cut.

Return these fields:

- tone, from -1 to 1: how good or bad the news is for the issuer's shareholders. -1 is clearly \
bad (profit warning, loss, targets cut, dividend cut, serious litigation). 0 is neutral, \
routine, immaterial or administrative. 1 is clearly good (results well up, targets raised). \
Judge the substance, not the wording: companies describe almost everything in positive \
language.

- surprise, from -1 to 1: how the news compares with what investors most likely expected just \
before this announcement. Base it on evidence in the filing: earlier guidance or targets it \
refers to, expectations or consensus figures it cites, and words such as raised, lowered, \
warning, ahead of or below expectations. Positive means better than expected, negative means \
worse. Use exactly 0 when the outcome matches earlier statements (for example guidance \
confirmed) or when the filing gives no basis to judge. Most filings are 0.

- uncertainty, from 0 to 1: how much the text hedges or stresses risks and an unclear outlook. \
0 is factual and certain. 1 is dominated by uncertainty (guidance suspended, effects that cannot \
be quantified, many conditions).

- topic: the single best category for the main subject:
  results: financial results for a year, half-year or quarter that include profit or margin \
figures.
  sales_update: revenue, orders, traffic or volumes for a period without profit figures \
(quarterly sales, monthly traffic).
  guidance: the main news is a new or changed target or outlook outside a results release \
(profit warning, upgrade, guidance withdrawn).
  m_and_a: the issuer buys, sells or merges businesses, or is part of a takeover bid.
  financing: bonds, loans, credit lines, capital increases, credit ratings.
  shareholder_returns: dividends and share buyback programmes.
  contract_win: a specific order, contract or commercial partnership with a named customer \
or partner.
  product_pipeline: products and research: clinical trial results, regulatory approval of a \
product, product launches.
  strategy_operations: the issuer's own plans and organisation: strategy plans, investor \
days, restructuring, sites opened or closed, investment projects. Not a customer contract.
  governance: board and management changes, shareholder meetings, remuneration.
  legal_regulatory: litigation, fines, investigations, sanctions and regulatory decisions \
about the issuer (not product approvals).
  esg: sustainability, climate and social commitments.
  other: none of the above, including purely administrative notices.

- confidence, from 0 to 1: how sure you are of your tone and surprise ratings given the text \
you saw. Lower it when the text is short, cut off before the key facts, mostly tables, or mixes \
good and bad news.

- rationale: one plain sentence of at most 30 words, in English, naming the facts that drove \
your ratings. Plain text only: no quotation marks, brackets, braces or tags.
"""

TRUNCATION_NOTE: str = "[The document continues beyond this point but was cut off here.]"


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the start of the text, cut at a line break when one is near, and say if we cut.

    The cap is given in characters, not tokens: counting tokens needs a network call per
    document. `max_chars` is derived from the token cap and a characters-per-token ratio
    measured on real documents (see `pilot.py`), so the cut is deterministic and offline.
    """
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    last_break = cut.rfind("\n")
    # Only step back to the line break if that loses less than 5% of the allowed text.
    if last_break >= int(max_chars * 0.95):
        cut = cut[:last_break]
    return cut.rstrip(), True


def describe_size(revenue_eur_bn: float) -> str:
    """'about EUR 26 billion' / 'about EUR 600 million': a rounded size the model can use."""
    if revenue_eur_bn >= 10:
        return f"about EUR {revenue_eur_bn:.0f} billion"
    if revenue_eur_bn >= 1:
        return f"about EUR {revenue_eur_bn:.1f} billion"
    return f"about EUR {revenue_eur_bn * 1000:.0f} million"


def build_user_message(
    issuer: str, revenue_eur_bn: float, title: str, text: str, max_chars: int
) -> tuple[str, bool]:
    """The user message for one document, plus whether the text was cut.

    The issuer's name is given so "the issuer's shareholders" is unambiguous even when a
    release is signed by two companies. No date is added: an explicit date would make it
    easier for the model to recall what happened afterwards.
    """
    body, was_cut = truncate_text(text, max_chars)
    parts = [
        f"Issuer: {issuer}",
        f"Issuer's approximate annual revenue: {describe_size(revenue_eur_bn)}",
        f"Title: {title.strip()}",
        "",
        "<document>",
        body,
    ]
    if was_cut:
        parts.append(TRUNCATION_NOTE)
    parts.append("</document>")
    return "\n".join(parts), was_cut
