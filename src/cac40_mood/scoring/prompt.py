"""The scoring prompt and the rule for cutting long documents.

`PROMPT_VERSION` is part of the cache key. Any change to the instructions, the schema or the
way the document is presented needs a new version, so old and new scores are never mixed.
"""

from __future__ import annotations

import re

PROMPT_VERSION: str = "v1"

SYSTEM_PROMPT: str = """\
You read one official announcement published by a listed French company (regulated information \
filed with the market regulator AMF) and rate it for a research project on how markets react \
to such announcements. The text is in French or English. It may be cut off after the first \
pages; a note at the end says so.

Rate only what this text says. Do not use anything you may know about what happened to the \
company, its results or its share price afterwards.

Return these fields:

- tone, from -1 to 1: how good or bad the news is for the company's shareholders. -1 is \
clearly bad (profit warning, loss, targets cut, dividend cut, serious litigation). 0 is neutral, \
purely factual or administrative. 1 is clearly good (results well up, targets raised, a large \
contract won). Judge the substance, not the wording: companies describe almost everything in \
positive language, so routine or modest news should stay near 0.

- surprise, from -1 to 1: whether the document itself shows that the outcome differs from what \
was expected or previously announced. Positive means better than expected (targets exceeded, \
guidance raised). Negative means worse (targets missed, guidance lowered, a warning). Use 0 \
when the outcome is in line with earlier statements, when the text gives no basis to tell, or \
when the news is routine.

- uncertainty, from 0 to 1: how much the text hedges or stresses risks and an unclear outlook. \
0 is factual and certain. 1 is dominated by uncertainty (guidance suspended, effects that cannot \
be quantified, many conditions).

- topic: the single best category for the main subject.

- confidence, from 0 to 1: how sure you are of your tone and surprise ratings given the text \
you saw. Lower it when the text is short, cut off before the key facts, mostly tables, or mixes \
good and bad news.

- rationale: at most 30 words, in English, naming the facts that drove your ratings.
"""

TRUNCATION_NOTE: str = "[The document continues beyond this point but was cut off here.]"
OMISSION_NOTE: str = "[Part of the document is left out here.]"

# A short line that contains one of these words is treated as the heading of the outlook or
# guidance section ("Perspectives 2024", "Objectifs confirmés", "Outlook", "2024 guidance").
OUTLOOK_HEADING = re.compile(
    r"^[^\n]{0,50}\b(perspectives?|objectifs?|outlook|guidance)\b[^\n]{0,50}$",
    re.IGNORECASE | re.MULTILINE,
)


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


def outlook_excerpt(text: str, search_from: int, max_chars: int) -> str:
    """The outlook / guidance section that starts after `search_from`, or '' if none is found.

    Why: results releases put the outlook near the end, beyond the cut. That section says
    whether targets were raised, confirmed or lowered, which is what "surprise" is about.
    """
    match = OUTLOOK_HEADING.search(text, search_from)
    if match is None or max_chars <= 0:
        return ""
    return text[match.start() : match.start() + max_chars].rstrip()


def build_user_message(
    title: str, text: str, max_chars: int, outlook_chars: int = 0
) -> tuple[str, bool]:
    """The user message for one document, plus whether any text was left out.

    The start of the document is always kept (up to `max_chars`). With `outlook_chars` > 0 the
    outlook section found later in the document is appended, marked as a separate excerpt.

    No date and no firm identifier are added: the model should rate the text, and an explicit
    date would make it easier for the model to recall what happened afterwards.
    """
    body, was_cut = truncate_text(text, max_chars)
    parts = [f"Title: {title.strip()}", "", "<document>", body]
    if was_cut:
        excerpt = outlook_excerpt(text, len(body), outlook_chars)
        if excerpt:
            parts.extend([OMISSION_NOTE, excerpt])
        parts.append(TRUNCATION_NOTE)
    parts.append("</document>")
    return "\n".join(parts), was_cut
