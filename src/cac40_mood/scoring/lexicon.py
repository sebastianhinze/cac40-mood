"""Baseline 1: a word-list ("lexicon") tone score for French and English filings.

How it works: count words that signal good news, bad news or uncertainty, then

    tone        = (positive - negative) / (positive + negative)      (0 if neither occurs)
    uncertainty = uncertainty words per 100 words

Why a hand-written list (DECISIONS #49): no openly licensed French finance lexicon was found;
the standard English list (Loughran-McDonald) has no published French version. The lists below
are short on purpose, can be read in full, and follow the same idea as Loughran-McDonald:
finance-specific words, not general emotion words.

Matching rules, all deliberately simple:
- a list entry is a word beginning ("stem"): `progress` matches progresse, progression, ...;
- entries ending in `=` must match the whole word (`gain=` matches "gain", not "gainsay");
- a match is ignored when one of the three words before it is a negation ("ne", "pas",
  "sans", "not", ...): "le chiffre d'affaires ne progresse pas" does not count as positive.
This is a baseline: it reads no context, so "réduit sa dette" and "réduit son dividende" look
the same to it. That weakness is the point of comparing it with the LLM.
"""

from __future__ import annotations

import re

WORD = re.compile(r"[a-zàâäçéèêëîïôöùûüÿœ]+")
NEGATION_WINDOW: int = 3

POSITIVE: dict[str, tuple[str, ...]] = {
    "fr": (
        "hausse",
        "croissan",
        "progress",
        "amélior",
        "augment",
        "bénéfic",
        "rentab",
        "record",
        "solide",
        "robust",
        "dynami",
        "succès",
        "réussi",
        "favorabl",
        "excellen",
        "performan",
        "surperform",
        "dépass",
        "relèv",
        "rehauss",
        "accélér",
        "rebond",
        "redress",
        "reprise",
        "expansion",
        "renforc",
        "résilien",
        "profitab",
        "remport",
        "gagn",
        "leader",
        "innov",
        "opportun",
        "supérieur",
        "positi",
        "encourag",
        "satisf",
        "soutenu",
        "vigoureu",
        "gain=",
        "gains=",
        "fort=",
        "forte=",
        "fortes=",
        "forts=",
        "fortement",
    ),
    "en": (
        "increas",
        "growth",
        "grow",
        "grew",
        "improv",
        "strong",
        "record",
        "profit",
        "gain",
        "success",
        "favorabl",
        "favourabl",
        "robust",
        "solid",
        "exceed",
        "outperform",
        "raised=",
        "raises=",
        "raising=",
        "upgrad",
        "accelerat",
        "rebound",
        "recover",
        "resilien",
        "won=",
        "wins=",
        "award",
        "achiev",
        "expan",
        "strengthen",
        "positiv",
        "leader",
        "innovat",
        "opportunit",
        "benefit",
        "higher",
        "above=",
    ),
}

NEGATIVE: dict[str, tuple[str, ...]] = {
    "fr": (
        "baisse",
        "recul",
        "diminu",
        "perte",
        "défici",
        "dépréci",
        "détérior",
        "dégrad",
        "chute",
        "effondr",
        "ralenti",
        "contraction",
        "faibl",
        "difficil",
        "crise",
        "litige",
        "amende",
        "sanction",
        "condamn",
        "enquête",
        "fraude",
        "licenci",
        "restructur",
        "fermeture",
        "suppression",
        "avertissement",
        "abaiss",
        "inférieur",
        "négati",
        "défavorabl",
        "pénalis",
        "retard",
        "annul",
        "suspen",
        "échec",
        "déceva",
        "décept",
        "défaut",
        "faillite",
        "insolv",
        "dévalu",
        "moins-value",
        "sous-perform",
        "affaibl",
        "érosion",
        "repli",
        "sévère",
        "grave",
        "préjudic",
        "menac",
        "conflit",
        "pénurie",
        "grève",
    ),
    "en": (
        "decreas",
        "declin",
        "loss",
        "lower",
        "fell=",
        "fall=",
        "falls=",
        "fallen=",
        "drop",
        "weak",
        "deteriorat",
        "impair",
        "downgrad",
        "cut=",
        "cuts=",
        "warning",
        "litigation",
        "penalt",
        "investigat",
        "fraud",
        "layoff",
        "restructur",
        "closure",
        "delay",
        "cancel",
        "suspend",
        "fail",
        "shortfall",
        "negativ",
        "advers",
        "below=",
        "slowdown",
        "slower",
        "contraction",
        "crisis",
        "default",
        "bankrupt",
        "disappoint",
        "shortage",
        "strike=",
    ),
}

UNCERTAINTY: dict[str, tuple[str, ...]] = {
    "fr": (
        "incert",
        "risqu",
        "volatil",
        "pourrait",
        "pourraient",
        "éventuel",
        "possibl",
        "probabl",
        "hypothès",
        "aléa",
        "imprévis",
        "instab",
        "approximati",
        "susceptibl",
        "provisoire",
        "dépendr",
        "fluctu",
        "anticip",
        "prévis",
        "estimation",
    ),
    "en": (
        "uncertain",
        "risk",
        "volatil",
        "may=",
        "might=",
        "could=",
        "possibl",
        "probabl",
        "approximat",
        "depend",
        "fluctuat",
        "contingen",
        "unpredict",
        "unforeseen",
        "assum",
        "estimat",
        "anticipat",
    ),
}

NEGATIONS: frozenset[str] = frozenset(
    {
        "ne",
        "n",
        "pas",
        "non",
        "aucun",
        "aucune",
        "sans",
        "ni",
        "jamais",
        "not",
        "no",
        "without",
        "never",
        "neither",
        "nor",
    }
)


def tokenize(text: str) -> list[str]:
    """Lower-case words; apostrophes and punctuation split words ("n'a" -> "n", "a")."""
    return WORD.findall(text.lower())


def matches(word: str, entries: tuple[str, ...]) -> bool:
    """True if `word` starts with a stem, or equals an entry that ends in '='."""
    for entry in entries:
        if entry.endswith("="):
            if word == entry[:-1]:
                return True
        elif word.startswith(entry):
            return True
    return False


def lexicon_scores(text: str, language: str) -> dict[str, float]:
    """Counts and scores for one document. Unknown languages are scored with the French lists,
    since nearly all filings are French."""
    language = language if language in POSITIVE else "fr"
    words = tokenize(text)
    counts = {"positive": 0, "negative": 0, "uncertainty": 0}
    lists = {
        "positive": POSITIVE[language],
        "negative": NEGATIVE[language],
        "uncertainty": UNCERTAINTY[language],
    }
    for position, word in enumerate(words):
        for name, entries in lists.items():
            if not matches(word, entries):
                continue
            previous = words[max(0, position - NEGATION_WINDOW) : position]
            negated = name != "uncertainty" and any(w in NEGATIONS for w in previous)
            if not negated:
                counts[name] += 1
    polar = counts["positive"] + counts["negative"]
    return {
        "lex_words": len(words),
        "lex_positive": counts["positive"],
        "lex_negative": counts["negative"],
        "lex_uncertain": counts["uncertainty"],
        "lex_tone": (counts["positive"] - counts["negative"]) / polar if polar else 0.0,
        "lex_uncertainty": 100.0 * counts["uncertainty"] / len(words) if words else 0.0,
    }
