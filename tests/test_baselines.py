"""Baselines: the word-list scorer and the helper functions around the local model.

The local model itself is not downloaded in tests; its sentence splitting and the aggregation
of sentence probabilities are tested with made-up numbers.
"""

import numpy as np
import pytest

from cac40_mood.scoring import lexicon
from cac40_mood.scoring.local_model import MODELS, document_scores, split_sentences

# ------------------------------------------------------------------ lexicon


def test_tokenizer_lowercases_and_splits_on_apostrophes() -> None:
    assert lexicon.tokenize("L'EBITDA n'a pas progressé : -3,2 %") == [
        "l", "ebitda", "n", "a", "pas", "progressé",
    ]  # fmt: skip


def test_stems_match_word_beginnings_and_exact_entries_match_whole_words() -> None:
    assert lexicon.matches("progression", ("progress",))
    assert not lexicon.matches("regress", ("progress",))
    assert lexicon.matches("gain", ("gain=",))
    assert not lexicon.matches("gainsay", ("gain=",))


def test_french_counts_and_tone() -> None:
    text = (
        "Le chiffre d'affaires est en hausse et la rentabilité progresse, malgré une perte en Asie."
    )
    scores = lexicon.lexicon_scores(text, "fr")
    assert (scores["lex_positive"], scores["lex_negative"]) == (3, 1)  # hausse, rentab, progress
    assert scores["lex_tone"] == pytest.approx((3 - 1) / 4)


def test_negated_words_are_not_counted() -> None:
    plain = lexicon.lexicon_scores("Le chiffre d'affaires progresse.", "fr")
    negated = lexicon.lexicon_scores("Le chiffre d'affaires ne progresse pas.", "fr")
    english = lexicon.lexicon_scores("Margins did not improve.", "en")
    assert plain["lex_positive"] == 1
    assert negated["lex_positive"] == 0 and negated["lex_tone"] == 0.0
    assert english["lex_positive"] == 0


def test_english_lists_are_used_for_english_text() -> None:
    scores = lexicon.lexicon_scores("Revenue fell 6% and the group issued a warning.", "en")
    assert (scores["lex_positive"], scores["lex_negative"]) == (0, 2)
    assert scores["lex_tone"] == -1.0


def test_uncertainty_is_per_hundred_words_and_ignores_negation() -> None:
    scores = lexicon.lexicon_scores(
        "Les perspectives restent incertaines et les risques élevés.", "fr"
    )
    assert scores["lex_uncertain"] == 2
    assert scores["lex_uncertainty"] == pytest.approx(100 * 2 / scores["lex_words"])


def test_text_without_signal_words_and_empty_text_score_zero() -> None:
    assert (
        lexicon.lexicon_scores("L'assemblée générale se tiendra le 12 mai.", "fr")["lex_tone"]
        == 0.0
    )
    empty = lexicon.lexicon_scores("", "fr")
    assert empty["lex_words"] == 0 and empty["lex_tone"] == 0.0 and empty["lex_uncertainty"] == 0.0


def test_unknown_language_falls_back_to_french() -> None:
    assert lexicon.lexicon_scores("Forte hausse du résultat.", "")["lex_positive"] == 2


def test_no_stem_is_listed_as_both_positive_and_negative() -> None:
    for language in ("fr", "en"):
        overlap = set(lexicon.POSITIVE[language]) & set(lexicon.NEGATIVE[language])
        assert not overlap, overlap


# -------------------------------------------------------------- local model


def test_sentence_splitter_keeps_prose_and_drops_tables_and_headers() -> None:
    text = (
        "COMMUNIQUÉ DE PRESSE\n"
        "Le chiffre d'affaires du groupe progresse de 8 % sur l'exercice. "
        "Le résultat opérationnel courant atteint 1,2 milliard d'euros.\n"
        "2023 2022 Var. 12 345 11 432 +8,0 % 1 204 1 100 +9,5 %\n"
        "Page 1 / 4\n"
        "Le conseil propose un dividende de 2,10 euros par action à l'assemblée générale."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Le chiffre d'affaires") and sentences[2].startswith(
        "Le conseil"
    )


def test_sentence_splitter_respects_the_limit() -> None:
    text = " ".join(
        f"La phrase numéro {i} décrit une activité du groupe en détail." for i in range(30)
    )
    assert len(split_sentences(text, limit=10)) == 10


def test_document_tone_is_the_average_of_positive_minus_negative() -> None:
    labels = ["positive", "neutral", "negative"]
    probabilities = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
    scores = document_scores(probabilities, labels)
    assert scores["lm_tone"] == pytest.approx((0.9 + 0.0 - 0.8) / 3)
    assert scores["lm_share_positive"] == pytest.approx(1 / 3)
    assert scores["lm_share_negative"] == pytest.approx(1 / 3)
    assert scores["lm_sentences"] == 3


def test_label_order_of_each_model_is_respected() -> None:
    """French model: positive/neutral/negative. FinBERT: positive/negative/neutral."""
    probabilities = np.array([[0.7, 0.2, 0.1]])
    french = document_scores(probabilities, ["positive", "neutral", "negative"])
    english = document_scores(probabilities, ["positive", "negative", "neutral"])
    assert french["lm_tone"] == pytest.approx(0.6)
    assert english["lm_tone"] == pytest.approx(0.5)


def test_document_without_prose_sentences_scores_zero() -> None:
    scores = document_scores(np.empty((0, 3)), ["positive", "neutral", "negative"])
    assert scores == {
        "lm_sentences": 0,
        "lm_tone": 0.0,
        "lm_share_positive": 0.0,
        "lm_share_negative": 0.0,
    }


def test_model_versions_are_pinned() -> None:
    for model_id, revision in MODELS.values():
        assert "/" in model_id and len(revision) == 40
