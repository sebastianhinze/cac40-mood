"""Baseline 2: a small sentiment classifier that runs on this machine (no API).

Models (DECISIONS #50), both fine-tuned on the Financial PhraseBank, a set of finance-news
sentences labelled positive / neutral / negative:
- French text: `bardsai/finance-sentiment-fr-base` (CamemBERT; trained on the French
  translation of the PhraseBank; MIT licence);
- English text: `ProsusAI/finbert` (BERT; trained on the original English PhraseBank;
  CC BY-NC-SA licence, fine for this non-commercial project).
Using the same training data in both languages makes the two as comparable as two models can be.
The exact model versions are pinned.

These models read one sentence at a time (they were trained on single sentences), so a
document is scored like this:
1. split the text the LLM would see into sentences and keep the ones that look like prose;
2. classify up to the first 100 of them;
3. document tone = average over sentences of P(positive) - P(negative), between -1 and 1.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODELS: dict[str, tuple[str, str]] = {
    "fr": ("bardsai/finance-sentiment-fr-base", "98f660ba2ca64140df78c1a29b91dc8b6beafb62"),
    "en": ("ProsusAI/finbert", "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"),
}

MAX_SENTENCES: int = 100
MAX_TOKENS: int = 128  # PhraseBank sentences are short; longer ones are cut
BATCH_SIZE: int = 64
MIN_WORDS: int = 6
MIN_LETTER_SHARE: float = 0.6

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str, limit: int = MAX_SENTENCES) -> list[str]:
    """The first `limit` prose sentences of a text.

    A piece counts as prose if it has at least six words and at least 60% letters; that drops
    table rows, page headers and lists of figures, which the classifier was never trained on.
    """
    sentences = []
    for piece in SENTENCE_END.split(text):
        piece = " ".join(piece.split())
        if len(piece.split()) < MIN_WORDS:
            continue
        letters = sum(character.isalpha() for character in piece)
        if letters / len(piece) < MIN_LETTER_SHARE:
            continue
        sentences.append(piece)
        if len(sentences) == limit:
            break
    return sentences


def pick_device() -> str:
    """Apple-silicon GPU if available, otherwise the CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_classifier(
    language: str, device: str
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, list[str]]:
    """Return (tokenizer, model, label names in the model's output order)."""
    model_id, revision = MODELS[language]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, use_safetensors=True
    )
    model.to(device).eval()
    labels = [model.config.id2label[index].lower() for index in range(model.config.num_labels)]
    return tokenizer, model, labels


def classify(
    tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, sentences: list[str], device: str
) -> np.ndarray:
    """Class probabilities for each sentence, in the model's own label order."""
    outputs = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, model.config.num_labels))


def document_scores(probabilities: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Aggregate sentence probabilities into document scores.

    `labels` gives the meaning of each probability column, because the two models order their
    classes differently.
    """
    if len(probabilities) == 0:
        return {
            "lm_sentences": 0,
            "lm_tone": 0.0,
            "lm_share_positive": 0.0,
            "lm_share_negative": 0.0,
        }
    positive = probabilities[:, labels.index("positive")]
    negative = probabilities[:, labels.index("negative")]
    winner = probabilities.argmax(axis=1)
    return {
        "lm_sentences": len(probabilities),
        "lm_tone": float((positive - negative).mean()),
        "lm_share_positive": float((winner == labels.index("positive")).mean()),
        "lm_share_negative": float((winner == labels.index("negative")).mean()),
    }
