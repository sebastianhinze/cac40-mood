"""Shared plumbing for every LLM request: cache key, cache files, one direct call.

Both ways of calling the model (one request at a time, or many through the Batches API) store
the model's raw answer in the same cache under the same key, so a document that was answered
one way is never paid for again the other way.

The cache key is a hash of the complete request: model, instructions, JSON schema, thinking
setting, output limit and the message with the document text. Changing any of them, however
slightly, gives a different key, so answers produced under one prompt or setting can never be
returned for another (DECISIONS #36).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import anthropic

from cac40_mood.scoring.spend import SpendLog, worst_case_cost_usd


def cache_key(request: dict[str, Any], prompt_version: str, repeat: int = 1) -> str:
    """sha256 over the prompt version, the repeat number and the complete request.

    The request is written as JSON with sorted keys so the same request always gives the same
    text, and therefore the same hash.
    """
    canonical_request = json.dumps(request, sort_keys=True, ensure_ascii=False)
    joined = "\n".join([prompt_version, f"repeat={repeat}", canonical_request])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    """Two-character sub-folders keep any one folder from holding thousands of files."""
    return cache_dir / key[:2] / f"{key}.json"


def read_cached(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """The stored answer for this key, or None. Old-format files (no 'text') count as absent."""
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    answer = json.loads(path.read_text())
    return answer if "text" in answer else None


def store_answer(cache_dir: Path, key: str, answer: dict[str, Any]) -> None:
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(answer, ensure_ascii=False, indent=1))


def answer_from_message(message: anthropic.types.Message, batch: bool) -> dict[str, Any]:
    """Keep only what later steps need from an API response: the text, why it stopped, tokens."""
    text = next((block.text for block in message.content if block.type == "text"), "")
    return {
        "text": text,
        "stop_reason": message.stop_reason,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "model": message.model,
        "batch": batch,
    }


def request_chars(request: dict[str, Any]) -> int:
    """Characters of instructions plus message, for the worst-case cost bound."""
    return len(request["system"]) + sum(len(m["content"]) for m in request["messages"])


def call_once(
    client: anthropic.Anthropic,
    request: dict[str, Any],
    key: str,
    label: str,
    doc_id: str,
    cache_dir: Path,
    spend_log: SpendLog,
) -> dict[str, Any]:
    """Answer one request now (not through a batch): cache, cap check, call, log, store."""
    cached = read_cached(cache_dir, key)
    if cached is not None:
        return {**cached, "from_cache": True}
    worst_case = worst_case_cost_usd(
        request["model"], request_chars(request), request["max_tokens"]
    )
    spend_log.check_can_spend(worst_case)
    answer = answer_from_message(client.messages.create(**request), batch=False)
    spend_log.record(
        request["model"], label, doc_id, answer["input_tokens"], answer["output_tokens"]
    )
    store_answer(cache_dir, key, answer)
    return {**answer, "from_cache": False}
