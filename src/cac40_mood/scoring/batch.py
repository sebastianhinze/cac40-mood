"""Run many requests through the Message Batches API (half price, answers within hours).

How a run works
1. Requests whose answer is already in the cache are not sent.
2. Requests that sit in a batch submitted earlier (by a run that was interrupted) are not sent
   again either: the state file remembers every submitted batch until its results are stored.
3. The worst-case cost of everything about to be sent is checked against the spend cap.
4. The rest is split into chunks (a batch may hold at most 256 MB) and submitted.
5. The run waits, polling once a minute, and stores each answer in the cache with its cost.

A request that failed or expired is not cached, so the next run sends it again.
Run one batch job at a time: two jobs started together could both submit the same documents.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from cac40_mood.scoring.llm import answer_from_message, read_cached, request_chars, store_answer
from cac40_mood.scoring.spend import BATCH_DISCOUNT, SpendLog, worst_case_cost_usd

# Limits per batch are 100,000 requests and 256 MB; we stay well below both so one upload
# stays small enough to retry comfortably.
MAX_REQUESTS_PER_CHUNK: int = 2_000
MAX_BYTES_PER_CHUNK: int = 60_000_000
POLL_SECONDS: float = 60.0


@dataclass(frozen=True)
class BatchItem:
    """One request: `custom_id` names it inside the batch, `key` is its cache key."""

    custom_id: str
    request: dict[str, Any]
    key: str


def split_into_chunks(items: list[BatchItem]) -> list[list[BatchItem]]:
    """Group items so that no chunk exceeds the request or size limit."""
    chunks: list[list[BatchItem]] = []
    current: list[BatchItem] = []
    current_bytes = 0
    for item in items:
        size = len(json.dumps(item.request, ensure_ascii=False).encode("utf-8"))
        too_many = len(current) >= MAX_REQUESTS_PER_CHUNK
        too_big = current_bytes + size > MAX_BYTES_PER_CHUNK
        if current and (too_many or too_big):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(item)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


def read_open_batches(state_path: Path) -> dict[str, dict[str, str]]:
    """Batches that were submitted but whose results are not stored yet: {batch_id: {id: key}}."""
    if not state_path.exists():
        return {}
    submitted: dict[str, dict[str, str]] = {}
    for line in state_path.read_text().splitlines():
        entry = json.loads(line)
        if entry["event"] == "submitted":
            submitted[entry["batch_id"]] = entry["keys"]
        elif entry["event"] == "collected":
            submitted.pop(entry["batch_id"], None)
    return submitted


def _append_state(state_path: Path, entry: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    entry["at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    with state_path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


def worst_case_batch_cost(items: list[BatchItem]) -> float:
    """Upper bound for sending these items at batch prices."""
    total = sum(
        worst_case_cost_usd(
            item.request["model"], request_chars(item.request), item.request["max_tokens"]
        )
        for item in items
    )
    return total * BATCH_DISCOUNT


def collect_batch(
    client: anthropic.Anthropic,
    batch_id: str,
    keys: dict[str, str],
    label: str,
    cache_dir: Path,
    spend_log: SpendLog,
) -> dict[str, int]:
    """Store every successful answer of a finished batch; count the outcomes."""
    outcomes = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
    for result in client.messages.batches.results(batch_id):
        outcome = result.result.type
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome != "succeeded":
            continue
        answer = answer_from_message(result.result.message, batch=True)
        spend_log.record(
            answer["model"],
            label,
            result.custom_id,
            answer["input_tokens"],
            answer["output_tokens"],
            batch=True,
        )
        store_answer(cache_dir, keys[result.custom_id], answer)
    return outcomes


def run_batch(
    client: anthropic.Anthropic,
    items: list[BatchItem],
    label: str,
    cache_dir: Path,
    spend_log: SpendLog,
    state_path: Path,
    poll_seconds: float = POLL_SECONDS,
) -> dict[str, dict[str, Any] | None]:
    """Get an answer for every item (None if the API failed it) and return {custom_id: answer}."""
    open_batches = read_open_batches(state_path)
    already_sent = {key for keys in open_batches.values() for key in keys.values()}
    to_send = [
        item
        for item in items
        if read_cached(cache_dir, item.key) is None and item.key not in already_sent
    ]
    if to_send:
        spend_log.check_can_spend(worst_case_batch_cost(to_send))
    for chunk in split_into_chunks(to_send):
        batch = client.messages.batches.create(
            requests=[{"custom_id": item.custom_id, "params": item.request} for item in chunk]
        )
        keys = {item.custom_id: item.key for item in chunk}
        _append_state(
            state_path, {"event": "submitted", "batch_id": batch.id, "label": label, "keys": keys}
        )
        open_batches[batch.id] = keys
        print(f"submitted batch {batch.id} with {len(chunk)} requests", flush=True)

    for batch_id, keys in open_batches.items():
        while client.messages.batches.retrieve(batch_id).processing_status != "ended":
            time.sleep(poll_seconds)
        outcomes = collect_batch(client, batch_id, keys, label, cache_dir, spend_log)
        _append_state(
            state_path, {"event": "collected", "batch_id": batch_id, "outcomes": outcomes}
        )
        print(f"collected batch {batch_id}: {outcomes}", flush=True)

    return {item.custom_id: read_cached(cache_dir, item.key) for item in items}
