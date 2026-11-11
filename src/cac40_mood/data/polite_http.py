"""A small HTTP client that behaves well towards public servers.

Why: the filings come from free government servers. We identify ourselves with a generic
User-Agent, never send requests faster than the configured rate, retry a few times on temporary
errors, and never download a file twice (the file on disk is the cache).
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

RETRY_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS: int = 4


class PoliteSession:
    """Wrap `requests.Session` with a User-Agent, a minimum gap between requests, and retries."""

    def __init__(self, user_agent: str, min_seconds_between_requests: float) -> None:
        self._session = requests.Session()
        self._session.headers["User-Agent"] = user_agent
        self._min_gap = min_seconds_between_requests
        self._last_request_at = 0.0

    def _wait_for_turn(self) -> None:
        """Sleep just long enough that two requests are never closer than the minimum gap."""
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)
        self._last_request_at = time.monotonic()

    def get(
        self, url: str, params: dict[str, str] | None = None, timeout: float = 120.0
    ) -> requests.Response:
        """GET with rate limiting. Temporary failures are retried with a growing pause.

        A 4xx answer (other than 429) is returned to the caller: it is a fact about the resource
        (for example a PDF that no longer exists), not something a retry can fix.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self._wait_for_turn()
            try:
                response = self._session.get(url, params=params, timeout=timeout)
            except requests.RequestException as error:
                last_error = error
            else:
                if response.status_code not in RETRY_STATUS_CODES:
                    return response
                last_error = requests.HTTPError(f"HTTP {response.status_code} for {url}")
            time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"Giving up on {url} after {MAX_ATTEMPTS} attempts") from last_error


def download_to_file(session: PoliteSession, url: str, destination: Path) -> str:
    """Download `url` to `destination` unless the file is already there.

    Returns 'cached', 'downloaded' or 'missing' (HTTP 404 and friends).
    The body is written to a temporary '.part' file first and renamed at the end, so an
    interrupted run never leaves a half-written file that a later run would mistake for a cache hit.
    """
    if destination.exists():
        return "cached"
    response = session.get(url)
    if response.status_code != 200:
        return "missing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_bytes(response.content)
    partial.rename(destination)
    return "downloaded"
