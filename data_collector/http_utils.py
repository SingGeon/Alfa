"""Shared HTTP-with-retry helper used by every external API client here.

Centralizing this is what satisfies the "rate limits + downtime handling"
requirement for *all* external APIs (market data, news) instead of
reimplementing backoff per-client.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_session = requests.Session()

# Hard ceiling on any single backoff sleep. A 429 can carry a Retry-After of
# a minute or more (seen from CoinGecko's free tier); honoring that verbatim
# would block the request - and, on a single-threaded dev server, every
# other request - for that long. Capping it keeps us responsive; the caller
# still gets a real error (or falls back to another source) if the wait
# wasn't enough.
_MAX_BACKOFF_SECONDS = 8.0


class ExternalAPIError(RuntimeError):
    """Raised when an external API is unreachable after retries."""


def get_json(
    url: str, params: dict | None = None, headers: dict | None = None,
    max_retries: int = 3, timeout: int = 10,
) -> Any:
    """GET a JSON endpoint with exponential backoff on 429/5xx/network errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        is_last = attempt == max_retries - 1
        try:
            resp = _session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = min(float(resp.headers.get("Retry-After", 2 ** attempt)), _MAX_BACKOFF_SECONDS)
                logger.warning("Rate limited by %s, waiting %.1fs", url, wait)
                if not is_last:
                    time.sleep(wait)
                continue
            if 400 <= resp.status_code < 500:
                # A client error (bad symbol, bad params, ...) won't fix
                # itself on retry - fail immediately instead of burning
                # several seconds of backoff on something that will never
                # succeed (seen scanning the scout universe: coins with no
                # real Binance pair get a 400 on every attempt).
                raise ExternalAPIError(f"{url} returned {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 500:
                logger.warning("%s returned %s, retrying", url, resp.status_code)
                if not is_last:
                    time.sleep(min(2 ** attempt, _MAX_BACKOFF_SECONDS))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request to %s failed (attempt %d/%d): %s", url, attempt + 1, max_retries, exc)
            if not is_last:
                time.sleep(min(2 ** attempt, _MAX_BACKOFF_SECONDS))
    raise ExternalAPIError(f"All {max_retries} attempts to reach {url} failed") from last_exc
