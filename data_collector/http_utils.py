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


class ExternalAPIError(RuntimeError):
    """Raised when an external API is unreachable after retries."""


def get_json(
    url: str, params: dict | None = None, headers: dict | None = None,
    max_retries: int = 3, timeout: int = 10,
) -> Any:
    """GET a JSON endpoint with exponential backoff on 429/5xx/network errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                logger.warning("Rate limited by %s, waiting %.1fs", url, wait)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                logger.warning("%s returned %s, retrying", url, resp.status_code)
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request to %s failed (attempt %d/%d): %s", url, attempt + 1, max_retries, exc)
            time.sleep(2 ** attempt)
    raise ExternalAPIError(f"All {max_retries} attempts to reach {url} failed") from last_exc
