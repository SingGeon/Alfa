"""DeFiLlama integration (no API key needed - fully public, free).

Gives us a real "how much is actually being built/used on Ethereum" signal
(total value locked in DeFi protocols) - the on-chain equivalent of the
"utilitatea rețelei" factor (dApps/DeFi adoption) that raw price/volume data
can't capture on its own. See ml/features.py's tvl_momentum.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from data_collector.http_utils import ExternalAPIError, get_json

logger = logging.getLogger(__name__)

TVL_HISTORY_URL = "https://api.llama.fi/v2/historicalChainTvl/Ethereum"

# The full history is ~3000+ daily points but changes once a day at most -
# no reason to re-fetch it on every prediction request. TTL matches the
# order of magnitude of other slow-moving caches in this project (news RSS).
_CACHE_TTL_SECONDS = 6 * 3600
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, dict]] = {}


def get_ethereum_tvl_by_date() -> dict:
    """Returns {date: raw_tvl_usd} for every day DeFiLlama has data for.

    Empty dict (never raises) if the request fails - TVL is a nice-to-have
    feature, not something that should take the whole prediction down.
    """
    cached = _cache.get("tvl")
    if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        raw = get_json(TVL_HISTORY_URL, max_retries=1, timeout=10)
    except ExternalAPIError as exc:
        logger.warning("DeFiLlama TVL history failed: %s", exc)
        return cached[1] if cached else {}

    by_date = {
        datetime.fromtimestamp(point["date"], tz=timezone.utc).date(): point["tvl"]
        for point in raw
        if point.get("tvl") is not None
    }
    with _cache_lock:
        _cache["tvl"] = (time.monotonic(), by_date)
    return by_date
