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
from data_collector.l2_registry import ETHEREUM_L2S

logger = logging.getLogger(__name__)

TVL_HISTORY_URL = "https://api.llama.fi/v2/historicalChainTvl/Ethereum"
CHAINS_URL = "https://api.llama.fi/v2/chains"

# Tier thresholds are TVL-only - a real, live, hard-to-fake number - not a
# claim about audits, insurance, or bug bounties, which DeFiLlama's free API
# doesn't expose and which this app has no way to verify. Labelled as such
# wherever it's shown (see the /api/l2 docstring and the frontend copy).
_ESTABLISHED_TVL_USD = 300_000_000
_GROWING_TVL_USD = 30_000_000
_ESTABLISHED_MIN_AGE_YEARS = 2

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


def _tier(tvl_usd: float, launch_year: int) -> str:
    age_years = datetime.now(timezone.utc).year - launch_year
    if tvl_usd >= _ESTABLISHED_TVL_USD and age_years >= _ESTABLISHED_MIN_AGE_YEARS:
        return "established"
    if tvl_usd >= _GROWING_TVL_USD:
        return "growing"
    return "emerging"


def get_l2_tvls() -> list[dict]:
    """Live TVL for the curated Ethereum L2s in l2_registry.py, ranked by
    TVL, each tagged with a TVL/age-based tier (see the module docstring
    for why this is a heuristic, not a security or insurance claim).

    Empty list (never raises) if the request fails - same "nice to have,
    not on the critical path" treatment as get_ethereum_tvl_by_date().
    """
    cached = _cache.get("l2")
    if cached and time.monotonic() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        chains = get_json(CHAINS_URL, max_retries=1, timeout=10)
    except ExternalAPIError as exc:
        logger.warning("DeFiLlama chains list failed: %s", exc)
        return cached[1] if cached else []

    tvl_by_name = {c["name"]: c.get("tvl") for c in chains if c.get("tvl") is not None}

    results = []
    for l2 in ETHEREUM_L2S:
        tvl = tvl_by_name.get(l2["name"])
        if tvl is None:
            # l2_registry.py's own docstring warns DeFiLlama renames chains
            # occasionally - without this, a renamed entry just vanishes
            # from /api/l2 with no signal anywhere that the registry needs
            # a matching update.
            logger.warning(
                "L2 registry entry %r has no matching DeFiLlama chain (renamed upstream?) - dropping it from /api/l2",
                l2["name"],
            )
            continue
        results.append({
            "name": l2["name"],
            "display_name": l2["display_name"],
            "launch_year": l2["launch_year"],
            "tvl_usd": tvl,
            "tier": _tier(tvl, l2["launch_year"]),
        })
    results.sort(key=lambda r: r["tvl_usd"], reverse=True)

    with _cache_lock:
        _cache["l2"] = (time.monotonic(), results)
    return results
