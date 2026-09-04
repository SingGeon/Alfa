"""Orchestrates one full scout scan: universe -> per-asset analysis -> storage.

Wired into run_scheduler.py on its own (long) interval - a full scan trains
a real model per asset, so it's deliberately a background job, never run
inline on a request.
"""
from __future__ import annotations

import logging

from database import repository
from scout.pipeline import analyze_crypto_asset, analyze_stock_asset
from scout.universe import (
    get_crypto_mega_cap_universe,
    get_crypto_universe,
    get_stock_mega_cap_universe,
    get_stock_universe,
)

logger = logging.getLogger(__name__)

CRYPTO_UNIVERSE_SIZE = 40
STOCK_UNIVERSE_SIZE = 40
CRYPTO_MEGA_CAP_UNIVERSE_SIZE = 30
STOCK_MEGA_CAP_UNIVERSE_SIZE = 25
# Below this, the model itself is telling us it found no real signal - the
# predicted % is noise, not a genuine opportunity. Showing it anyway (just
# ranked low) is what produced results like a 0/100-confidence "+65%" pick.
MIN_CONFIDENCE = 25

# A mega-cap (BTC/ETH, NVDA/AAPL-scale stocks) is efficiently priced and
# rarely moves double digits in a week - underdogs only need a positive
# call to qualify, but a mega-cap has to clear a much higher bar to be
# worth surfacing outside the "underdog" framing at all. This is the
# escape hatch for "one of the big ones is genuinely about to move a lot",
# not a relaxation of the underdog focus by default - see the conversation
# that added this (Scout AI structurally can't have called Nvidia's move
# without it; this lets a comparably dramatic mega-cap call through).
MEGA_CAP_MIN_CHANGE_PCT = 10.0
MEGA_CAP_MIN_CONFIDENCE = 40


def _scan(universe, analyze, id_field: str, tier: str) -> list[dict]:
    results = []
    for asset in universe:
        try:
            result = analyze(asset)
        except Exception:
            logger.exception("%s analysis crashed for %s", tier, asset.get(id_field))
            continue
        if result:
            result["tier"] = tier
            results.append(result)
    return results


def run_scout_scan() -> int:
    """Scan the underdog universe (+ a separate, stricter mega-cap pass),
    score each asset, persist the ranked results.

    Returns the number of assets successfully scored. Failures on
    individual assets (missing data, a bad ticker, a flaky fetch) are
    logged and skipped - one bad asset never aborts the whole scan.
    """
    underdog_results = _scan(get_crypto_universe(CRYPTO_UNIVERSE_SIZE), analyze_crypto_asset, "coin_id", "underdog")
    underdog_results += _scan(get_stock_universe(STOCK_UNIVERSE_SIZE), analyze_stock_asset, "ticker", "underdog")

    mega_cap_results = _scan(
        get_crypto_mega_cap_universe(CRYPTO_MEGA_CAP_UNIVERSE_SIZE), analyze_crypto_asset, "coin_id", "mega_cap"
    )
    mega_cap_results += _scan(
        get_stock_mega_cap_universe(STOCK_MEGA_CAP_UNIVERSE_SIZE), analyze_stock_asset, "ticker", "mega_cap"
    )

    scanned = len(underdog_results) + len(mega_cap_results)
    total_universe = (
        CRYPTO_UNIVERSE_SIZE + STOCK_UNIVERSE_SIZE + CRYPTO_MEGA_CAP_UNIVERSE_SIZE + STOCK_MEGA_CAP_UNIVERSE_SIZE
    )

    # This is a "will it rise" scout, not a general watchlist - a declining
    # prediction isn't a lesser opportunity, it's a different question
    # nobody asked. Drop it rather than rank it low. Same logic for
    # confidence: below the bar the model found no real signal, so the
    # predicted % is noise, not an opportunity worth showing at all.
    underdog_results = [
        r for r in underdog_results if r["predicted_change_pct"] > 0 and r["confidence"] >= MIN_CONFIDENCE
    ]
    mega_cap_results = [
        r for r in mega_cap_results
        if r["predicted_change_pct"] >= MEGA_CAP_MIN_CHANGE_PCT and r["confidence"] >= MEGA_CAP_MIN_CONFIDENCE
    ]

    results = underdog_results + mega_cap_results
    results.sort(key=lambda r: r["score"], reverse=True)
    repository.save_scout_results(results)
    logger.info(
        "Scout scan complete: %d/%d assets scored, %d underdog + %d mega-cap kept",
        scanned, total_universe, len(underdog_results), len(mega_cap_results),
    )
    return len(results)
