"""HTTP routes. Thin: validation + delegating to database/repository or
api/services, then serializing to JSON.



    ----------RUN Command----------
cd /home/singeon/Documents/Alfa
.venv/bin/python run_api.py

cd /home/singeon/Documents/Alfa
.venv/bin/python run_scheduler.py


"""
from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, request

import config
from api.services import InsufficientDataError, run_combined_summary, run_outlook, run_prediction
from data_collector.market_data import SUPPORTED_INTERVALS, MarketDataError, get_current_price
from database import repository

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# Short server-side cache for the current-price ticker. get_current_price()
# tries Binance first (weight-1 call, generous ~1200/min limit) and only
# falls back to CoinGecko's much stricter free-tier limit on failure, so
# this just needs to survive a burst of tabs polling in the same instant -
# not protect against sustained per-second polling the way it did when
# CoinGecko was the primary source.
_PRICE_CACHE_TTL_SECONDS = 3.0
_price_cache: dict[str, tuple[float, dict]] = {}


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.get("/price/current")
def price_current():
    coin_id = request.args.get("coin_id", config.COIN_ID)

    cached = _price_cache.get(coin_id)
    if cached and time.monotonic() - cached[0] < _PRICE_CACHE_TTL_SECONDS:
        return jsonify(cached[1])

    try:
        result = get_current_price(coin_id)
    except MarketDataError as exc:
        return jsonify({"error": str(exc)}), 502
    _price_cache[coin_id] = (time.monotonic(), result)
    return jsonify(result)


@api_bp.get("/price/history")
def price_history():
    coin_id = request.args.get("coin_id", config.COIN_ID)
    interval = request.args.get("interval", "1h")
    limit = int(request.args.get("limit", 200))
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"interval must be one of {sorted(SUPPORTED_INTERVALS)}"}), 400
    candles = repository.get_candles(coin_id, interval, limit)
    for c in candles:
        c["timestamp"] = c["timestamp"].isoformat()
    return jsonify({"coin_id": coin_id, "interval": interval, "candles": candles})


@api_bp.get("/predict")
def predict():
    coin_id = request.args.get("coin_id", config.COIN_ID)
    interval = request.args.get("interval", "1h")
    steps = int(request.args.get("steps", config.PREDICTION_HORIZON_HOURS))
    use_sentiment = request.args.get("use_sentiment", "true").lower() != "false"

    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"interval must be one of {sorted(SUPPORTED_INTERVALS)}"}), 400
    if not (1 <= steps <= 500):
        return jsonify({"error": "steps must be between 1 and 500"}), 400

    try:
        return jsonify(run_prediction(coin_id, interval, steps, use_sentiment))
    except InsufficientDataError as exc:
        return jsonify({"error": str(exc)}), 409


@api_bp.get("/outlook")
def outlook():
    coin_id = request.args.get("coin_id", config.COIN_ID)
    use_sentiment = request.args.get("use_sentiment", "true").lower() != "false"
    try:
        return jsonify(run_outlook(coin_id, use_sentiment))
    except InsufficientDataError as exc:
        return jsonify({"error": str(exc)}), 409


@api_bp.get("/news")
def news():
    limit = int(request.args.get("limit", 50))
    items = repository.get_recent_news(limit)
    for item in items:
        if item.get("published_at"):
            item["published_at"] = item["published_at"].isoformat()
    return jsonify({"news": items})


@api_bp.get("/sentiment")
def sentiment_summary():
    from datetime import datetime, timedelta, timezone

    from nlp.sentiment import aggregate_daily

    days = int(request.args.get("days", 7))
    since = datetime.now(timezone.utc) - timedelta(days=days)
    articles = repository.get_news_since(since)
    daily = aggregate_daily(articles)
    overall_avg = sum(daily.values()) / len(daily) if daily else 0.0
    return jsonify(
        {
            "days": days,
            "articles_analyzed": len(articles),
            "overall_avg_compound": round(overall_avg, 4),
            "daily": {str(day): round(score, 4) for day, score in sorted(daily.items())},
        }
    )


@api_bp.get("/gas")
def gas_price():
    from data_collector.etherscan_client import get_gas_price_gwei, is_configured

    if not is_configured():
        return jsonify({"error": "ETHERSCAN_API_KEY not configured"}), 501
    gas = get_gas_price_gwei()
    if gas is None:
        return jsonify({"error": "Etherscan gas oracle unavailable"}), 502
    return jsonify(gas)


@api_bp.get("/scout")
def scout():
    """Ranked underdog-asset scan results (see scout/). Precomputed by the
    scheduler (scout/scanner.py), never run inline here - a full scan
    trains a real model per asset and takes minutes.
    """
    asset_type = request.args.get("asset_type")
    if asset_type not in (None, "crypto", "stock"):
        return jsonify({"error": "asset_type must be 'crypto' or 'stock'"}), 400
    limit = int(request.args.get("limit", 100))
    results = repository.get_scout_results(asset_type, limit)
    for r in results:
        if r.get("scanned_at"):
            r["scanned_at"] = r["scanned_at"].isoformat()
    return jsonify({"results": results, "count": len(results)})


# Detail involves a real per-asset train (candles + news + a fresh model,
# ~3s live - see scout/pipeline.py's module docstring for why this isn't
# precomputed for every scanned asset the way the compact scan result is).
# Caching the response means a re-click, an accidental double-load, or a
# page refresh on the same asset is instant instead of re-paying that cost;
# TTL matches the scan cadence (config.SCOUT_INTERVAL_MINUTES) so this never
# serves something older than the next scan would have replaced anyway.
_scout_detail_cache: dict[tuple, tuple[float, dict]] = {}
_SCOUT_DETAIL_CACHE_TTL_SECONDS = config.SCOUT_INTERVAL_MINUTES * 60


@api_bp.get("/scout/detail")
def scout_detail():
    """Full chart/prediction/news detail for one scout asset, computed live
    on request (not precomputed - this is a per-click view, not a scan).
    """
    from scout import pipeline

    asset_type = request.args.get("asset_type")
    asset_id = request.args.get("id")
    if asset_type not in ("crypto", "stock") or not asset_id:
        return jsonify({"error": "asset_type ('crypto'|'stock') and id are required"}), 400

    cache_key = (asset_type, asset_id)
    cached = _scout_detail_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < _SCOUT_DETAIL_CACHE_TTL_SECONDS:
        return jsonify(cached[1])

    # Reuse the symbol/name the last scan already resolved for this asset,
    # rather than re-hitting the universe APIs just for a lookup. Fall back
    # to a pinned bookmark: a pinned asset can legitimately drop out of the
    # scan (filters changed, it stopped rising) without becoming unviewable.
    existing = repository.get_scout_results(asset_type, limit=500)
    match = next((r for r in existing if r["id"] == asset_id), None)
    if not match:
        match = repository.get_pinned_asset(asset_type, asset_id)
    if not match:
        return jsonify({"error": f"{asset_id!r} isn't in the latest scan - try again after the next scout scan"}), 404

    try:
        if asset_type == "crypto":
            detail = pipeline.detail_for_crypto(
                asset_id, match["symbol"], match["name"], f"{match['symbol']}USDT"
            )
        else:
            detail = pipeline.detail_for_stock(asset_id, match["name"])
    except Exception:
        logger.exception("Scout detail analysis failed for %s/%s", asset_type, asset_id)
        return jsonify({"error": "Analysis failed"}), 502

    if detail is None:
        return jsonify({"error": "Insufficient data to analyze this asset"}), 409
    _scout_detail_cache[cache_key] = (time.monotonic(), detail)
    return jsonify(detail)


_scout_price_cache: dict[tuple, tuple[float, dict]] = {}
_SCOUT_PRICE_CACHE_TTL_SECONDS = 3.0


@api_bp.get("/scout/price")
def scout_price():
    """Lightweight live quote for the detail page's price ticker - no
    candles, no news, no model, so it's safe to poll every few seconds.
    """
    from scout import pipeline

    asset_type = request.args.get("asset_type")
    asset_id = request.args.get("id")
    if asset_type not in ("crypto", "stock") or not asset_id:
        return jsonify({"error": "asset_type ('crypto'|'stock') and id are required"}), 400

    key = (asset_type, asset_id)
    cached = _scout_price_cache.get(key)
    if cached and time.monotonic() - cached[0] < _SCOUT_PRICE_CACHE_TTL_SECONDS:
        return jsonify(cached[1])

    if asset_type == "crypto":
        existing = repository.get_scout_results("crypto", limit=500)
        match = next((r for r in existing if r["id"] == asset_id), None)
        if not match:
            match = repository.get_pinned_asset("crypto", asset_id)
        if not match:
            return jsonify({"error": f"{asset_id!r} isn't in the latest scan"}), 404
        quote = pipeline.get_live_price_crypto(f"{match['symbol']}USDT", asset_id)
    else:
        quote = pipeline.get_live_price_stock(asset_id)

    if quote is None:
        return jsonify({"error": "Live price unavailable"}), 502
    _scout_price_cache[key] = (time.monotonic(), quote)
    return jsonify(quote)


@api_bp.get("/scout/pins")
def scout_pins():
    """Bookmarked assets (see scout.html's "Fixate" tab) - independent of
    whatever the latest scan happens to contain right now.
    """
    asset_type = request.args.get("asset_type")
    if asset_type not in (None, "crypto", "stock"):
        return jsonify({"error": "asset_type must be 'crypto' or 'stock'"}), 400
    pins = repository.get_pinned_assets(asset_type)
    for p in pins:
        if p.get("pinned_at"):
            p["pinned_at"] = p["pinned_at"].isoformat()
    return jsonify({"pins": pins})


@api_bp.post("/scout/pins")
def scout_pin_add():
    body = request.get_json(silent=True) or {}
    asset_type = body.get("asset_type")
    asset_id = body.get("id")
    symbol = body.get("symbol")
    name = body.get("name")
    if asset_type not in ("crypto", "stock") or not asset_id or not symbol or not name:
        return jsonify({"error": "asset_type ('crypto'|'stock'), id, symbol and name are required"}), 400
    repository.pin_asset(asset_type, asset_id, symbol, name)
    return jsonify({"status": "pinned"})


@api_bp.delete("/scout/pins")
def scout_pin_remove():
    asset_type = request.args.get("asset_type")
    asset_id = request.args.get("id")
    if asset_type not in ("crypto", "stock") or not asset_id:
        return jsonify({"error": "asset_type ('crypto'|'stock') and id are required"}), 400
    repository.unpin_asset(asset_type, asset_id)
    return jsonify({"status": "unpinned"})


@api_bp.get("/summary")
def summary():
    coin_id = request.args.get("coin_id", config.COIN_ID)
    interval = request.args.get("interval", "1h")
    if interval not in SUPPORTED_INTERVALS:
        return jsonify({"error": f"interval must be one of {sorted(SUPPORTED_INTERVALS)}"}), 400
    try:
        result = run_combined_summary(coin_id, interval)
    except InsufficientDataError as exc:
        return jsonify({"error": str(exc)}), 409
    for item in result.get("recent_news", []):
        if item.get("published_at"):
            item["published_at"] = item["published_at"].isoformat()
    return jsonify(result)
