"""HTTP routes. Thin: validation + delegating to database/repository or
api/services, then serializing to JSON.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

import config
from api.services import InsufficientDataError, run_combined_summary, run_prediction
from data_collector.market_data import SUPPORTED_INTERVALS, MarketDataError, get_current_price
from database import repository

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.get("/price/current")
def price_current():
    coin_id = request.args.get("coin_id", config.COIN_ID)
    try:
        return jsonify(get_current_price(coin_id))
    except MarketDataError as exc:
        return jsonify({"error": str(exc)}), 502


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
