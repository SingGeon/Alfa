"""Fills in what really happened for each logged prediction.

Runs every minute (see evaluation/jobs.py). For every `pending` prediction
it appends the real close of each forecast candle that has closed since the
last run, and once the last forecast candle has closed (resolved_at) it
computes the error, the direction hit and marks the prediction completed.

Price source: Binance klines (same exchange and candle convention the
models train on - a forecast point at timestamp T is the close of the
candle that *opens* at T). If Binance fails after retries, or returns a
hole for a candle that has already closed, CoinGecko's price history is
used for the missing points (last price at or before that candle's close).
A prediction whose final price still can't be found config.EVAL_EXPIRE_HOURS
after it resolved is marked `expired` rather than left pending forever.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import config
from data_collector.http_utils import ExternalAPIError, get_json
from evaluation import storage
from evaluation.storage import interval_delta, parse_iso, to_iso

logger = logging.getLogger(__name__)

_BINANCE_PAGE_LIMIT = 1000


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def compute_errors(price_at_prediction: float, predicted_final_price: float, actual_final_price: float) -> dict:
    """abs_error in USD, pct_error signed (positive = model predicted too
    high), direction_correct = model called the same side of
    price_at_prediction (up / down / flat) as what really happened."""
    abs_error = abs(predicted_final_price - actual_final_price)
    pct_error = (predicted_final_price - actual_final_price) / actual_final_price * 100 if actual_final_price else 0.0
    direction_correct = _sign(predicted_final_price - price_at_prediction) == _sign(actual_final_price - price_at_prediction)
    return {
        "abs_error": round(abs_error, 4),
        "pct_error": round(pct_error, 4),
        "direction_correct": direction_correct,
    }


def fetch_binance_candles(interval: str, start: datetime, end: datetime, now: datetime) -> dict[str, dict]:
    """{candle open time (ISO): {"open", "high", "low", "close"}} for every
    *closed* candle opening in [start, end]. Pages through Binance's
    1000-candle limit."""
    closes: dict[str, dict] = {}
    now_ms = int(now.timestamp() * 1000)
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while cursor <= end_ms:
        raw = get_json(
            f"{config.BINANCE_API_BASE}/klines",
            params={
                "symbol": config.BINANCE_SYMBOL, "interval": interval,
                "startTime": cursor, "endTime": end_ms, "limit": _BINANCE_PAGE_LIMIT,
            },
            max_retries=3, timeout=10,
        )
        if not raw:
            break
        for k in raw:
            if int(k[6]) < now_ms:  # k[6] = close time; skip the still-forming candle
                closes[to_iso(datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc))] = {
                    "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
                }
        if len(raw) < _BINANCE_PAGE_LIMIT:
            break
        cursor = int(raw[-1][0]) + 1
    return closes


def fetch_coingecko_candles(interval: str, timestamps: list[str]) -> dict[str, dict]:
    """Fallback: approximate each candle's close as the last CoinGecko price
    point at or before that candle's close time. CoinGecko's granularity
    is automatic (5-minute for a range under a day, hourly up to 90 days),
    so a point is only accepted within a tolerance of the close time. Only
    a close is known this way - no open/high/low."""
    if not timestamps:
        return {}
    delta = interval_delta(interval)
    close_times = {ts: parse_iso(ts) + delta for ts in timestamps}
    tolerance = max(delta, timedelta(hours=1))
    start = min(close_times.values()) - tolerance
    end = max(close_times.values())
    data = get_json(
        f"{config.COINGECKO_API_BASE}/coins/{config.COIN_ID}/market_chart/range",
        params={"vs_currency": config.VS_CURRENCY, "from": int(start.timestamp()), "to": int(end.timestamp())},
        max_retries=2, timeout=10,
    )
    points = sorted(
        (datetime.fromtimestamp(ms / 1000, tz=timezone.utc), float(price)) for ms, price in data.get("prices", [])
    )
    closes: dict[str, dict] = {}
    for ts, close_time in close_times.items():
        best = None
        for point_time, price in points:
            if point_time > close_time:
                break
            best = (point_time, price)
        if best and close_time - best[0] <= tolerance:
            closes[ts] = {"close": best[1]}
    return closes


def _fetch_candles(interval: str, needed: set[str], now: datetime) -> dict[str, dict]:
    ordered = sorted(needed)
    closes: dict[str, dict] = {}
    try:
        closes = fetch_binance_candles(interval, parse_iso(ordered[0]), parse_iso(ordered[-1]), now)
    except ExternalAPIError as exc:
        logger.warning("Binance klines failed for %s evaluation (%s), trying CoinGecko", interval, exc)
    missing = [ts for ts in ordered if ts not in closes]
    if missing:
        try:
            closes.update(fetch_coingecko_candles(interval, missing))
        except (ExternalAPIError, KeyError, TypeError, ValueError) as exc:
            logger.warning("CoinGecko fallback failed for %s evaluation (%s)", interval, exc)
    return closes


def complete_pending_predictions(now: datetime | None = None) -> dict:
    """One evaluation pass. Returns {"updated", "completed", "expired",
    "updated_ids", "completed_ids"} for logging and so the caller can redraw
    the visual snapshot of whatever gained real candles, and chart what
    just completed."""
    now = now or datetime.now(timezone.utc)
    pending = storage.get_pending()
    result = {"updated": 0, "completed": 0, "expired": 0, "updated_ids": [], "completed_ids": []}
    if not pending:
        return result

    # Which closed candles each prediction still lacks, grouped by interval
    # so one API call (or a few pages) covers every prediction at that interval.
    needed_by_interval: dict[str, set[str]] = defaultdict(set)
    needed_by_prediction: dict[int, list[str]] = {}
    for pred in pending:
        delta = interval_delta(pred["interval"])
        have = {p["timestamp"] for p in pred["actual_path"]}
        needed = [
            p["timestamp"] for p in pred["predicted_path"]
            if p["timestamp"] not in have and parse_iso(p["timestamp"]) + delta <= now
        ]
        needed_by_prediction[pred["id"]] = needed
        needed_by_interval[pred["interval"]].update(needed)

    closes_by_interval = {
        interval: _fetch_candles(interval, needed, now) for interval, needed in needed_by_interval.items() if needed
    }

    for pred in pending:
        closes = closes_by_interval.get(pred["interval"], {})
        # "price" is the close (what the error is computed on); open/high/low
        # are kept when known so the visual snapshot can draw real candles.
        found = [
            {"timestamp": ts, "price": closes[ts]["close"], **{k: v for k, v in closes[ts].items() if k != "close"}}
            for ts in needed_by_prediction[pred["id"]] if ts in closes
        ]
        actual_path = sorted(pred["actual_path"] + found, key=lambda p: p["timestamp"])
        resolved_at = parse_iso(pred["resolved_at"])

        if now < resolved_at:
            if found:
                storage.update_actual_path(pred["id"], actual_path)
                result["updated"] += 1
                result["updated_ids"].append(pred["id"])
            continue

        final = next((p["price"] for p in actual_path if p["timestamp"] == pred["target_time"]), None)
        if final is not None:
            errors = compute_errors(pred["price_at_prediction"], pred["predicted_final_price"], final)
            storage.complete_prediction(pred["id"], actual_path=actual_path, actual_final_price=final, completed_at=now, **errors)
            result["completed"] += 1
            result["completed_ids"].append(pred["id"])
        elif now >= resolved_at + timedelta(hours=config.EVAL_EXPIRE_HOURS):
            logger.warning(
                "Prediction %s (%s %s) expired: no real price found for %s",
                pred["id"], pred["interval"], pred["model_name"], pred["target_time"],
            )
            storage.expire_prediction(pred["id"], actual_path)
            result["expired"] += 1
        elif found:
            storage.update_actual_path(pred["id"], actual_path)
            result["updated"] += 1
            result["updated_ids"].append(pred["id"])

    return result
