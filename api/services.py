"""Business logic shared between API routes - keeps routes.py thin.

This is the "combined prediction module": it wires together candles from
the database, the sentiment scores from collected news, the PricePredictor
(ml.price_predictor) and the confidence/narrative layer
(ml.combined_predictor).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import config
from data_collector import market_data
from data_collector.defillama_client import get_ethereum_tvl_by_date
from data_collector.http_utils import ExternalAPIError
from database import repository
from ml.combined_predictor import compute_confidence, generate_narrative_summary
from ml.features import build_feature_frame
from ml.price_predictor import PricePredictor
from nlp.sentiment import aggregate_daily

logger = logging.getLogger(__name__)

MIN_CANDLES_FOR_TRAINING = 40

# Training (fit) doesn't depend on `steps` - only predict() does - so the
# fitted model is cached per (coin_id, interval, use_sentiment) instead of
# retrained on every request. Without this, one 30s dashboard refresh cycle
# (chart + daily outlook + weekly outlook + summary) retrained up to 4
# separate models back to back.
#
# 900s (15min), matching the collector's own refresh cadence (see
# run_scheduler.py) - this never serves something staler than the
# underlying candle data already is, and it's deliberately generous: with
# warm_prediction_cache() keeping this refreshed from a background thread
# (see run_api.py) well before it can expire, TTL only matters as a
# fallback if that thread ever stalls - and a longer TTL degrades that
# failure into "briefly stale but still fast" instead of "every request
# pays the ~6s training cost again".
_PREDICTION_CACHE_TTL_SECONDS = 900
_prediction_cache: dict[tuple, dict] = {}
_prediction_cache_lock = threading.Lock()


class InsufficientDataError(RuntimeError):
    pass


def _sentiment_by_date(lookback_days: int = 30) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles = repository.get_news_since(since)
    return aggregate_daily(articles)


def _btc_close_series(interval: str, coin_id: str):
    """BTC's own close series at the same interval, for ml/features.py's
    btc_return_1 (see its docstring) - None (feature degrades to neutral
    0.0) if this coin already *is* BTC, or if the fetch fails.
    """
    import pandas as pd

    if coin_id == "bitcoin":
        return None
    try:
        btc_candles = market_data.get_ohlc_candles(interval=interval, coin_id="bitcoin", symbol="BTCUSDT")
    except ExternalAPIError as exc:
        logger.warning("BTC candle fetch failed (%s) - btc_return_1 will default neutral", exc)
        return None
    if not btc_candles:
        return None
    sorted_candles = sorted(btc_candles, key=lambda c: c["timestamp"])
    index = pd.to_datetime([c["timestamp"] for c in sorted_candles], utc=True)
    return pd.Series([c["close"] for c in sorted_candles], index=index)


def _get_trained_model(coin_id: str, interval: str, use_sentiment: bool):
    """Return (fitted predictor, feature frame, sentiment_avg), from cache when fresh.

    Holds the lock across the whole cache-check-then-train-then-write
    section, not just the dict access: with the old narrower lock, two
    concurrent requests for the same cold key (e.g. the dashboard's
    Promise.all firing /api/predict and /api/summary for the same interval
    at once) would both see "not cached" and both pay the full ~6s training
    cost independently. Training is CPU-bound and GIL-bound anyway - a
    thread "waiting" here isn't losing any real parallelism, just avoiding
    duplicate work. See warm_prediction_cache() for the actual fix to
    "training happens at all on a user's request" - this lock is a
    correctness backstop, not the primary latency fix.
    """
    key = (coin_id, interval, use_sentiment)
    with _prediction_cache_lock:
        now = time.monotonic()
        cached = _prediction_cache.get(key)
        if cached and now - cached["cached_at"] < _PREDICTION_CACHE_TTL_SECONDS:
            return cached["predictor"], cached["df"], cached["sentiment_avg"]

        candles = repository.get_candles(coin_id, interval, limit=1000)
        if len(candles) < MIN_CANDLES_FOR_TRAINING:
            raise InsufficientDataError(
                f"Only {len(candles)} {interval} candles stored for {coin_id}; "
                f"need >= {MIN_CANDLES_FOR_TRAINING}. Run the collector (run_scheduler.py) longer, "
                f"or seed data with `python -m data_collector.jobs`."
            )

        sentiment_map = _sentiment_by_date() if use_sentiment else {}
        btc_close = _btc_close_series(interval, coin_id)
        tvl_by_date = get_ethereum_tvl_by_date() if coin_id == "ethereum" else {}
        df = build_feature_frame(candles, sentiment_map, btc_close=btc_close, tvl_by_date=tvl_by_date)
        train_df = df.dropna()

        predictor = PricePredictor(backend=config.PREDICTION_BACKEND)
        predictor.fit(train_df)
        sentiment_avg = sum(sentiment_map.values()) / len(sentiment_map) if sentiment_map else 0.0

        _prediction_cache[key] = {
            "predictor": predictor,
            "df": df,
            "sentiment_avg": sentiment_avg,
            "cached_at": now,
        }
        return predictor, df, sentiment_avg


# The dashboard only ever needs three (coin, interval) combos: the chart's
# default interval and the two outlook legs. Keeping this list small and
# explicit (rather than "whatever was last requested") means warm-up work
# stays cheap and bounded regardless of how many intervals SUPPORTED_INTERVALS
# grows to.
_WARM_UP_INTERVALS = ("1h", "1d", "1w")


def warm_prediction_cache(coin_id: str = config.COIN_ID) -> None:
    """Proactively (re)train and cache the models a fresh dashboard load
    needs, so a real user's request always hits a warm cache instead of
    paying the ~6s-per-interval training cost itself.

    Meant to be called on a background timer (see run_api.py) - each call
    is a full retrain (no shortcut for "already warm enough"), so the
    caller controls the cadence, not this function.
    """
    for interval in _WARM_UP_INTERVALS:
        try:
            _get_trained_model(coin_id, interval, True)
        except InsufficientDataError:
            logger.info("Skipping cache warm-up for %s/%s - not enough candles yet", coin_id, interval)
        except Exception:
            logger.exception("Cache warm-up failed for %s/%s", coin_id, interval)


def run_prediction(
    coin_id: str = config.COIN_ID,
    interval: str = "1h",
    steps: int = config.PREDICTION_HORIZON_HOURS,
    use_sentiment: bool = True,
) -> dict:
    """Forecast `steps` periods ahead using a (cached, per-interval) trained model.

    Returns predictions + a confidence score derived from interval width
    and sentiment/direction agreement. Raises InsufficientDataError if the
    collector hasn't gathered enough history yet.
    """
    predictor, df, sentiment_avg = _get_trained_model(coin_id, interval, use_sentiment)
    predictions = predictor.predict(df, steps=steps, interval=interval)
    confidence = compute_confidence(predictions, sentiment_avg)

    result = {
        "coin_id": coin_id,
        "interval": interval,
        "steps": steps,
        "backend": config.PREDICTION_BACKEND,
        "used_sentiment": use_sentiment,
        "sentiment_avg": sentiment_avg,
        "confidence": confidence,
        "last_known_price": float(df["close"].iloc[-1]),
        "predictions": [
            {
                "timestamp": p["timestamp"].isoformat(),
                "predicted_price": round(p["predicted_price"], 2),
                "lower": round(p["lower"], 2),
                "upper": round(p["upper"], 2),
            }
            for p in predictions
        ],
    }
    repository.save_prediction(coin_id, result)
    return result


def _outlook_leg(coin_id: str, interval: str, use_sentiment: bool) -> dict:
    prediction = run_prediction(coin_id=coin_id, interval=interval, steps=1, use_sentiment=use_sentiment)
    last = prediction["last_known_price"]
    point = prediction["predictions"][0]
    change_pct = (point["predicted_price"] - last) / last * 100 if last else 0.0
    return {
        "interval": interval,
        "timestamp": point["timestamp"],
        "last_known_price": last,
        "predicted_price": point["predicted_price"],
        "lower": point["lower"],
        "upper": point["upper"],
        "change_pct": round(change_pct, 2),
        "confidence": prediction["confidence"],
    }


def run_outlook(coin_id: str = config.COIN_ID, use_sentiment: bool = True) -> dict:
    """Quick daily (next 1d candle) + weekly (next 1w candle) % price outlook.

    Independent of whatever interval/steps the chart is showing - always
    the single next-day and next-week prediction, for the +/-% tiles.
    """
    return {
        "coin_id": coin_id,
        "daily": _outlook_leg(coin_id, "1d", use_sentiment),
        "weekly": _outlook_leg(coin_id, "1w", use_sentiment),
    }


def run_combined_summary(coin_id: str = config.COIN_ID, interval: str = "1h") -> dict:
    """Prediction + narrative summary + supporting news, for the dashboard's daily digest."""
    prediction = run_prediction(coin_id=coin_id, interval=interval)
    recent_news = repository.get_recent_news(limit=10)

    predictions_for_narrative = [
        {"predicted_price": p["predicted_price"]} for p in prediction["predictions"]
    ]
    narrative = generate_narrative_summary(predictions_for_narrative, prediction["sentiment_avg"], recent_news)

    return {**prediction, "narrative": narrative, "recent_news": recent_news}
