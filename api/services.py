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
from evaluation.recorder import record_prediction
from ml.combined_predictor import compute_confidence, generate_narrative_summary
from ml.features import build_feature_frame
from ml.price_predictor import _INTERVAL_TIMEDELTA, PricePredictor
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

# One lock per (coin, interval, use_sentiment, backend) key, created lazily.
# Serializes concurrent cold requests for the *same* key (see
# _get_trained_model's docstring) without serializing the whole cache -
# see the same docstring for the incident this replaced.
_key_locks: dict[tuple, threading.Lock] = {}


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


PREDICTION_BACKENDS = ("sklearn",)

# See ml.price_predictor.MODEL_VARIANTS for what these actually change -
# exposed end-to-end (here, the routes, and the dashboard's own model
# toggle) once tuning the GBR hyperparameters turned out to be a genuine
# trade-off (better price MAPE, worse direction accuracy and a visibly
# flatter forecast in calm markets) rather than a strict win, so both
# stay user-selectable instead of picking one side for everyone.
MODEL_VARIANTS = ("tuned", "legacy")


def _get_trained_model(
    coin_id: str, interval: str, use_sentiment: bool, backend: str | None = None, model_variant: str = "tuned",
):
    """Return (fitted predictor, feature frame, sentiment_avg, trained_at), from cache when fresh.

    The per-key locking below (rather than one lock for the whole
    cache-check-then-train-then-write section) dates from when the model
    selector could also pick LSTM/ensemble (since removed - see
    ml/price_predictor.py's module docstring for why): a single slow cold
    training (10-13s) blocked every other prediction request app-wide for
    its entire duration, including ones for already-warm, completely
    unrelated keys that should have returned in well under a second
    (confirmed directly - a warm sklearn request fired mid-training took
    8.2s of a 9.9s unrelated training run, instead of its usual ~0.25s).
    Kept even with sklearn as the only backend now: it's still correct and
    cheap, and re-adding a second backend later wouldn't need to rediscover
    this.  The fast path below only ever holds the small, dict-only
    `_prediction_cache_lock` - training itself is guarded by a per-key
    lock, so a slow cold train for one key can only ever block *other
    requests for that same key* (the original problem this locking existed
    to solve - two concurrent requests both training the same cold key
    independently), never unrelated ones.

    `backend` is part of the cache key for the same forward-compatibility
    reason, and defaults to config.PREDICTION_BACKEND when not given
    explicitly.
    """
    backend = backend or config.PREDICTION_BACKEND
    if backend not in PREDICTION_BACKENDS:
        # api/routes.py validates this for every HTTP entry point before it
        # ever reaches here, but a direct caller (data_collector.jobs, a
        # future script) bypasses that - fail with a clear message instead
        # of an opaque error surfacing deep inside PricePredictor.fit().
        raise ValueError(f"Unknown prediction backend {backend!r}; must be one of {PREDICTION_BACKENDS}")
    if model_variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model_variant {model_variant!r}; must be one of {MODEL_VARIANTS}")
    key = (coin_id, interval, use_sentiment, backend, model_variant)

    def _fresh_cached():
        cached = _prediction_cache.get(key)
        if cached and time.monotonic() - cached["cached_at"] < _PREDICTION_CACHE_TTL_SECONDS:
            return cached["predictor"], cached["df"], cached["sentiment_avg"], cached["cached_at"]
        return None

    with _prediction_cache_lock:
        hit = _fresh_cached()
    if hit:
        return hit

    with _prediction_cache_lock:
        key_lock = _key_locks.setdefault(key, threading.Lock())

    with key_lock:
        # Re-check: another thread may have trained this exact key while
        # we were waiting on key_lock.
        with _prediction_cache_lock:
            hit = _fresh_cached()
        if hit:
            return hit

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

        predictor = PricePredictor(backend=backend, model_variant=model_variant)
        predictor.fit(train_df)
        sentiment_avg = sum(sentiment_map.values()) / len(sentiment_map) if sentiment_map else 0.0
        now = time.monotonic()

        with _prediction_cache_lock:
            _prediction_cache[key] = {
                "predictor": predictor,
                "df": df,
                "sentiment_avg": sentiment_avg,
                "cached_at": now,
            }
        return predictor, df, sentiment_avg, now


# The dashboard only ever needs a handful of (coin, interval) combos: the
# chart's default interval, the two outlook legs, and 15m (tracked by the
# accuracy chart's interval selector, and by data_collector.jobs's
# recurring snapshot job - without it here, every snapshot would pay a
# cold-train instead of hitting the warm cache). Keeping this list small
# and explicit (rather than "whatever was last requested") means warm-up
# work stays cheap and bounded regardless of how many intervals
# SUPPORTED_INTERVALS grows to.
_WARM_UP_INTERVALS = ("15m", "1h", "1d", "1w")

# Every backend the model selector can pick (see PREDICTION_BACKENDS) -
# a single-element tuple now that sklearn is the only backend, kept for
# when a second one gets added rather than hardcoding just "sklearn" here.
_WARM_UP_BACKENDS = PREDICTION_BACKENDS


def warm_prediction_cache(coin_id: str = config.COIN_ID) -> None:
    """Proactively (re)train and cache the models a fresh dashboard load
    needs, so a real user's request always hits a warm cache instead of
    paying the training cost itself.

    Meant to be called on a background timer (see run_api.py). Each call
    checks freshness per (interval, backend) combo the same way a real
    request would (_get_trained_model's own TTL check) - only combos whose
    cache has actually expired since the last cycle pay to retrain, not
    every combo on every call - so the caller only controls the *ceiling*
    on how stale a combo can get, not how often real training happens.
    """
    for interval in _WARM_UP_INTERVALS:
        for backend in _WARM_UP_BACKENDS:
            for model_variant in MODEL_VARIANTS:
                try:
                    _get_trained_model(coin_id, interval, True, backend, model_variant)
                except InsufficientDataError:
                    logger.info(
                        "Skipping cache warm-up for %s/%s/%s/%s - not enough candles yet",
                        coin_id, interval, backend, model_variant,
                    )
                except Exception:
                    logger.exception("Cache warm-up failed for %s/%s/%s/%s", coin_id, interval, backend, model_variant)


def run_prediction(
    coin_id: str = config.COIN_ID,
    interval: str = "1h",
    steps: int = config.PREDICTION_HORIZON_HOURS,
    use_sentiment: bool = True,
    backend: str | None = None,
    model_variant: str = "tuned",
) -> dict:
    """Forecast `steps` periods ahead using a (cached, per-interval) trained model.

    `backend` (see ml.price_predictor.PREDICTION_BACKENDS) defaults to
    config.PREDICTION_BACKEND when None. `model_variant` ("tuned" | "legacy",
    see MODEL_VARIANTS) picks which GradientBoostingRegressor hyperparameters
    train with - see ml.price_predictor's module-level comment for why both
    stay selectable instead of just one replacing the other.

    Returns predictions + a confidence score derived from interval width
    and sentiment/direction agreement. Raises InsufficientDataError if the
    collector hasn't gathered enough history yet.
    """
    backend = backend or config.PREDICTION_BACKEND
    predictor, df, sentiment_avg, trained_at = _get_trained_model(
        coin_id, interval, use_sentiment, backend, model_variant
    )
    predictions = predictor.predict(df, steps=steps, interval=interval)
    confidence = compute_confidence(predictions, sentiment_avg)

    # Real, model-derived answer to "how much does sentiment actually
    # factor into this" (see PricePredictor.get_feature_importance).
    # trained_ago_seconds is concrete evidence that the "automatic
    # recalibration" the sentiment toggle implies is actually happening -
    # every cache expiry (_PREDICTION_CACHE_TTL_SECONDS) retrains from
    # scratch on whatever sentiment/price data has arrived since.
    importances = predictor.get_feature_importance() if use_sentiment else None
    sentiment_contribution_pct = (
        round(importances["sentiment"] * 100, 1) if importances and "sentiment" in importances else None
    )

    result = {
        "coin_id": coin_id,
        "interval": interval,
        "steps": steps,
        "backend": backend,
        "model_variant": model_variant,
        "used_sentiment": use_sentiment,
        "sentiment_avg": sentiment_avg,
        "sentiment_contribution_pct": sentiment_contribution_pct,
        "trained_ago_seconds": round(time.monotonic() - trained_at, 1),
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
    # Evaluation log (evaluation/): records what the model said so it can be
    # scored against the real price later. Never raises.
    record_prediction(result, df, sentiment_avg)
    return result


def _outlook_leg(
    coin_id: str, interval: str, use_sentiment: bool, backend: str | None = None, model_variant: str = "tuned",
) -> dict:
    prediction = run_prediction(
        coin_id=coin_id, interval=interval, steps=1, use_sentiment=use_sentiment, backend=backend,
        model_variant=model_variant,
    )
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


def run_outlook(
    coin_id: str = config.COIN_ID, use_sentiment: bool = True, backend: str | None = None,
    model_variant: str = "tuned",
) -> dict:
    """Quick daily (next 1d candle) + weekly (next 1w candle) % price outlook.

    Independent of whatever interval/steps the chart is showing - always
    the single next-day and next-week prediction, for the +/-% tiles.
    """
    return {
        "coin_id": coin_id,
        "daily": _outlook_leg(coin_id, "1d", use_sentiment, backend, model_variant),
        "weekly": _outlook_leg(coin_id, "1w", use_sentiment, backend, model_variant),
    }


def get_prediction_accuracy(
    coin_id: str = config.COIN_ID, interval: str = "1d", days: int | None = None, limit: int | None = None,
    backend: str | None = None, model_variant: str | None = None,
) -> dict:
    """Compare each one-step-ahead forecast recorded over the last `days`
    against the actual close that later materialized for that same candle,
    so the dashboard can show how closely the model has tracked reality
    over time (a month, a year), not just its current live forecast.

    `limit`, when given, trims the result to the most recent N verified
    predictions - "the last 100 predictions at 1h" reads naturally for a
    fast interval, where "the last 30 days" would mean many hundreds of
    points; `days` alone (limit=None) is what the monthly/yearly accuracy
    chart uses, where a fixed calendar window is what makes sense. If both
    are omitted, `days` defaults to 30. When only `limit` is given, `days`
    is sized from the interval's own duration so the lookback window is
    just wide enough to contain that many predictions even for slow
    intervals (1w) without over-fetching for fast ones (15m).
    """
    interval_delta = _INTERVAL_TIMEDELTA.get(interval, timedelta(days=1))
    if days is None:
        if limit is None:
            days = 30
        else:
            # 3x safety margin: predictions aren't recorded on a perfectly
            # even cadence (model warm-up gaps, missed scheduler runs), so
            # a bare limit*interval window would sometimes come up short.
            days = max(1, min(730, int((interval_delta * limit * 3) / timedelta(days=1)) + 1))

    since = datetime.now(timezone.utc) - timedelta(days=days)
    records = repository.get_prediction_accuracy_series(coin_id, interval, since, backend=backend, model_variant=model_variant)

    # How many *candles* (not days) the `days` lookback window holds
    # depends entirely on the interval's own granularity - 1 candle/day
    # only holds for interval="1d". Getting this wrong under-fetches
    # candles for anything faster (e.g. "1h" needs ~24x as many for the
    # same calendar window) and silently drops most of `records` below
    # for lack of a matching actual price.
    candle_limit = max(int(timedelta(days=days) / interval_delta) + 30, 60)
    candles = repository.get_candles(coin_id, interval, limit=candle_limit)
    actual_by_timestamp = {repository.iso_utc(c["timestamp"]): c["close"] for c in candles}

    points = []
    for record in records:
        actual = actual_by_timestamp.get(record["target"])
        if actual is None:
            continue
        error_pct = (record["predicted_price"] - actual) / actual * 100 if actual else None
        points.append({
            "timestamp": record["target"],
            "predicted_price": round(record["predicted_price"], 2),
            "actual_price": round(actual, 2),
            "error_pct": round(error_pct, 2) if error_pct is not None else None,
        })

    if limit is not None:
        points = points[-limit:]
    errors = [abs(p["error_pct"]) for p in points if p["error_pct"] is not None]

    return {
        "coin_id": coin_id,
        "interval": interval,
        "backend": backend,
        "days": days,
        "limit": limit,
        "count": len(points),
        "points": points,
        "mape": round(sum(errors) / len(errors), 2) if errors else None,
    }


def run_combined_summary(
    coin_id: str = config.COIN_ID, backend: str | None = None, lang: str = "en", model_variant: str = "tuned",
) -> dict:
    """Prediction + narrative summary + supporting news, for the dashboard's
    daily digest - always the next-day (interval="1d", steps=1) forecast,
    same as the "24h prediction" outlook tile (run_outlook's daily leg),
    regardless of whatever interval the main chart happens to be showing.

    Before this, the narrative used the chart's *own* selected interval
    plus a fixed 24-step horizon - a real, reported bug: at the dashboard's
    default "1h" that's a 24-*hour* forecast from an hourly model (which
    can - and did - disagree in direction with the 1d model's own 24h
    outlook, both shown side by side under "24h"), and at "1d" or "1w"
    it silently became a 24-day or 24-week forecast mislabeled as "daily".
    """
    prediction = run_prediction(coin_id=coin_id, interval="1d", steps=1, backend=backend, model_variant=model_variant)
    recent_news = repository.get_recent_news(limit=10)

    if not prediction["predictions"]:
        # predictor.predict() can return zero steps (not raise) when the
        # most recent candle lacks enough rolling history for even one
        # step (see PricePredictor._iterate_recursive) - treat that the
        # same as "not enough data yet" rather than crashing on [0] below.
        raise InsufficientDataError(
            f"The 1d model could not generate a prediction for {coin_id} right now "
            f"(insufficient history for the most recent candle)."
        )

    point = prediction["predictions"][0]
    narrative = generate_narrative_summary(
        prediction["last_known_price"], point["predicted_price"], prediction["sentiment_avg"], recent_news, lang=lang,
    )

    return {**prediction, "narrative": narrative, "recent_news": recent_news}
