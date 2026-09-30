"""The actual work each scheduled job performs. Kept separate from the
APScheduler wiring (run_scheduler.py) so jobs can also be triggered
on-demand (e.g. from a CLI or a Flask admin route) without needing a
running scheduler.
"""
from __future__ import annotations

import logging

import config
from data_collector import market_data, news_collector
from database import repository
from nlp import sentiment

logger = logging.getLogger(__name__)

# Intervals the "Precizia predicției" accuracy charts let you pick between
# (see frontend/web/app.js's accuracyIntervalGroup). Recorded here on a
# fixed schedule rather than relying on run_prediction() being called
# incidentally by a live dashboard request: that only saved a snapshot for
# whatever interval a browser tab happened to have open at the time, so
# 15m/1h accuracy history only accumulated by coincidence instead of
# continuously.
_PREDICTION_SNAPSHOT_INTERVALS = ("15m", "1h", "1d")

# Sklearn is the only backend now (see ml/price_predictor.py's module
# docstring for why LSTM was removed) - a single-element tuple rather than
# a hardcoded call so a future second backend just means adding to this list.
_PREDICTION_SNAPSHOT_BACKENDS = ("sklearn",)


def collect_market_data_job(coin_id: str = config.COIN_ID) -> None:
    """Fetch current price + latest candles for every supported interval, persist to Mongo."""
    try:
        snapshot = market_data.get_current_price(coin_id)
        repository.save_price_snapshot(coin_id, snapshot)
        logger.info("Saved price snapshot for %s: %s", coin_id, snapshot.get("price"))
    except Exception:
        logger.exception("Failed to collect current price for %s", coin_id)

    for interval in market_data.SUPPORTED_INTERVALS:
        try:
            candles = market_data.get_ohlc_candles(interval=interval, coin_id=coin_id)
            repository.save_candles(coin_id, interval, candles)
            logger.info("Saved %d candles (%s) for %s", len(candles), interval, coin_id)
        except Exception:
            logger.exception("Failed to collect %s candles for %s", interval, coin_id)


def collect_news_job() -> None:
    """Fetch recent articles, score sentiment, persist new ones (dedup by URL)."""
    try:
        articles = news_collector.collect_news()
        scored = [sentiment.score_article(a) for a in articles]
        inserted = repository.save_news_items(scored)
        logger.info("Collected %d articles, %d new", len(scored), inserted)
    except Exception:
        logger.exception("Failed to collect news")


def record_prediction_snapshots_job(coin_id: str = config.COIN_ID) -> None:
    """Persist a fresh one-step-ahead forecast for each (interval, backend,
    model_variant) combo the accuracy charts track, independent of whether
    anyone has the dashboard open right now. Both model variants (see
    ml.price_predictor.MODEL_VARIANTS) are recorded every cycle - not just
    whichever one a user happens to have the dashboard's model toggle set
    to - so the "other model" comparison line on the accuracy chart always
    has real, continuously-tracked history to show, not just whatever
    sparse points a toggle click happened to produce. Imported lazily to
    avoid a module-level import cycle risk with api.services (which itself
    imports from data_collector) - there isn't actually one today, but
    this keeps it that way regardless of how either module's imports evolve.
    """
    from api.services import MODEL_VARIANTS, InsufficientDataError, run_prediction

    for interval in _PREDICTION_SNAPSHOT_INTERVALS:
        for backend in _PREDICTION_SNAPSHOT_BACKENDS:
            for model_variant in MODEL_VARIANTS:
                try:
                    run_prediction(coin_id, interval=interval, steps=1, backend=backend, model_variant=model_variant)
                except InsufficientDataError:
                    logger.info(
                        "Skipping prediction snapshot for %s/%s/%s/%s - not enough candles yet",
                        coin_id, interval, backend, model_variant,
                    )
                except Exception:
                    logger.exception(
                        "Failed to record prediction snapshot for %s/%s/%s/%s", coin_id, interval, backend, model_variant
                    )


def backfill_prediction_history(coin_id: str = config.COIN_ID, interval: str = "1d", max_days: int = 60) -> int:
    """Walk-forward backtest: for each of the last `max_days` real candles
    that has no prediction recorded for it yet, retrain a model using only
    candles that existed *strictly before* that candle and save what it
    would have predicted then - a genuine point-in-time backtest, not a
    live prediction recorded late.

    This is the only way to close a gap in the "Precizia predicției" chart
    left by a stretch where nothing was running: a live prediction for day
    D can only ever be recorded by something running on day D-1, and once
    that window has passed there's no way to get a *live* one for D
    anymore - "today" predicting D after the fact would be cheating (D's
    real candle would already be sitting right there to peek at). Walk-
    forward backtesting sidesteps that by re-deriving each day's forecast
    from a candle history truncated to exactly what would have been known
    at the time, which is exactly what a live run on that day would have
    trained on anyway.

    Not wired into the scheduler - unlike record_prediction_snapshots_job,
    this doesn't need to run continuously (nothing new to backfill once
    the app has been running normally), so it's meant to be triggered
    manually after a gap (e.g. `python -c "from data_collector.jobs import
    backfill_prediction_history as b; b()"`) rather than on a timer.

    Returns how many days were backfilled.
    """
    from api.services import MIN_CANDLES_FOR_TRAINING, _btc_close_series, _sentiment_by_date
    from data_collector.defillama_client import get_ethereum_tvl_by_date
    from ml.combined_predictor import compute_confidence
    from ml.features import build_feature_frame
    from ml.price_predictor import PricePredictor

    already_predicted = repository.get_predicted_target_timestamps(coin_id, interval)
    # One batched fetch instead of one per candidate: each candidate's
    # training history is "up to 1000 candles strictly before it", and
    # consecutive candidates' windows overlap almost completely (day D's
    # window and D-1's share ~999 candles), so a single fetch of the most
    # recent (1000 + max_days) candles covers every candidate's window and
    # is sliced in memory below instead of re-querying Mongo per day.
    all_candles = repository.get_candles(coin_id, interval, limit=1000 + max_days)
    candidates = all_candles[-max_days:] if len(all_candles) > max_days else all_candles
    candidates_offset = len(all_candles) - len(candidates)
    sentiment_map = _sentiment_by_date(lookback_days=max_days + 5)
    tvl_by_date = get_ethereum_tvl_by_date() if coin_id == "ethereum" else {}
    btc_close = _btc_close_series(interval, coin_id)

    backfilled = 0
    for i, candle in enumerate(candidates):
        target_iso = repository.iso_utc(candle["timestamp"])
        if target_iso in already_predicted:
            continue

        global_idx = candidates_offset + i
        history = all_candles[max(0, global_idx - 1000):global_idx]
        if len(history) < MIN_CANDLES_FOR_TRAINING:
            logger.info("Skipping backfill for %s/%s target %s - not enough prior history", coin_id, interval, target_iso)
            continue

        try:
            df = build_feature_frame(history, sentiment_map, btc_close=btc_close, tvl_by_date=tvl_by_date)
            train_df = df.dropna()
            if len(train_df) < 30:
                continue
            predictor = PricePredictor(backend="sklearn", ensemble_size=1)
            predictor.fit(train_df)
            predictions = predictor.predict(df, steps=1, interval=interval)
        except Exception:
            logger.exception("Backfill training failed for %s/%s target %s", coin_id, interval, target_iso)
            continue

        if not predictions:
            continue
        point = predictions[0]
        predictions_field = [{
            "timestamp": repository.iso_utc(point["timestamp"]),
            "predicted_price": round(point["predicted_price"], 2),
            "lower": round(point["lower"], 2),
            "upper": round(point["upper"], 2),
        }]
        sentiment_avg = float(df["sentiment"].mean()) if "sentiment" in df.columns else 0.0
        importances = predictor.get_feature_importance()
        sentiment_contribution_pct = (
            round(importances["sentiment"] * 100, 1) if importances and "sentiment" in importances else None
        )
        # Same document shape run_prediction() writes for a live snapshot
        # (api/services.py) - a backfilled row that's missing fields a live
        # one always has would be indistinguishable from a live one to any
        # future reader, and would KeyError/blank out on those fields.
        repository.save_prediction(coin_id, {
            "interval": interval,
            "steps": 1,
            "backend": "sklearn",
            "used_sentiment": True,
            "sentiment_avg": sentiment_avg,
            "sentiment_contribution_pct": sentiment_contribution_pct,
            "trained_ago_seconds": 0.0,
            "confidence": compute_confidence(predictions_field, sentiment_avg),
            "last_known_price": float(df["close"].iloc[-1]),
            "predictions": predictions_field,
        })
        backfilled += 1

    logger.info("Backfilled %d/%d candidate days for %s/%s", backfilled, len(candidates), coin_id, interval)
    return backfilled


def backfill_signal_history(
    coin_id: str = config.COIN_ID, interval: str = "1h", steps: int = 24, max_candidates: int = 50,
) -> int:
    """Walk-forward backtest of the dashboard's "Trading signal" card
    itself (ml/trading_signal.py), not just the raw price prediction: for
    each of the last `max_candidates` real candles old enough that the
    outcome `steps` periods later is already known, retrain a model using
    only data that existed at that point, work out what the signal card
    would have shown (same RSI/MACD/trend/sentiment/confidence rule), and
    record whether that call actually played out.

    This is the only honest way to answer "does this signal actually
    work" - checking it against *live* calls would take steps*interval of
    real time per data point (24 hours per point at the default 1h/24-step
    setting) to accumulate any history at all, and idempotent (skips
    already-recorded entry timestamps, see
    repository.get_signal_recorded_timestamps) - safe to call repeatedly
    (e.g. on every collector cycle) to grow the track record over time,
    the same pattern backfill_prediction_history above uses.

    Returns how many candidates were newly recorded.
    """
    from api.services import MIN_CANDLES_FOR_TRAINING, _btc_close_series, _sentiment_by_date
    from ml.combined_predictor import compute_confidence
    from ml.features import build_feature_frame
    from ml.price_predictor import PricePredictor
    from ml.trading_signal import compute_signal_label

    already_recorded = repository.get_signal_recorded_timestamps(coin_id, interval)
    all_candles = repository.get_candles(coin_id, interval, limit=1000 + max_candidates + steps)
    n = len(all_candles)
    # Only candidates whose real outcome (steps candles later) has already
    # happened - a candidate from the last `steps` periods is still "live",
    # its outcome isn't knowable yet.
    usable_end = n - steps
    start = max(0, usable_end - max_candidates)
    if usable_end <= start:
        return 0

    sentiment_map = _sentiment_by_date(lookback_days=max_candidates + steps + 5)
    btc_close = _btc_close_series(interval, coin_id)

    recorded = 0
    for i in range(start, usable_end):
        entry_candle = all_candles[i]
        entry_iso = repository.iso_utc(entry_candle["timestamp"])
        if entry_iso in already_recorded:
            continue

        history = all_candles[max(0, i + 1 - 1000): i + 1]
        if len(history) < MIN_CANDLES_FOR_TRAINING:
            continue
        try:
            df = build_feature_frame(history, sentiment_map, btc_close=btc_close)
            train_df = df.dropna()
            if len(train_df) < 30:
                continue
            predictor = PricePredictor(backend="sklearn", ensemble_size=1)
            predictor.fit(train_df)
            predictions = predictor.predict(df, steps=steps, interval=interval)
        except Exception:
            logger.exception("Signal backfill training failed for %s/%s at %s", coin_id, interval, entry_iso)
            continue
        if not predictions:
            continue

        last_close = float(df["close"].iloc[-1])
        rsi = df["rsi_14"].iloc[-1]
        macd = df["macd"].iloc[-1]
        macd_signal_val = df["macd_signal"].iloc[-1]
        sentiment_avg = float(df["sentiment"].mean()) if "sentiment" in df.columns else 0.0
        trend_pct = (predictions[-1]["predicted_price"] - last_close) / last_close * 100 if last_close else 0.0
        rsi_val = float(rsi) if rsi == rsi else None  # NaN check
        macd_bullish = bool(macd > macd_signal_val) if macd == macd and macd_signal_val == macd_signal_val else None
        confidence = compute_confidence(predictions, sentiment_avg)
        signal = compute_signal_label(trend_pct, rsi_val, macd_bullish, sentiment_avg, confidence)

        target_candle = all_candles[i + steps]
        actual_price = float(target_candle["close"])
        correct = None
        if signal == "buy":
            correct = actual_price > last_close
        elif signal == "sell":
            correct = actual_price < last_close

        repository.save_signal_record(coin_id, {
            "interval": interval,
            "steps": steps,
            "timestamp": entry_candle["timestamp"],
            "signal": signal,
            "entry_price": round(last_close, 2),
            "target_timestamp": target_candle["timestamp"],
            "actual_price": round(actual_price, 2),
            "correct": correct,
            "confidence": confidence,
            "trend_pct": round(trend_pct, 3),
        })
        recorded += 1

    logger.info(
        "Signal backfill complete for %s/%s: %d/%d candidates newly recorded",
        coin_id, interval, recorded, usable_end - start,
    )
    return recorded


def run_all_once() -> None:
    """Convenience for manual runs / first-time seeding.

    Also self-heals the "1d" accuracy chart's history on every startup:
    backfill_prediction_history is idempotent and only ever trains for
    days that are still actually missing, so this is cheap (a no-op) on
    a normal restart and only does real work right after a stretch where
    nothing was running - closing that gap automatically instead of
    needing someone to remember to run it by hand.
    """
    collect_market_data_job()
    collect_news_job()
    record_prediction_snapshots_job()
    backfill_prediction_history(interval="1d", max_days=60)
    backfill_signal_history(interval="1h", steps=24, max_candidates=20)
