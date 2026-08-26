"""Business logic shared between API routes - keeps routes.py thin.

This is the "combined prediction module": it wires together candles from
the database, the sentiment scores from collected news, the PricePredictor
(ml.price_predictor) and the confidence/narrative layer
(ml.combined_predictor).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import config
from database import repository
from ml.combined_predictor import compute_confidence, generate_narrative_summary
from ml.features import build_feature_frame
from ml.price_predictor import PricePredictor
from nlp.sentiment import aggregate_daily

logger = logging.getLogger(__name__)

MIN_CANDLES_FOR_TRAINING = 40


class InsufficientDataError(RuntimeError):
    pass


def _sentiment_by_date(lookback_days: int = 30) -> dict:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    articles = repository.get_news_since(since)
    return aggregate_daily(articles)


def run_prediction(
    coin_id: str = config.COIN_ID,
    interval: str = "1h",
    steps: int = config.PREDICTION_HORIZON_HOURS,
    use_sentiment: bool = True,
) -> dict:
    """Train on stored candle history and forecast `steps` periods ahead.

    Returns predictions + a confidence score derived from interval width
    and sentiment/direction agreement. Raises InsufficientDataError if the
    collector hasn't gathered enough history yet.
    """
    candles = repository.get_candles(coin_id, interval, limit=1000)
    if len(candles) < MIN_CANDLES_FOR_TRAINING:
        raise InsufficientDataError(
            f"Only {len(candles)} {interval} candles stored for {coin_id}; "
            f"need >= {MIN_CANDLES_FOR_TRAINING}. Run the collector (run_scheduler.py) longer, "
            f"or seed data with `python -m data_collector.jobs`."
        )

    sentiment_map = _sentiment_by_date() if use_sentiment else {}
    df = build_feature_frame(candles, sentiment_map)
    train_df = df.dropna()

    predictor = PricePredictor(backend=config.PREDICTION_BACKEND)
    predictor.fit(train_df)
    predictions = predictor.predict(df, steps=steps, interval=interval)

    sentiment_avg = sum(sentiment_map.values()) / len(sentiment_map) if sentiment_map else 0.0
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


def run_combined_summary(coin_id: str = config.COIN_ID, interval: str = "1h") -> dict:
    """Prediction + narrative summary + supporting news, for the dashboard's daily digest."""
    prediction = run_prediction(coin_id=coin_id, interval=interval)
    recent_news = repository.get_recent_news(limit=10)

    predictions_for_narrative = [
        {"predicted_price": p["predicted_price"]} for p in prediction["predictions"]
    ]
    narrative = generate_narrative_summary(predictions_for_narrative, prediction["sentiment_avg"], recent_news)

    return {**prediction, "narrative": narrative, "recent_news": recent_news}
