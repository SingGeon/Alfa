"""The one hook between the existing prediction code and the evaluation log.

api.services.run_prediction calls record_prediction() right after it has
built its result - the models themselves are untouched; this only writes
down what they said. It must never break a prediction, so every failure is
logged and swallowed.
"""
from __future__ import annotations

import logging

import config
from evaluation import storage

logger = logging.getLogger(__name__)


def model_name_for(backend: str, model_variant: str) -> str:
    return f"{backend}-{model_variant}"


def _signal(result: dict, df, sentiment_avg: float) -> str | None:
    """Same "Trading signal" rule as the dashboard card and
    data_collector.jobs.backfill_signal_history (ml.trading_signal)."""
    from ml.trading_signal import compute_signal_label

    preds = result["predictions"]
    last_close = result["last_known_price"]
    if not preds or not last_close:
        return None
    trend_pct = (preds[-1]["predicted_price"] - last_close) / last_close * 100
    rsi = df["rsi_14"].iloc[-1] if "rsi_14" in df.columns else float("nan")
    macd = df["macd"].iloc[-1] if "macd" in df.columns else float("nan")
    macd_sig = df["macd_signal"].iloc[-1] if "macd_signal" in df.columns else float("nan")
    rsi_val = float(rsi) if rsi == rsi else None  # NaN check
    macd_bullish = bool(macd > macd_sig) if macd == macd and macd_sig == macd_sig else None
    return compute_signal_label(trend_pct, rsi_val, macd_bullish, sentiment_avg, result["confidence"])


def _history(df) -> list[dict]:
    """The last real candles the model saw, for the visual snapshot."""
    tail = df[["open", "high", "low", "close"]].tail(config.EVAL_SNAPSHOT_HISTORY_CANDLES)
    return [
        {"timestamp": storage.to_iso(ts.to_pydatetime()), **{k: round(float(v), 2) for k, v in row.items()}}
        for ts, row in tail.iterrows()
    ]


def record_prediction(result: dict, df, sentiment_avg: float, diagnostics: dict | None = None) -> int | None:
    """Log the prediction and, if it's a new one, draw its visual snapshot."""
    try:
        try:
            signal = _signal(result, df, sentiment_avg)
        except Exception:
            logger.exception("Could not compute trading signal for evaluation log")
            signal = None
        try:
            history = _history(df)
        except Exception:
            logger.exception("Could not extract candle history for evaluation log")
            history = []
        prediction_id = storage.log_prediction(
            interval=result["interval"],
            model_name=model_name_for(result["backend"], result["model_variant"]),
            price_at_prediction=result["last_known_price"],
            predictions=result["predictions"],
            confidence=result.get("confidence"),
            signal=signal,
            history=history,
            diagnostics=diagnostics,
        )
        if prediction_id:
            from evaluation import charts  # matplotlib only loads once there's something to draw

            charts.refresh_snapshots([prediction_id])
        return prediction_id
    except Exception:
        logger.exception("Failed to record prediction in the evaluation log")
        return None
