"""Feature engineering shared by all prediction backends.

Turns raw OHLC candles (+ optional daily sentiment scores) into a tidy
DataFrame of technical indicators. Keeping this separate from
price_predictor.py means a new backend (Prophet, LSTM, ...) can reuse the
exact same features instead of re-deriving them.
"""
from __future__ import annotations

import pandas as pd

# Lags expressed in "number of candles back", independent of the interval
# (1h/1d/1w) so the same code path works for any granularity.
LAG_STEPS = (1, 2, 3, 6, 12, 24)

FEATURE_COLUMNS = [
    "close",
    "return_1",
    "sma_6",
    "sma_24",
    "ema_12",
    "rsi_14",
    "volatility_24",
    *[f"lag_{n}" for n in LAG_STEPS],
    "sentiment",
]


def candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Sort candles chronologically and index by timestamp."""
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(candles).sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp")


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add moving averages, RSI, volatility and lag features on `close`."""
    df = df.copy()
    df["return_1"] = df["close"].pct_change()
    df["sma_6"] = df["close"].rolling(6).mean()
    df["sma_24"] = df["close"].rolling(24).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["rsi_14"] = _rsi(df["close"], 14)
    df["volatility_24"] = df["return_1"].rolling(24).std()
    for n in LAG_STEPS:
        df[f"lag_{n}"] = df["close"].shift(n)
    return df


def merge_sentiment(df: pd.DataFrame, sentiment_by_date: dict) -> pd.DataFrame:
    """Attach a daily sentiment score to each candle (matched by calendar date).

    `sentiment_by_date` maps `date` -> score in [-1, 1]. Missing days are
    forward-filled, then defaulted to 0 (neutral) if none precede them.
    """
    df = df.copy()
    if not sentiment_by_date:
        df["sentiment"] = 0.0
        return df
    sentiment_series = pd.Series(sentiment_by_date)
    sentiment_series.index = pd.to_datetime(sentiment_series.index)
    dates = pd.to_datetime(df.index.date)
    df["sentiment"] = pd.Series(dates, index=df.index).map(
        lambda d: sentiment_series.get(d)
    )
    df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
    return df


def build_feature_frame(candles: list[dict], sentiment_by_date: dict | None = None) -> pd.DataFrame:
    """One-call pipeline: candles -> full feature DataFrame (NaNs kept, caller drops them)."""
    df = candles_to_dataframe(candles)
    df = add_technical_features(df)
    df = merge_sentiment(df, sentiment_by_date or {})
    return df


def make_supervised(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS):
    """Build (X, y) where y is the next-step log return of `close`.

    Predicting the return rather than the raw price keeps the target
    roughly stationary, which matters a lot for tree-based models that
    can't extrapolate beyond the price range they were trained on.
    """
    import numpy as np

    data = df.copy()
    data["target"] = np.log(data["close"].shift(-1) / data["close"])
    data = data.dropna(subset=[*feature_cols, "target"])
    X = data[feature_cols]
    y = data["target"]
    return X, y
