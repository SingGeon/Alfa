"""Feature engineering shared by all prediction backends.

Turns raw OHLC candles (+ optional daily sentiment scores) into a tidy
DataFrame of technical indicators. Keeping this separate from
price_predictor.py means a new backend (Prophet, LSTM, ...) can reuse the
exact same features instead of re-deriving them.
"""
from __future__ import annotations

import numpy as np
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
    "macd",
    "macd_signal",
    "bb_position",
    "volume_ratio",
    "taker_buy_ratio",
    "btc_return_1",
    "tvl_momentum",
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
    """Add moving averages, RSI, MACD, Bollinger position, volume ratio and
    lag features on `close` (+ `volume`, which used to be fetched and then
    thrown away before training - one of the most standard technical
    signals alongside price, so it's worth the model actually seeing it).
    """
    df = df.copy()
    df["return_1"] = df["close"].pct_change()
    df["sma_6"] = df["close"].rolling(6).mean()
    df["sma_24"] = df["close"].rolling(24).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["rsi_14"] = _rsi(df["close"], 14)
    df["volatility_24"] = df["return_1"].rolling(24).std()

    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = df["ema_12"] - ema_26
    df["macd"] = macd_line
    df["macd_signal"] = macd_line.ewm(span=9, adjust=False).mean()

    bb_std = df["close"].rolling(24).std()
    bb_upper = df["sma_24"] + 2 * bb_std
    bb_lower = df["sma_24"] - 2 * bb_std
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, 1e-9)

    # Ratio rather than raw volume: stationary across price/activity regimes,
    # and a spike above 1 (unusually high recent volume) is the actual
    # signal, not the absolute number.
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(24).mean().replace(0, 1e-9)

    # Cerere vs. ofertă: what share of this candle's volume was buyer-
    # initiated (taker-buy), straight from Binance's own kline breakdown -
    # not a derived guess. Only Binance-sourced candles carry this (the
    # CoinGecko fallback and yfinance stocks don't), so it's missing
    # entirely for those; 0.5 (neutral - no lean either way) rather than
    # dropping the row, same reasoning as sentiment's neutral default below.
    if "taker_buy_volume" in df.columns:
        df["taker_buy_ratio"] = (df["taker_buy_volume"] / df["volume"].replace(0, np.nan)).fillna(0.5)
    else:
        df["taker_buy_ratio"] = 0.5

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


def merge_btc_returns(df: pd.DataFrame, btc_close: pd.Series | None) -> pd.DataFrame:
    """Attach BTC's own period return, aligned by exact candle timestamp
    (both exchanges' klines share the same interval grid, so this is a
    direct reindex, not a date-bucket approximation like sentiment/TVL).

    A real, honest "which way is the broader crypto market leaning" signal
    - most alts (ETH included) move with BTC far more than they move on
    their own fundamentals. `btc_close` is None for stocks (not fetched -
    doesn't apply) and for BTC itself.
    """
    df = df.copy()
    if btc_close is None or btc_close.empty:
        df["btc_return_1"] = 0.0
        return df
    # replace inf: a 0 (or repeated-identical) price anywhere in btc_close
    # would make pct_change() divide by zero - shouldn't happen for real
    # BTC data, but sklearn hard-fails on a single inf slipping into
    # training, so guard it the same way tvl_momentum below has to for
    # real (documented) zero-TVL history.
    btc_return = btc_close.pct_change().replace([np.inf, -np.inf], np.nan)
    df["btc_return_1"] = btc_return.reindex(df.index, method="ffill")
    df["btc_return_1"] = df["btc_return_1"].fillna(0.0)
    return df


def merge_defi_tvl(df: pd.DataFrame, tvl_by_date: dict | None) -> pd.DataFrame:
    """Attach Ethereum DeFi TVL day-over-day % change ("network utility"
    momentum, see DeFiLlama), matched by calendar date like sentiment.

    Deliberately a *change*, not the raw TVL level: the level trends with
    overall market size and would just echo price, which is circular -
    the momentum (growing/shrinking faster than usual) is the actual signal.
    `tvl_by_date` maps `date` -> raw TVL; only ever passed for Ethereum
    itself (see api/services.py) - None elsewhere.
    """
    df = df.copy()
    if not tvl_by_date:
        df["tvl_momentum"] = 0.0
        return df
    tvl_series = pd.Series(tvl_by_date).sort_index()
    tvl_series.index = pd.to_datetime(tvl_series.index)
    # DeFiLlama's Ethereum history starts in 2017, years before DeFi
    # existed - those early days are real, documented tvl=0 rows, and
    # pct_change() out of 0 is +/-inf. sklearn hard-fails on a single inf
    # reaching training (seen live: /api/outlook's weekly leg trains on
    # candles back to 2017 and hit exactly this), so treat inf as "no
    # signal yet" the same as any other gap - ffill/fillna below carries
    # the last real momentum value forward past it.
    tvl_change = tvl_series.pct_change().replace([np.inf, -np.inf], np.nan)
    dates = pd.to_datetime(df.index.date)
    df["tvl_momentum"] = pd.Series(dates, index=df.index).map(lambda d: tvl_change.get(d))
    df["tvl_momentum"] = df["tvl_momentum"].ffill().fillna(0.0)
    return df


def build_feature_frame(
    candles: list[dict],
    sentiment_by_date: dict | None = None,
    btc_close: pd.Series | None = None,
    tvl_by_date: dict | None = None,
) -> pd.DataFrame:
    """One-call pipeline: candles -> full feature DataFrame (NaNs kept, caller drops them)."""
    df = candles_to_dataframe(candles)
    df = add_technical_features(df)
    df = merge_sentiment(df, sentiment_by_date or {})
    df = merge_btc_returns(df, btc_close)
    df = merge_defi_tvl(df, tvl_by_date)
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
