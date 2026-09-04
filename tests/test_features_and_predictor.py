import numpy as np
import pandas as pd

from ml.features import build_feature_frame, make_supervised
from ml.price_predictor import PricePredictor


def test_build_feature_frame_has_expected_columns(synthetic_candles):
    df = build_feature_frame(synthetic_candles(50))
    for col in (
        "sma_6", "sma_24", "ema_12", "rsi_14", "volatility_24", "lag_1", "sentiment",
        "taker_buy_ratio", "btc_return_1", "tvl_momentum",
    ):
        assert col in df.columns


def test_merge_sentiment_defaults_to_zero_when_missing(synthetic_candles):
    df = build_feature_frame(synthetic_candles(10), sentiment_by_date=None)
    assert (df["sentiment"] == 0.0).all()


def test_taker_buy_ratio_defaults_neutral_without_binance_data(synthetic_candles):
    # synthetic_candles has no taker_buy_volume field (mirrors CoinGecko
    # fallback / yfinance stocks) - should default neutral, not drop rows.
    df = build_feature_frame(synthetic_candles(10))
    assert (df["taker_buy_ratio"] == 0.5).all()


def test_btc_return_1_defaults_zero_when_missing(synthetic_candles):
    df = build_feature_frame(synthetic_candles(10), btc_close=None)
    assert (df["btc_return_1"] == 0.0).all()


def test_btc_return_1_reflects_real_btc_moves(synthetic_candles):
    candles = synthetic_candles(10)
    index = pd.to_datetime([c["timestamp"] for c in candles], utc=True)
    btc_close = pd.Series([100.0 * (1.01**i) for i in range(len(candles))], index=index)
    df = build_feature_frame(candles, btc_close=btc_close)
    # Each step is a real +1% BTC move, so every return after the first is ~0.01.
    assert df["btc_return_1"].iloc[2:].round(4).eq(0.0100).all()


def test_tvl_momentum_defaults_zero_when_missing(synthetic_candles):
    df = build_feature_frame(synthetic_candles(10), tvl_by_date=None)
    assert (df["tvl_momentum"] == 0.0).all()


def test_tvl_momentum_handles_real_zero_history(synthetic_candles):
    # DeFiLlama's real Ethereum TVL history starts at 0 for its first many
    # days (pre-DeFi, 2017) - pct_change() out of 0 is +/-inf, which used
    # to crash sklearn training with "Input X contains infinity" the first
    # time a prediction trained on candles reaching that far back (hit live
    # via /api/outlook's weekly leg).
    candles = synthetic_candles(10)
    dates = [c["timestamp"].date().isoformat() for c in candles]
    tvl_by_date = {d: 0.0 for d in dates[:5]} | {d: 1000.0 * (i + 1) for i, d in enumerate(dates[5:])}
    df = build_feature_frame(candles, tvl_by_date=tvl_by_date)
    assert np.isfinite(df["tvl_momentum"]).all()


def test_make_supervised_drops_na_rows(synthetic_candles):
    df = build_feature_frame(synthetic_candles(50))
    X, y = make_supervised(df)
    assert len(X) == len(y)
    assert not X.isna().any().any()


def test_predictor_fit_predict_shape(synthetic_candles):
    df = build_feature_frame(synthetic_candles(200))
    predictor = PricePredictor(backend="sklearn")
    predictor.fit(df.dropna())
    out = predictor.predict(df, steps=6, interval="1h")
    assert len(out) == 6
    for step in out:
        assert step["lower"] <= step["upper"]
        assert "predicted_price" in step
        assert "timestamp" in step


def test_predictor_raises_without_fit(synthetic_candles):
    df = build_feature_frame(synthetic_candles(50))
    predictor = PricePredictor(backend="sklearn")
    try:
        predictor.predict(df, steps=1, interval="1h")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_predictor_unknown_backend_raises():
    try:
        PricePredictor(backend="does-not-exist").fit(None)
        assert False, "expected ValueError"
    except ValueError:
        pass
