import pytest
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




# --- confidence band calibration (see ml/price_predictor.py) -----------------

def _fitted_forecast(synthetic_candles, n=300, steps=24):
    df = build_feature_frame(synthetic_candles(n))
    predictor = PricePredictor(backend="sklearn", model_variant="tuned").fit(df.dropna())
    return predictor, predictor.predict(df, steps=steps, interval="1h")


def test_band_widens_with_horizon(synthetic_candles):
    _, out = _fitted_forecast(synthetic_candles)
    widths = [np.log(p["upper"] / p["lower"]) for p in out]
    assert all(b > a for a, b in zip(widths, widths[1:]))
    # half-width grows as k ** BAND_HORIZON_EXPONENT from the step-1 width
    from ml.price_predictor import BAND_HORIZON_EXPONENT

    assert widths[-1] / widths[0] == pytest.approx(24 ** BAND_HORIZON_EXPONENT, rel=1e-6)


def test_point_always_inside_band(synthetic_candles):
    _, out = _fitted_forecast(synthetic_candles)
    for p in out:
        assert p["lower"] <= p["predicted_price"] <= p["upper"]


def test_conformal_calibration_runs_on_long_history(synthetic_candles):
    predictor, _ = _fitted_forecast(synthetic_candles, n=300, steps=1)
    assert predictor._cqr_calibration_rows > 0


def test_conformal_calibration_skipped_on_short_history(synthetic_candles):
    # 130 candles -> ~105 training rows after the 24-candle rolling warm-up: < 120
    predictor, out = _fitted_forecast(synthetic_candles, n=130, steps=3)
    assert predictor._cqr_calibration_rows == 0
    assert predictor._cqr_q == 0.0
    assert len(out) == 3


def test_taker_buy_ratio_stays_real_in_recursion(synthetic_candles):
    candles = synthetic_candles(200)
    for i, c in enumerate(candles):
        c["taker_buy_volume"] = c["volume"] * (0.3 if i % 2 else 0.7)
    df = build_feature_frame(candles)
    real_last_ratio = df["taker_buy_ratio"].iloc[-1]
    assert real_last_ratio != 0.5

    predictor = PricePredictor(backend="sklearn", ensemble_size=1).fit(df.dropna())
    seen = []

    class Spy:
        def __init__(self, model):
            self.model = model

        def predict(self, row):
            seen.append(float(row["taker_buy_ratio"].iloc[0]))
            return self.model.predict(row)

    predictor._models = [Spy(m) for m in predictor._models]
    predictor.predict(df, steps=3, interval="1h")
    assert seen[0] == pytest.approx(real_last_ratio)  # step 1: the real candle's own value
    assert seen[1] == pytest.approx(0.5)              # synthetic candles: neutral


def test_legacy_variant_keeps_uncalibrated_band(synthetic_candles):
    # "legacy" is the fixed reference: no CQR, no horizon-scaled band.
    df = build_feature_frame(synthetic_candles(300))
    predictor = PricePredictor(backend="sklearn", model_variant="legacy").fit(df.dropna())
    out = predictor.predict(df, steps=24, interval="1h")
    assert predictor._cqr_calibration_rows == 0
    assert predictor._cqr_q == 0.0
    widths = [np.log(p["upper"] / p["lower"]) for p in out]
    from ml.price_predictor import BAND_HORIZON_EXPONENT

    assert widths[-1] / widths[0] != pytest.approx(24 ** BAND_HORIZON_EXPONENT, rel=1e-3)
