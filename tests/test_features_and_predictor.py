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




# --- evolving population ("tuned", ml/evolution.py) ----------------------------

def _fitted_forecast(synthetic_candles, n=500, steps=24, variant="tuned"):
    df = build_feature_frame(synthetic_candles(n))
    predictor = PricePredictor(backend="sklearn", model_variant=variant).fit(df.dropna())
    preds, diag = predictor.predict_with_diagnostics(df, steps=steps, interval="1h")
    return predictor, preds, diag


def test_tuned_is_an_evolving_population(synthetic_candles):
    from ml.evolution import POPULATION

    predictor, preds, diag = _fitted_forecast(synthetic_candles)
    assert predictor._evolution is not None
    assert len(preds) == 24
    assert diag["method"] == "evolution"
    assert diag["population"] == POPULATION
    assert diag["leaders"] and all(o["voting"] for o in diag["leaders"])
    assert diag["emotion"] in {"fear", "caution", "calm", "confidence", "euphoria"}


def test_tuned_point_inside_band(synthetic_candles):
    _, preds, _ = _fitted_forecast(synthetic_candles, steps=30)
    for p in preds:
        assert p["lower"] <= p["predicted_price"] <= p["upper"]


def test_legacy_learns_from_market_context(synthetic_candles):
    df = build_feature_frame(synthetic_candles(300))
    df["funding"] = np.where(np.arange(len(df)) > 150, 1e-4, np.nan)  # source starts later: NaN -> 0
    df["fear_greed"] = 0.2
    predictor = PricePredictor(backend="sklearn", model_variant="legacy").fit(df.dropna(subset=["close"]).dropna(subset=["rsi_14", "lag_24", "volatility_24"]))
    importances = predictor.get_feature_importance()
    assert "funding" in importances and "fear_greed" in importances
    preds = predictor.predict(df, steps=5, interval="1h")
    assert len(preds) == 5


def test_tuned_is_deterministic(synthetic_candles):
    _, a, _ = _fitted_forecast(synthetic_candles)
    _, b, _ = _fitted_forecast(synthetic_candles)
    assert [p["predicted_price"] for p in a] == [p["predicted_price"] for p in b]


def test_legacy_stays_recursive_without_diagnostics(synthetic_candles):
    predictor, preds, diag = _fitted_forecast(synthetic_candles, n=300, variant="legacy")
    assert predictor._evolution is None
    assert diag is None
    assert len(preds) == 24


def test_walk_forward_direction_ignores_flat_forecasts():
    from ml.walk_forward import BacktestResult, WindowResult

    def window(pred, actual):
        return WindowResult(origin_close=100.0, predicted=[pred], lower=[90.0], upper=[110.0], actual=[actual])

    result = BacktestResult(windows=[
        window(101.0, 102.0),   # called up, went up
        window(99.0, 102.0),    # called down, went up
        window(100.0, 103.0),   # flat: calls no direction
        window(100.0, 97.0),    # flat
    ])
    summary = result.summarize()
    assert summary["windows"] == 4
    assert summary["direction_calls"] == 2
    assert summary["direction_accuracy"] == 0.5

    flat_only = BacktestResult(windows=[window(100.0, 101.0)]).summarize()
    assert flat_only["direction_calls"] == 0
    assert flat_only["direction_accuracy"] is None
    assert flat_only["direction_se"] is None
