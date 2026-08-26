from ml.features import build_feature_frame, make_supervised
from ml.price_predictor import PricePredictor


def test_build_feature_frame_has_expected_columns(synthetic_candles):
    df = build_feature_frame(synthetic_candles(50))
    for col in ("sma_6", "sma_24", "ema_12", "rsi_14", "volatility_24", "lag_1", "sentiment"):
        assert col in df.columns


def test_merge_sentiment_defaults_to_zero_when_missing(synthetic_candles):
    df = build_feature_frame(synthetic_candles(10), sentiment_by_date=None)
    assert (df["sentiment"] == 0.0).all()


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
