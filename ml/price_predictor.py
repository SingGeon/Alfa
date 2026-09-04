"""Pluggable price forecaster.

Default backend is scikit-learn (GradientBoostingRegressor), which needs no
extra dependencies and works well on the small candle histories a free
market-data API gives you. Prophet and LSTM backends are provided for when
you have more history / heavier infra, gated behind optional imports so the
base install stays light.

All backends implement the same contract:
    fit(df)               -> None
    predict(df, steps)    -> list[{"timestamp", "predicted_price", "lower", "upper"}]

`df` is the output of ml.features.build_feature_frame().
"""
from __future__ import annotations

import logging
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, add_technical_features, make_supervised

logger = logging.getLogger(__name__)

_INTERVAL_TIMEDELTA = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}

# Small ensemble of differently-seeded models, averaged at predict time,
# instead of one single fit. A single GradientBoostingRegressor's output
# depends noticeably on its random_state on datasets this small (a few
# hundred to ~1000 rows); averaging several smooths that noise out into a
# steadier point estimate and a less erratic confidence interval, at the
# (one-time, now cached - see api/services.py) cost of training 3x as many
# trees.
_ENSEMBLE_SEEDS = (0, 21, 42)


class PricePredictor:
    def __init__(self, backend: str = "sklearn", feature_cols: list[str] | None = None, ensemble_size: int = len(_ENSEMBLE_SEEDS)):
        self.backend = backend
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.ensemble_size = max(1, min(ensemble_size, len(_ENSEMBLE_SEEDS)))
        self._models = []          # ensemble of mean predictors (sklearn backend)
        self._models_lower = []    # ensemble of lower-quantile predictors
        self._models_upper = []    # ensemble of upper-quantile predictors
        self._model = None          # single-model backends (prophet)
        self._last_close = None
        self._fitted = False

    # -- public API -----------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PricePredictor":
        if self.backend == "sklearn":
            self._fit_sklearn(df)
        elif self.backend == "prophet":
            self._fit_prophet(df)
        elif self.backend == "lstm":
            self._fit_lstm(df)
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame, steps: int, interval: str = "1h") -> list[dict]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if self.backend == "sklearn":
            return self._predict_sklearn(df, steps, interval)
        if self.backend == "prophet":
            return self._predict_prophet(df, steps, interval)
        if self.backend == "lstm":
            return self._predict_lstm(df, steps, interval)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PricePredictor":
        return joblib.load(path)

    # -- sklearn backend (default) --------------------------------------

    def _fit_sklearn(self, df: pd.DataFrame) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        X, y = make_supervised(df, self.feature_cols)
        if len(X) < 30:
            raise ValueError(
                f"Not enough history to train ({len(X)} rows) - need at least 30. "
                "Let the collector run longer or backfill more candles."
            )
        # Ensemble: same hyperparameters, different random_state, averaged
        # at predict time (see _ENSEMBLE_SEEDS). Quantile-loss models give a
        # real, data-driven confidence interval instead of a fixed +/-X% band.
        # ensemble_size lets latency-sensitive callers (an on-demand detail
        # view someone is actively waiting on) trade a bit of smoothing for
        # speed - see PricePredictor.__init__.
        seeds = _ENSEMBLE_SEEDS[: self.ensemble_size]
        self._models = [GradientBoostingRegressor(random_state=s).fit(X, y) for s in seeds]
        self._models_lower = [
            GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=s).fit(X, y) for s in seeds
        ]
        self._models_upper = [
            GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=s).fit(X, y) for s in seeds
        ]

    def _predict_sklearn(self, df: pd.DataFrame, steps: int, interval: str) -> list[dict]:
        step_delta = _INTERVAL_TIMEDELTA[interval]
        history = add_technical_features(df[["open", "high", "low", "close", "volume"]].copy())
        # None of sentiment/btc_return_1/tvl_momentum extrapolate from OHLCV
        # alone (they come from merge_sentiment/merge_btc_returns/
        # merge_defi_tvl in build_feature_frame, which this recursive
        # re-derivation deliberately skips - recomputing them per future
        # step would need data we don't have yet) - hold each at its last
        # known real value for every future step instead.
        last_sentiment = df["sentiment"].iloc[-1] if "sentiment" in df.columns and len(df) else 0.0
        history["sentiment"] = df["sentiment"] if "sentiment" in df.columns else last_sentiment
        last_btc_return = df["btc_return_1"].iloc[-1] if "btc_return_1" in df.columns and len(df) else 0.0
        history["btc_return_1"] = df["btc_return_1"] if "btc_return_1" in df.columns else last_btc_return
        last_tvl_momentum = df["tvl_momentum"].iloc[-1] if "tvl_momentum" in df.columns and len(df) else 0.0
        history["tvl_momentum"] = df["tvl_momentum"] if "tvl_momentum" in df.columns else last_tvl_momentum

        results = []
        last_timestamp = history.index[-1]
        for _ in range(steps):
            row = history.iloc[[-1]][self.feature_cols]
            if row.isna().any(axis=None):
                # Not enough rolling history yet (e.g. right at the start) - stop early
                # rather than feeding the model garbage.
                logger.warning("Insufficient rolling history for a further step, stopping early")
                break
            log_return = float(np.mean([m.predict(row)[0] for m in self._models]))
            log_return_low = float(np.mean([m.predict(row)[0] for m in self._models_lower]))
            log_return_high = float(np.mean([m.predict(row)[0] for m in self._models_upper]))

            last_close = float(history["close"].iloc[-1])
            predicted_price = last_close * np.exp(log_return)
            lower = last_close * np.exp(min(log_return_low, log_return_high))
            upper = last_close * np.exp(max(log_return_low, log_return_high))

            last_timestamp = last_timestamp + step_delta
            results.append(
                {
                    "timestamp": last_timestamp,
                    "predicted_price": predicted_price,
                    "lower": lower,
                    "upper": upper,
                }
            )

            # Append the predicted candle so the next iteration's rolling
            # features (SMA/RSI/lags/...) account for it (recursive forecasting).
            new_row = {
                "open": last_close,
                "high": max(last_close, predicted_price),
                "low": min(last_close, predicted_price),
                "close": predicted_price,
                "volume": history["volume"].iloc[-1],
                "sentiment": last_sentiment,
            }
            history = pd.concat([history, pd.DataFrame([new_row], index=[last_timestamp])])
            history = add_technical_features(history[["open", "high", "low", "close", "volume"]])
            history["sentiment"] = history["sentiment"] if "sentiment" in history else last_sentiment
            history.loc[last_timestamp, "sentiment"] = last_sentiment
            history["btc_return_1"] = history["btc_return_1"] if "btc_return_1" in history else last_btc_return
            history.loc[last_timestamp, "btc_return_1"] = last_btc_return
            history["tvl_momentum"] = history["tvl_momentum"] if "tvl_momentum" in history else last_tvl_momentum
            history.loc[last_timestamp, "tvl_momentum"] = last_tvl_momentum

        return results

    # -- Prophet backend (optional) --------------------------------------

    def _fit_prophet(self, df: pd.DataFrame) -> None:
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError(
                "backend='prophet' requires the `prophet` package: pip install prophet"
            ) from exc
        data = df.reset_index()[["timestamp", "close"]].rename(columns={"timestamp": "ds", "close": "y"})
        data["ds"] = data["ds"].dt.tz_localize(None)
        self._model = Prophet(interval_width=0.8)
        self._model.fit(data)

    def _predict_prophet(self, df: pd.DataFrame, steps: int, interval: str) -> list[dict]:
        freq = {"15m": "15min", "1h": "h", "4h": "4h", "1d": "D", "1w": "W"}[interval]
        future = self._model.make_future_dataframe(periods=steps, freq=freq)
        forecast = self._model.predict(future)
        tail = forecast.tail(steps)
        return [
            {
                "timestamp": row.ds.to_pydatetime(),
                "predicted_price": row.yhat,
                "lower": row.yhat_lower,
                "upper": row.yhat_upper,
            }
            for row in tail.itertuples()
        ]

    # -- LSTM backend (optional) ------------------------------------------

    def _fit_lstm(self, df: pd.DataFrame) -> None:
        raise NotImplementedError(
            "backend='lstm' requires TensorFlow (pip install tensorflow) and is a stub here - "
            "swap in a Sequential(LSTM(...)) model trained on windowed `close` sequences. "
            "The sklearn backend is the supported default; this is an extension point."
        )

    def _predict_lstm(self, df: pd.DataFrame, steps: int, interval: str) -> list[dict]:
        raise NotImplementedError("See _fit_lstm")
