"""Pluggable price forecaster.

Default (and only exposed) backend is scikit-learn (GradientBoostingRegressor),
which needs no extra dependencies and works well on the small candle
histories a free market-data API gives you. Prophet is a leftover extension
point for heavier infra (needs `pip install prophet`, not installed by
default, not exposed in the dashboard's model selector).

An LSTM backend (and an "ensemble" blending it with sklearn) used to live
here too. Removed: measured directly against the trivial "predict no
change" baseline, it never beat it - every architecture tried (more units,
dropout, more epochs) landed 16x-345x *worse* than doing nothing, because a
few hundred noisy hourly candles simply don't carry enough single-step
signal for a recurrent net to learn from without just memorizing noise.
Once self-calibrated to that measured skill it degenerated to predicting
flat, which made it dead weight: a second full training pass (plus
TensorFlow itself, a heavy dependency) for a prediction that was
functionally "no change" anyway. Pattern recognition (ml/pattern_recognition.py)
replaces it as the dashboard's secondary signal, since geometric/candlestick
patterns don't need a trained model to be honest about what they found -
they're either present in the price series or they aren't.

The backend implements:
    fit(df)               -> None
    predict(df, steps)    -> list[{"timestamp", "predicted_price", "lower", "upper"}]

`df` is the output of ml.features.build_feature_frame().
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
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

# GradientBoostingRegressor hyperparameters, chosen against sklearn's own
# defaults (n_estimators=100, max_depth=3, learning_rate=0.1, subsample=1.0,
# min_samples_leaf=1) via walk-forward validation (expanding-window
# TimeSeriesSplit for search, then confirmed with the same per-day
# retrain-from-scratch backtest data_collector.jobs.
# backfill_prediction_history uses in production - not a proxy metric -
# on real ETH candles).
#
# A first attempt picked the single lowest-MAPE corner of the search grid
# (n_estimators=30, max_depth=2, learning_rate=0.02, min_samples_leaf=20)
# and it was a mistake: on 1h it drove the recursive multi-step forecast's
# predicted log-returns down to ~8% of the target's own variance (was 56%
# at these defaults), i.e. it collapsed toward predicting the unconditional
# mean for nearly every input - the exact degenerate "flat line" failure
# this file's own docstring already documents for why LSTM was removed.
# Financial returns are mostly noise at these horizons, so pure MAPE
# rewards shrinking toward "predict nothing" almost without limit; MAPE
# alone can't tell a genuinely-regularized model apart from one that just
# stopped differentiating between inputs. These settings are the mildest
# regularization step that still gets nearly all of the walk-forward MAPE
# improvement while keeping the recursive forecast visibly responsive:
# confirmed directly on the real recursive predict() path (not a proxy) by
# comparing each interval's simulated 24-step-ahead (14-step for 1d)
# forecast range against how much price actually moved over the same real
# window, across ~25-30 different historical starting points per interval.
# MAPE improvement over defaults, walk-forward-confirmed per interval:
# 15m 0.209%->0.196% (-6.1%), 1h 0.380%->0.329% (-13.4%),
# 1d 2.441%->2.204% (-9.7%) - direction accuracy and confidence-interval
# calibration (~80% of actuals inside [lower, upper]) improved alongside
# MAPE on all three, and the same one set of values won independently on
# all three, so there's no need to special-case per interval.
#
# Still a real trade-off, not a strict win, confirmed with a fresh spot
# check on the most recent live data (not just the original backtest
# window): on the last 60 real 1h candles, "tuned" was only slightly ahead
# on price MAPE (0.240% vs 0.267%) and actually behind on direction
# accuracy (61.7% vs 66.7%) - and its recursive forecast visibly hugs a
# flat line in a calm market (0.13% predicted 24h range vs "legacy"'s
# 0.69%), which reads as "not predicting anything" even when it's the more
# statistically honest answer. Kept both variants selectable (see
# MODEL_VARIANTS in api/services.py and the dashboard's model toggle)
# rather than picking one side of that trade-off for everyone.
MODEL_VARIANTS = ("tuned", "legacy")

_TUNED_GBR_PARAMS = dict(
    n_estimators=50,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.7,
    min_samples_leaf=10,
)
# sklearn's own defaults (n_estimators=100, max_depth=3, learning_rate=0.1,
# subsample=1.0, min_samples_leaf=1) - the pre-tuning behavior, kept
# reachable rather than deleted once tuning turned out to be a genuine
# trade-off and not a strict improvement (see above).
_LEGACY_GBR_PARAMS: dict = {}

_GBR_PARAMS_BY_VARIANT = {"tuned": _TUNED_GBR_PARAMS, "legacy": _LEGACY_GBR_PARAMS}

# --- Confidence band calibration --------------------------------------------
#
# Measured on a walk-forward backtest (ml/walk_forward.py), the raw quantile
# band ([alpha=0.1, alpha=0.9] GBR models, re-evaluated on every synthetic
# candle of the recursive forecast) held the real price only ~26% of the time
# over 24 steps, against the 80% it is meant to. Three reasons, all fixed below:
#  - quantile models under-cover out of sample -> split-conformal correction
#    (CQR, Romano et al. 2019): fit the quantile models on the older 75% of
#    rows, measure how far the newest 25% fall outside, widen by that amount;
#  - each step's band was one step wide around the previous *prediction*, so
#    it never grew with the horizon -> half-width grows as k ** exponent;
#  - past step 1 the quantile models only ever see synthetic candles (whose
#    volatility_24 collapses toward 0), so their width shrinks -> the per-step
#    width is frozen at step 1, the only one computed on real candles.
_BAND_ALPHA = 0.2            # 1 - target coverage of [lower, upper]
_CQR_MIN_ROWS = 120          # below this, too few calibration rows - skip CQR
_CQR_CALIBRATION_FRACTION = 0.25
# 80% band half-width of real k-step ETH log returns grows as k ** e. Measured
# (not fitted per model: that was too noisy) on real Binance candles: 1h 0.55,
# 1d 0.57 (stable across both halves of each history). Plain random-walk
# scaling would be 0.5; crypto's fatter tails over longer horizons push it up.
BAND_HORIZON_EXPONENT = 0.57
# Everything above (CQR, horizon-scaled band, taker_buy_volume carried through
# the recursion) applies to "tuned" only. "legacy" is kept exactly as it was
# before, as the fixed reference the tuned model is compared against.
_CALIBRATED_VARIANTS = ("tuned",)


class PricePredictor:
    def __init__(
        self,
        backend: str = "sklearn",
        feature_cols: list[str] | None = None,
        ensemble_size: int = len(_ENSEMBLE_SEEDS),
        model_variant: str = "tuned",
    ):
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model_variant: {model_variant!r}; must be one of {MODEL_VARIANTS}")
        self.backend = backend
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self.ensemble_size = max(1, min(ensemble_size, len(_ENSEMBLE_SEEDS)))
        self.model_variant = model_variant
        self._models = []          # ensemble of mean predictors (sklearn backend)
        self._models_lower = []    # ensemble of lower-quantile predictors
        self._models_upper = []    # ensemble of upper-quantile predictors
        self._cqr_q = 0.0           # conformal widening of the quantile band (log-return units)
        self._cqr_calibration_rows = 0  # rows the widening was measured on (0 = skipped)
        self._model = None          # single-model backends (prophet)
        self._last_close = None
        self._fitted = False

    # -- public API -----------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "PricePredictor":
        if self.backend == "sklearn":
            self._fit_sklearn(df)
        elif self.backend == "prophet":
            self._fit_prophet(df)
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
        raise ValueError(f"Unknown backend: {self.backend!r}")

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PricePredictor":
        return joblib.load(path)

    def get_feature_importance(self) -> dict[str, float] | None:
        """{feature_name: relative_importance in [0, 1], summing to ~1}
        from the trained model's own learned weights - not a made-up
        number, and not a per-prediction attribution (that would need
        something like SHAP) but the model's overall, global read on how
        much each feature has mattered across all of training. Lets the
        dashboard answer "how much does sentiment actually factor into
        this model" with a real figure instead of just the on/off toggle.

        None for backends with no native notion of feature importance.
        """
        if self.backend == "sklearn":
            if not self._models:
                return None
            importances = np.mean([m.feature_importances_ for m in self._models], axis=0)
            return dict(zip(self.feature_cols, (float(v) for v in importances)))
        return None

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
        gbr_params = _GBR_PARAMS_BY_VARIANT[self.model_variant]
        self._models = [GradientBoostingRegressor(random_state=s, **gbr_params).fit(X, y) for s in seeds]

        # Conformalized quantile regression (see _BAND_ALPHA's comment): the
        # quantile models train on the older rows only, so the newest rows
        # (in time order - never shuffled) are an honest out-of-sample check
        # of how far reality lands outside their band.
        calibrated = self.model_variant in _CALIBRATED_VARIANTS
        if calibrated and len(X) >= _CQR_MIN_ROWS:
            split = int(len(X) * (1 - _CQR_CALIBRATION_FRACTION))
            X_fit, y_fit, X_cal, y_cal = X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]
        else:
            X_fit, y_fit, X_cal, y_cal = X, y, None, None
        low_alpha, high_alpha = (_BAND_ALPHA / 2, 1 - _BAND_ALPHA / 2) if calibrated else (0.1, 0.9)
        self._models_lower = [
            GradientBoostingRegressor(loss="quantile", alpha=low_alpha, random_state=s, **gbr_params).fit(X_fit, y_fit)
            for s in seeds
        ]
        self._models_upper = [
            GradientBoostingRegressor(loss="quantile", alpha=high_alpha, random_state=s, **gbr_params).fit(X_fit, y_fit)
            for s in seeds
        ]
        self._cqr_q = 0.0
        self._cqr_calibration_rows = 0 if X_cal is None else len(X_cal)
        if X_cal is not None:
            lo = np.mean([m.predict(X_cal) for m in self._models_lower], axis=0)
            hi = np.mean([m.predict(X_cal) for m in self._models_upper], axis=0)
            scores = np.maximum(lo - y_cal.to_numpy(), y_cal.to_numpy() - hi)
            n = len(scores)
            level = min(1.0, math.ceil((n + 1) * (1 - _BAND_ALPHA)) / n)
            self._cqr_q = float(np.quantile(scores, level, method="higher"))

    def _predict_sklearn(self, df: pd.DataFrame, steps: int, interval: str) -> list[dict]:
        def step_fn(history: pd.DataFrame, with_band: bool) -> tuple[float, float, float] | None:
            row = history.iloc[[-1]][self.feature_cols]
            if row.isna().any(axis=None):
                return None
            log_return = float(np.mean([m.predict(row)[0] for m in self._models]))
            if not with_band:
                return log_return, log_return, log_return
            log_return_low = float(np.mean([m.predict(row)[0] for m in self._models_lower])) - self._cqr_q
            log_return_high = float(np.mean([m.predict(row)[0] for m in self._models_upper])) + self._cqr_q
            return log_return, log_return_low, log_return_high

        return self._iterate_recursive(df, steps, interval, step_fn)

    # -- shared recursive multi-step loop --------------------------------
    #
    # Forecasts one step at a time and feeds its own prediction back in as
    # a synthetic candle before forecasting the next, so later steps'
    # rolling features (SMA/RSI/lags/...) account for the model's own
    # trajectory rather than staying frozen at the last real candle. Kept
    # as its own method (rather than inlined into _predict_sklearn) since
    # this loop used to be shared with the LSTM backend too - now sklearn's
    # only caller, but the separation still keeps the recursive-forecast
    # mechanics isolated from the model-specific "how do I produce one
    # step's log-return" logic in step_fn.

    def _iterate_recursive(
        self,
        df: pd.DataFrame,
        steps: int,
        interval: str,
        step_fn: Callable[[pd.DataFrame, bool], tuple[float, float, float] | None],
    ) -> list[dict]:
        """Band: step_fn's [low, high] is only asked for at step 1 (real
        candles). Its distance below/above the point is then held fixed and
        scaled by k ** BAND_HORIZON_EXPONENT around the cumulative forecast
        at step k, so the band widens with the horizon and always contains
        the point."""
        step_delta = _INTERVAL_TIMEDELTA[interval]
        calibrated = self.model_variant in _CALIBRATED_VARIANTS
        # taker_buy_volume is carried along so taker_buy_ratio keeps its real
        # values on real candles (it used to be dropped here, which turned the
        # feature into a constant 0.5 at predict time only).
        carried = ("open", "high", "low", "close", "volume", "taker_buy_volume") if calibrated else ("open", "high", "low", "close", "volume")
        ohlcv = [c for c in carried if c in df.columns]
        history = add_technical_features(df[ohlcv].copy())
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
        origin_close = float(history["close"].iloc[-1])
        cumulative = 0.0
        below = above = 0.0
        for k in range(1, steps + 1):
            step = step_fn(history, k == 1 or not calibrated)
            if step is None:
                # Not enough rolling history yet (e.g. right at the start) - stop early
                # rather than feeding the model garbage.
                logger.warning("Insufficient rolling history for a further step, stopping early")
                break
            log_return, log_return_low, log_return_high = step
            if k == 1:
                # max(0, ...) keeps the point inside the band even if the
                # quantile models disagree with the mean model.
                below = max(0.0, log_return - min(log_return_low, log_return_high))
                above = max(0.0, max(log_return_low, log_return_high) - log_return)

            last_close = float(history["close"].iloc[-1])
            predicted_price = last_close * np.exp(log_return)
            cumulative += log_return
            if calibrated:
                growth = k ** BAND_HORIZON_EXPONENT
                lower = origin_close * np.exp(cumulative - below * growth)
                upper = origin_close * np.exp(cumulative + above * growth)
            else:
                # legacy: one-step band around the previous (predicted) close,
                # point estimate included so it always falls inside.
                lower = last_close * np.exp(min(log_return_low, log_return_high, log_return))
                upper = last_close * np.exp(max(log_return_low, log_return_high, log_return))

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
            if "taker_buy_volume" in history.columns:
                new_row["taker_buy_volume"] = 0.5 * new_row["volume"]  # neutral: no buy/sell lean
            history = pd.concat([history, pd.DataFrame([new_row], index=[last_timestamp])])
            history = add_technical_features(history[ohlcv])
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
