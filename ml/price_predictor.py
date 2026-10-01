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
    predict_with_diagnostics(df, steps) -> (that list, diagnostics dict | None)

Two sklearn variants (MODEL_VARIANTS):
  - "tuned" is an evolving population of small models with emotions
    (ml/evolution.py): each one lives only as long as it beats "no change"
    on real outcomes, the dead are replaced by children of the strongest,
    the forecast is the strongest adults' vote, and its band is calibrated
    on the population's own past errors. Diagnostics per prediction;
  - "legacy" is the original recursive forecaster, kept exactly as it was
    as the fixed reference ("control") the tuned one is measured against.

`df` is the output of ml.features.build_feature_frame().
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from ml.evolution import EvolutionaryForecaster, evolve
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

# Variants that are an evolving population (ml/evolution.py) instead of the
# recursive forecaster. "legacy" is deliberately not in it: it stays the
# unchanged recursive reference.
_EVOLUTION_VARIANTS = ("tuned",)

# Market context (ml/market_context.py) the recursive model also learns from
# when the caller added it to the frame. Missing values (a source that was
# down, funding before it existed) become 0 = neutral, since
# GradientBoostingRegressor can't take NaN. Like sentiment, each is held at
# its last known value over the forecast.
LEGACY_CONTEXT_COLUMNS = ("fear_greed", "fear_greed_chg", "funding", "premium")


class PricePredictor:
    def __init__(
        self,
        backend: str = "sklearn",
        feature_cols: list[str] | None = None,
        ensemble_size: int = len(_ENSEMBLE_SEEDS),
        model_variant: str = "tuned",
        evolution_key: str | None = None,
    ):
        """`evolution_key` (e.g. "ethereum_1h"): the "tuned" population for
        it keeps living across fits and app restarts (ml/evolution.evolve)
        instead of being replayed from scratch on each fit."""
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model_variant: {model_variant!r}; must be one of {MODEL_VARIANTS}")
        self.backend = backend
        self.feature_cols = feature_cols or FEATURE_COLUMNS
        self._fit_cols = list(self.feature_cols)  # + the context columns present at fit time
        self.ensemble_size = max(1, min(ensemble_size, len(_ENSEMBLE_SEEDS)))
        self.model_variant = model_variant
        self.evolution_key = evolution_key
        self._models = []          # ensemble of mean predictors (sklearn backend)
        self._models_lower = []    # ensemble of lower-quantile predictors
        self._models_upper = []    # ensemble of upper-quantile predictors
        self._evolution: EvolutionaryForecaster | None = None  # see _EVOLUTION_VARIANTS
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
        return self.predict_with_diagnostics(df, steps, interval)[0]

    def predict_with_diagnostics(self, df: pd.DataFrame, steps: int, interval: str = "1h") -> tuple[list[dict], dict | None]:
        """Same forecast as predict(), plus the diagnostics log of an
        evolution variant (None for the others). Returned rather than stored on self:
        one fitted predictor is shared by concurrent requests (api/services)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if self.backend == "sklearn":
            if self._evolution is not None:
                return self._predict_evolution(df, steps, interval)
            return self._predict_sklearn(df, steps, interval), None
        if self.backend == "prophet":
            return self._predict_prophet(df, steps, interval), None
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
            return dict(zip(self._fit_cols, (float(v) for v in importances)))
        return None

    # -- sklearn backend (default) --------------------------------------

    def _predict_evolution(self, df: pd.DataFrame, steps: int, interval: str) -> tuple[list[dict], dict]:
        path, diagnostics = self._evolution.predict(df, steps)
        origin = float(df["close"].iloc[-1])
        step_delta = _INTERVAL_TIMEDELTA[interval]
        ts = df.index[-1]
        diagnostics["origin_close"] = origin
        results = []
        for log_return, log_low, log_high in path:
            ts = ts + step_delta
            results.append({
                "timestamp": ts,
                "predicted_price": origin * np.exp(log_return),
                "lower": origin * np.exp(log_low),
                "upper": origin * np.exp(log_high),
            })
        return results, diagnostics

    def _fit_sklearn(self, df: pd.DataFrame) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        if self.model_variant in _EVOLUTION_VARIANTS:
            self._evolution = evolve(df, self.evolution_key)
            self._models = []  # ridge organisms: no tree feature importances
            return

        context = [c for c in LEGACY_CONTEXT_COLUMNS if c in df.columns]
        self._fit_cols = list(self.feature_cols) + context
        df = df.assign(**{c: df[c].fillna(0.0) for c in context})
        X, y = make_supervised(df, self._fit_cols)
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
        self._models_lower = [
            GradientBoostingRegressor(loss="quantile", alpha=0.1, random_state=s, **gbr_params).fit(X, y)
            for s in seeds
        ]
        self._models_upper = [
            GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=s, **gbr_params).fit(X, y)
            for s in seeds
        ]

    def _predict_sklearn(self, df: pd.DataFrame, steps: int, interval: str) -> list[dict]:
        def step_fn(history: pd.DataFrame) -> tuple[float, float, float] | None:
            row = history.iloc[[-1]][self._fit_cols]
            if row.isna().any(axis=None):
                return None
            log_return = float(np.mean([m.predict(row)[0] for m in self._models]))
            log_return_low = float(np.mean([m.predict(row)[0] for m in self._models_lower]))
            log_return_high = float(np.mean([m.predict(row)[0] for m in self._models_upper]))
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
        step_fn: Callable[[pd.DataFrame], tuple[float, float, float] | None],
    ) -> list[dict]:
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
        context = [c for c in self._fit_cols if c in LEGACY_CONTEXT_COLUMNS]
        last_context = {c: (float(df[c].fillna(0.0).iloc[-1]) if c in df.columns and len(df) else 0.0) for c in context}
        for c in context:
            history[c] = df[c].fillna(0.0) if c in df.columns else last_context[c]

        results = []
        last_timestamp = history.index[-1]
        for _ in range(steps):
            step = step_fn(history)
            if step is None:
                # Not enough rolling history yet (e.g. right at the start) - stop early
                # rather than feeding the model garbage.
                logger.warning("Insufficient rolling history for a further step, stopping early")
                break
            log_return, log_return_low, log_return_high = step

            last_close = float(history["close"].iloc[-1])
            predicted_price = last_close * np.exp(log_return)
            # Including log_return itself in the min/max (not just the two
            # quantile bounds) guarantees the point estimate always falls
            # inside [lower, upper] - true by construction for sklearn's
            # same-distribution quantile models.
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
            history = pd.concat([history, pd.DataFrame([new_row], index=[last_timestamp])])
            history = add_technical_features(history[["open", "high", "low", "close", "volume"]])
            history["sentiment"] = history["sentiment"] if "sentiment" in history else last_sentiment
            history.loc[last_timestamp, "sentiment"] = last_sentiment
            history["btc_return_1"] = history["btc_return_1"] if "btc_return_1" in history else last_btc_return
            history.loc[last_timestamp, "btc_return_1"] = last_btc_return
            history["tvl_momentum"] = history["tvl_momentum"] if "tvl_momentum" in history else last_tvl_momentum
            history.loc[last_timestamp, "tvl_momentum"] = last_tvl_momentum
            for c in context:
                history[c] = history[c] if c in history else last_context[c]
                history.loc[last_timestamp, c] = last_context[c]

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
