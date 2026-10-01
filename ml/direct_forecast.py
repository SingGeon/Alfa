"""Direct multi-horizon forecasting (the "tuned" model variant).

Why not recursive: the recursive loop (PricePredictor._iterate_recursive,
still used by "legacy") feeds each predicted candle back in as if it were
real. Those synthetic candles have almost no range, so after ~20 steps
volatility_24 falls below the 5th percentile of anything seen in training
and rsi_14 leaves its training range in most runs - the model is then
extrapolating on inputs it has never seen. Result: a flat line for the
regularized model, erratic jumps for the unregularized one.

Here instead, one model per horizon h in DIRECT_HORIZONS is trained on
log(close[t+h] / close[t]) using only the features of candle t - the last
*real* candle at predict time. No synthetic candle ever reaches a model.
Per horizon:
  - mean model: small ensemble of GradientBoostingRegressor;
  - band: quantile models (10% / 90%) + split-conformal widening (CQR)
    measured on the newest rows, so each horizon's band is calibrated on
    that horizon's own real outcomes;
  - shrinkage toward "no change": final return = alpha * model return,
    alpha in [0, 1] picked on the same held-out newest rows (the one that
    minimizes MAE there). alpha ~ 0 means the model showed no skill over
    "price stays where it is" on data it hadn't seen, and the forecast
    honestly says so.
Steps between two trained horizons are interpolated (linear in k, in
log-return space); steps past the last one keep its drift per step and
widen the band as k ** BAND_HORIZON_EXPONENT.

Guards (each logs a warning and is recorded in the diagnostics):
  - every feature of the predict row is clipped to [p1, p99] of training;
  - the change between two consecutive steps is capped at the step-1 band
    half-width (a sudden jump between neighbouring horizons' models).

Every predict() also returns a diagnostics dict (features used, what was
clipped, per-horizon raw/shrunk returns, alpha, band, per-step path) that
the evaluation log stores next to the prediction.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DIRECT_HORIZONS = (1, 2, 4, 8, 12, 24)
_MIN_TRAIN_ROWS = 30
_FEATURE_CLIP_QUANTILES = (0.01, 0.99)
_SHRINK_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


class DirectForecaster:
    def __init__(self, feature_cols: list[str], gbr_params: dict, seeds: tuple[int, ...], band_alpha: float,
                 cqr_min_rows: int, cqr_calibration_fraction: float, horizon_exponent: float):
        self.feature_cols = feature_cols
        self.gbr_params = gbr_params
        self.seeds = seeds
        self.band_alpha = band_alpha
        self.cqr_min_rows = cqr_min_rows
        self.cqr_calibration_fraction = cqr_calibration_fraction
        self.horizon_exponent = horizon_exponent
        self.horizons: dict[int, dict] = {}
        self.clip_low: pd.Series | None = None
        self.clip_high: pd.Series | None = None

    # -- training ----------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        data = df.dropna(subset=self.feature_cols)
        X_all = data[self.feature_cols]
        close = df["close"]
        self.clip_low = X_all.quantile(_FEATURE_CLIP_QUANTILES[0])
        self.clip_high = X_all.quantile(_FEATURE_CLIP_QUANTILES[1])
        low_q, high_q = self.band_alpha / 2, 1 - self.band_alpha / 2

        def gbr(seed, **extra):
            return GradientBoostingRegressor(random_state=seed, **self.gbr_params, **extra)

        self.horizons = {}
        for h in DIRECT_HORIZONS:
            y_all = np.log(close.shift(-h) / close).reindex(X_all.index)
            mask = y_all.notna()
            X, y = X_all[mask], y_all[mask]
            if len(X) < _MIN_TRAIN_ROWS:
                continue
            entry = {"train_rows": len(X), "cal_rows": 0, "cqr_q": 0.0, "alpha": 0.0, "holdout_skill": None}

            if len(X) >= self.cqr_min_rows:
                split = int(len(X) * (1 - self.cqr_calibration_fraction))
                # Drop the last h fit rows: their targets reach into the
                # calibration window, which would leak it into training.
                fit_end = max(split - h, _MIN_TRAIN_ROWS)
                X_fit, y_fit = X.iloc[:fit_end], y.iloc[:fit_end]
                X_cal, y_cal = X.iloc[split:], y.iloc[split:].to_numpy()

                # alpha: shrinkage toward "no change", chosen out of sample.
                probe = gbr(self.seeds[0]).fit(X_fit, y_fit).predict(X_cal)
                errors = [np.abs(a * probe - y_cal).mean() for a in _SHRINK_GRID]
                best = int(np.argmin(errors))
                baseline_err = np.abs(y_cal).mean()
                entry["alpha"] = float(_SHRINK_GRID[best])
                entry["holdout_skill"] = float(1 - errors[-1] / baseline_err) if baseline_err else None

                lower = [gbr(s, loss="quantile", alpha=low_q).fit(X_fit, y_fit) for s in self.seeds]
                upper = [gbr(s, loss="quantile", alpha=high_q).fit(X_fit, y_fit) for s in self.seeds]
                lo = np.mean([m.predict(X_cal) for m in lower], axis=0)
                hi = np.mean([m.predict(X_cal) for m in upper], axis=0)
                scores = np.maximum(lo - y_cal, y_cal - hi)
                n = len(scores)
                level = min(1.0, math.ceil((n + 1) * (1 - self.band_alpha)) / n)
                entry["cqr_q"] = float(np.quantile(scores, level, method="higher"))
                entry["cal_rows"] = n
            else:
                # Too little history to measure skill: alpha stays 0 (no
                # evidence the model beats "no change"), band uncorrected.
                lower = [gbr(s, loss="quantile", alpha=low_q).fit(X, y) for s in self.seeds]
                upper = [gbr(s, loss="quantile", alpha=high_q).fit(X, y) for s in self.seeds]

            entry["models"] = [gbr(s).fit(X, y) for s in self.seeds]
            entry["lower"] = lower
            entry["upper"] = upper
            self.horizons[h] = entry

        if not self.horizons:
            raise ValueError(
                f"Not enough history to train ({len(X_all)} rows) - need at least {_MIN_TRAIN_ROWS}. "
                "Let the collector run longer or backfill more candles."
            )

    @property
    def mean_models_h1(self) -> list:
        first = min(self.horizons)
        return self.horizons[first]["models"]

    # -- prediction ----------------------------------------------------------

    def _clip_row(self, row: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        raw = row.iloc[0]
        clipped = raw.clip(self.clip_low, self.clip_high)
        changed = {
            col: {"raw": float(raw[col]), "clipped_to": float(clipped[col])}
            for col in self.feature_cols
            if raw[col] != clipped[col]
        }
        if changed:
            logger.warning("Direct forecast: features outside training [p1, p99], clipped: %s", sorted(changed))
        return clipped.to_frame().T.astype(float), changed

    def predict(self, df: pd.DataFrame, steps: int) -> tuple[list[tuple[float, float, float]], dict]:
        """([(log_return, log_low, log_high) cumulative from the last real
        close, one per step 1..steps], diagnostics)."""
        row = df[self.feature_cols].iloc[[-1]]
        if row.isna().any(axis=None):
            missing = [c for c in self.feature_cols if pd.isna(row.iloc[0][c])]
            logger.warning("Direct forecast: last candle has missing features %s, no forecast", missing)
            return [], {"method": "direct", "error": "missing_features", "missing": missing}
        row_c, clipped = self._clip_row(row)

        per_h = {}
        for h, e in self.horizons.items():
            raw = float(np.mean([m.predict(row_c)[0] for m in e["models"]]))
            point = e["alpha"] * raw
            lo = float(np.mean([m.predict(row_c)[0] for m in e["lower"]])) - e["cqr_q"]
            hi = float(np.mean([m.predict(row_c)[0] for m in e["upper"]])) + e["cqr_q"]
            lo, hi = min(lo, hi, point), max(lo, hi, point)
            per_h[h] = {"h": h, "raw_return": raw, "alpha": e["alpha"], "return": point, "low": lo, "high": hi,
                        "cqr_q": e["cqr_q"], "cal_rows": e["cal_rows"], "holdout_skill": e["holdout_skill"]}

        known = sorted(per_h)
        path, step_log = [], []
        first_half_width = None
        prev_point = 0.0
        for k in range(1, steps + 1):
            if k in per_h:
                p = per_h[k]
                point, below, above, source = p["return"], p["return"] - p["low"], p["high"] - p["return"], "model"
            elif k < known[-1]:
                a = max(h for h in known if h < k) if any(h < k for h in known) else known[0]
                b = min(h for h in known if h > k)
                w = (k - a) / (b - a) if b != a else 0.0
                pa, pb = per_h[a], per_h[b]
                point = pa["return"] + w * (pb["return"] - pa["return"])
                below = (pa["return"] - pa["low"]) + w * ((pb["return"] - pb["low"]) - (pa["return"] - pa["low"]))
                above = (pa["high"] - pa["return"]) + w * ((pb["high"] - pb["return"]) - (pa["high"] - pa["return"]))
                source = "interpolated"
            else:
                last = per_h[known[-1]]
                scale = k / known[-1]
                growth = scale ** self.horizon_exponent
                point = last["return"] * scale
                below, above = (last["return"] - last["low"]) * growth, (last["high"] - last["return"]) * growth
                source = "extrapolated"

            if first_half_width is None:
                first_half_width = max((below + above) / 2, 1e-12)
            step_return = point - prev_point
            guarded = abs(step_return) > first_half_width
            if guarded:
                point = prev_point + math.copysign(first_half_width, step_return)
                logger.warning(
                    "Direct forecast: step %d moved %.4f in log-return (> step-1 band half-width %.4f), capped",
                    k, step_return, first_half_width,
                )
            path.append((point, point - below, point + above))
            step_log.append({"k": k, "source": source, "return": point, "low": point - below, "high": point + above,
                             "step_return": point - prev_point, "capped": guarded})
            prev_point = point

        diagnostics = {
            "method": "direct",
            "features": {c: float(row.iloc[0][c]) for c in self.feature_cols},
            "clipped_features": clipped,
            "horizons": [per_h[h] for h in known],
            "steps": step_log,
        }
        return path, diagnostics
