"""Statistical validation of rule-based trading signals against future returns.

Chapter 7 of "Algorithmic Trading Hands-On Approach Using Python" (Ziad
Francis): before trusting a pattern/indicator because it "looks right",
test whether it actually correlates with what happens next - a one-way
ANOVA comparing future returns when the signal fired vs when it didn't,
and a chi-squared independence test between "signal present" and "future
direction" (up/flat/down). A pattern that can't clear either bar is
decoration, not information.

Generic: works on any boolean signal aligned to a price Series' index -
callers (ml.pattern_recognition, ml.box_breakout, ...) build the signal
array from their own detection logic; this module only scores it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_SAMPLES = 20          # below this, a p-value is noise, not evidence
FLAT_THRESHOLD_PCT = 0.1  # |future return| below this % counts as "flat"


def label_future_return(close: pd.Series, horizon: int) -> pd.DataFrame:
    """Future log return (%) over `horizon` bars from each bar, and its
    category (up/flat/down). The last `horizon` rows have no future to
    look at (NaN/None) - never backfilled, so a validator can't score on
    information from the future."""
    log_close = np.log(close.to_numpy(dtype=float))
    n = len(log_close)
    pct = np.full(n, np.nan)
    if horizon < n:
        pct[:-horizon] = (log_close[horizon:] - log_close[:-horizon]) * 100
    category = np.full(n, None, dtype=object)
    up, down = pct > FLAT_THRESHOLD_PCT, pct < -FLAT_THRESHOLD_PCT
    category[up] = "up"
    category[down] = "down"
    category[~up & ~down & ~np.isnan(pct)] = "flat"
    return pd.DataFrame({"future_return_pct": pct, "category": category}, index=close.index)


def validate_signal(signal: np.ndarray | pd.Series, future: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> dict:
    """Does `signal` (boolean, True where a pattern/rule fired) carry real
    information about `future` (label_future_return's output)?

    Two different angles on the same question - either clearing the bar
    counts as real signal, not just one lucky test:
      - ANOVA (f_oneway): is the mean future return different when the
        signal fired vs when it didn't?
      - Chi-squared: is (signal present/absent) independent of (future
        category up/flat/down)?

    Returns n_signal/n_baseline, mean future return in each group, both
    p-values (None if not computable), `significant` (needs n_signal >=
    min_samples AND at least one p-value < 0.05), and `low_sample`."""
    signal = np.asarray(signal, dtype=bool)
    ret = future["future_return_pct"].to_numpy(dtype=float)
    cat = future["category"].to_numpy(dtype=object)
    valid = ~np.isnan(ret)
    signal, ret, cat = signal[valid], ret[valid], cat[valid]

    n_signal, n_baseline = int(signal.sum()), int((~signal).sum())
    out = {
        "n_signal": n_signal, "n_baseline": n_baseline,
        "mean_return_signal_pct": round(float(ret[signal].mean()), 4) if n_signal else None,
        "mean_return_baseline_pct": round(float(ret[~signal].mean()), 4) if n_baseline else None,
        "anova_p": None, "chi2_p": None, "significant": False,
        "low_sample": n_signal < min_samples,
    }
    if n_signal < 2 or n_baseline < 2:
        return out

    out["anova_p"] = round(float(stats.f_oneway(ret[signal], ret[~signal]).pvalue), 4)

    table = pd.crosstab(signal, cat)
    if table.shape[0] == 2 and table.shape[1] >= 2:
        try:
            out["chi2_p"] = round(float(stats.chi2_contingency(table.to_numpy())[1]), 4)
        except ValueError:
            pass   # a category with zero occurrences in one row - leave chi2_p as None

    out["significant"] = (not out["low_sample"]) and any(
        p is not None and p < 0.05 for p in (out["anova_p"], out["chi2_p"])
    )
    return out
