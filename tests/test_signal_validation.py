import numpy as np
import pandas as pd

from ml import signal_validation as sv


def test_label_future_return_direction_and_category():
    # A clean uptrend: every bar's horizon-ahead return is positive and "up".
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 130, 30), index=idx)
    out = sv.label_future_return(close, horizon=5)
    assert (out["future_return_pct"].iloc[:-5] > 0).all()
    assert (out["category"].iloc[:-5] == "up").all()
    # The last `horizon` rows have no future to look at.
    assert out["future_return_pct"].iloc[-5:].isna().all()
    assert out["category"].iloc[-5:].apply(lambda x: x is None).all()


def test_label_future_return_flat_band():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    close = pd.Series([100.0, 100.0005, 100.0], index=idx)   # ~0.0005% move: inside the flat band
    out = sv.label_future_return(close, horizon=1)
    assert out["category"].iloc[0] == "flat"


def test_validate_signal_detects_a_real_effect():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.random(n) < 0.2                      # ~40 occurrences
    ret = np.where(signal, 2.0, 0.0) + rng.normal(0, 0.05, n)   # signal rows clearly higher
    future = pd.DataFrame({"future_return_pct": ret, "category": np.where(ret > 0.1, "up", "flat")})

    out = sv.validate_signal(signal, future)
    assert out["n_signal"] == int(signal.sum())
    assert not out["low_sample"]
    assert out["mean_return_signal_pct"] > out["mean_return_baseline_pct"]
    assert out["anova_p"] < 0.05
    assert out["significant"] is True


def test_validate_signal_low_sample_is_never_significant():
    rng = np.random.default_rng(1)
    n = 200
    signal = np.zeros(n, dtype=bool)
    signal[:5] = True                                   # only 5 occurrences: below MIN_SAMPLES
    ret = np.where(signal, 5.0, 0.0) + rng.normal(0, 0.01, n)   # would otherwise look very significant
    future = pd.DataFrame({"future_return_pct": ret, "category": np.where(ret > 0.1, "up", "flat")})

    out = sv.validate_signal(signal, future)
    assert out["low_sample"] is True
    assert out["significant"] is False


def test_validate_signal_no_effect_when_identical():
    n = 200
    signal = np.arange(n) % 3 == 0
    ret = np.zeros(n)   # same (zero) return whether or not the signal fired
    future = pd.DataFrame({"future_return_pct": ret, "category": np.full(n, "flat", dtype=object)})

    out = sv.validate_signal(signal, future)
    assert out["mean_return_signal_pct"] == 0.0 == out["mean_return_baseline_pct"]
    assert out["significant"] is False


def test_validate_signal_ignores_rows_with_no_future():
    n = 10
    signal = np.array([True, False] * 5)
    ret = np.array([1.0, 0.0] * 4 + [np.nan, np.nan])
    future = pd.DataFrame({"future_return_pct": ret, "category": ["up", "flat"] * 4 + [None, None]})

    out = sv.validate_signal(signal, future)
    assert out["n_signal"] + out["n_baseline"] == 8
