from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from evaluation import charts
from ml import monte_carlo as mc

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ohlc_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    idx = pd.DatetimeIndex([T0 + timedelta(hours=i) for i in range(n)])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 100.0}, index=idx)


def test_apply_blur_keeps_ohlc_invariants():
    df = _ohlc_df()
    out = mc.apply_blur(df, sigma_pct=3.0, seed=1)
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()
    assert len(out) == len(df)


def test_apply_blur_with_zero_sigma_is_a_noop():
    df = _ohlc_df()
    out = mc.apply_blur(df, sigma_pct=0.0, seed=1)
    for col in ("open", "high", "low", "close"):
        assert np.allclose(out[col].to_numpy(), df[col].to_numpy())


def test_apply_blur_is_deterministic_given_a_seed():
    df = _ohlc_df()
    a = mc.apply_blur(df, sigma_pct=2.0, seed=42)
    b = mc.apply_blur(df, sigma_pct=2.0, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_run_monte_carlo_on_a_deterministic_strategy_with_no_noise():
    df = _ohlc_df()

    def strategy_fn(variant):
        return {"total_return_pct": float(variant["close"].iloc[-1] / variant["close"].iloc[0] - 1) * 100,
                "max_drawdown_pct": 0.0}

    results = mc.run_monte_carlo(strategy_fn, df, n_runs=10, sigma_pct=0.0, seed=0)
    assert len(results) == 10
    rets = [r["total_return_pct"] for r in results]
    assert all(r == pytest.approx(rets[0]) for r in rets)   # no noise -> every run is identical


def test_run_monte_carlo_skips_runs_that_raise():
    df = _ohlc_df()

    def strategy_fn(variant):
        raise ValueError("boom")

    assert mc.run_monte_carlo(strategy_fn, df, n_runs=5, sigma_pct=1.0) == []


def test_summarize_distribution_reports_median_min_max():
    results = [{"total_return_pct": v, "max_drawdown_pct": abs(v)} for v in (-5.0, 0.0, 10.0)]
    out = mc.summarize_distribution(results, keys=("total_return_pct",), drawdown_key="max_drawdown_pct")
    assert out["n_runs"] == 3
    assert out["total_return_pct"] == {"median": 0.0, "min": -5.0, "max": 10.0, "n": 3}
    assert isinstance(out["bimodal_drawdown"], bool)


def test_is_bimodal_detects_two_separated_clusters():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-10, 1, 200), rng.normal(10, 1, 200)])
    assert mc._is_bimodal(values) is True


def test_is_bimodal_false_for_a_single_cluster():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 200)
    assert mc._is_bimodal(values) is False


def test_is_bimodal_false_for_too_few_samples():
    assert mc._is_bimodal(np.array([1.0, 2.0, -1.0, 5.0])) is False


def test_render_monte_carlo_report_draws_a_png():
    rng = np.random.default_rng(0)
    results = [{"total_return_pct": float(v), "max_drawdown_pct": float(abs(v))}
               for v in rng.normal(5, 2, 60)]
    path = charts.render_monte_carlo_report(results, "Box Breakout - ETHUSDT 1h", "test_report")
    assert path == charts.monte_carlo_report_path("test_report")
    with Image.open(path) as img:
        assert img.size[0] > 0 and img.size[1] > 0


def test_render_monte_carlo_report_with_equity_curves_and_sortino():
    rng = np.random.default_rng(0)
    results = [{
        "total_return_pct": float(v), "max_drawdown_pct": float(abs(v)), "sortino": float(v / 10),
        "equity_curve": list(np.cumprod(1 + rng.normal(0.01, 0.05, rng.integers(5, 20)))),
    } for v in rng.normal(5, 2, 40)]
    path = charts.render_monte_carlo_report(results, "Box Breakout - ETHUSDT 1h", "test_report_full")
    with Image.open(path) as img:
        assert img.size[0] > 0 and img.size[1] > 0
