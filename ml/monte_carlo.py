"""Monte Carlo robustness testing (Chapter 10 of "Algorithmic Trading Hands-On Approach Using
Python", Ziad Francis): does a strategy's backtest result hold up under small, realistic price
noise, or is it a fluke of one specific historical path? Settings are never re-optimized here,
only replayed - exactly the book's "applying normally distributed blur" method (p.272-279),
including its headline warning sign: a *bimodal* drawdown distribution, meaning two different
outcome "regimes" hiding inside what looked like one result.

    .venv/bin/python -m ml.monte_carlo                    # box breakout, ETHUSDT 1h
    .venv/bin/python -m ml.monte_carlo --interval 4h --runs 1000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

OHLC_COLS = ("open", "high", "low", "close")


def apply_blur(df: pd.DataFrame, sigma_pct: float, seed: int | None = None) -> pd.DataFrame:
    """A new OHLC frame with small gaussian noise (sigma_pct, e.g. 2.0 = 2% std) applied bar by
    bar - every OHLC column of a given bar scaled by the *same* random factor, so the bar's own
    candle shape (body/wicks) survives; high/low re-clamped afterward so the result stays a
    valid OHLC bar (high = max of the four, low = min)."""
    rng = np.random.default_rng(seed)
    factor = 1 + rng.normal(0.0, sigma_pct / 100, len(df))
    cols = [c for c in OHLC_COLS if c in df.columns]
    out = df.copy()
    for col in cols:
        out[col] = df[col].to_numpy(float) * factor
    out["high"] = out[cols].max(axis=1)
    out["low"] = out[cols].min(axis=1)
    return out


def realistic_sigma_pct(df: pd.DataFrame, scale: float = 0.5) -> float:
    """A sensible default noise level for apply_blur, calibrated to *this data's own* bar-to-bar
    volatility (its 1-step log-return std, as a %) - not a fixed number copied from the book. A
    1-3% blur that's small next to a daily-bar asset's normal ~1-3% daily moves would dwarf an
    hourly series whose bars typically move ~0.3-0.7%; used uncalibrated, the "robustness" result
    degenerates into pure injected noise outrunning the real signal, not a genuine stress test.
    `scale` keeps the perturbation a *fraction* of normal volatility (Ch. 10's blur is meant to
    be small), not a change of volatility regime."""
    close = df["close"].to_numpy(float)
    ret_std = float(np.std(np.diff(np.log(close))))
    return round(ret_std * 100 * scale, 4)


def run_monte_carlo(strategy_fn, df: pd.DataFrame, n_runs: int = 500, sigma_pct: float | None = None,
                     seed: int = 0) -> list[dict]:
    """Reruns `strategy_fn(df_variant) -> dict` (a summarize()/Result-shaped result) `n_runs`
    times on independently blurred variants of `df`, with the strategy's settings held fixed -
    never re-optimized, only replayed. `sigma_pct` defaults to realistic_sigma_pct(df) - this
    data's own bar-to-bar volatility, not a fixed book-generic number (see its docstring for
    why that matters). Returns the list of per-run result dicts (any run that raises is skipped,
    not counted)."""
    sigma_pct = realistic_sigma_pct(df) if sigma_pct is None else sigma_pct
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_runs):
        variant = apply_blur(df, sigma_pct, seed=int(rng.integers(0, 2**31 - 1)))
        try:
            out.append(strategy_fn(variant))
        except Exception:
            continue
    return out


def _is_bimodal(values: np.ndarray, min_samples: int = 20) -> bool:
    """A dependency-light bimodality check (Ch. 10's warning, p.274): KDE the distribution and
    look for two peaks separated by a valley at least 20% below the shorter of the two - a
    single smooth hump is normal sampling variability, two separated humps mean two different
    outcome "regimes" hiding in what looked like one result, which is the book's actual warning
    sign (and exactly the kind of thing a single backtest can't show you)."""
    values = values[np.isfinite(values)]
    if len(values) < min_samples or np.std(values) == 0:
        return False
    kde = gaussian_kde(values)
    xs = np.linspace(values.min(), values.max(), 200)
    density = kde(xs)
    peaks = [i for i in range(1, len(density) - 1) if density[i] > density[i - 1] and density[i] > density[i + 1]]
    if len(peaks) < 2:
        return False
    top_two = sorted((density[i] for i in peaks), reverse=True)[:2]
    valley = min(density[peaks[i]:peaks[i + 1] + 1].min() for i in range(len(peaks) - 1))
    return bool(valley < 0.8 * top_two[1])


def summarize_distribution(
    results: list[dict],
    keys: tuple[str, ...] = ("total_return_pct", "max_drawdown_pct", "sortino", "calmar"),
    drawdown_key: str = "max_drawdown_pct",
) -> dict:
    """median/min/max of each metric across every Monte Carlo run, plus `n_runs` and a
    bimodality flag on `drawdown_key` specifically - the book's own headline finding (p.274)."""
    out: dict = {"n_runs": len(results)}
    for k in keys:
        vals = np.array([r[k] for r in results if r.get(k) is not None], dtype=float)
        out[k] = {
            "median": round(float(np.median(vals)), 3) if len(vals) else None,
            "min": round(float(vals.min()), 3) if len(vals) else None,
            "max": round(float(vals.max()), 3) if len(vals) else None,
            "n": len(vals),
        }
    dd = np.array([r[drawdown_key] for r in results if r.get(drawdown_key) is not None], dtype=float)
    out["bimodal_drawdown"] = _is_bimodal(dd)
    return out


def main() -> None:
    from ml import box_breakout as bb
    from ml.features import candles_to_dataframe
    from ml.walk_forward import fetch_binance

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--sigma-pct", type=float, default=None,
                         help="default: this data's own bar-to-bar return std x 0.5 (see realistic_sigma_pct)")
    args = parser.parse_args()

    rows = fetch_binance("ETHUSDT", args.interval, 20_000)
    df = candles_to_dataframe(rows)
    s = bb.BoxSettings()
    sigma_pct = args.sigma_pct if args.sigma_pct is not None else realistic_sigma_pct(df)

    def strategy_fn(variant: pd.DataFrame) -> dict:
        boxes = bb.detect_boxes(variant, s)
        trades = bb.simulate(variant, boxes, s)
        result = bb.summarize(variant, trades, s)
        result["equity_curve"] = bb.equity_curve(trades, s.fee_pct).tolist()
        return result

    base = strategy_fn(df)
    results = run_monte_carlo(strategy_fn, df, n_runs=args.runs, sigma_pct=sigma_pct)
    dist = summarize_distribution(results)
    print(f"Base run (no noise): return {base['total_return_pct']:+.1f}%, drawdown {base['max_drawdown_pct']:.1f}%, "
          f"{base['trades']} trades")
    print(f"\n{dist['n_runs']} Monte Carlo runs, {sigma_pct}% noise:")
    for k in ("total_return_pct", "max_drawdown_pct", "sortino", "calmar"):
        d = dist[k]
        print(f"  {k}: median {d['median']}, range [{d['min']}, {d['max']}] (n={d['n']})")
    if dist["bimodal_drawdown"]:
        print("\n  WARNING: drawdown distribution looks bimodal - two different outcome regimes "
              "hiding in this result (see Ch. 10, p.274).")

    from evaluation import charts
    path = charts.render_monte_carlo_report(results, f"Box Breakout - ETHUSDT {args.interval}", "box_breakout")
    print(f"\n  Report saved to {path}")


if __name__ == "__main__":
    main()
