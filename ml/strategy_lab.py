"""Strategy lab: which way of trading ETH makes the most money after fees?

The direction of the next candles is not predictable here (see README,
"Walk-forward backtest": ~50% over nine years), but how much the price will
move is (the volatility of the last day predicts the next day's with a
correlation of ~0.66). Every strategy below trades on that, alone or
combined, on 100 EUR (virtual) with Binance's 0.1% fee per side:

- buy_hold       buy ETH once and keep it (the baseline to beat)
- vol_target     long ETH, sized by the volatility forecast: smaller when the
                 market is wild, all-in when it is calm (option 3)
- range          mean reversion inside the forecast range: buy low in the
                 band, sell at its middle (short above it on long_short), stand
                 aside when the forecast says the market is too wild or
                 trending (option 2)
- range_vt       range, each trade sized like vol_target (2 + 3)
- core_range     a vol_target core plus range trades on top, capped at 100%
                 of the money (2 + 3, the core earns the trend, the range the
                 swings)

Option 1, the range forecast itself, is scored too: how often the real
high/low of the next H candles stayed inside the forecast band.

Honest by construction:
- The volatility model is refit every year only on the candles before it.
- Each strategy's settings are picked on TRAIN (up to 2022) only, by the
  most money with a drawdown limit; TEST (2023 on) is scored once with the
  settings fixed, so a good TEST number is not a fit to the past.
- A decision at a candle's close only earns from the next candle on.

    .venv/bin/python -m ml.strategy_lab                  # 1h and 4h, ~2017 to now
    .venv/bin/python -m ml.strategy_lab --interval 15m
"""
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.walk_forward import fetch_binance

FEE = 0.001
START = 100.0
H = 24                                   # forecast horizon, in candles
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
MAX_TRAIN_DRAWDOWN = 0.6                 # settings losing more than this on TRAIN are not picked
CANDLES = {"15m": 330_000, "1h": 83_000, "4h": 21_000, "1d": 3_500}
PER_YEAR = {"15m": 35_040, "1h": 8_760, "4h": 2_190, "1d": 365}
VOL_WINDOWS = {"1d": (5, 20, 60)}          # in candles; 720 daily candles would be 2 years
DEFAULT_VOL_WINDOWS = (24, 168, 720)
MIN_FIT_ROWS = {"1d": 250}


# -- the forecast ----------------------------------------------------------

def forecast_volatility(df: pd.DataFrame, interval: str = "1h") -> pd.Series:
    """Per-candle volatility expected over the next H candles (std of one
    candle's log return), out of sample: for every year, a regression of the
    log of the realized next-H volatility on the logs of the past 24/168/720
    candles' volatility, fit only on candles whose next H had closed before
    that year began."""
    r = np.log(df["close"]).diff()
    windows = VOL_WINDOWS.get(interval, DEFAULT_VOL_WINDOWS)
    X = pd.concat({f"v{w}": np.log(r.rolling(w).std()) for w in windows}, axis=1)
    y = np.log(r[::-1].rolling(H).std()[::-1].shift(-1))
    out = pd.Series(np.nan, index=df.index)
    years = sorted(set(df.index.year))
    for year in years[1:]:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        fit_rows = X.index < start
        fit_rows[max(0, fit_rows.sum() - H):] = False      # their target must have closed
        ok = fit_rows & X.notna().all(axis=1).values & y.notna().values
        if ok.sum() < MIN_FIT_ROWS.get(interval, 500):
            continue
        A = np.column_stack([X[ok].values, np.ones(ok.sum())])
        coef, *_ = np.linalg.lstsq(A, y[ok].values, rcond=None)
        rows = (X.index >= start) & (X.index < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
        Xr = X[rows]
        good = Xr.notna().all(axis=1)
        out[Xr.index[good]] = np.exp(np.column_stack([Xr[good].values, np.ones(good.sum())]) @ coef)
    return out


def range_coverage(df: pd.DataFrame, sigma: pd.Series, k: float = 1.28) -> dict:
    """Option 1: the forecast band close*exp(+-k*sigma*sqrt(H)); how often the
    real high and low of the next H candles stayed inside it."""
    lc = np.log(df["close"])
    width = k * sigma * math.sqrt(H)
    fut_hi = np.log(df["high"])[::-1].rolling(H).max()[::-1].shift(-1)
    fut_lo = np.log(df["low"])[::-1].rolling(H).min()[::-1].shift(-1)
    ok = sigma.notna() & fut_hi.notna()
    inside_hi = (fut_hi[ok] <= lc[ok] + width[ok]).mean()
    inside_lo = (fut_lo[ok] >= lc[ok] - width[ok]).mean()
    end_in = ((lc.shift(-H) - lc).abs()[ok] <= width[ok]).mean()
    return {"high_inside": round(inside_hi, 3), "low_inside": round(inside_lo, 3),
            "close_after_H_inside": round(end_in, 3), "mean_band_pct": round(float((np.exp(width[ok]) - 1).mean() * 100), 2)}


# -- the strategies: a target position (fraction of the money in ETH, <0 short) per candle

def pos_buy_hold(ctx, p) -> np.ndarray:
    return np.ones(ctx["n"])


def pos_vol_target(ctx, p) -> np.ndarray:
    """Rebalanced only when the target drifts more than `band` from what is
    held: every candle's small change would be paid for in fees."""
    ann = ctx["sigma"] * math.sqrt(ctx["per_year"])
    want = np.nan_to_num(np.clip(p["target"] / ann, 0.0, 1.0))
    band = p.get("band", 0.2)
    pos = np.empty_like(want)
    cur = 0.0
    for t, w in enumerate(want):
        if abs(w - cur) > band or (w == 0.0) != (cur == 0.0):
            cur = w
        pos[t] = cur
    return pos


def pos_range(ctx, p, size: np.ndarray | None = None) -> np.ndarray:
    """Buy when the close is k band-widths below its moving middle, sell back
    at the middle; short the mirror image on long_short. Stand aside (and
    close) when the forecast volatility is in the wildest part of history or
    the last week trended too hard."""
    n, z, trend, wild = ctx["n"], ctx["z"][p["center"]], ctx["trend"], ctx["vol_rank"] > p["vol_cap"]
    pos = np.zeros(n)
    cur = 0.0
    k, stop, short = p["k"], p["k"] + p["stop"], p["long_short"]
    for t in range(n):
        zt = z[t]
        if not np.isfinite(zt) or wild[t] or abs(trend[t]) > p["trend_max"]:
            cur = 0.0
        elif cur > 0:
            if zt >= p["exit"] or zt < -stop:
                cur = 0.0
        elif cur < 0:
            if zt <= -p["exit"] or zt > stop:
                cur = 0.0
        elif -stop <= zt < -k:
            cur = 1.0
        elif short and stop >= zt > k:
            cur = -1.0
        pos[t] = cur
    return pos if size is None else pos * size


def pos_range_vt(ctx, p) -> np.ndarray:
    return pos_range(ctx, p, size=pos_vol_target(ctx, p))


def pos_box(ctx, p) -> np.ndarray:
    return ctx["box"]


def pos_core_box(ctx, p) -> np.ndarray:
    """A vol_target core (earns the calm uptrends) plus the Box Breakout
    trades (ride big breakouts either way, the crashes included)."""
    return np.clip(p["core"] * pos_vol_target(ctx, p) + p["box_w"] * ctx["box"], -1.0, 1.0)


def pos_core_range(ctx, p) -> np.ndarray:
    core = p["core"] * pos_vol_target(ctx, p)
    return np.clip(core + pos_range(ctx, p), -1.0, 1.0)


GRIDS = {
    "buy_hold": {},
    "vol_target": {"target": [0.15, 0.25, 0.4, 0.6, 0.9], "band": [0.1, 0.25]},
    "range": {"center": [24, 72, 168], "k": [1.0, 1.5, 2.0], "exit": [0.0, 0.5], "stop": [1.0, 3.0],
              "vol_cap": [0.8, 1.01], "trend_max": [2.0, 99.0], "long_short": [False, True]},
    "range_vt": {"center": [24, 72], "k": [1.0, 1.5, 2.0], "exit": [0.0, 0.5], "stop": [1.0, 3.0],
                 "vol_cap": [0.8, 1.01], "trend_max": [2.0, 99.0], "long_short": [False, True], "target": [0.5, 0.9]},
    "core_range": {"center": [24, 72], "k": [1.0, 1.5, 2.0], "exit": [0.0], "stop": [1.0, 3.0],
                   "vol_cap": [1.01], "trend_max": [2.0, 99.0], "long_short": [False, True],
                   "target": [0.5, 0.9], "core": [0.3, 0.5, 0.7]},
    "box": {},
    "core_box": {"target": [0.25, 0.5, 0.9], "band": [0.1, 0.25], "core": [0.3, 0.5, 0.7, 1.0],
                 "box_w": [0.3, 0.5, 0.7, 1.0]},
}
POSITIONS = {"buy_hold": pos_buy_hold, "vol_target": pos_vol_target, "range": pos_range,
             "range_vt": pos_range_vt, "core_range": pos_core_range, "box": pos_box, "core_box": pos_core_box}


# -- money -----------------------------------------------------------------

@dataclass
class Result:
    money: float
    ret_pct: float
    cagr_pct: float
    max_dd_pct: float
    trades: int
    fees: float
    exposure_pct: float
    sharpe: float


def wealth(pos: np.ndarray, r_next: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Money after each candle in `mask` (START before the first), same
    accounting as simulate()."""
    idx = np.flatnonzero(mask)
    p = np.nan_to_num(pos)[idx]
    turnover = np.abs(p - np.concatenate([[0.0], p[:-1]]))
    return START * np.cumprod((1 + p * np.expm1(r_next[idx])) * (1 - FEE * turnover))


def simulate(pos: np.ndarray, r_next: np.ndarray, per_year: int, mask: np.ndarray) -> Result:
    """Hold pos[t] from candle t's close to t+1's, pay FEE on every change,
    over the candles in `mask` only (starting flat with START)."""
    pos = np.where(mask, np.nan_to_num(pos), 0.0)
    idx = np.flatnonzero(mask)
    p = pos[idx]
    prev = np.concatenate([[0.0], p[:-1]])
    turnover = np.abs(p - prev)
    gross = p * np.expm1(r_next[idx])
    step = (1 + gross) * (1 - FEE * turnover)
    wealth = START * np.cumprod(step)
    fees = float((np.concatenate([[START], wealth[:-1]]) * FEE * turnover).sum())
    peak = np.maximum.accumulate(np.concatenate([[START], wealth]))
    dd = 1 - np.concatenate([[START], wealth]) / peak
    rets = np.log(step)
    years = len(idx) / per_year
    final = float(wealth[-1])
    return Result(
        money=round(final, 2), ret_pct=round((final / START - 1) * 100, 1),
        cagr_pct=round(((final / START) ** (1 / years) - 1) * 100, 1) if years > 0 and final > 0 else -100.0,
        max_dd_pct=round(float(dd.max()) * 100, 1), trades=int((turnover > 1e-9).sum()), fees=round(fees, 2),
        exposure_pct=round(float(np.abs(p).mean()) * 100, 1),
        sharpe=round(float(rets.mean() / rets.std() * math.sqrt(per_year)), 2) if rets.std() > 0 else 0.0,
    )


def context(df: pd.DataFrame, interval: str) -> dict:
    sigma = forecast_volatility(df, interval)
    lc = np.log(df["close"])
    band = sigma * math.sqrt(H)
    train = df.index < TRAIN_END
    # Where today's forecast sits among TRAIN's forecasts (no look at TEST).
    ref = np.sort(sigma[train].dropna().values)
    vol_rank = np.searchsorted(ref, sigma.fillna(np.inf).values) / max(len(ref), 1)
    return {
        "box": box_positions(df, interval),
        "n": len(df), "per_year": PER_YEAR[interval], "sigma": sigma.values,
        "z": {c: ((lc - lc.ewm(span=c, adjust=False).mean()) / band).values for c in (24, 72, 168)},
        "trend": ((lc - lc.shift(168)) / (sigma * math.sqrt(168))).values,
        "vol_rank": vol_rank,
        "r_next": lc.diff().shift(-1).fillna(0.0).values,
        "train": train & sigma.notna().values, "test": (df.index >= TRAIN_END) & sigma.notna().values,
    }


def box_positions(df: pd.DataFrame, interval: str) -> np.ndarray:
    """The app's walk-forward Box Breakout (ml/box_breakout.run_adaptive:
    settings re-chosen every 100 bars from the 500 before) as a position per
    candle: held from the entry candle's close to the exit candle's. Scored
    close to close here, like every other strategy, not at its exact stop and
    target fills."""
    from ml import box_breakout

    rows = [{"timestamp": t.to_pydatetime(), **r} for t, r in df[["open", "high", "low", "close", "volume"]].iterrows()]
    res = box_breakout.run_adaptive(rows, interval)
    where = {t: i for i, t in enumerate(df.index)}
    pos = np.zeros(len(df))
    for tr in res["trades"]:
        a = where[pd.Timestamp(tr["entry_time"])]
        b = where[pd.Timestamp(tr["exit_time"])] if tr["exit_time"] else len(df)
        pos[a:b] = 1.0 if tr["side"] == "long" else -1.0
    return pos


def best_settings(name: str, ctx: dict) -> tuple[dict, Result]:
    grid = GRIDS[name]
    keys = list(grid)
    best = None
    for values in itertools.product(*grid.values()) if keys else [()]:
        p = dict(zip(keys, values))
        res = simulate(POSITIONS[name](ctx, p), ctx["r_next"], ctx["per_year"], ctx["train"])
        if res.max_dd_pct > MAX_TRAIN_DRAWDOWN * 100 and name != "buy_hold":
            continue
        if best is None or res.money > best[1].money:
            best = (p, res)
    return best if best else ({}, None)


def run(interval: str) -> list[dict]:
    rows = fetch_binance("ETHUSDT", interval, CANDLES[interval])
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    ctx = context(df, interval)
    test_from = df.index[ctx["test"]][0]
    print(f"\n== ETH {interval}: {len(df)} candles {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d} | "
          f"TRAIN {df.index[ctx['train']][0]:%Y-%m} .. 2022-12, TEST {test_from:%Y-%m} .. now", flush=True)
    sig = pd.Series(ctx["sigma"], index=df.index)
    print(f"  option 1, range forecast (80% band, TEST only): "
          f"{range_coverage(df[df.index >= TRAIN_END], sig[sig.index >= TRAIN_END])}", flush=True)
    out = []
    for name in POSITIONS:
        p, train_res = best_settings(name, ctx)
        if train_res is None:
            print(f"  {name}: no setting kept its drawdown under {MAX_TRAIN_DRAWDOWN:.0%} on TRAIN")
            continue
        test_res = simulate(POSITIONS[name](ctx, p), ctx["r_next"], ctx["per_year"], ctx["test"])
        out.append({"interval": interval, "strategy": name, "settings": p, "train": train_res, "test": test_res})
        print(f"  {name:<11} TRAIN {train_res.money:>9.2f} EUR (dd {train_res.max_dd_pct:>5.1f}%) | "
              f"TEST {test_res.money:>8.2f} EUR {test_res.ret_pct:>+7.1f}% dd {test_res.max_dd_pct:>5.1f}% "
              f"sharpe {test_res.sharpe:>5.2f} trades {test_res.trades:>5} fees {test_res.fees:>6.2f} "
              f"in market {test_res.exposure_pct:>5.1f}% | {p}", flush=True)
    return out


def robustness(interval: str, name: str = "core_box") -> None:
    """Every setting of `name` scored on TEST (not just the one TRAIN picked)
    and the picked one year by year, next to buy & hold: a result that only
    one lucky setting reaches is not a result."""
    rows = fetch_binance("ETHUSDT", interval, CANDLES[interval])
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    ctx = context(df, interval)
    grid = GRIDS[name]
    bh = simulate(np.ones(ctx["n"]), ctx["r_next"], ctx["per_year"], ctx["test"])
    res = [simulate(POSITIONS[name](ctx, dict(zip(grid, v))), ctx["r_next"], ctx["per_year"], ctx["test"])
           for v in itertools.product(*grid.values())]
    m, dd = np.array([r.money for r in res]), np.array([r.max_dd_pct for r in res])
    print(f"\n== {name} {interval}, TEST from {df.index[ctx['test']][0]:%Y-%m}: all {len(res)} settings -> "
          f"money median {np.median(m):.0f} (min {m.min():.0f}, max {m.max():.0f}) EUR, beat buy & hold "
          f"({bh.money:.0f} EUR) in {np.mean(m > bh.money):.0%}; drawdown median {np.median(dd):.0f}% "
          f"(buy & hold {bh.max_dd_pct:.0f}%)")
    p, _ = best_settings(name, ctx)
    print(f"  picked on TRAIN: {p}")
    for y in sorted(set(df.index[ctx["test"]].year)):
        mk = ctx["test"] & (df.index.year == y)
        a = simulate(POSITIONS[name](ctx, p), ctx["r_next"], ctx["per_year"], mk)
        b = simulate(np.ones(ctx["n"]), ctx["r_next"], ctx["per_year"], mk)
        print(f"  {y}: {a.ret_pct:+7.1f}% (dd {a.max_dd_pct:4.0f}%, {a.trades} trades) vs buy & hold "
              f"{b.ret_pct:+7.1f}% (dd {b.max_dd_pct:4.0f}%)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", action="append", help="repeatable; default 1h and 4h")
    parser.add_argument("--robust", metavar="STRATEGY", help="score every setting of STRATEGY on TEST, and year by year")
    args = parser.parse_args()
    for interval in args.interval or ["1h", "4h"]:
        if args.robust:
            robustness(interval, args.robust)
        else:
            run(interval)


if __name__ == "__main__":
    main()
