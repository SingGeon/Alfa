"""An evolving population of trading strategies, one per interval.

ml/strategy_lab.py found the pieces that work and the ones that don't, but
picked each strategy's settings by hand on one TRAIN period. Here the
pieces become genes and selection picks them, separately on every
interval, only ever from the past:

- core, target, band   how much ETH it holds, sized by the volatility
                       forecast (strategy_lab.pos_vol_target)
- box_w                how much it follows the app's walk-forward Box
                       Breakout (long and short breakouts)
- range_w, center, k,  how much it trades the swings inside the forecast
  exit, stop,          range, around which moving middle, how far out it
  trend_max,           buys, where it takes profit and gives up, when the
  long_short           market trends too hard for it, and whether it shorts

A strategy's position is core*vol_target + box_w*box + range_w*range,
capped at all of its money (no leverage).

Every month (RESELECT_DAYS) each strategy is scored on the ~4 years before
(LOOKBACK_DAYS; less while there isn't that much history yet): its log growth after fees, minus DD_PENALTY x its worst
drop. The weakest REPLACE are replaced by mutated children of two strong
ones (tournament), scored on the same past year. The fund then holds, for
the coming month, the average position of the TOP_K best, averaged again
over independent populations (SEEDS). Nothing it
does at a candle depends on a later candle, so its whole history is out of
sample; TEST (2023 on) is reported next to strategy_lab's for comparison.

    .venv/bin/python -m ml.strategy_evolution                    # 15m 1h 4h 1d
    .venv/bin/python -m ml.strategy_evolution --interval 4h
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from ml import strategy_lab as lab
from ml.walk_forward import fetch_binance

POPULATION = 24
TOP_K = 4
REPLACE = 6
RESELECT_DAYS = 30
LOOKBACK_DAYS = 1500       # ~4 years: last year's winner is too often next year's loser
DD_PENALTY = 1.0
SEEDS = (1, 2, 3)          # the fund averages independent populations: one alone is partly luck

BOUNDS = {"core": (0.0, 1.0), "target": (0.1, 1.2), "band": (0.05, 0.3), "box_w": (0.0, 1.0),
          "range_w": (0.0, 1.0), "k": (1.0, 2.5), "exit": (0.0, 0.8), "stop": (0.5, 3.0), "trend_max": (1.0, 6.0)}
CHOICES = {"center": (24, 72, 168), "long_short": (False, True)}
TREND_OFF = 5.5            # trend_max above this: never stands aside for a trend


def random_genes(rng: np.random.Generator) -> dict:
    g = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in BOUNDS.items()}
    g.update({k: v[rng.integers(len(v))] for k, v in CHOICES.items()})
    return g


def child(a: dict, b: dict, rng: np.random.Generator) -> dict:
    g = {}
    for k, (lo, hi) in BOUNDS.items():
        g[k] = float(np.clip((a[k] + b[k]) / 2 + rng.normal(0, 0.1 * (hi - lo)), lo, hi))
    for k, v in CHOICES.items():
        g[k] = (a if rng.random() < 0.5 else b)[k]
        if rng.random() < 0.1:
            g[k] = v[rng.integers(len(v))]
    return g


def parts(ctx: dict, g: dict) -> dict[str, np.ndarray]:
    """A strategy's three pieces, each already weighted by its gene."""
    out = {"core": g["core"] * lab.pos_vol_target(ctx, g), "box": g["box_w"] * ctx["box"],
           "range": np.zeros(ctx["n"])}
    if g["range_w"] > 0.02:
        rp = {**g, "vol_cap": 1.01, "trend_max": 99.0 if g["trend_max"] > TREND_OFF else g["trend_max"]}
        out["range"] = g["range_w"] * lab.pos_range(ctx, rp)
    return out


def positions(ctx: dict, g: dict) -> np.ndarray:
    return np.clip(sum(parts(ctx, g).values()), -1.0, 1.0)


def score(pos: np.ndarray, r_next: np.ndarray) -> float:
    """Log growth after fees over this stretch, minus DD_PENALTY x its worst
    drop (as a fraction): the most money, but not by risking all of it."""
    prev = np.concatenate([[pos[0]], pos[:-1]])
    step = (1 + pos * np.expm1(r_next)) * (1 - lab.FEE * np.abs(pos - prev))
    logw = np.cumsum(np.log(np.maximum(step, 1e-12)))
    dd = 1 - np.exp(logw - np.maximum.accumulate(np.maximum(logw, 0.0)))
    return float(logw[-1] - DD_PENALTY * dd.max())


def evolve(ctx: dict, seed: int = SEEDS[0]) -> tuple[np.ndarray, list[dict], list[dict]]:
    """The fund's position per candle, the TOP_K's genes at every
    reselection, and the TOP_K at the last one (genes, past-years score)."""
    rng = np.random.default_rng(seed)
    per_day = ctx["per_year"] / 365
    every, lookback, start = int(RESELECT_DAYS * per_day), int(LOOKBACK_DAYS * per_day), int(365 * per_day)
    r_next = ctx["r_next"]
    first = int(np.flatnonzero(np.isfinite(ctx["sigma"]))[0])
    pop = [random_genes(rng) for _ in range(POPULATION)]
    pos = [positions(ctx, g) for g in pop]
    fund = np.zeros(ctx["n"])
    log, final = [], []
    for t in range(first + start, ctx["n"], every):
        win = slice(max(first, t - lookback), t)   # returns that had closed by t
        sc = np.array([score(p[win], r_next[win]) for p in pos])
        order = np.argsort(sc)[::-1]
        strong = order[:POPULATION // 2]
        for i in order[-REPLACE:]:
            a, b = (pop[max(rng.choice(strong, 2), key=lambda j: sc[j])] for _ in range(2))
            pop[i] = child(a, b, rng)
            pos[i] = positions(ctx, pop[i])
            sc[i] = score(pos[i][win], r_next[win])
        top = np.argsort(sc)[::-1][:TOP_K]
        fund[t:t + every] = np.mean([pos[i][t:t + every] for i in top], axis=0)
        log.append({"t": t, "score": float(sc[top[0]]),
                    **{k: float(np.mean([pop[i][k] for i in top])) for k in (*BOUNDS, "long_short")}})
        final = [{"genes": dict(pop[i]), "score": float(sc[i])} for i in top]
    return fund, log, final


def run(interval: str, candles: int | None = None) -> dict:
    t0 = time.time()
    rows = fetch_binance("ETHUSDT", interval, candles or lab.CANDLES[interval])
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    ctx = lab.context(df, interval)
    runs = [evolve(ctx, s) for s in SEEDS]
    fund, log = np.mean([f for f, _, _ in runs], axis=0), runs[0][1]
    live = np.zeros(ctx["n"], dtype=bool)
    live[log[0]["t"]:] = True
    ones = np.ones(ctx["n"])
    print(f"\n== ETH {interval}: {len(df)} candles, the fund trades from {df.index[log[0]['t']]:%Y-%m-%d} "
          f"({len(log)} monthly reselections, {time.time() - t0:.0f}s)")
    out = {"interval": interval}
    for name, mask in (("whole life", live), ("TEST 2023+", live & ctx["test"])):
        f = lab.simulate(fund, ctx["r_next"], ctx["per_year"], mask)
        b = lab.simulate(ones, ctx["r_next"], ctx["per_year"], mask)
        out[name] = (f, b)
        print(f"  {name:<11} fund {f.money:>8.2f} EUR {f.ret_pct:>+8.1f}% dd {f.max_dd_pct:>5.1f}% sharpe {f.sharpe:>5.2f} "
              f"trades {f.trades:>5} fees {f.fees:>6.2f} | buy & hold {b.money:>8.2f} EUR {b.ret_pct:>+8.1f}% dd {b.max_dd_pct:>5.1f}%")
    for y in sorted(set(df.index[live].year)):
        mk = live & (df.index.year == y)
        f, b = lab.simulate(fund, ctx["r_next"], ctx["per_year"], mk), lab.simulate(ones, ctx["r_next"], ctx["per_year"], mk)
        print(f"    {y}: fund {f.ret_pct:>+7.1f}% (dd {f.max_dd_pct:>3.0f}%) vs buy & hold {b.ret_pct:>+7.1f}% (dd {b.max_dd_pct:>3.0f}%)")
    g = pd.DataFrame(log)
    g["year"] = df.index[g["t"]].year
    print("  what the best strategies carried (yearly average of the top 4's genes):")
    print(g.groupby("year")[["core", "target", "box_w", "range_w", "long_short"]].mean().round(2).to_string())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", action="append", help="repeatable; default 15m 1h 4h 1d")
    parser.add_argument("--candles", type=int, help="history length (default: everything since 2017; on 15m the "
                        "walk-forward Box Breakout costs ~2h on all of it, 150000 is ~35 min)")
    args = parser.parse_args()
    for interval in args.interval or ["15m", "1h", "4h", "1d"]:
        run(interval, args.candles)


if __name__ == "__main__":
    main()
