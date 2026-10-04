"""The live strategy fund: ml/strategy_evolution.py on the full ETH history,
for the intervals where it beat buy & hold out of sample (4h and 1d; on 1h
fees ate it, 15m is still to be tested - see README, "Strategy fund").

compute() replays everything since Binance listed ETHUSDT (August 2017),
so the same candles always give the same fund, and saves one JSON report
per interval to data/strategy_fund/. It takes ~1.5 min on 4h, ~15s on 1d,
so the app runs it in its own process when a new candle has closed
(refresh_all, see run_api.py) and the API only reads the saved report.

    .venv/bin/python -m ml.strategy_fund                 # 4h and 1d
    .venv/bin/python -m ml.strategy_fund --interval 1d
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml import strategy_evolution as se
from ml import strategy_lab as lab
from ml.walk_forward import fetch_binance

logger = logging.getLogger(__name__)

FUND_INTERVALS = ("4h", "1d")
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "strategy_fund"
CURVE_POINTS = 1500
INTERVAL_SECONDS = {"4h": 4 * 3600, "1d": 86400}


def report_path(interval: str) -> Path:
    return STATE_DIR / f"ethereum_{interval}.json"


def load(interval: str) -> dict | None:
    path = report_path(interval)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        logger.exception("Strategy fund: could not read %s", path)
        return None


def _result(r: lab.Result) -> dict:
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in r.__dict__.items()}


def _curve(df: pd.DataFrame, fund: np.ndarray, r_next: np.ndarray, live: np.ndarray) -> list[dict]:
    idx = np.flatnonzero(live)
    f, b = lab.wealth(fund, r_next, live), lab.wealth(np.ones(len(df)), r_next, live)
    # wealth[k] is the money once candle idx[k]+1 has closed; start at 100 the candle before.
    times = [df.index[idx[0]]] + [df.index[min(i + 1, len(df) - 1)] for i in idx]
    f, b, pos = np.concatenate([[lab.START], f]), np.concatenate([[lab.START], b]), np.concatenate([[0.0], fund[idx]])
    keep = np.unique(np.linspace(0, len(times) - 1, min(CURVE_POINTS, len(times))).astype(int))
    return [{"time": times[i].isoformat(), "fund": round(float(f[i]), 2), "buy_hold": round(float(b[i]), 2),
             "position": round(float(pos[i]), 3)} for i in keep]


def compute(interval: str) -> dict:
    """Replay the whole history and build the report (see module docstring)."""
    if interval not in FUND_INTERVALS:
        raise ValueError(f"The strategy fund runs on {FUND_INTERVALS} only")
    t0 = time.time()
    rows = fetch_binance("ETHUSDT", interval, lab.CANDLES[interval])
    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    ctx = lab.context(df, interval)
    runs = [se.evolve(ctx, s) for s in se.SEEDS]
    fund = np.mean([f for f, _, _ in runs], axis=0)
    start = runs[0][1][0]["t"]
    live = np.zeros(ctx["n"], dtype=bool)
    live[start:] = True
    ones = np.ones(ctx["n"])
    r_next = ctx["r_next"]

    def pair(mask):
        return {"fund": _result(lab.simulate(fund, r_next, ctx["per_year"], mask)),
                "buy_hold": _result(lab.simulate(ones, r_next, ctx["per_year"], mask))}

    years = []
    for y in sorted(set(df.index[live].year)):
        p = pair(live & (df.index.year == y))
        years.append({"year": int(y), "fund_pct": p["fund"]["ret_pct"], "fund_dd": p["fund"]["max_dd_pct"],
                      "buy_hold_pct": p["buy_hold"]["ret_pct"], "buy_hold_dd": p["buy_hold"]["max_dd_pct"],
                      "trades": p["fund"]["trades"]})

    # Why it holds what it holds: every top strategy's three pieces at the
    # last closed candle, averaged like the fund averages them.
    last = ctx["n"] - 1
    top = [s for _, _, final in runs for s in final]
    pieces = {k: 0.0 for k in ("core", "box", "range")}
    strategies = []
    for s in top:
        pr = se.parts(ctx, s["genes"])
        now = {k: float(v[last]) for k, v in pr.items()}
        for k in pieces:
            pieces[k] += now[k] / len(top)
        strategies.append({"genes": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["genes"].items()},
                           "score": round(s["score"], 4), "now": {k: round(v, 3) for k, v in now.items()},
                           "position": round(float(np.clip(sum(now.values()), -1, 1)), 3)})
    genes_over_time = []
    for i, entry in enumerate(runs[0][1]):
        row = {"time": df.index[entry["t"]].isoformat()}
        for k in ("core", "target", "box_w", "range_w", "long_short"):
            row[k] = round(float(np.mean([r[1][i][k] for r in runs])), 3)
        genes_over_time.append(row)

    sigma = ctx["sigma"][last]
    report = {
        "interval": interval,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "compute_seconds": round(time.time() - t0, 1),
        "candles": len(df),
        "history_from": df.index[0].isoformat(),
        "last_candle": df.index[last].isoformat(),
        "trading_from": df.index[start].isoformat(),
        "test_from": lab.TRAIN_END.isoformat(),
        "price": float(df["close"].iloc[-1]),
        "now": {
            "position": round(float(fund[last]), 3),
            "pieces": {k: round(v, 3) for k, v in pieces.items()},
            "box": int(ctx["box"][last]),
            "vol_forecast_annual_pct": round(float(sigma * math.sqrt(ctx["per_year"]) * 100), 1) if np.isfinite(sigma) else None,
            "next_reselection": (df.index[start] + pd.Timedelta(seconds=len(runs[0][1]) * int(
                se.RESELECT_DAYS * ctx["per_year"] / 365) * INTERVAL_SECONDS[interval])).isoformat(),
        },
        "whole_life": pair(live),
        "test": pair(live & ctx["test"]),
        "years": years,
        "curve": _curve(df, fund, r_next, live),
        "strategies": strategies,
        "genes_over_time": genes_over_time,
        "settings": {"population": se.POPULATION, "top_k": se.TOP_K, "replace": se.REPLACE,
                     "reselect_days": se.RESELECT_DAYS, "lookback_days": se.LOOKBACK_DAYS,
                     "dd_penalty": se.DD_PENALTY, "seeds": list(se.SEEDS), "fee": lab.FEE},
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = report_path(interval).with_suffix(".tmp")
    tmp.write_text(json.dumps(report))
    tmp.replace(report_path(interval))
    logger.info("Strategy fund %s computed in %.0fs", interval, report["compute_seconds"])
    return report


def is_stale(interval: str, now: datetime | None = None) -> bool:
    """True when a candle has closed since the saved report's last one."""
    rep = load(interval)
    if rep is None:
        return True
    now = now or datetime.now(timezone.utc)
    last = datetime.fromisoformat(rep["last_candle"])
    return (now - last).total_seconds() >= 2 * INTERVAL_SECONDS[interval]


def refresh_all() -> None:
    """Recompute every interval whose report is behind (run in its own process)."""
    logging.basicConfig(level=logging.INFO)
    for interval in FUND_INTERVALS:
        if is_stale(interval):
            try:
                compute(interval)
            except Exception:
                logger.exception("Strategy fund %s: compute failed", interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", action="append", choices=FUND_INTERVALS, help="repeatable; default 4h and 1d")
    args = parser.parse_args()
    for interval in args.interval or FUND_INTERVALS:
        rep = compute(interval)
        t, w = rep["test"], rep["whole_life"]
        print(f"{interval}: position now {rep['now']['position']:+.2f} | TEST {t['fund']['money']:.2f} EUR "
              f"(buy & hold {t['buy_hold']['money']:.2f}) | since {rep['trading_from'][:10]} {w['fund']['money']:.2f} EUR "
              f"(buy & hold {w['buy_hold']['money']:.2f}) | {rep['compute_seconds']:.0f}s")


if __name__ == "__main__":
    main()
