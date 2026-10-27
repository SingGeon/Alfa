"""Background-computed Monte Carlo robustness report for the box breakout strategy (ml/box_breakout.py,
ml/monte_carlo.py - Ch. 10 of "Algorithmic Trading Hands-On Approach Using Python"): replays the
dashboard's *current* adaptively-chosen settings (ml.box_breakout.run_adaptive's "current" segment)
hundreds of times against small, realistic price noise.

Expensive (re-simulates the strategy MC_RUNS times over thousands of candles - tens of seconds to a
couple of minutes), so this follows ml/strategy_fund.py's exact save/load/staleness pattern: a cron
job (see run_api.py) recomputes it periodically in its own process, and the API route just reads the
saved JSON + PNG - fast and synchronous, safe to call from a Flask request.

    .venv/bin/python -m evaluation.box_breakout_robustness          # 1h and 4h
    .venv/bin/python -m evaluation.box_breakout_robustness --interval 1h
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from database import repository
from evaluation import charts as eval_charts
from ml import box_breakout as bb
from ml import monte_carlo as mc
from ml.features import candles_to_dataframe

logger = logging.getLogger(__name__)

MC_INTERVALS = ("1h", "4h")   # the rest would cost too much CPU for the benefit - same reasoning
                              # as ml/strategy_fund.py's FUND_INTERVALS restriction
MC_RUNS = 300
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "box_breakout_monte_carlo"
REFRESH_STALE_AFTER = timedelta(hours=24)   # a health-check, not a live number: no need to
                                             # recompute on every candle close


def report_path(interval: str) -> Path:
    return STATE_DIR / f"ethereum_{interval}.json"


def report_png_name(interval: str) -> str:
    return f"box_breakout_{interval}"


def report_png_path(interval: str) -> Path:
    return eval_charts.monte_carlo_report_path(report_png_name(interval))


def load(interval: str) -> dict | None:
    path = report_path(interval)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        logger.exception("Box breakout Monte Carlo: could not read %s", path)
        return None


def compute(interval: str) -> dict:
    """Recompute the report for `interval`: the box breakout strategy's currently-active,
    adaptively-chosen settings, replayed MC_RUNS times on blurred variants of the real candle
    history. Saves both the JSON summary and the PNG report."""
    if interval not in MC_INTERVALS:
        raise ValueError(f"Box breakout Monte Carlo only runs on {MC_INTERVALS}")
    t0 = time.time()
    candles = repository.get_candles("ethereum", interval, limit=3000)
    adaptive = bb.run_adaptive(candles, interval)
    settings = adaptive.get("current", {}).get("settings")
    if not settings:
        raise ValueError(f"{interval}: no active settings right now (the strategy is standing aside)")
    s = bb.BoxSettings(**settings)
    df = candles_to_dataframe(bb.closed_candles(candles, interval))

    def strategy_fn(variant):
        boxes = bb.detect_boxes(variant, s)
        trades = bb.simulate(variant, boxes, s)
        out = bb.summarize(variant, trades, s)
        out["equity_curve"] = bb.equity_curve(trades, s.fee_pct).tolist()
        return out

    base = strategy_fn(df)
    sigma_pct = mc.realistic_sigma_pct(df)
    results = mc.run_monte_carlo(strategy_fn, df, n_runs=MC_RUNS, sigma_pct=sigma_pct)
    dist = mc.summarize_distribution(results)
    eval_charts.render_monte_carlo_report(
        results, f"Box Breakout robustness - ETHUSDT {interval}", report_png_name(interval),
    )

    report = {
        "interval": interval,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "compute_seconds": round(time.time() - t0, 1),
        "settings": settings,
        "sigma_pct": sigma_pct,
        "base": {
            "total_return_pct": base["total_return_pct"],
            "max_drawdown_pct": base["max_drawdown_pct"],
            "trades": base["trades"],
        },
        **dist,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = report_path(interval).with_suffix(".tmp")
    tmp.write_text(json.dumps(report))
    tmp.replace(report_path(interval))
    logger.info("Box breakout Monte Carlo %s computed in %.0fs", interval, report["compute_seconds"])
    return report


def is_stale(interval: str, now: datetime | None = None) -> bool:
    rep = load(interval)
    if rep is None:
        return True
    now = now or datetime.now(timezone.utc)
    computed_at = datetime.fromisoformat(rep["computed_at"])
    return now - computed_at >= REFRESH_STALE_AFTER


def refresh_all() -> None:
    """Recompute every interval whose report is stale (run in its own process)."""
    logging.basicConfig(level=logging.INFO)
    for interval in MC_INTERVALS:
        if is_stale(interval):
            try:
                compute(interval)
            except Exception:
                logger.exception("Box breakout Monte Carlo %s: compute failed", interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", action="append", choices=MC_INTERVALS, help="repeatable; default 1h and 4h")
    args = parser.parse_args()
    for interval in args.interval or MC_INTERVALS:
        rep = compute(interval)
        print(f"{interval}: base return {rep['base']['total_return_pct']:+.1f}% -> MC median "
              f"{rep['total_return_pct']['median']:+.1f}% (drawdown median {rep['max_drawdown_pct']['median']:.1f}%, "
              f"bimodal={rep['bimodal_drawdown']}) in {rep['compute_seconds']:.0f}s")


if __name__ == "__main__":
    main()
