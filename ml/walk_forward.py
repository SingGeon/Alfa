"""Walk-forward backtest of PricePredictor on real candles.

Expanding window: for each forecast origin i, a fresh model is trained on
every candle up to and including i (nothing after it), then asked for an
H-step recursive forecast, which is scored against the real closes
i+1..i+H. Features only ever use data up to the row they describe (rolling
windows look backwards; the target is dropped for the last training row),
so no future information reaches training or prediction.

Metrics (see summarize()):
- coverage of [lower, upper]: at step 1, over the whole trajectory, at the last step
- last-step error vs the "no change" baseline: mean |log(pred_H / actual_H)|
  divided by mean |log(close_i / actual_H)| (< 1 = beats the baseline)
- last-step direction accuracy, with its standard error
- spread of predicted H-step returns vs real ones (a ratio near 0 means the
  forecast has collapsed into a flat "straight line")

Usage (fetches closed Binance candles, needs network):
    python -m ml.walk_forward --interval 1h --candles 4000 --windows 40 --horizon 24 --variant tuned --features relative
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, RELATIVE_FEATURE_COLUMNS, build_feature_frame
from ml.price_predictor import PricePredictor


@dataclass
class WindowResult:
    origin_close: float
    predicted: list[float]
    lower: list[float]
    upper: list[float]
    actual: list[float]


@dataclass
class BacktestResult:
    windows: list[WindowResult] = field(default_factory=list)

    def summarize(self) -> dict:
        w = [r for r in self.windows if len(r.predicted) == len(r.actual) and r.predicted]
        n = len(w)
        if not n:
            return {"windows": 0}
        horizon = len(w[0].predicted)
        inside = np.array([[lo <= a <= hi for lo, hi, a in zip(r.lower, r.upper, r.actual)] for r in w], dtype=float)
        pred_h = np.array([math.log(r.predicted[-1] / r.origin_close) for r in w])
        act_h = np.array([math.log(r.actual[-1] / r.origin_close) for r in w])
        model_err = np.abs(pred_h - act_h).mean()
        baseline_err = np.abs(act_h).mean()
        hits = (np.sign(pred_h) == np.sign(act_h)).astype(float)
        p = hits.mean()
        width_h = np.array([math.log(r.upper[-1] / r.lower[-1]) for r in w])
        width_1 = np.array([math.log(r.upper[0] / r.lower[0]) for r in w])
        return {
            "windows": n,
            "horizon": horizon,
            "coverage_step1": round(inside[:, 0].mean(), 3),
            "coverage_trajectory": round(inside.mean(), 3),
            "coverage_last": round(inside[:, -1].mean(), 3),
            "error_vs_no_change": round(model_err / baseline_err, 3) if baseline_err else None,
            "direction_accuracy": round(p, 3),
            "direction_se": round(math.sqrt(p * (1 - p) / n), 3),
            "pred_spread_ratio": round(pred_h.std() / act_h.std(), 3) if act_h.std() else None,
            "mean_width_step1_pct": round(width_1.mean() * 100, 3),
            "mean_width_last_pct": round(width_h.mean() * 100, 3),
        }


def origins_for(n_candles: int, windows: int, horizon: int, min_train: int) -> list[int]:
    """`windows` evenly spaced forecast origins, each leaving `horizon` real
    candles after it and at least `min_train` before it."""
    last = n_candles - horizon - 1
    if last < min_train:
        return []
    step = max(1, (last - min_train) // max(windows - 1, 1))
    return list(range(last, min_train - 1, -step))[:windows][::-1]


def run(
    candles: list[dict],
    btc_candles: list[dict] | None,
    interval: str,
    horizon: int,
    windows: int,
    min_train: int = 500,
    predictor_factory=lambda: PricePredictor(model_variant="tuned"),
) -> BacktestResult:
    btc_close = None
    if btc_candles:
        btc = pd.DataFrame(btc_candles)
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
        btc_close = btc.set_index("timestamp")["close"]

    result = BacktestResult()
    for i in origins_for(len(candles), windows, horizon, min_train):
        history = candles[: i + 1]
        cutoff = pd.Timestamp(history[-1]["timestamp"])
        btc_hist = btc_close[btc_close.index <= cutoff] if btc_close is not None else None
        # Sentiment/TVL are left out: their history is date-bucketed and not
        # available point-in-time here, and including them would risk leaking
        # later-in-the-day information into earlier candles.
        df = build_feature_frame(history, {}, btc_close=btc_hist, tvl_by_date=None)
        predictor = predictor_factory()
        predictor.fit(df.dropna())
        preds = predictor.predict(df, steps=horizon, interval=interval)
        result.windows.append(WindowResult(
            origin_close=float(history[-1]["close"]),
            predicted=[p["predicted_price"] for p in preds],
            lower=[p["lower"] for p in preds],
            upper=[p["upper"] for p in preds],
            actual=[float(c["close"]) for c in candles[i + 1: i + 1 + horizon]],
        ))
    return result


def fetch_binance(symbol: str, interval: str, total: int) -> list[dict]:
    """Closed Binance klines, oldest first, paging back from now."""
    import time
    from datetime import datetime, timezone

    import requests

    out: list = []
    end = None
    while len(out) < total:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            params["endTime"] = end
        page = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15).json()
        if not page:
            break
        out = page + out
        end = page[0][0] - 1
    now_ms = time.time() * 1000
    return [
        {
            "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]), "taker_buy_volume": float(k[9]),
        }
        for k in out if k[6] < now_ms
    ][-total:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--candles", type=int, default=4000)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--variant", default="tuned", choices=("tuned", "legacy"))
    parser.add_argument("--features", default="price", choices=("price", "relative"))
    args = parser.parse_args()

    eth = fetch_binance("ETHUSDT", args.interval, args.candles)
    btc = fetch_binance("BTCUSDT", args.interval, args.candles)
    feature_cols = RELATIVE_FEATURE_COLUMNS if args.features == "relative" else FEATURE_COLUMNS
    result = run(
        eth, btc, args.interval, args.horizon, args.windows, args.min_train,
        predictor_factory=lambda: PricePredictor(model_variant=args.variant, feature_cols=feature_cols),
    )
    summary = result.summarize()
    for key, value in summary.items():
        print(f"{key:>24}: {value}")
    if summary.get("windows", 0) < 30:
        print("WARNING: fewer than 30 windows - small differences are noise.")


if __name__ == "__main__":
    main()
