"""Dynamic Box Breakout Strategy - Darvas-style consolidation boxes, volume-
confirmed breakouts, and a simulated (backtested) trade log.

Rule-based, like ml/pattern_recognition.py: nothing here is learned, every
rule is plain arithmetic over closed candles, so the result is fully
reproducible and can be checked bar by bar on the chart.

BOX DETECTION
    At bar i, take the `lookback` (X) bars that end `padding` (Y) bars ago:
    ceiling = their highest high, floor = their lowest low. If the Y bars
    since then all stayed inside [floor, ceiling] (no high above the
    ceiling, no low below the floor), the range has held: a box is
    established at bar i. It is then extended forward, bar by bar, until a
    candle *closes* outside it. While a box is active no new one is searched
    for; the search restarts on the bar after the breakout.

ENTRY (on the close of the breakout candle - no look-ahead)
    long:  close > ceiling * (1 + breakout_buffer)
    short: close < floor   * (1 - breakout_buffer)
    Both need the volume filter: that candle's volume > its `vol_ma` (Z)
    bar simple moving average. A breakout without volume still ends the box
    (the range is broken) but opens no trade - that's the fakeout filter.
    One position at a time. While in a trade, a confirmed breakout the
    other way closes it at that close and opens the opposite one
    (reverse_on_opposite, like a Pine strategy.entry in the other
    direction); one the same way is ignored.
    A box taller than max_box_pct (0 = no limit) isn't a consolidation:
    its breakout opens no trade. Neither does one whose target would be at
    or below 0 (a short whose risk is larger than the price itself).

RISK MANAGEMENT
    stop loss: long at floor * (1 - stop_buffer), short at
        ceiling * (1 + stop_buffer) - the far side of the box it broke out of.
    exit_mode "rr": take profit at entry +/- rr * risk (default 1:2).
    exit_mode "trail": no fixed target; each new box established while in
        the trade moves the stop to its floor (long) / ceiling (short), only
        ever in the trade's favor.
    Exits are checked from the bar after entry, on each bar's high/low. When
    one bar touches both the stop and the target, the stop is assumed first
    (a candle doesn't say which came first; assuming the worst keeps the
    backtest honest). A stop that the bar opens past fills at the open.
    Every exit costs `fee_pct` per side.

Only closed candles are used: a still-open last candle (its period hasn't
ended yet) would make the latest box/signal flicker until it closes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ml.features import candles_to_dataframe

INTERVAL_DELTA = {
    "15m": timedelta(minutes=15), "1h": timedelta(hours=1), "4h": timedelta(hours=4),
    "1d": timedelta(days=1), "1w": timedelta(weeks=1),
}
EXIT_MODES = ("rr", "trail")


@dataclass
class BoxSettings:
    lookback: int = 20            # X: bars whose high/low define the box
    padding: int = 5              # Y: bars the range must hold before the box counts
    vol_ma: int = 14              # Z: volume moving average period for the filter
    rr: float = 2.0               # take profit at rr x risk (exit_mode "rr")
    exit_mode: str = "rr"         # "rr" or "trail"
    breakout_buffer_pct: float = 0.0  # close must clear the boundary by this % ("cleanly")
    stop_buffer_pct: float = 0.1      # stop this % beyond the opposite boundary
    fee_pct: float = 0.1              # per side (entry and exit)
    max_box_pct: float = 0.0          # skip boxes taller than this % of the floor (0 = no limit)
    reverse_on_opposite: bool = True  # opposite breakout closes and reverses the trade

    def validate(self) -> None:
        if not 2 <= self.lookback <= 500:
            raise ValueError("lookback must be between 2 and 500")
        if not 1 <= self.padding <= 200:
            raise ValueError("padding must be between 1 and 200")
        if not 2 <= self.vol_ma <= 500:
            raise ValueError("vol_ma must be between 2 and 500")
        if not 0.1 <= self.rr <= 20:
            raise ValueError("rr must be between 0.1 and 20")
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode must be one of {EXIT_MODES}")
        for name in ("breakout_buffer_pct", "stop_buffer_pct", "fee_pct"):
            if not 0 <= getattr(self, name) <= 10:
                raise ValueError(f"{name} must be between 0 and 10")
        if not 0 <= self.max_box_pct <= 1000:
            raise ValueError("max_box_pct must be between 0 and 1000")


@dataclass
class Box:
    start: int                 # first bar of the lookback window
    established: int           # bar at which the range had held for `padding` bars
    end: int                   # last bar inside the box (the bar before the breakout, or the last bar)
    ceiling: float
    floor: float
    breakout: str | None = None        # "up" / "down" / None (still active)
    breakout_bar: int | None = None
    volume_confirmed: bool | None = None


@dataclass
class Trade:
    side: str                  # "long" / "short"
    entry_bar: int
    entry_price: float
    stop: float
    initial_stop: float
    target: float | None
    exit_bar: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None     # "stop" / "target" / "trail_stop" / "reverse" / None (open)
    stop_moves: list[dict] = field(default_factory=list)

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.initial_stop)


def closed_candles(candles: list[dict], interval: str, now: datetime | None = None) -> list[dict]:
    """Drop a trailing candle whose period hasn't ended yet."""
    if not candles or interval not in INTERVAL_DELTA:
        return candles
    now = now or datetime.now(timezone.utc)
    last = pd.Timestamp(candles[-1]["timestamp"])
    last = last.tz_localize("UTC") if last.tzinfo is None else last
    return candles[:-1] if last + INTERVAL_DELTA[interval] > now else candles


def detect_boxes(df: pd.DataFrame, s: BoxSettings) -> list[Box]:
    high, low, close = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    n = len(df)
    up_mult, down_mult = 1 + s.breakout_buffer_pct / 100, 1 - s.breakout_buffer_pct / 100
    boxes: list[Box] = []
    active: Box | None = None
    i = s.lookback + s.padding - 1
    while i < n:
        if active is not None:
            if close[i] > active.ceiling * up_mult or close[i] < active.floor * down_mult:
                active.breakout = "up" if close[i] > active.ceiling * up_mult else "down"
                active.breakout_bar = i
                active.end = i - 1
                active = None
                # Neither the old box's bars nor the breakout candle itself
                # (its range spans both levels) are reused: the next box's
                # lookback window starts on the bar after the breakout.
                i += s.lookback + s.padding
                continue
            active.end = i
            i += 1
            continue
        window_end = i - s.padding                      # last bar of the lookback window
        window_start = window_end - s.lookback + 1
        ceiling = high[window_start:window_end + 1].max()
        floor = low[window_start:window_end + 1].min()
        held = high[window_end + 1:i + 1].max() <= ceiling and low[window_end + 1:i + 1].min() >= floor
        if held and ceiling > floor:
            active = Box(start=window_start, established=i, end=i, ceiling=float(ceiling), floor=float(floor))
            boxes.append(active)
        i += 1
    return boxes


def simulate(df: pd.DataFrame, boxes: list[Box], s: BoxSettings) -> list[Trade]:
    open_, high, low = df["open"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float)
    close, volume = df["close"].to_numpy(float), df["volume"].to_numpy(float)
    vol_ma = pd.Series(volume).rolling(s.vol_ma).mean().to_numpy()
    for b in boxes:
        if b.breakout_bar is not None:
            v = vol_ma[b.breakout_bar]
            b.volume_confirmed = bool(not math.isnan(v) and volume[b.breakout_bar] > v)

    breakouts = {b.breakout_bar: b for b in boxes if b.breakout_bar is not None}
    established = {b.established: b for b in boxes}
    stop_mult = s.stop_buffer_pct / 100
    trades: list[Trade] = []
    pos: Trade | None = None

    for i in range(len(df)):
        if pos is not None and i > pos.entry_bar:
            long = pos.side == "long"
            stop_hit = low[i] <= pos.stop if long else high[i] >= pos.stop
            target_hit = pos.target is not None and (high[i] >= pos.target if long else low[i] <= pos.target)
            if stop_hit:
                gapped = open_[i] <= pos.stop if long else open_[i] >= pos.stop
                pos.exit_bar, pos.exit_price = i, float(open_[i] if gapped else pos.stop)
                pos.exit_reason = "trail_stop" if pos.stop_moves else "stop"
            elif target_hit:
                gapped = open_[i] >= pos.target if long else open_[i] <= pos.target
                pos.exit_bar, pos.exit_price, pos.exit_reason = i, float(open_[i] if gapped else pos.target), "target"
            if pos.exit_bar is not None:
                pos = None
            elif s.exit_mode == "trail" and i in established:
                # Trail to the newly formed box's far side, only in the trade's favor.
                b = established[i]
                new_stop = b.floor * (1 - stop_mult) if long else b.ceiling * (1 + stop_mult)
                if (long and new_stop > pos.stop) or (not long and new_stop < pos.stop):
                    pos.stop = float(new_stop)
                    pos.stop_moves.append({"bar": i, "stop": pos.stop})

        b = breakouts.get(i)
        if b is None or not b.volume_confirmed:
            continue
        if s.max_box_pct and (b.ceiling / b.floor - 1) * 100 > s.max_box_pct:
            continue
        side = "long" if b.breakout == "up" else "short"
        if pos is not None:
            if pos.side == side or not s.reverse_on_opposite:
                continue
            pos.exit_bar, pos.exit_price, pos.exit_reason = i, float(close[i]), "reverse"
            pos = None
        entry = float(close[i])
        if side == "long":
            stop = b.floor * (1 - stop_mult)
            target = entry + s.rr * (entry - stop) if s.exit_mode == "rr" else None
            valid = stop < entry
        else:
            stop = b.ceiling * (1 + stop_mult)
            target = entry - s.rr * (stop - entry) if s.exit_mode == "rr" else None
            valid = stop > entry and (target is None or target > 0)
        if valid:
            pos = Trade(side, i, entry, float(stop), float(stop), None if target is None else float(target))
            trades.append(pos)
    return trades


def trade_return(t: Trade, fee_pct: float, mark_price: float | None = None) -> float:
    """Net fractional return of one trade (full position), fees on both sides."""
    exit_price = t.exit_price if t.exit_price is not None else mark_price
    gross = exit_price / t.entry_price - 1 if t.side == "long" else 1 - exit_price / t.entry_price
    return gross - 2 * fee_pct / 100


def summarize(df: pd.DataFrame, trades: list[Trade], s: BoxSettings) -> dict:
    closed = [t for t in trades if t.exit_bar is not None]
    rets = np.array([trade_return(t, s.fee_pct) for t in closed])
    r_multiples = np.array([
        (t.exit_price - t.entry_price) / t.risk if t.side == "long" else (t.entry_price - t.exit_price) / t.risk
        for t in closed if t.risk > 0
    ])
    equity = np.cumprod(1 + rets) if len(rets) else np.array([1.0])
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))
    drawdown = 1 - np.concatenate([[1.0], equity]) / peak
    wins, losses = rets[rets > 0], rets[rets <= 0]
    close = df["close"].to_numpy(float)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in {
        "trades": len(closed),
        "open_trades": len(trades) - len(closed),
        "longs": sum(t.side == "long" for t in closed),
        "shorts": sum(t.side == "short" for t in closed),
        "win_rate_pct": round(len(wins) / len(rets) * 100, 1) if len(rets) else None,
        "total_return_pct": round((equity[-1] - 1) * 100, 2) if len(rets) else 0.0,
        "avg_trade_pct": round(rets.mean() * 100, 3) if len(rets) else None,
        "avg_r": round(r_multiples.mean(), 2) if len(r_multiples) else None,
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if len(losses) and losses.sum() < 0 else None,
        "max_drawdown_pct": round(drawdown.max() * 100, 2),
        "buy_hold_pct": round((close[-1] / close[0] - 1) * 100, 2) if len(close) else None,
        "bars": len(df),
        "from": df.index[0].isoformat() if len(df) else None,
        "to": df.index[-1].isoformat() if len(df) else None,
    }.items()}


def run(candles: list[dict], interval: str, settings: BoxSettings | None = None, now: datetime | None = None) -> dict:
    """Everything the dashboard's Box Breakout panel needs, in one call."""
    s = settings or BoxSettings()
    s.validate()
    df = candles_to_dataframe(closed_candles(candles, interval, now))
    if len(df) < s.lookback + s.padding + 1:
        raise ValueError(f"Need at least {s.lookback + s.padding + 1} closed candles, have {len(df)}")
    boxes = detect_boxes(df, s)
    trades = simulate(df, boxes, s)
    ts = [t.isoformat() for t in df.index]
    last_close = float(df["close"].iloc[-1])

    def box_json(b: Box) -> dict:
        d = asdict(b)
        d.update(start_time=ts[b.start], established_time=ts[b.established], end_time=ts[b.end],
                 breakout_time=ts[b.breakout_bar] if b.breakout_bar is not None else None)
        return d

    def trade_json(t: Trade) -> dict:
        d = asdict(t)
        d["entry_time"] = ts[t.entry_bar]
        d["exit_time"] = ts[t.exit_bar] if t.exit_bar is not None else None
        d["stop_moves"] = [{**m, "time": ts[m["bar"]]} for m in t.stop_moves]
        d["return_pct"] = round(trade_return(t, s.fee_pct, last_close) * 100, 3)
        return d

    active = boxes[-1] if boxes and boxes[-1].breakout is None else None
    open_trade = trades[-1] if trades and trades[-1].exit_bar is None else None
    return {
        "settings": asdict(s),
        "boxes": [box_json(b) for b in boxes],
        "trades": [trade_json(t) for t in trades],
        "stats": summarize(df, trades, s),
        "active_box": box_json(active) if active else None,
        "open_trade": trade_json(open_trade) if open_trade else None,
        "last_close": last_close,
        "last_time": ts[-1],
    }


# --- Adaptive mode: the algorithm picks its own settings ----------------------
#
# Nobody tunes X / Y / R:R by hand. Every REOPT_EVERY bars the algorithm
# re-runs the strategy with every combination in PARAM_GRID on the TRAIN_BARS
# bars *before* that point (never after), and keeps the one that would have
# made the most (sum of log returns, fees included, at least MIN_TRAIN_TRADES
# closed trades). It trades the next REOPT_EVERY bars with it. If no
# combination made money on that window, it stands aside for those bars:
# "when to change" (every REOPT_EVERY bars, or never trading at all when the
# market didn't reward breakouts) and "what to change" (box length, hold,
# exit) both come from the data, not from the user.
#
# All stats are measured only on bars the choice had not seen (walk-forward,
# out of sample) - the warm-up bars before the first choice are not traded.

TRAIN_BARS = 500
REOPT_EVERY = 100
MIN_TRAIN_TRADES = 3
PARAM_GRID = [
    dict(lookback=lb, padding=pd_, exit_mode=mode, rr=rr)
    for lb in (10, 20, 30, 50)
    for pd_ in (3, 5, 8)
    for mode, rrs in (("rr", (1.5, 2.0, 3.0)), ("trail", (2.0,)))
    for rr in rrs
]


def _score(df: pd.DataFrame, s: BoxSettings) -> tuple[float, int]:
    trades = [t for t in simulate(df, detect_boxes(df, s), s) if t.exit_bar is not None]
    return float(sum(math.log(max(1e-9, 1 + trade_return(t, s.fee_pct))) for t in trades)), len(trades)


def choose_settings(train: pd.DataFrame) -> tuple[BoxSettings | None, dict]:
    """Best PARAM_GRID entry on `train`, or None (stand aside) if none made money."""
    best, best_score, best_n = None, 0.0, 0
    for params in PARAM_GRID:
        s = BoxSettings(**params)
        if len(train) < s.lookback + s.padding + 2:
            continue
        score, n = _score(train, s)
        if n >= MIN_TRAIN_TRADES and score > best_score:
            best, best_score, best_n = s, score, n
    return best, {"train_return_pct": round((math.exp(best_score) - 1) * 100, 2) if best else None,
                  "train_trades": best_n, "candidates": len(PARAM_GRID)}


def box_stats(df: pd.DataFrame, boxes: list[Box]) -> dict:
    """How the boxes themselves behaved: size, duration, and how often a
    confirmed breakout followed through (reached one box height beyond the
    broken side) before closing back inside the box."""
    close, high, low = df["close"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float)
    broken = [b for b in boxes if b.breakout_bar is not None]
    confirmed = [b for b in broken if b.volume_confirmed]
    follow = 0
    for b in confirmed:
        height = b.ceiling - b.floor
        for j in range(b.breakout_bar + 1, len(df)):
            if b.breakout == "up" and high[j] >= b.ceiling + height or b.breakout == "down" and low[j] <= b.floor - height:
                follow += 1
                break
            if b.floor <= close[j] <= b.ceiling:
                break
    return {
        "boxes": len(boxes),
        "avg_height_pct": round(float(np.mean([(b.ceiling / b.floor - 1) * 100 for b in boxes])), 2) if boxes else None,
        "avg_duration_bars": round(float(np.mean([b.end - b.start + 1 for b in boxes])), 1) if boxes else None,
        "breakouts_up": sum(b.breakout == "up" for b in broken),
        "breakouts_down": sum(b.breakout == "down" for b in broken),
        "volume_confirmed_pct": round(len(confirmed) / len(broken) * 100, 1) if broken else None,
        "follow_through_pct": round(follow / len(confirmed) * 100, 1) if confirmed else None,
    }


def run_adaptive(candles: list[dict], interval: str, now: datetime | None = None) -> dict:
    """Walk-forward Box Breakout: settings re-chosen from past data every
    REOPT_EVERY bars; boxes, trades and stats only from out-of-sample bars."""
    df = candles_to_dataframe(closed_candles(candles, interval, now))
    n = len(df)
    train_bars = min(TRAIN_BARS, n // 2)
    if n < 200:
        raise ValueError(f"Need at least 200 closed candles, have {n}")
    ts = [t.isoformat() for t in df.index]
    last_close = float(df["close"].iloc[-1])

    segments, boxes, trades = [], [], []
    for start in range(train_bars, n, REOPT_EVERY):
        end = min(start + REOPT_EVERY, n)
        s, info = choose_settings(df.iloc[start - train_bars:start])
        segments.append({"from_bar": start, "to_bar": end - 1, "from_time": ts[start], "to_time": ts[end - 1],
                         "settings": asdict(s) if s else None, **info})
        # Run on everything up to now (boxes need their history, exits their
        # future), keep what *starts* inside this segment. Standing aside,
        # boxes are still drawn (default settings) but nothing is traded.
        seg_boxes = detect_boxes(df, s or BoxSettings())
        seg_trades = simulate(df, seg_boxes, s) if s else []
        if not s:
            for b in seg_boxes:  # volume flag only (simulate() normally sets it)
                if b.breakout_bar is not None and b.volume_confirmed is None:
                    v = df["volume"].rolling(BoxSettings().vol_ma).mean().iloc[b.breakout_bar]
                    b.volume_confirmed = bool(not math.isnan(v) and df["volume"].iloc[b.breakout_bar] > v)
        boxes += [b for b in seg_boxes if start <= b.established < end]
        for t in seg_trades:
            if not start <= t.entry_bar < end:
                continue
            prev = trades[-1] if trades else None
            if prev and (prev.exit_bar is None or t.entry_bar < prev.exit_bar):
                continue  # one position at a time across segments
            trades.append(t)
        segments[-1]["_boxes"] = seg_boxes

    oos = df.iloc[train_bars:]
    current = segments[-1]
    fee = BoxSettings().fee_pct
    stats = summarize(oos, trades, BoxSettings())
    active = None
    if current.get("_boxes"):
        last_box = current["_boxes"][-1]
        active = last_box if last_box.breakout is None else None
    open_trade = trades[-1] if trades and trades[-1].exit_bar is None else None

    def box_json(b: Box) -> dict:
        d = asdict(b)
        d.update(start_time=ts[b.start], established_time=ts[b.established], end_time=ts[b.end],
                 breakout_time=ts[b.breakout_bar] if b.breakout_bar is not None else None)
        return d

    def trade_json(t: Trade) -> dict:
        d = asdict(t)
        d["entry_time"] = ts[t.entry_bar]
        d["exit_time"] = ts[t.exit_bar] if t.exit_bar is not None else None
        d["stop_moves"] = [{**m, "time": ts[m["bar"]]} for m in t.stop_moves]
        d["return_pct"] = round(trade_return(t, fee, last_close) * 100, 3)
        return d

    return {
        "mode": "adaptive",
        "train_bars": train_bars,
        "reopt_every": REOPT_EVERY,
        "current": {k: v for k, v in current.items() if not k.startswith("_")},
        "next_reopt_in_bars": current["from_bar"] + REOPT_EVERY - (n - 1),
        "segments": [{k: v for k, v in seg.items() if not k.startswith("_")} for seg in segments],
        "boxes": [box_json(b) for b in boxes],
        "trades": [trade_json(t) for t in trades],
        "stats": stats,
        "box_stats": box_stats(df, boxes),
        "active_box": box_json(active) if active else None,
        "open_trade": trade_json(open_trade) if open_trade else None,
        "last_close": last_close,
        "last_time": ts[-1],
        "oos_from": ts[train_bars],
    }
