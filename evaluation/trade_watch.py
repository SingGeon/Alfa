"""Watches ml.evolution's fund trades (per coin/interval) and draws a snapshot the instant one
opens or closes - never on a timer, never a forecast (see evaluation.charts' trade-snapshot
section and memory feedback-chart-entry-exit-not-prediction). The model decides *when* to
trade; this records exactly those moments, nothing else.

Each check compares the live population's current trade list against what was last seen
(persisted in a small JSON file, survives restarts) to tell apart three cases: a brand-new
entry, a flip (the open trade just closed *and* a new one opened in the same tick), and a
plain close. Cheap enough to run every minute alongside the rest of evaluation.jobs - it's
just a trade-list diff, no model inference.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config
from database import repository
from evaluation import charts, storage

logger = logging.getLogger(__name__)

_CONTEXT_BEFORE = 60    # candles of context before the trade's entry
_CONTEXT_AFTER = 20     # candles of context past its most recent event


def _state_path() -> Path:
    # A function, not a module-level constant: config.EVAL_DATA_DIR is monkeypatched per-test
    # (tests/conftest.py's evaluation_data_dir), and a constant computed once at import time
    # would freeze the first value it ever saw instead of tracking that.
    return Path(config.EVAL_DATA_DIR) / "trade_watch_state.json"


def _load_state() -> dict:
    path = _state_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.exception("Could not read %s, starting fresh", path)
    return {}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _context_candles(coin_id: str, interval: str, trade: dict) -> list[dict]:
    """Real candles around the trade (repository.get_candles, native datetimes) converted to
    the ISO-string shape evaluation.charts._draw_candles expects."""
    cutoff = storage.parse_iso(trade.get("exit_time") or trade["entry_time"])
    raw = repository.get_candles(coin_id, interval, limit=max(500, (_CONTEXT_BEFORE + _CONTEXT_AFTER) * 5))
    if not raw:
        return []

    def ts(c: dict) -> datetime:
        t = c["timestamp"]
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)

    idx = next((i for i, c in enumerate(raw) if ts(c) > cutoff), len(raw))
    window = raw[max(0, idx - _CONTEXT_BEFORE): min(len(raw), idx + _CONTEXT_AFTER)]
    return [{"timestamp": storage.to_iso(ts(c)), "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"]} for c in window]


def check_interval(coin_id: str, interval: str, model_name: str) -> int:
    """Draws any new entry/exit snapshot for `interval`'s live population since the last
    check. Returns how many were drawn."""
    from ml import evolution_report

    ev = evolution_report.current(f"{coin_id}_{interval}")
    if ev is None:
        return 0
    trades = ev.fund_trades_report()["trades"]
    if not trades:
        return 0
    latest = trades[-1]
    key = f"{coin_id}_{interval}"
    state = _load_state()
    last = state.get(key)
    drawn = 0

    def draw(trade: dict, kind: str) -> None:
        nonlocal drawn
        try:
            candles = _context_candles(coin_id, interval, trade)
            charts.render_trade_snapshot(trade, candles, interval, model_name, kind)
            drawn += 1
        except Exception:
            logger.exception("Failed to draw %s trade snapshot for %s/%s", kind, coin_id, interval)

    if last is None or last.get("entry_bar") != latest["entry_bar"]:
        # A new leg started. If the previous one was still open as of the last check, it must
        # have just closed in the same tick (a flip) - find and draw its exit first.
        if last is not None and last.get("entry_bar") is not None and last.get("exit_bar") is None:
            closed = next((t for t in reversed(trades) if t["entry_bar"] == last["entry_bar"]), None)
            if closed is not None and closed.get("exit_bar") is not None:
                draw(closed, "exit")
        draw(latest, "entry")
    elif latest.get("exit_bar") is not None and (last is None or last.get("exit_bar") is None):
        draw(latest, "exit")

    state[key] = {"entry_bar": latest["entry_bar"], "exit_bar": latest.get("exit_bar")}
    _save_state(state)
    return drawn


def check_all(coin_id: str | None = None) -> int:
    from evaluation.recorder import model_name_for

    coin_id = coin_id or config.COIN_ID
    model_name = model_name_for("sklearn", "tuned")
    drawn = 0
    for interval in config.EVAL_SNAPSHOT_INTERVALS:
        drawn += check_interval(coin_id, interval, model_name)
    return drawn
