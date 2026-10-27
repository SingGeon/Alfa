from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ml import box_breakout as bb
from ml.features import candles_to_dataframe

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
NOW = T0 + timedelta(days=365)


def candle(i, o, h, l, c, v=100.0):
    return {"timestamp": T0 + timedelta(hours=i), "open": o, "high": h, "low": l, "close": c, "volume": v}


def range_then(after, n_range=30, low=99.0, high=101.0, vol=100.0):
    """n_range candles oscillating inside [low, high], then `after` (list of (o, h, l, c, v))."""
    out = []
    for i in range(n_range):
        c = 100.5 if i % 2 else 99.5
        out.append(candle(i, 100.0, high if i % 5 == 0 else 100.8, low if i % 5 == 1 else 99.2, c, vol))
    for j, (o, h, l, c, v) in enumerate(after):
        out.append(candle(n_range + j, o, h, l, c, v))
    return out


SETTINGS = dict(lookback=10, padding=3, vol_ma=5, fee_pct=0.0, stop_buffer_pct=0.0)


def test_box_is_detected_and_extended_until_breakout():
    candles = range_then([(100, 103, 100, 102.5, 300)])  # closes above the 101 ceiling
    out = bb.run(candles, "1h", bb.BoxSettings(**SETTINGS), now=NOW)
    box = out["boxes"][0]
    assert box["ceiling"] == 101.0 and box["floor"] == 99.0
    assert box["breakout"] == "up" and box["volume_confirmed"] is True
    assert box["end"] == box["breakout_bar"] - 1


def test_long_entry_stop_and_rr_target():
    candles = range_then([(100, 103, 100, 103.0, 300)])
    trade = bb.run(candles, "1h", bb.BoxSettings(**SETTINGS, rr=2.0), now=NOW)["trades"][0]
    assert trade["side"] == "long"
    assert trade["entry_price"] == 103.0
    assert trade["initial_stop"] == 99.0            # box floor
    assert trade["target"] == pytest.approx(103 + 2 * 4)


def test_breakout_without_volume_opens_no_trade():
    candles = range_then([(100, 103, 100, 102.5, 50)])  # volume below its MA
    out = bb.run(candles, "1h", bb.BoxSettings(**SETTINGS), now=NOW)
    assert out["boxes"][0]["volume_confirmed"] is False
    assert out["trades"] == []


def test_short_hits_target():
    after = [(100, 100, 97, 97.0, 300)] + [(97, 97.5, 90, 90.5, 100)]
    out = bb.run(range_then(after), "1h", bb.BoxSettings(**SETTINGS, rr=1.0), now=NOW)
    t = out["trades"][0]
    assert t["side"] == "short" and t["initial_stop"] == 101.0
    assert t["target"] == pytest.approx(93.0)
    assert t["exit_reason"] == "target" and t["exit_price"] == pytest.approx(93.0)
    assert out["stats"]["win_rate_pct"] == 100.0


def test_stop_wins_when_bar_touches_both():
    after = [(100, 103, 100, 103.0, 300), (103, 120, 98, 104, 100)]
    t = bb.run(range_then(after), "1h", bb.BoxSettings(**SETTINGS), now=NOW)["trades"][0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == 99.0


def test_gap_through_stop_fills_at_open():
    after = [(100, 103, 100, 103.0, 300), (95, 96, 94, 95, 100)]
    t = bb.run(range_then(after), "1h", bb.BoxSettings(**SETTINGS), now=NOW)["trades"][0]
    assert t["exit_price"] == 95.0


def test_short_with_target_below_zero_is_skipped():
    # 10x-tall box: risk exceeds the entry price, a 1:2 target would be negative.
    candles = range_then([(10, 10, 5, 5.0, 300)], low=9.0, high=100.0)
    out = bb.run(candles, "1h", bb.BoxSettings(**SETTINGS), now=NOW)
    assert out["boxes"][0]["breakout"] == "down"
    assert out["trades"] == []


def test_max_box_pct_filters_tall_boxes():
    candles = range_then([(100, 103, 100, 103.0, 300)])  # box height ~2%
    assert bb.run(candles, "1h", bb.BoxSettings(**SETTINGS, max_box_pct=1.0), now=NOW)["trades"] == []
    assert len(bb.run(candles, "1h", bb.BoxSettings(**SETTINGS, max_box_pct=5.0), now=NOW)["trades"]) == 1


def test_still_open_last_candle_is_dropped():
    candles = range_then([(100, 103, 100, 103.0, 300)])
    almost_now = candles[-1]["timestamp"] + timedelta(minutes=30)
    out = bb.run(candles, "1h", bb.BoxSettings(**SETTINGS), now=almost_now)
    assert out["trades"] == []  # the breakout candle hasn't closed yet
    assert out["active_box"] is not None


def test_no_lookahead_signal_only_uses_past_bars():
    candles = range_then([(100, 103, 100, 103.0, 300)] + [(103, 104, 102, 103, 100)] * 5)
    full = bb.run(candles, "1h", bb.BoxSettings(**SETTINGS), now=NOW)
    cut = bb.run(candles[:31], "1h", bb.BoxSettings(**SETTINGS), now=NOW)
    assert full["trades"][0]["entry_time"] == cut["trades"][0]["entry_time"]
    assert full["trades"][0]["entry_price"] == cut["trades"][0]["entry_price"]


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        bb.run(range_then([]), "1h", bb.BoxSettings(exit_mode="nope"), now=NOW)
    with pytest.raises(ValueError):
        bb.run(range_then([])[:5], "1h", bb.BoxSettings(), now=NOW)


def _trending_boxes(n_cycles=40, seed=0):
    """Ranges that repeatedly break out upward and keep going: breakouts pay."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out, price, i = [], 100.0, 0
    for _ in range(n_cycles):
        for _ in range(25):  # range
            c = price + rng.uniform(-0.5, 0.5)
            out.append(candle(i, price, price + 0.8, price - 0.8, c, 100)); i += 1
        for k in range(10):  # breakout leg up, with volume on its first bar
            o = price; price *= 1.01
            out.append(candle(i, o, price + 0.2, o - 0.1, price, 400 if k == 0 else 100)); i += 1
    return out


def test_adaptive_chooses_settings_from_past_only():
    candles = _trending_boxes()
    out = bb.run_adaptive(candles, "1h", now=NOW)
    seg = out["segments"][0]
    assert seg["from_bar"] == out["train_bars"]
    assert all(t["entry_bar"] >= out["train_bars"] for t in out["trades"])  # nothing traded in warm-up
    # Changing the future must not change the first choice.
    cut = bb.run_adaptive(candles[: seg["from_bar"] + 5], "1h", now=NOW)
    assert cut["segments"][0]["settings"] == seg["settings"]


def test_adaptive_trades_when_breakouts_pay():
    out = bb.run_adaptive(_trending_boxes(), "1h", now=NOW)
    assert out["current"]["settings"] is not None
    assert out["stats"]["trades"] > 0
    assert out["stats"]["total_return_pct"] > 0


def test_adaptive_stands_aside_when_nothing_made_money():
    # Every breakout is a fakeout: it closes outside with volume, then the
    # next bar slams through the opposite side (stop), and the range resumes.
    import numpy as np
    rng = np.random.default_rng(1)
    candles, i = [], 0
    for cycle in range(30):
        for _ in range(40):
            c = 100 + rng.uniform(-0.5, 0.5)
            candles.append(candle(i, 100, 100.8, 99.2, c, 100)); i += 1
        if cycle % 2:
            candles.append(candle(i, 100, 102.5, 99.9, 102.0, 400)); i += 1   # fake breakout up
            candles.append(candle(i, 102, 102.1, 97.0, 100.0, 100)); i += 1   # back through the floor
        else:
            candles.append(candle(i, 100, 100.1, 97.5, 98.0, 400)); i += 1    # fake breakout down
            candles.append(candle(i, 98, 103.0, 97.9, 100.0, 100)); i += 1   # back through the ceiling
    out = bb.run_adaptive(candles, "1h", now=NOW)
    assert all(s["settings"] is None for s in out["segments"])
    assert out["trades"] == []


def test_box_stats_counts_follow_through():
    out = bb.run_adaptive(_trending_boxes(), "1h", now=NOW)
    st = out["box_stats"]
    assert st["boxes"] > 0 and st["breakouts_up"] > 0
    assert 0 <= st["follow_through_pct"] <= 100


# -- risk-adjusted metrics (Ch. 9: Sharpe/Sortino/Calmar, SQN, sample-size discipline) ----------

def test_summarize_computes_sortino_calmar_sqn():
    idx = pd.DatetimeIndex([T0 + timedelta(days=i) for i in range(6)])
    df = pd.DataFrame({"close": [100.0] * 6}, index=idx)
    trades = [
        bb.Trade("long", 0, 100.0, 90.0, 90.0, 120.0, exit_bar=1, exit_price=120.0, exit_reason="target"),
        bb.Trade("long", 1, 100.0, 90.0, 90.0, 120.0, exit_bar=2, exit_price=90.0, exit_reason="stop"),
        bb.Trade("long", 2, 100.0, 90.0, 90.0, 120.0, exit_bar=3, exit_price=110.0, exit_reason="target"),
        bb.Trade("long", 3, 100.0, 90.0, 90.0, 120.0, exit_bar=4, exit_price=95.0, exit_reason="stop"),
    ]
    out = bb.summarize(df, trades, bb.BoxSettings(fee_pct=0.0))
    assert out["win_rate_pct"] == 50.0
    assert out["sortino"] == pytest.approx(1.5, abs=0.01)
    assert out["sqn"] is not None and out["calmar"] is not None
    assert out["low_sample"] is True   # 4 trades, below LOW_SAMPLE_TRADES


def test_equity_curve_matches_cumulative_trade_returns():
    trades = [
        bb.Trade("long", 0, 100.0, 90.0, 90.0, 120.0, exit_bar=1, exit_price=120.0, exit_reason="target"),
        bb.Trade("long", 1, 100.0, 90.0, 90.0, 120.0, exit_bar=2, exit_price=90.0, exit_reason="stop"),
    ]
    curve = bb.equity_curve(trades, fee_pct=0.0)
    assert curve[0] == pytest.approx(1.2)          # +20% after the first trade
    assert curve[1] == pytest.approx(1.2 * 0.9)    # then -10% after the second


def test_equity_curve_with_no_closed_trades_is_flat():
    assert bb.equity_curve([], fee_pct=0.0).tolist() == [1.0]


def test_low_sample_flag_clears_above_threshold():
    idx = pd.DatetimeIndex([T0 + timedelta(hours=i) for i in range(bb.LOW_SAMPLE_TRADES + 5)])
    df = pd.DataFrame({"close": [100.0] * len(idx)}, index=idx)
    trades = [
        bb.Trade("long", i, 100.0, 90.0, 90.0, 120.0, exit_bar=i + 1,
                 exit_price=120.0 if i % 2 else 90.0, exit_reason="target")
        for i in range(bb.LOW_SAMPLE_TRADES + 1)
    ]
    out = bb.summarize(df, trades, bb.BoxSettings(fee_pct=0.0))
    assert out["trades"] == bb.LOW_SAMPLE_TRADES + 1
    assert out["low_sample"] is False


# -- exit-management variants (Ch. 9 case studies: fixed target, ATR stop, trailing, breakeven) --

def test_atr_stop_overrides_the_box_based_stop():
    candles = range_then([(100, 103, 100, 103.0, 300)])
    s = bb.BoxSettings(**SETTINGS, stop_atr_mult=1.0)
    trade = bb.run(candles, "1h", s, now=NOW)["trades"][0]
    assert trade["initial_stop"] != 99.0   # no longer the plain box floor

    df = candles_to_dataframe(bb.closed_candles(candles, "1h", NOW))
    atr = bb._atr(df)
    assert trade["initial_stop"] == pytest.approx(trade["entry_price"] - atr[trade["entry_bar"]])


def test_breakeven_moves_stop_once_r_target_is_reached():
    after = [
        (100, 103, 100, 103.0, 300),     # breakout: entry 103, stop 99 (floor), risk 4
        (104, 109, 103.5, 108.0, 100),   # close 108 -> 1.25R in favor -> stop moves to breakeven (103)
        (107, 108, 101, 102.0, 100),     # low 101 would blow through the old stop (99)... but not the new one
    ]
    s = bb.BoxSettings(**SETTINGS, rr=10.0, breakeven_at_r=1.0)
    trade = bb.run(range_then(after), "1h", s, now=NOW)["trades"][0]
    assert trade["stop_moves"][0]["reason"] == "breakeven"
    assert trade["exit_reason"] == "breakeven_stop"
    assert trade["exit_price"] == pytest.approx(103.0)   # exited near breakeven, not at the original stop


def test_compare_exit_management_runs_all_four_variants_on_the_same_signals():
    candles = _trending_boxes()
    df = candles_to_dataframe(bb.closed_candles(candles, "1h", NOW))
    s = bb.BoxSettings(**SETTINGS)
    boxes = bb.detect_boxes(df, s)
    out = bb.compare_exit_management(df, boxes, s)
    assert set(out) == {"fixed_target", "atr_stop", "trailing_stop", "breakeven_dynamic"}
    for variant in out.values():
        assert "sortino" in variant and "sqn" in variant and "calmar" in variant
        assert isinstance(variant["equity_curve"], list) and len(variant["equity_curve"]) >= 1
        assert variant["equity_curve"][0] != 0  # a real cumulative-return curve, not left empty/zeroed


# -- statistical validation of the breakout signal itself (Ch. 7) --------------------------------

def test_validate_breakout_signal_scores_up_and_down_separately():
    candles = _trending_boxes()
    df = candles_to_dataframe(bb.closed_candles(candles, "1h", NOW))
    s = bb.BoxSettings(**SETTINGS)
    boxes = bb.detect_boxes(df, s)
    bb.simulate(df, boxes, s)   # sets volume_confirmed on each box
    out = bb.validate_breakout_signal(df, boxes, horizon=5)
    assert set(out) == {"up", "down"}
    # This fixture only ever breaks out up and keeps going: a real, scoreable signal.
    assert out["up"]["n_signal"] > 0
    assert out["up"]["mean_return_signal_pct"] > out["up"]["mean_return_baseline_pct"]
