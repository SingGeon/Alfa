"""ml/pattern_recognition.py - candlestick (exact OHLC construction) and
geometric (synthetic price paths shaped like the textbook pattern) tests.
Deliberately not using the `synthetic_candles` fixture (random-walk noise)
here - pattern detection needs OHLC shaped *exactly* like the pattern
being tested, not merely plausible-looking price data.
"""
from datetime import datetime, timedelta, timezone

from ml import pattern_recognition as pr


def _ramp(start, end, steps):
    """`steps` linearly-spaced closes from start to end, inclusive of both."""
    return [start + (end - start) * i / (steps - 1) for i in range(steps)]


def _hill(peak, half_width=2, step=1.0):
    """A small rounded peak (strictly up then strictly down) rather than a
    single-point spike or a flat multi-candle plateau - both of those
    defeat scipy's argrelextrema (a plateau has no strict local max at
    all; an exact single-tick spike can tie with its smoothed neighbors
    under the rolling-mean smoothing _find_extrema applies first). A few
    small, strictly-monotonic steps on each side survive smoothing while
    still reading as "the same peak level" once refined.
    """
    up = [peak - step * (half_width - i) for i in range(half_width)]
    down = [peak - step * (i + 1) for i in range(half_width)]
    return up + [peak] + down


def _valley(bottom, half_width=2, step=1.0):
    down = [bottom + step * (half_width - i) for i in range(half_width)]
    up = [bottom + step * (i + 1) for i in range(half_width)]
    return down + [bottom] + up


def _candles_from_closes(closes, interval_hours=1, wick=0.3):
    """Turn a plain list of closing prices into a candle list with a tiny
    wick on each side - keeps OHLC tightly tracking `closes` (what the
    geometric detectors, which key off `close`, actually look at) while
    staying valid OHLC (high >= max(open, close), low <= min(open, close)).
    """
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = []
    prev_close = closes[0]
    for i, close in enumerate(closes):
        open_ = prev_close
        candles.append({
            "timestamp": start + timedelta(hours=i * interval_hours),
            "open": open_,
            "high": max(open_, close) + wick,
            "low": min(open_, close) - wick,
            "close": close,
            "volume": 1000,
        })
        prev_close = close
    return candles


def _candle(o, h, l, c, ts):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000}


# -- candlestick patterns ------------------------------------------------

def test_bullish_engulfing_detected():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(100, 101, 90, 92, ts),
        _candle(91, 105, 90, 103, ts + timedelta(hours=1)),
    ]
    found = pr.detect_candlestick_patterns(candles)
    names = [p["name"] for p in found]
    assert "Bullish Engulfing" in names
    match = next(p for p in found if p["name"] == "Bullish Engulfing")
    assert match["direction"] == "bullish"


def test_bearish_engulfing_detected():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(90, 101, 89, 100, ts),
        _candle(101, 102, 87, 88, ts + timedelta(hours=1)),
    ]
    found = pr.detect_candlestick_patterns(candles)
    names = [p["name"] for p in found]
    assert "Bearish Engulfing" in names


def test_doji_detected():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [_candle(100, 110, 90, 100.05, ts)]
    found = pr.detect_candlestick_patterns(candles)
    assert any(p["name"] == "Doji" for p in found)


def test_hammer_after_downtrend_not_hanging_man():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # A clear downtrend, then a hammer shape on the last candle: body is
    # ~12% of the range (small, but not so small it reads as a Doji), sits
    # near the top, with a long lower wick and a negligible upper one.
    candles = [_candle(110 - i, 111 - i, 109 - i, 110 - i - 1, ts + timedelta(hours=i)) for i in range(6)]
    candles.append(_candle(100, 102.5, 90, 102, ts + timedelta(hours=6)))
    found = pr.detect_candlestick_patterns(candles)
    match = next((p for p in found if p["index"] == len(candles) - 1), None)
    assert match is not None
    assert match["name"] == "Hammer"
    assert match["direction"] == "bullish"


def test_hanging_man_after_uptrend():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [_candle(100 + i, 101 + i, 99 + i, 100 + i + 1, ts + timedelta(hours=i)) for i in range(6)]
    candles.append(_candle(106, 108.5, 96, 108, ts + timedelta(hours=6)))
    found = pr.detect_candlestick_patterns(candles)
    match = next((p for p in found if p["index"] == len(candles) - 1), None)
    assert match is not None
    assert match["name"] == "Hanging Man"
    assert match["direction"] == "bearish"


def test_three_white_soldiers_detected():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [
        _candle(100, 105.5, 99.5, 105, ts),
        _candle(105.2, 110.5, 104.8, 110, ts + timedelta(hours=1)),
        _candle(110.2, 115.5, 109.8, 115, ts + timedelta(hours=2)),
    ]
    found = pr.detect_candlestick_patterns(candles)
    assert any(p["name"] == "Three White Soldiers" for p in found)


def test_no_patterns_on_plain_repeated_candles():
    # A moderate body (half the range, balanced wicks) repeated identically
    # matches no candlestick shape at all: not a Doji (body isn't tiny), not
    # a Marubozu (wicks aren't negligible), not a Hammer/Inverted Hammer
    # (body isn't near either edge), and repeating the exact same candle
    # can't engulf/harami/tweezer itself or chain into three soldiers
    # (body/range 0.5 is below the 0.6 "strong" bar three-soldiers needs).
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = [_candle(100, 103, 99, 102, ts + timedelta(hours=i)) for i in range(10)]
    found = pr.detect_candlestick_patterns(candles)
    assert found == []


# -- geometric patterns ---------------------------------------------------

def test_double_top_detected_and_confirmed():
    # Rise to a peak, pull back to a trough, rise to a similar second peak,
    # then break down through the trough - a textbook confirmed double top.
    closes = (
        _ramp(100, 128, 9) + _hill(130) + _ramp(126, 116, 7) + _valley(114)
        + _ramp(116, 126, 7) + _hill(130) + _ramp(126, 90, 10)
    )
    candles = _candles_from_closes(closes)
    patterns = pr.detect_geometric_patterns(candles)
    names = [p["name"] for p in patterns]
    assert "Double Top" in names
    dt = next(p for p in patterns if p["name"] == "Double Top")
    assert dt["direction"] == "bearish"
    assert dt["confirmed"] is True
    assert dt["target"] < dt["neckline"]


def test_double_bottom_detected_and_confirmed():
    # Mirror image of the double top above.
    closes = (
        _ramp(130, 102, 9) + _valley(100) + _ramp(104, 114, 7) + _hill(116)
        + _ramp(114, 104, 7) + _valley(100) + _ramp(104, 140, 10)
    )
    candles = _candles_from_closes(closes)
    patterns = pr.detect_geometric_patterns(candles)
    names = [p["name"] for p in patterns]
    assert "Double Bottom" in names
    db = next(p for p in patterns if p["name"] == "Double Bottom")
    assert db["direction"] == "bullish"
    assert db["confirmed"] is True
    assert db["target"] > db["neckline"]


def test_ascending_triangle_detected():
    # Flat resistance around 140 (each peak the same rounded-hill shape),
    # rising support (each trough's valley meaningfully higher than the last).
    closes = [100]
    support = 100
    for _ in range(4):
        closes += _ramp(closes[-1], 137, 6) + _hill(140)
        support += 10
        closes += _ramp(closes[-1], support, 6) + _valley(support)
    candles = _candles_from_closes(closes)
    patterns = pr.detect_geometric_patterns(candles)
    names = [p["name"] for p in patterns]
    assert "Ascending Triangle" in names


def test_geometric_patterns_need_at_least_20_candles():
    candles = _candles_from_closes([100, 101, 99, 102])
    assert pr.detect_geometric_patterns(candles) == []


def test_candlestick_patterns_on_empty_input():
    assert pr.detect_candlestick_patterns([]) == []


def test_analyze_combines_both():
    candles = _candles_from_closes([100 + (i % 5) for i in range(60)])
    result = pr.analyze(candles)
    assert "candlestick" in result
    assert "geometric" in result
