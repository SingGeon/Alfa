"""Rule-based chart pattern recognition - candlestick (1-3 candle OHLC
relationships) and geometric (swing-extrema based) patterns.

Extrema detection and the four classic reversal patterns (Double Top,
Double Bottom, Head & Shoulders, Inverse Head & Shoulders) adapt the
approach from github.com/tysoncung/crypto-chart-patterns (MIT license,
(c) 2024 Tyson Cung, branch enhance-patterns-and-testing): scipy's
argrelextrema on a smoothed close-price series, refined by locating the
exact local high/low within a small window - smoothing first avoids
flagging every minor tick as its own "swing", then refining recovers the
real (unsmoothed) price so pattern levels reflect actual tradeable prices.

The remaining pattern types (Triangle, Wedge, Flag, Pennant, Channel, Cup
and Handle) are adapted from that same repo's enhanced_patterns.py, with
two corrections made here: triangles and wedges fit *separate* trendlines
to highs and lows (the source fits one line through a mixed sequence of
both, which conflates the two boundaries into a single average trend and
can't actually distinguish a triangle from a wedge); and every pattern
below reports whether its breakout has actually happened yet
("confirmed") plus a measured-move price target, neither of which the
source computes - a pattern still "forming" and one that already broke
out are very different things to tell a user.

Deliberately rule-based, not ML: same reasoning as removing the LSTM
backend (see ml/price_predictor.py's module docstring) - a few hundred
noisy candles isn't enough to *learn* what a double top looks like, but
the classic definitions don't need learning, they need arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from ml.features import candles_to_dataframe

# Only the tail of the loaded history is relevant to "what might happen
# from here" - an ancient double top from a year ago isn't a live signal.
RECENT_WINDOW = 150

EXTREMA_SMOOTHING = 3
EXTREMA_WINDOW_RANGE = 3

PEAK_TOLERANCE = 0.025    # two peaks/troughs count as "the same level" within 2.5%
MIN_LEG_PCT = 0.015       # the trough/peak between two tops/bottoms must differ by >= 1.5%, or it's noise
FLAT_SLOPE_PCT = 0.0006   # trendline slope below this (as % of price per candle) counts as "roughly flat"

POLE_MIN_MOVE_PCT = 0.08  # a flag/pennant's pole must move >= 8% to count as a real pole, not just drift
POLE_MAX_BARS = 12
CONSOLIDATION_BARS = 8

CHANNEL_WINDOW = 20


def _note(en: str, ro: str, lang: str) -> str:
    """Pick the localized wording for a pattern's `note` field. Pattern
    `name`s themselves stay in English everywhere (standard TA terminology,
    also matched by name in frontend/web/src/components/PriceChart.tsx's TWO_LINE_PATTERNS) -
    only the free-text explanation is translated.
    """
    return ro if lang == "ro" else en


@dataclass
class Pivot:
    index: int
    price: float


@dataclass
class Pattern:
    name: str
    direction: str  # "bullish" | "bearish" | "neutral"
    confirmed: bool
    note: str
    levels: list[dict] = field(default_factory=list)   # [{"index", "price"}, ...] - the pivots that define the shape
    neckline: float | None = None
    target: float | None = None
    # For two-boundary-line patterns only (triangle/wedge/channel): how many
    # of the leading entries in `levels` belong to the upper line - the rest
    # are the lower line. `highs + lows` aren't always equal-length (triangle/
    # wedge keep up to the last 4 pivots on *each* side independently), so a
    # renderer can't assume a plain half/half split.
    upper_count: int | None = None


# -- extrema detection ------------------------------------------------

def _find_extrema(closes: np.ndarray) -> tuple[list[Pivot], list[Pivot]]:
    """(highs, lows) - local turning points in `closes`, sorted by index."""
    n = len(closes)
    if n < 2 * EXTREMA_WINDOW_RANGE + 2:
        return [], []

    smooth = pd.Series(closes).rolling(window=EXTREMA_SMOOTHING, min_periods=1).mean().values
    local_max_pos = argrelextrema(smooth, np.greater)[0]
    local_min_pos = argrelextrema(smooth, np.less)[0]

    def refine(positions, is_max: bool) -> list[Pivot]:
        out = {}
        for pos in positions:
            if pos <= EXTREMA_WINDOW_RANGE or pos >= n - EXTREMA_WINDOW_RANGE:
                continue
            window = closes[pos - EXTREMA_WINDOW_RANGE: pos + EXTREMA_WINDOW_RANGE + 1]
            local_idx = int(np.argmax(window)) if is_max else int(np.argmin(window))
            # int(): pos comes from argrelextrema as numpy.int64, which
            # json.dumps can't serialize - every index that flows into the
            # API response (Pivot.index, and everywhere it's echoed back in
            # a pattern's "levels") needs to be a plain Python int.
            real_idx = int(pos) - EXTREMA_WINDOW_RANGE + local_idx
            out[real_idx] = closes[real_idx]
        return [Pivot(index=i, price=float(p)) for i, p in sorted(out.items())]

    return refine(local_max_pos, True), refine(local_min_pos, False)


# -- shared helpers -----------------------------------------------------

def _measured_target(direction: str, breakout_price: float, pattern_height: float) -> float:
    """"Measured move": the standard TA rule of thumb that a confirmed
    pattern tends to travel about as far past its breakout as the pattern
    itself was tall.
    """
    return breakout_price - pattern_height if direction == "bearish" else breakout_price + pattern_height


def _line_through(a: Pivot, b: Pivot) -> tuple[float, float, float]:
    """(slope_per_bar, normalized_slope, avg_price) for the 2-point line a->b.

    Not a least-squares fit - with only a handful of pivots per side, a
    line through the extremes is both simpler and less sensitive to a
    single noisy middle point than a full regression would be. Slope is
    also expressed as a fraction of price per candle, so "flat vs rising
    vs falling" classification doesn't depend on ETH's absolute price level.
    """
    avg_price = (a.price + b.price) / 2
    slope_per_bar = (b.price - a.price) / max(b.index - a.index, 1)
    return slope_per_bar, slope_per_bar / avg_price, avg_price


# -- classic reversal patterns (Double Top/Bottom, Head & Shoulders) ----

def _detect_double_top_bottom(highs: list[Pivot], lows: list[Pivot], last_price: float, lang: str = "en") -> Pattern | None:
    for i in range(len(highs) - 2, max(-1, len(highs) - 4), -1):
        h1, h2 = highs[i], highs[i + 1]
        if abs(h1.price - h2.price) / h1.price > PEAK_TOLERANCE:
            continue
        trough = next((l for l in lows if h1.index < l.index < h2.index), None)
        if not trough or (h1.price - trough.price) / h1.price < MIN_LEG_PCT:
            continue
        confirmed = last_price < trough.price
        return Pattern(
            name="Double Top", direction="bearish", confirmed=confirmed,
            levels=[{"index": h1.index, "price": h1.price}, {"index": h2.index, "price": h2.price},
                    {"index": trough.index, "price": trough.price}],
            neckline=trough.price,
            target=_measured_target("bearish", trough.price, h1.price - trough.price),
            note=_note(
                "confirmed: price broke below the neckline - the decline is expected to continue" if confirmed
                else "forming: two similar peaks - watch for a break below the neckline",
                "confirmat: prețul a rupt sub linia gâtului - se anticipează continuarea scăderii" if confirmed
                else "în formare: două vârfuri similare - urmărește ruperea sub linia gâtului",
                lang,
            ),
        )

    for i in range(len(lows) - 2, max(-1, len(lows) - 4), -1):
        l1, l2 = lows[i], lows[i + 1]
        if abs(l1.price - l2.price) / l1.price > PEAK_TOLERANCE:
            continue
        peak = next((h for h in highs if l1.index < h.index < l2.index), None)
        if not peak or (peak.price - l1.price) / l1.price < MIN_LEG_PCT:
            continue
        confirmed = last_price > peak.price
        return Pattern(
            name="Double Bottom", direction="bullish", confirmed=confirmed,
            levels=[{"index": l1.index, "price": l1.price}, {"index": l2.index, "price": l2.price},
                    {"index": peak.index, "price": peak.price}],
            neckline=peak.price,
            target=_measured_target("bullish", peak.price, peak.price - l1.price),
            note=_note(
                "confirmed: price broke above the neckline - the rally is expected to continue" if confirmed
                else "forming: two similar troughs - watch for a break above the neckline",
                "confirmat: prețul a rupt peste linia gâtului - se anticipează continuarea creșterii" if confirmed
                else "în formare: două minime similare - urmărește ruperea peste linia gâtului",
                lang,
            ),
        )
    return None


def _detect_triple_top_bottom(highs: list[Pivot], lows: list[Pivot], last_price: float, lang: str = "en") -> Pattern | None:
    if len(highs) >= 3:
        a, b, c = highs[-3:]
        spread = (max(a.price, b.price, c.price) - min(a.price, b.price, c.price)) / a.price
        if spread <= PEAK_TOLERANCE:
            troughs = [l for l in lows if a.index < l.index < c.index]
            neckline = min((t.price for t in troughs), default=min(a.price, b.price, c.price) * (1 - MIN_LEG_PCT))
            confirmed = last_price < neckline
            return Pattern(
                name="Triple Top", direction="bearish", confirmed=confirmed,
                levels=[{"index": p.index, "price": p.price} for p in (a, b, c)],
                neckline=neckline, target=_measured_target("bearish", neckline, a.price - neckline),
                note=_note(
                    "confirmed: three equal peaks, price broke support" if confirmed else "forming: three peaks at similar levels",
                    "confirmat: trei vârfuri egale, prețul a rupt suportul" if confirmed else "în formare: trei vârfuri la niveluri similare",
                    lang,
                ),
            )
    if len(lows) >= 3:
        a, b, c = lows[-3:]
        spread = (max(a.price, b.price, c.price) - min(a.price, b.price, c.price)) / a.price
        if spread <= PEAK_TOLERANCE:
            peaks = [h for h in highs if a.index < h.index < c.index]
            neckline = max((p.price for p in peaks), default=max(a.price, b.price, c.price) * (1 + MIN_LEG_PCT))
            confirmed = last_price > neckline
            return Pattern(
                name="Triple Bottom", direction="bullish", confirmed=confirmed,
                levels=[{"index": p.index, "price": p.price} for p in (a, b, c)],
                neckline=neckline, target=_measured_target("bullish", neckline, neckline - a.price),
                note=_note(
                    "confirmed: three equal troughs, price broke resistance" if confirmed else "forming: three troughs at similar levels",
                    "confirmat: trei minime egale, prețul a rupt rezistența" if confirmed else "în formare: trei minime la niveluri similare",
                    lang,
                ),
            )
    return None


def _detect_head_and_shoulders(highs: list[Pivot], lows: list[Pivot], last_price: float, lang: str = "en") -> Pattern | None:
    if len(highs) >= 3:
        ls, head, rs = highs[-3:]
        shoulders_match = abs(ls.price - rs.price) / ls.price < PEAK_TOLERANCE * 1.5
        head_is_highest = head.price > ls.price * (1 + MIN_LEG_PCT) and head.price > rs.price * (1 + MIN_LEG_PCT)
        if shoulders_match and head_is_highest:
            troughs = [l for l in lows if ls.index < l.index < rs.index]
            if len(troughs) >= 2:
                neckline = (troughs[0].price + troughs[-1].price) / 2
                confirmed = last_price < neckline
                return Pattern(
                    name="Head and Shoulders", direction="bearish", confirmed=confirmed,
                    levels=[{"index": p.index, "price": p.price} for p in (ls, head, rs)],
                    neckline=neckline, target=_measured_target("bearish", neckline, head.price - neckline),
                    note=_note(
                        "confirmed: price broke below the neckline after head and shoulders" if confirmed
                        else "forming: head and shoulders - watch for a break below the neckline",
                        "confirmat: prețul a rupt sub linia gâtului după cap și umeri" if confirmed
                        else "în formare: cap și umeri - urmărește ruperea sub linia gâtului",
                        lang,
                    ),
                )
    if len(lows) >= 3:
        ls, head, rs = lows[-3:]
        shoulders_match = abs(ls.price - rs.price) / ls.price < PEAK_TOLERANCE * 1.5
        head_is_lowest = head.price < ls.price * (1 - MIN_LEG_PCT) and head.price < rs.price * (1 - MIN_LEG_PCT)
        if shoulders_match and head_is_lowest:
            peaks = [h for h in highs if ls.index < h.index < rs.index]
            if len(peaks) >= 2:
                neckline = (peaks[0].price + peaks[-1].price) / 2
                confirmed = last_price > neckline
                return Pattern(
                    name="Inverse Head and Shoulders", direction="bullish", confirmed=confirmed,
                    levels=[{"index": p.index, "price": p.price} for p in (ls, head, rs)],
                    neckline=neckline, target=_measured_target("bullish", neckline, neckline - head.price),
                    note=_note(
                        "confirmed: price broke above the neckline after an inverse head and shoulders" if confirmed
                        else "forming: inverse head and shoulders - watch for a break above the neckline",
                        "confirmat: prețul a rupt peste linia gâtului după cap și umeri inversat" if confirmed
                        else "în formare: cap și umeri inversat - urmărește ruperea peste linia gâtului",
                        lang,
                    ),
                )
    return None


# -- triangles and wedges ------------------------------------------------

def _detect_triangle_or_wedge(highs: list[Pivot], lows: list[Pivot], last_price: float, lang: str = "en") -> Pattern | None:
    highs, lows = highs[-4:], lows[-4:]
    if len(highs) < 2 or len(lows) < 2:
        return None

    _, norm_upper, _ = _line_through(highs[0], highs[-1])
    _, norm_lower, _ = _line_through(lows[0], lows[-1])
    upper_dir = "up" if norm_upper > FLAT_SLOPE_PCT else "down" if norm_upper < -FLAT_SLOPE_PCT else "flat"
    lower_dir = "up" if norm_lower > FLAT_SLOPE_PCT else "down" if norm_lower < -FLAT_SLOPE_PCT else "flat"

    points = highs + lows
    prices = [p.price for p in points]
    pattern_height = max(prices) - min(prices)
    levels = [{"index": p.index, "price": p.price} for p in points]
    upper_b, lower_b = highs[-1], lows[-1]

    if upper_dir == "flat" and lower_dir == "up":
        return Pattern(name="Ascending Triangle", direction="bullish", confirmed=last_price > upper_b.price,
                        levels=levels, upper_count=len(highs), target=_measured_target("bullish", upper_b.price, pattern_height),
                        note=_note(
                            "horizontal resistance + progressively higher lows - usually breaks upward",
                            "rezistență orizontală + minime din ce în ce mai sus - de obicei se rupe în sus",
                            lang,
                        ))
    if upper_dir == "down" and lower_dir == "flat":
        return Pattern(name="Descending Triangle", direction="bearish", confirmed=last_price < lower_b.price,
                        levels=levels, upper_count=len(highs), target=_measured_target("bearish", lower_b.price, pattern_height),
                        note=_note(
                            "horizontal support + progressively lower highs - usually breaks downward",
                            "suport orizontal + maxime din ce în ce mai jos - de obicei se rupe în jos",
                            lang,
                        ))
    if upper_dir == "down" and lower_dir == "up":
        mid_price = (upper_b.price + lower_b.price) / 2
        return Pattern(name="Symmetrical Triangle", direction="bullish" if last_price > mid_price else "bearish",
                        confirmed=False, levels=levels, upper_count=len(highs), target=None,
                        note=_note(
                            "converging lines - direction depends on which side it breaks (undecided yet)",
                            "linii convergente - direcția depinde de partea pe care are loc ruperea (nedecisă încă)",
                            lang,
                        ))
    if upper_dir == "up" and lower_dir == "down":
        return Pattern(name="Expanding Triangle", direction="neutral", confirmed=False, levels=levels, upper_count=len(highs), target=None,
                        note=_note(
                            "rising volatility, diverging lines - unstable market, no clear direction",
                            "volatilitate în creștere, linii divergente - piață instabilă, fără direcție clară",
                            lang,
                        ))
    if upper_dir == "up" and lower_dir == "up" and norm_upper < norm_lower:
        return Pattern(name="Rising Wedge", direction="bearish", confirmed=last_price < lower_b.price,
                        levels=levels, upper_count=len(highs), target=_measured_target("bearish", lower_b.price, pattern_height),
                        note=_note(
                            "both lines rise but converge - usually ends with a downside break",
                            "ambele linii urcă dar converg - de obicei se termină cu o rupere în jos",
                            lang,
                        ))
    if upper_dir == "down" and lower_dir == "down" and norm_upper > norm_lower:
        return Pattern(name="Falling Wedge", direction="bullish", confirmed=last_price > upper_b.price,
                        levels=levels, upper_count=len(highs), target=_measured_target("bullish", upper_b.price, pattern_height),
                        note=_note(
                            "both lines fall but converge - usually ends with an upside break",
                            "ambele linii coboară dar converg - de obicei se termină cu o rupere în sus",
                            lang,
                        ))
    return None


# -- flags and pennants ---------------------------------------------------

def _detect_flag_or_pennant(closes: np.ndarray, highs_arr: np.ndarray, lows_arr: np.ndarray, lang: str = "en") -> Pattern | None:
    n = len(closes)
    if n < POLE_MAX_BARS + CONSOLIDATION_BARS + 2:
        return None
    consol_start = n - CONSOLIDATION_BARS
    consol_high = float(np.max(highs_arr[consol_start:]))
    consol_low = float(np.min(lows_arr[consol_start:]))
    consol_range_pct = (consol_high - consol_low) / consol_low
    # A real consolidation is noticeably tighter than the pole that made it -
    # otherwise it's just continued trending, not a pause.
    if consol_range_pct > POLE_MIN_MOVE_PCT * 0.6:
        return None

    for pole_bars in range(POLE_MAX_BARS, 3, -1):
        pole_start = consol_start - pole_bars
        if pole_start < 0:
            continue
        pole_open, pole_close = closes[pole_start], closes[consol_start - 1]
        pole_move_pct = (pole_close - pole_open) / pole_open
        if abs(pole_move_pct) < POLE_MIN_MOVE_PCT:
            continue

        bullish_pole = pole_move_pct > 0
        consol_slope_pct = (closes[-1] - closes[consol_start]) / closes[consol_start]
        against_pole = consol_slope_pct < 0.01 if bullish_pole else consol_slope_pct > -0.01
        if not against_pole:
            continue

        mid = consol_start + CONSOLIDATION_BARS // 2
        first_range = float(np.max(highs_arr[consol_start:mid]) - np.min(lows_arr[consol_start:mid]))
        second_range = float(np.max(highs_arr[mid:]) - np.min(lows_arr[mid:]))
        is_pennant = second_range < first_range * 0.7

        shape_name = "Pennant" if is_pennant else "Flag"
        direction = "bullish" if bullish_pole else "bearish"
        breakout_price = consol_high if bullish_pole else consol_low
        pole_height = abs(pole_close - pole_open)
        return Pattern(
            name=f"{'Bullish' if bullish_pole else 'Bearish'} {shape_name}", direction=direction, confirmed=False,
            levels=[{"index": pole_start, "price": float(pole_open)}, {"index": consol_start - 1, "price": float(pole_close)},
                    {"index": consol_start, "price": consol_high}, {"index": n - 1, "price": consol_low}],
            neckline=breakout_price, target=_measured_target(direction, breakout_price, pole_height),
            note=_note(
                "consolidation after a strong rally - usually continues upward on breakout" if bullish_pole
                else "consolidation after a strong decline - usually continues downward on breakout",
                "consolidare după o urcare puternică - de obicei continuă în sus la rupere" if bullish_pole
                else "consolidare după o scădere puternică - de obicei continuă în jos la rupere",
                lang,
            ),
        )
    return None


# -- channels -------------------------------------------------------------

def _detect_channel(highs_arr: np.ndarray, lows_arr: np.ndarray, offset: int, lang: str = "en") -> Pattern | None:
    """Parallel trendline channel over the most recent CHANNEL_WINDOW
    candles - fit with a proper least-squares line (numpy polyfit) rather
    than the 2-point lines used for triangles/wedges above, since a
    channel's trendlines are meant to represent *all* the highs/lows in
    the window, not just its first and last pivot.
    """
    n = len(highs_arr)
    if n < CHANNEL_WINDOW:
        return None
    window_highs = highs_arr[-CHANNEL_WINDOW:]
    window_lows = lows_arr[-CHANNEL_WINDOW:]
    x = np.arange(CHANNEL_WINDOW)

    high_slope, high_intercept = np.polyfit(x, window_highs, 1)
    low_slope, low_intercept = np.polyfit(x, window_lows, 1)
    avg_price = (window_highs.mean() + window_lows.mean()) / 2
    norm_high, norm_low = high_slope / avg_price, low_slope / avg_price

    # Parallel: both trendlines sloping the same way, within 40% of each
    # other's steepness - a real channel's bounds move together.
    if norm_high * norm_low <= 0 or abs(norm_high - norm_low) > max(abs(norm_high), abs(norm_low)) * 0.4:
        return None

    avg_slope = (norm_high + norm_low) / 2
    if abs(avg_slope) < FLAT_SLOPE_PCT:
        name, direction = "Horizontal Channel", "neutral"
    elif avg_slope > 0:
        name, direction = "Ascending Channel", "bullish"
    else:
        name, direction = "Descending Channel", "bearish"

    start_idx, end_idx = offset, offset + CHANNEL_WINDOW - 1
    levels = [
        {"index": start_idx, "price": float(high_intercept)},
        {"index": end_idx, "price": float(high_slope * (CHANNEL_WINDOW - 1) + high_intercept)},
        {"index": start_idx, "price": float(low_intercept)},
        {"index": end_idx, "price": float(low_slope * (CHANNEL_WINDOW - 1) + low_intercept)},
    ]
    return Pattern(
        name=name, direction=direction, confirmed=False, levels=levels, upper_count=2, target=None,
        note=_note(
            "price is oscillating between two parallel lines - potential to trade the range",
            "prețul oscilează între două linii paralele - potențial de tranzacționare între margini",
            lang,
        ),
    )


# -- cup and handle ---------------------------------------------------------

CUP_WINDOW = 30
HANDLE_WINDOW = 10


def _detect_cup_and_handle(closes: np.ndarray, lang: str = "en") -> Pattern | None:
    n = len(closes)
    if n < CUP_WINDOW + HANDLE_WINDOW:
        return None
    cup = closes[-(CUP_WINDOW + HANDLE_WINDOW): n - HANDLE_WINDOW]
    handle = closes[-HANDLE_WINDOW:]

    left_high = float(np.max(cup[:5]))
    right_high = float(np.max(cup[-5:]))
    mid = len(cup) // 2
    bottom = float(np.min(cup[max(0, mid - 5): mid + 5]))

    # The two "rims" of the cup should be at a similar height, with a
    # meaningfully lower bottom in between - otherwise it's just drift.
    if abs(left_high - right_high) / left_high > 0.05 or bottom > left_high * 0.85:
        return None

    handle_high, handle_low = float(np.max(handle)), float(np.min(handle))
    # The handle is a small pullback near the cup's rim, not a fresh leg down.
    if not (handle_low > right_high * 0.95 and handle_high < right_high * 1.02):
        return None

    offset = n - (CUP_WINDOW + HANDLE_WINDOW)
    return Pattern(
        name="Cup and Handle", direction="bullish", confirmed=closes[-1] > right_high,
        levels=[{"index": offset, "price": left_high}, {"index": offset + mid, "price": bottom},
                {"index": n - HANDLE_WINDOW, "price": right_high}, {"index": n - 1, "price": handle_low}],
        neckline=right_high, target=_measured_target("bullish", right_high, right_high - bottom),
        note=_note(
            "confirmed: price broke above the cup's rim - bullish continuation expected" if closes[-1] > right_high
            else "forming: cup and handle shape - watch for a break above the right rim",
            "confirmat: prețul a depășit marginea cupei - continuare bullish așteptată" if closes[-1] > right_high
            else "în formare: formă de cupă cu toartă - urmărește ruperea peste marginea dreaptă",
            lang,
        ),
    )


def detect_geometric_patterns(candles: list[dict], lang: str = "en") -> list[dict]:
    df = candles_to_dataframe(candles).reset_index()
    if len(df) < 20:
        return []
    recent = df.iloc[-RECENT_WINDOW:].reset_index(drop=True)
    offset = len(df) - len(recent)

    closes = recent["close"].to_numpy(dtype=float)
    highs_arr = recent["high"].to_numpy(dtype=float)
    lows_arr = recent["low"].to_numpy(dtype=float)
    last_price = float(closes[-1])
    highs, lows = _find_extrema(closes)

    candidates = [
        _detect_head_and_shoulders(highs, lows, last_price, lang),
        _detect_triple_top_bottom(highs, lows, last_price, lang),
        _detect_double_top_bottom(highs, lows, last_price, lang),
        _detect_triangle_or_wedge(highs, lows, last_price, lang),
        _detect_flag_or_pennant(closes, highs_arr, lows_arr, lang),
        _detect_channel(highs_arr, lows_arr, offset=0, lang=lang),
        _detect_cup_and_handle(closes, lang),
    ]
    patterns = [p for p in candidates if p is not None]

    # Prefer an already-confirmed (breakout happened) pattern over one
    # still forming, then prefer whichever pattern's most recent level is
    # closest to "now" - that's the one most relevant to the current price.
    patterns.sort(key=lambda p: (not p.confirmed, -max(l["index"] for l in p.levels)))

    results = []
    for p in patterns:
        levels = [
            {"index": lvl["index"] + offset, "timestamp": recent.iloc[lvl["index"]]["timestamp"].isoformat(), "price": round(lvl["price"], 2)}
            for lvl in p.levels
        ]
        results.append({
            "name": p.name, "direction": p.direction, "confirmed": p.confirmed, "note": p.note,
            "neckline": round(p.neckline, 2) if p.neckline is not None else None,
            "target": round(p.target, 2) if p.target is not None else None,
            "levels": levels,
            "upper_count": p.upper_count,
        })
    return results


# -- candlestick patterns (1-3 candles) -----------------------------------
#
# Each detector below mirrors the standard textbook definition for that
# shape - thresholds (e.g. "wick >= 2x body") aren't tuned/fitted to this
# app's data.

def _body(o, c) -> float:
    return abs(c - o)


def _rng(h, l) -> float:
    return max(h - l, 1e-9)


def _trend_before(closes: np.ndarray, i: int, lookback: int = 5) -> str:
    start = max(0, i - lookback)
    if start >= i:
        return "flat"
    change_pct = (closes[i - 1] - closes[start]) / closes[start]
    if change_pct > 0.008:
        return "up"
    if change_pct < -0.008:
        return "down"
    return "flat"


def _single_candle_pattern(o, h, l, c, trend: str, lang: str = "en") -> dict | None:
    b, r = _body(o, c), _rng(h, l)
    upper_wick, lower_wick = h - max(o, c), min(o, c) - l
    is_bull = c > o

    if b / r < 0.1:
        if lower_wick > r * 0.6 and upper_wick < r * 0.15:
            return {"name": "Dragonfly Doji", "direction": "bullish", "note": _note(
                "strong rejection of lower prices - possible reversal",
                "respingere puternică a prețurilor joase - posibilă revenire", lang)}
        if upper_wick > r * 0.6 and lower_wick < r * 0.15:
            return {"name": "Gravestone Doji", "direction": "bearish", "note": _note(
                "strong rejection of higher prices - possible drop",
                "respingere puternică a prețurilor mari - posibilă cădere", lang)}
        return {"name": "Doji", "direction": "neutral", "note": _note(
            "indecision - buyers and sellers are in balance",
            "indecizie - cumpărătorii și vânzătorii sunt în echilibru", lang)}

    if b / r > 0.92:
        return (
            {"name": "Bullish Marubozu", "direction": "bullish", "note": _note(
                "buyers were in full control for the entire candle", "control total al cumpărătorilor pe toată durata candelei", lang)}
            if is_bull else
            {"name": "Bearish Marubozu", "direction": "bearish", "note": _note(
                "sellers were in full control for the entire candle", "control total al vânzătorilor pe toată durata candelei", lang)}
        )

    small_body_near_top = b / r < 0.35 and lower_wick >= b * 2 and upper_wick < b * 0.6
    if small_body_near_top:
        if trend == "down":
            return {"name": "Hammer", "direction": "bullish", "note": _note(
                "rejection of lower prices after a decline - possible reversal",
                "respingere a prețurilor joase după o scădere - posibilă revenire", lang)}
        if trend == "up":
            return {"name": "Hanging Man", "direction": "bearish", "note": _note(
                "same shape as a Hammer, but after a rally - a warning of exhaustion",
                "aceeași formă ca un Hammer, dar după o creștere - avertisment de epuizare", lang)}

    small_body_near_bottom = b / r < 0.35 and upper_wick >= b * 2 and lower_wick < b * 0.6
    if small_body_near_bottom and trend == "down":
        return {"name": "Inverted Hammer", "direction": "bullish", "note": _note(
            "buyers tried to push the price up after a decline",
            "cumpărătorii au încercat să împingă prețul în sus după o scădere", lang)}

    return None


def _two_candle_pattern(prev, cur, lang: str = "en") -> dict | None:
    po, ph, pl, pc = prev
    o, h, l, c = cur
    prev_bull, prev_bear = pc > po, pc < po
    cur_bull, cur_bear = c > o, c < o
    body_prev, body_cur = _body(po, pc), _body(o, c)
    rng_prev = _rng(ph, pl)

    if prev_bear and cur_bull and o <= pc and c >= po and body_cur > body_prev:
        return {"name": "Bullish Engulfing", "direction": "bullish", "note": _note(
            "the green candle fully engulfed the prior red candle",
            "candela verde a acoperit complet candela roșie anterioară", lang)}
    if prev_bull and cur_bear and o >= pc and c <= po and body_cur > body_prev:
        return {"name": "Bearish Engulfing", "direction": "bearish", "note": _note(
            "the red candle fully engulfed the prior green candle",
            "candela roșie a acoperit complet candela verde anterioară", lang)}

    if prev_bear and cur_bull and o > pc and c < po and body_cur < body_prev * 0.6:
        return {"name": "Bullish Harami", "direction": "bullish", "note": _note(
            "small candle inside the prior one - the decline is losing steam",
            "candelă mică în interiorul celei anterioare - scăderea își pierde forța", lang)}
    if prev_bull and cur_bear and o < pc and c > po and body_cur < body_prev * 0.6:
        return {"name": "Bearish Harami", "direction": "bearish", "note": _note(
            "small candle inside the prior one - the rally is losing steam",
            "candelă mică în interiorul celei anterioare - creșterea își pierde forța", lang)}

    highs_match = abs(ph - h) / rng_prev < 0.1
    lows_match = abs(pl - l) / rng_prev < 0.1
    if highs_match and prev_bull and cur_bear:
        return {"name": "Tweezer Top", "direction": "bearish", "note": _note(
            "two matching peaks - the level rejected price twice",
            "două vârfuri identice - nivelul a respins prețul de două ori", lang)}
    if lows_match and prev_bear and cur_bull:
        return {"name": "Tweezer Bottom", "direction": "bullish", "note": _note(
            "two matching troughs - the level supported price twice",
            "două minime identice - nivelul a susținut prețul de două ori", lang)}

    return None


def _three_candle_pattern(c1, c2, c3, lang: str = "en") -> dict | None:
    o1, h1, l1, cl1 = c1
    o2, h2, l2, cl2 = c2
    o3, h3, l3, cl3 = c3
    body1, body2, body3 = _body(o1, cl1), _body(o2, cl2), _body(o3, cl3)
    mid1 = (o1 + cl1) / 2

    small_middle = body2 < body1 * 0.5 and body2 < body3 * 0.5
    if small_middle and cl1 < o1 and cl3 > o3 and cl3 > mid1:
        return {"name": "Morning Star", "direction": "bullish", "note": _note(
            "the decline exhausted itself, followed by a strong recovery on the third candle",
            "epuizarea scăderii, urmată de o revenire puternică pe a treia candelă", lang)}
    if small_middle and cl1 > o1 and cl3 < o3 and cl3 < mid1:
        return {"name": "Evening Star", "direction": "bearish", "note": _note(
            "the rally exhausted itself, followed by a strong drop on the third candle",
            "epuizarea creșterii, urmată de o cădere puternică pe a treia candelă", lang)}

    def strong_bull(o, h, l, cl):
        return cl > o and _body(o, cl) / _rng(h, l) > 0.6

    def strong_bear(o, h, l, cl):
        return cl < o and _body(o, cl) / _rng(h, l) > 0.6

    if strong_bull(o1, h1, l1, cl1) and strong_bull(o2, h2, l2, cl2) and strong_bull(o3, h3, l3, cl3) and cl2 > cl1 and cl3 > cl2:
        return {"name": "Three White Soldiers", "direction": "bullish", "note": _note(
            "three consecutive green candles, each higher than the last",
            "trei candele verzi consecutive, fiecare mai sus decât precedenta", lang)}
    if strong_bear(o1, h1, l1, cl1) and strong_bear(o2, h2, l2, cl2) and strong_bear(o3, h3, l3, cl3) and cl2 < cl1 and cl3 < cl2:
        return {"name": "Three Black Crows", "direction": "bearish", "note": _note(
            "three consecutive red candles, each lower than the last",
            "trei candele roșii consecutive, fiecare mai jos decât precedenta", lang)}

    return None


def detect_candlestick_patterns(candles: list[dict], lang: str = "en") -> list[dict]:
    df = candles_to_dataframe(candles).reset_index()
    if df.empty:
        return []
    recent = df.iloc[-RECENT_WINDOW:].reset_index(drop=True)
    offset = len(df) - len(recent)
    closes = recent["close"].to_numpy(dtype=float)

    found = []
    for i in range(len(recent)):
        row = recent.iloc[i]
        cur = (row["open"], row["high"], row["low"], row["close"])
        match = None
        if i >= 2:
            c1 = tuple(recent.iloc[i - 2][["open", "high", "low", "close"]])
            c2 = tuple(recent.iloc[i - 1][["open", "high", "low", "close"]])
            match = _three_candle_pattern(c1, c2, cur, lang)
        if not match and i >= 1:
            prev = tuple(recent.iloc[i - 1][["open", "high", "low", "close"]])
            match = _two_candle_pattern(prev, cur, lang)
        if not match:
            match = _single_candle_pattern(*cur, trend=_trend_before(closes, i), lang=lang)
        if match:
            found.append({
                "index": i + offset,
                "timestamp": row["timestamp"].isoformat(),
                **match,
            })
    return found


def analyze(candles: list[dict], lang: str = "en") -> dict:
    """Everything the dashboard's pattern-recognition panel needs, in one call."""
    return {
        "candlestick": detect_candlestick_patterns(candles, lang),
        "geometric": detect_geometric_patterns(candles, lang),
    }
