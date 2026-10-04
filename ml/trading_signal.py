"""Trading-signal scoring - the exact same weighted heuristic the
dashboard's "Trading signal" card computes client-side
(frontend/web/src/lib/indicators.ts's computeSignal), ported to Python so
data_collector.jobs.backfill_signal_history can evaluate what this rule
would have called historically against real outcomes, instead of the live
badge being an unverifiable black box.

Keep this in sync with indicators.ts by hand if that formula ever changes - there
is no shared implementation between the two, by necessity (one runs in the
browser against client-computed RSI/MACD, the other runs here against a
point-in-time retrained model for the backtest).
"""
from __future__ import annotations

TREND_THRESHOLD_PCT = 1.0
RSI_OVERSOLD = 30.0
RSI_OVERBOUGHT = 70.0
SENTIMENT_THRESHOLD = 0.1
SCORE_THRESHOLD = 1.5
MIN_CONFIDENCE = 40.0


def compute_signal_score(
    trend_pct: float, rsi: float | None, macd_bullish: bool | None, sentiment_avg: float,
) -> float:
    score = 0.0
    if trend_pct > TREND_THRESHOLD_PCT:
        score += 1
    elif trend_pct < -TREND_THRESHOLD_PCT:
        score -= 1
    if rsi is not None:
        if rsi < RSI_OVERSOLD:
            score += 1
        elif rsi > RSI_OVERBOUGHT:
            score -= 1
    if macd_bullish is not None:
        score += 1 if macd_bullish else -1
    if sentiment_avg > SENTIMENT_THRESHOLD:
        score += 0.5
    elif sentiment_avg < -SENTIMENT_THRESHOLD:
        score -= 0.5
    return score


def compute_signal_label(
    trend_pct: float, rsi: float | None, macd_bullish: bool | None, sentiment_avg: float, confidence: float,
) -> str:
    """"buy" | "sell" | "wait" - mirrors indicators.ts exactly: both a strong
    enough score AND a high enough model confidence are required, or a
    wildly volatile but low-confidence prediction (see the 1w interval's
    5/100-confidence +21% call, confirmed live) would otherwise fire a
    signal nobody should act on.
    """
    score = compute_signal_score(trend_pct, rsi, macd_bullish, sentiment_avg)
    if confidence >= MIN_CONFIDENCE and score >= SCORE_THRESHOLD:
        return "buy"
    if confidence >= MIN_CONFIDENCE and score <= -SCORE_THRESHOLD:
        return "sell"
    return "wait"
