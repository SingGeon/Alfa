// Technical indicators + the Buy/Sell/Wait heuristic, ported as-is from
// the original app.js. The signal formula must stay in sync with
// ml/trading_signal.py (the backend backtests this exact rule).
import type { Candle, PredictionPoint } from "./api";

export function sma(values: number[], period: number): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    out.push(i >= period - 1 ? sum / period : null);
  }
  return out;
}

export type TrendSignal = "buy" | "sell" | null;

/** Trend-filtered Bollinger Band entry signal: "buy" where a candle's low wicks below the lower
 * band and it closes back above it, while the last `trendWindow` candles (before this one) were
 * all trading above the SMA (an established uptrend) - "sell" the mirror case in a downtrend.
 * null everywhere else. A generic trend-filter + mean-reversion combination, not tied to either
 * indicator's own math - needs both SMA and Bollinger computed to mean anything.
 *
 * Uses the candle's low/high, not its open: on a continuous 24/7 market (crypto, no session
 * gaps) a candle's open is essentially the prior candle's close, so "open below the lower band"
 * collapses to "the prior close was below the lower band" - which can't coexist with that same
 * prior candle having been part of an "above the SMA" uptrend (the lower band always sits below
 * the SMA). The low/high of the *current* candle is the gapless-market equivalent of a forex-style
 * gap-through-and-recover open: it captures the same "touched the band, closed back past it"
 * shape without requiring an open-price gap that rarely happens here. */
export function smaBollingerSignals(
  lows: number[], highs: number[], closes: number[], smaVals: (number | null)[],
  bbUpper: (number | null)[], bbLower: (number | null)[], trendWindow = 5,
): TrendSignal[] {
  const n = closes.length;
  const out: TrendSignal[] = new Array(n).fill(null);
  for (let i = trendWindow; i < n; i++) {
    const up = bbUpper[i], lo = bbLower[i];
    if (up == null || lo == null) continue;
    let allAbove = true, allBelow = true;
    for (let j = i - trendWindow; j < i; j++) {
      const m = smaVals[j];
      if (m == null) { allAbove = allBelow = false; break; }
      if (closes[j] <= m) allAbove = false;
      if (closes[j] >= m) allBelow = false;
    }
    if (allAbove && lows[i] < lo && closes[i] > lo) out[i] = "buy";
    else if (allBelow && highs[i] > up && closes[i] < up) out[i] = "sell";
  }
  return out;
}

/** The slope of `series` over the trailing `window` points at each index, via closed-form
 * linear regression (x = 0..window-1) - the rate of change of an indicator over time (a
 * positive slope = rising, negative = falling). null until `window` values are available, or
 * wherever a value inside that window is itself null. */
export function rollingSlope(series: (number | null)[], window: number): (number | null)[] {
  const n = series.length;
  const out: (number | null)[] = new Array(n).fill(null);
  const xMean = (window - 1) / 2;
  let xVar = 0;
  for (let k = 0; k < window; k++) xVar += (k - xMean) ** 2;
  for (let i = window - 1; i < n; i++) {
    const ys: number[] = [];
    let ok = true;
    for (let k = 0; k < window; k++) {
      const v = series[i - window + 1 + k];
      if (v == null) { ok = false; break; }
      ys.push(v);
    }
    if (!ok) continue;
    const yMean = ys.reduce((a, b) => a + b, 0) / window;
    let sumXY = 0;
    for (let k = 0; k < window; k++) sumXY += (k - xMean) * (ys[k] - yMean);
    out[i] = xVar > 0 ? sumXY / xVar : 0;
  }
  return out;
}

export type TrendConfirmation = "up" | "down" | null;

/** "up" at index i where every one of `slopeSeries` agrees positive there (every moving average
 * rising at once - a confirmed uptrend), "down" where every one agrees negative, null wherever
 * they disagree or aren't available yet. Several moving averages of different lengths that all
 * point the same way carry more weight than any single one alone - a trend-confirmation lens,
 * not a complete trading strategy on its own. */
export function confirmedTrend(slopeSeries: (number | null)[][]): TrendConfirmation[] {
  const n = slopeSeries[0]?.length ?? 0;
  const out: TrendConfirmation[] = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const vals = slopeSeries.map((s) => s[i]);
    if (vals.some((v) => v == null)) continue;
    if (vals.every((v) => (v as number) > 0)) out[i] = "up";
    else if (vals.every((v) => (v as number) < 0)) out[i] = "down";
  }
  return out;
}

export type FractalKind = "high" | "low" | null;

/** A classic fractal pivot: candle i's high strictly exceeds the highs of the `window` candles
 * on each side (a high/bearish fractal - possible local top), or its low is strictly below the
 * lows of those same neighbors (a low/bullish fractal - possible local bottom). Raw candle
 * comparison, no smoothing - a different, simpler lens than the smoothed-extrema swing
 * detection the backend uses for fitting chart-pattern shapes. null for every other candle, and
 * for the first/last `window` (not enough neighbors on both sides to compare yet). */
export function findFractals(candles: { high: number; low: number }[], window = 2): FractalKind[] {
  const n = candles.length;
  const out: FractalKind[] = new Array(n).fill(null);
  for (let i = window; i < n - window; i++) {
    const h = candles[i].high, l = candles[i].low;
    let isHigh = true, isLow = true;
    for (let k = 1; k <= window && (isHigh || isLow); k++) {
      if (candles[i - k].high >= h || candles[i + k].high >= h) isHigh = false;
      if (candles[i - k].low <= l || candles[i + k].low <= l) isLow = false;
    }
    if (isHigh) out[i] = "high";
    else if (isLow) out[i] = "low";
  }
  return out;
}

/** Sorts `levels` and collapses any that land within `thresholdPct` of the previous kept one
 * into a single level (keeps the first of each cluster) - several fractals near the same price
 * become one key level instead of a cluttered handful of near-duplicate lines. */
function clusterLevels(levels: number[], thresholdPct: number): number[] {
  const sorted = [...levels].sort((a, b) => a - b);
  const out: number[] = [];
  for (const level of sorted) {
    const prev = out[out.length - 1];
    if (prev != null && level > 0 && Math.abs(level - prev) <= (thresholdPct / 100) * level) continue;
    out.push(level);
  }
  return out;
}

/** Support/resistance key levels: resistance = the highs of "high" fractal pivots
 * (findFractals), support = the lows of "low" pivots, each list clustered with clusterLevels so
 * nearby fractals collapse into one level rather than drawing a line per pivot. `window=3` is a
 * slightly wider pivot window than the standalone fractal indicator's default of 2, since a
 * level worth drawing on the chart should rest on a somewhat more significant local extreme.
 * `thresholdPct` is tuned for crypto's intraday volatility.
 *
 * Over hundreds of loaded candles, clustering alone still leaves dozens of pivots more than
 * `thresholdPct` apart - each kept as its own line, flooding the chart. `maxLevels` keeps only
 * the N closest to `currentPrice` on each side, the ones actually relevant to price right now. */
export function supportResistanceLevels(
  candles: { high: number; low: number }[], currentPrice: number, window = 3, thresholdPct = 1, maxLevels = 4,
): { support: number[]; resistance: number[] } {
  const fractals = findFractals(candles, window);
  const nearest = (levels: number[]) =>
    [...levels].sort((a, b) => Math.abs(a - currentPrice) - Math.abs(b - currentPrice)).slice(0, maxLevels).sort((a, b) => a - b);
  const resistance = nearest(clusterLevels(fractals.flatMap((k, i) => (k === "high" ? [candles[i].high] : [])), thresholdPct));
  const support = nearest(clusterLevels(fractals.flatMap((k, i) => (k === "low" ? [candles[i].low] : [])), thresholdPct));
  return { support, resistance };
}

export function ema(values: number[], period: number): number[] {
  const k = 2 / (period + 1);
  const out: number[] = [];
  let prev: number | null = null;
  for (const v of values) {
    prev = prev === null ? v : v * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

export function bollinger(values: number[], period: number, mult: number) {
  const middle = sma(values, period);
  const upper: (number | null)[] = [];
  const lower: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    const m = middle[i];
    if (m === null) {
      upper.push(null);
      lower.push(null);
      continue;
    }
    let sumSq = 0;
    for (let j = i - period + 1; j <= i; j++) sumSq += (values[j] - m) ** 2;
    const std = Math.sqrt(sumSq / period);
    upper.push(m + mult * std);
    lower.push(m - mult * std);
  }
  return { upper, middle, lower };
}

/** Cumulative VWAP over the loaded window (no session anchor - the chart
 *  spans 15m..1w, where a "trading day" anchor means different things). */
export function vwap(candles: Candle[]): (number | null)[] {
  let pv = 0;
  let vol = 0;
  return candles.map((c) => {
    const v = c.volume || 0;
    pv += ((c.high + c.low + c.close) / 3) * v;
    vol += v;
    return vol > 0 ? pv / vol : null;
  });
}

export function rsi(closes: number[], period = 14): (number | null)[] {
  const out: (number | null)[] = new Array(closes.length).fill(null);
  if (closes.length < period + 1) return out;
  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) gain += d;
    else loss -= d;
  }
  let avgGain = gain / period;
  let avgLoss = loss / period;
  out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = period + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    avgGain = (avgGain * (period - 1) + Math.max(d, 0)) / period;
    avgLoss = (avgLoss * (period - 1) + Math.max(-d, 0)) / period;
    out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

export function macd(closes: number[], fast = 12, slow = 26, signal = 9) {
  const f = ema(closes, fast);
  const s = ema(closes, slow);
  const line = closes.map((_, i) => f[i] - s[i]);
  const sig = ema(line, signal);
  return { line, signal: sig };
}

export type Regime = "calm" | "normal" | "volatile";

/** Recent (24-candle) return volatility vs. this asset's own longer average. */
export function volatilityRegime(closes: number[]): Regime | null {
  if (closes.length < 30) return null;
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) if (closes[i - 1]) rets.push((closes[i] - closes[i - 1]) / closes[i - 1]);
  const std = (xs: number[]) => {
    const m = xs.reduce((a, b) => a + b, 0) / xs.length;
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  };
  const base = std(rets);
  if (!base) return null;
  const ratio = std(rets.slice(-24)) / base;
  if (ratio >= 1.3) return "volatile";
  if (ratio <= 0.7) return "calm";
  return "normal";
}

export type SignalKind = "buy" | "sell" | "wait";

export interface SignalSummary {
  kind: SignalKind;
  score: number;
  trendPct: number;
  rsi: number | null;
  macdBullish: boolean | null;
  regime: Regime | null;
  low: PredictionPoint;
  high: PredictionPoint;
  spreadPct: number | null;
}

export function computeSignal(
  candles: Candle[],
  predictions: PredictionPoint[],
  basePrice: number,
  confidence: number,
  sentimentAvg: number,
): SignalSummary | null {
  if (!predictions.length) return null;
  const closes = candles.map((c) => c.close);
  const final = predictions[predictions.length - 1];
  const trendPct = ((final.predicted_price - basePrice) / basePrice) * 100;
  const low = predictions.reduce((a, b) => (b.predicted_price < a.predicted_price ? b : a));
  const high = predictions.reduce((a, b) => (b.predicted_price > a.predicted_price ? b : a));
  const spreadPct = low.predicted_price ? ((high.predicted_price - low.predicted_price) / low.predicted_price) * 100 : null;

  const r = closes.length >= 15 ? rsi(closes, 14).at(-1) ?? null : null;
  let macdBullish: boolean | null = null;
  if (closes.length >= 26 + 9) {
    const m = macd(closes);
    macdBullish = m.line.at(-1)! > m.signal.at(-1)!;
  }

  let score = 0;
  if (trendPct > 1) score += 1;
  else if (trendPct < -1) score -= 1;
  if (r !== null) {
    if (r < 30) score += 1;
    else if (r > 70) score -= 1;
  }
  if (macdBullish !== null) score += macdBullish ? 1 : -1;
  if (sentimentAvg > 0.1) score += 0.5;
  else if (sentimentAvg < -0.1) score -= 0.5;

  let kind: SignalKind = "wait";
  if (confidence >= 40 && score >= 1.5) kind = "buy";
  else if (confidence >= 40 && score <= -1.5) kind = "sell";

  return { kind, score, trendPct, rsi: r, macdBullish, regime: volatilityRegime(closes), low, high, spreadPct };
}

export const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];

const SWING_PIVOT_WINDOW = 5;   // candles on each side that must be lower/higher to confirm a pivot
const SWING_MIN_MOVE_PCT = 0.015; // a high/low pair under this % apart is noise, not a real swing

export interface Swing {
  high: number;
  low: number;
  /** Candle index of the high/low pivot, so a caller can look up its real timestamp. */
  highIndex: number;
  lowIndex: number;
}

export interface FibEntrySignal {
  side: "long" | "short";
  entry: number;
  stopLoss: number;
  takeProfit: number;
  swingLowIndex: number;
  swingHighIndex: number;
}

/** Fibonacci-retracement entry signal, trend-filtered: at each candle, take the high/low extremes
 * over a trailing window ending `gap` candles back (so the window never includes the candle being
 * evaluated or its very recent neighbors - no lookahead), and only when that swing is big enough
 * (`minSwingPct` of price) and the current close sits within `zonePct` of the 61.8% retracement
 * entry level, in the direction the trend filter agrees with, emit a signal: entry at the 61.8%
 * level, a stop beyond the swing's own extreme (where the setup is invalidated, not deeper into
 * the retracement), and a target at the opposite extreme. `trend` is any TrendConfirmation series
 * (e.g. from confirmedTrend over a few EMA slopes) - reused rather than rebuilt here. null
 * everywhere a signal doesn't fire. */
export function fibEntrySignals(
  candles: { high: number; low: number; close: number }[],
  trend: TrendConfirmation[],
  { backcandles = 40, gap = 5, zonePct = 0.1, minSwingPct = 1 } = {},
): (FibEntrySignal | null)[] {
  const n = candles.length;
  const out: (FibEntrySignal | null)[] = new Array(n).fill(null);
  for (let i = backcandles; i < n; i++) {
    const t = trend[i];
    if (t == null) continue;
    const start = i - backcandles;
    const end = i - gap;
    if (end <= start) continue;

    let maxPrice = -Infinity, minPrice = Infinity, maxIdx = -1, minIdx = -1;
    for (let j = start; j < end; j++) {
      if (candles[j].high > maxPrice) { maxPrice = candles[j].high; maxIdx = j; }
      if (candles[j].low < minPrice) { minPrice = candles[j].low; minIdx = j; }
    }
    if (minPrice <= 0) continue;
    const diff = maxPrice - minPrice;
    if ((diff / minPrice) * 100 < minSwingPct) continue;

    const close = candles[i].close;
    if (t === "down" && minIdx < maxIdx) {
      const entry = maxPrice - 0.618 * diff;
      if (entry > 0 && (Math.abs(close - entry) / entry) * 100 < zonePct) {
        out[i] = { side: "short", entry, stopLoss: maxPrice, takeProfit: minPrice, swingLowIndex: minIdx, swingHighIndex: maxIdx };
      }
    } else if (t === "up" && maxIdx < minIdx) {
      const entry = minPrice + 0.618 * diff;
      if (entry > 0 && (Math.abs(close - entry) / entry) * 100 < zonePct) {
        out[i] = { side: "long", entry, stopLoss: minPrice, takeProfit: maxPrice, swingLowIndex: minIdx, swingHighIndex: maxIdx };
      }
    }
  }
  return out;
}

/** The most recent real swing (a confirmed local high next to a confirmed local low, whichever
 * order they came in) big enough to count as an actual move - not just "whatever's in the
 * current zoom/scroll window". Used to anchor Fibonacci retracement levels to a fixed, real
 * price range instead of one that jumps around as the chart is panned. Returns null if there
 * isn't enough history or no pair clears SWING_MIN_MOVE_PCT. */
export function findLastSwing(candles: { high: number; low: number }[]): Swing | null {
  const n = candles.length;
  if (n < SWING_PIVOT_WINDOW * 2 + 1) return null;

  type Pivot = { i: number; price: number; kind: "high" | "low" };
  const pivots: Pivot[] = [];
  for (let i = SWING_PIVOT_WINDOW; i < n - SWING_PIVOT_WINDOW; i++) {
    const window = candles.slice(i - SWING_PIVOT_WINDOW, i + SWING_PIVOT_WINDOW + 1);
    const h = candles[i].high, l = candles[i].low;
    if (h >= Math.max(...window.map((c) => c.high))) pivots.push({ i, price: h, kind: "high" });
    else if (l <= Math.min(...window.map((c) => c.low))) pivots.push({ i, price: l, kind: "low" });
  }

  for (let j = pivots.length - 1; j > 0; j--) {
    const a = pivots[j], b = pivots[j - 1];
    if (a.kind === b.kind) continue;
    if (a.price <= 0 || b.price <= 0) continue;
    const high = a.kind === "high" ? a : b;
    const low = a.kind === "low" ? a : b;
    if ((high.price - low.price) / low.price >= SWING_MIN_MOVE_PCT) {
      return { high: high.price, low: low.price, highIndex: high.i, lowIndex: low.i };
    }
  }
  return null;
}

export type FutureLabel = "up" | "down" | null;

/** Compares the average close of the next N candles to the current candle's close - "up" if
 * that average clears a threshold above, "down" if below, null for "within range" or for the
 * last N candles (no future to look at yet - never backfilled, so this stays a hindsight
 * diagnostic, never a forward-looking prediction). A tool for eyeballing whether a candidate
 * threshold marks a reasonable density of candles, not a trading signal itself.
 *
 * The threshold isn't a fixed percentage (that only works tuned to one fixed series) - it's `k`
 * standard deviations of this series' own single-period returns, scaled by sqrt(N) for the
 * N-period average's lower variance. That keeps the marked density roughly stable (~10%)
 * whether the loaded candles are 15m or 1w, instead of flooding the chart on slow intervals and
 * going silent on fast ones. */
export function futurePriceLabels(closes: number[], N = 10, k = 1): FutureLabel[] {
  const n = closes.length;
  const out: FutureLabel[] = new Array(n).fill(null);
  const rets: number[] = [];
  for (let i = 1; i < n; i++) if (closes[i - 1]) rets.push((closes[i] - closes[i - 1]) / closes[i - 1]);
  if (!rets.length) return out;
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const std = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length);
  if (!std) return out;
  const thresholdPct = k * std * Math.sqrt(N) * 100;

  for (let i = 0; i + N < n; i++) {
    let sum = 0;
    for (let j = i + 1; j <= i + N; j++) sum += closes[j];
    const avg = sum / N;
    const pctDiff = ((avg - closes[i]) / closes[i]) * 100;
    if (pctDiff > thresholdPct) out[i] = "up";
    else if (pctDiff < -thresholdPct) out[i] = "down";
  }
  return out;
}
