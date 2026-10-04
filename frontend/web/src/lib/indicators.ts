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
