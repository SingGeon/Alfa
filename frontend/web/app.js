"use strict";

const API_BASE = "";

// Matches what the backend actually stores per interval (see jobs.py /
// market_data.py), so panning back isn't artificially capped at a few days.
const DEFAULT_HISTORY_LIMIT = 1000;

const RANGE_PRESETS = {
  "1S": { interval: "1h", limit: 168 },   // 1 săptămână
  "1L": { interval: "4h", limit: 180 },   // 1 lună
  "3L": { interval: "4h", limit: 540 },   // 3 luni
  "6L": { interval: "1d", limit: 182 },   // 6 luni
  "1A": { interval: "1d", limit: 365 },   // 1 an
  TOT: { interval: "1w", limit: 1000 },   // tot istoricul real disponibil (Binance ETHUSDT din 2017)
};

const state = {
  interval: "1h",
  steps: 24,
  useSentiment: true,
  historyLimit: DEFAULT_HISTORY_LIMIT,
  chartType: "candles",
  indicators: { sma: false, ema: false, bollinger: false, fibonacci: false },
};

const els = {
  price: document.getElementById("price"),
  change: document.getElementById("change"),
  source: document.getElementById("source"),
  pinnedTicker: document.getElementById("pinnedTicker"),
  updated: document.getElementById("updated"),
  refresh: document.getElementById("refresh"),
  errorBanner: document.getElementById("errorBanner"),
  intervalGroup: document.getElementById("intervalGroup"),
  rangeGroup: document.getElementById("rangeGroup"),
  chartTypeGroup: document.getElementById("chartTypeGroup"),
  indicatorsGroup: document.getElementById("indicatorsGroup"),
  steps: document.getElementById("steps"),
  stepsValue: document.getElementById("stepsValue"),
  useSentiment: document.getElementById("useSentiment"),
  chart: document.getElementById("chart"),
  chartWrap: document.querySelector(".chart-wrap"),
  chartEmpty: document.getElementById("chartEmpty"),
  liveJumpBtn: document.getElementById("liveJumpBtn"),
  chartTooltip: document.getElementById("chartTooltip"),
  confidence: document.getElementById("confidence"),
  sentimentAvg: document.getElementById("sentimentAvg"),
  backend: document.getElementById("backend"),
  outlookDailyPct: document.getElementById("outlookDailyPct"),
  outlookDailyTarget: document.getElementById("outlookDailyTarget"),
  outlookWeeklyPct: document.getElementById("outlookWeeklyPct"),
  outlookWeeklyTarget: document.getElementById("outlookWeeklyTarget"),
  trendPct: document.getElementById("trendPct"),
  minPredicted: document.getElementById("minPredicted"),
  maxPredicted: document.getElementById("maxPredicted"),
  predictionRows: document.getElementById("predictionRows"),
  narrative: document.getElementById("narrative"),
  newsList: document.getElementById("newsList"),
  gasBadge: document.getElementById("gasBadge"),
  signalBadge: document.getElementById("signalBadge"),
  bestBuy: document.getElementById("bestBuy"),
  bestSell: document.getElementById("bestSell"),
  rsiValue: document.getElementById("rsiValue"),
  macdValue: document.getElementById("macdValue"),
};

const COLORS = {
  up: "#199e70",
  down: "#e66767",
  pred: "#3987e5",
  sma: "#9085e9",
  ema: "#d55181",
  bollinger: "#c3c2b7",
  fibonacci: "#898781",
  grid: "#2c2c2a",
  text: "#898781",
  surface: "#1a1a19",
};

const SMA_PERIOD = 20;
const EMA_PERIOD = 20;
const BOLLINGER_PERIOD = 20;
const BOLLINGER_MULT = 2;
const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
const RSI_PERIOD = 14;
const MACD_FAST = 12;
const MACD_SLOW = 26;
const MACD_SIGNAL = 9;

let chart, candleSeries, volumeSeries, predSeries, upperSeries, lowerSeries;
let closePriceSeries, smaSeries, emaSeries, bbUpperSeries, bbMiddleSeries, bbLowerSeries;
let fibPriceLines = [];
let lastSortedCandles = [];
let lastTotalBars = null;
let forceFullView = false;
let liveCandleAnchor = null;

function initChart() {
  if (!window.LightweightCharts) {
    els.chartEmpty.hidden = false;
    els.chartEmpty.textContent = "Nu s-a putut încărca librăria de grafice (verifică conexiunea la internet).";
    return;
  }

  // Lightweight Charts formats UTCTimestamp values in UTC by default, not
  // the viewer's local timezone - on this dashboard that showed candles/
  // predictions looking hours "behind" wall-clock time. These formatters
  // render everything in the browser's own local time instead.
  const tickMarkFormatter = (time, tickMarkType) => {
    const d = new Date(time * 1000);
    const TMT = LightweightCharts.TickMarkType;
    if (tickMarkType === TMT.Year) return d.toLocaleDateString("ro-RO", { year: "numeric" });
    if (tickMarkType === TMT.Month) return d.toLocaleDateString("ro-RO", { month: "short", year: "numeric" });
    if (tickMarkType === TMT.DayOfMonth) return d.toLocaleDateString("ro-RO", { day: "2-digit", month: "short" });
    return d.toLocaleTimeString("ro-RO", { hour: "2-digit", minute: "2-digit" });
  };

  chart = LightweightCharts.createChart(els.chart, {
    layout: {
      background: { color: "transparent" },
      textColor: COLORS.text,
      fontFamily: "system-ui, -apple-system, 'Segoe UI', sans-serif",
    },
    grid: {
      vertLines: { color: COLORS.grid },
      horzLines: { color: COLORS.grid },
    },
    localization: {
      timeFormatter: (time) =>
        new Date(time * 1000).toLocaleString("ro-RO", {
          day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
        }),
    },
    rightPriceScale: { borderColor: COLORS.grid },
    timeScale: {
      borderColor: COLORS.grid,
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 8,
      tickMarkFormatter,
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    handleScroll: {
      mouseWheel: true,
      pressedMouseMove: true,
      horzTouchDrag: true,
      vertTouchDrag: true,
    },
    handleScale: {
      mouseWheel: true,
      pinch: true,
      axisPressedMouseMove: true,
    },
    kineticScroll: { touch: true, mouse: true },
  });

  // createChart() sizes to the container at that exact instant, which can
  // be before the surrounding layout has settled - forcing the current
  // size right away avoids computing the first zoom against a stale/too-
  // narrow width (see the same fix in detail.js for the bug this caused
  // there: a broken-looking flash of the wrong range on first paint).
  chart.applyOptions({ width: els.chartWrap.clientWidth, height: els.chartWrap.clientHeight });

  candleSeries = chart.addCandlestickSeries({
    upColor: COLORS.up,
    downColor: COLORS.down,
    borderUpColor: COLORS.up,
    borderDownColor: COLORS.down,
    wickUpColor: COLORS.up,
    wickDownColor: COLORS.down,
  });

  closePriceSeries = chart.addLineSeries({ color: COLORS.pred, lineWidth: 2, visible: false });

  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: "volume" },
    priceScaleId: "volume",
  });
  chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });

  smaSeries = chart.addLineSeries({ color: COLORS.sma, lineWidth: 2, visible: false, crosshairMarkerVisible: false });
  emaSeries = chart.addLineSeries({ color: COLORS.ema, lineWidth: 2, visible: false, crosshairMarkerVisible: false });
  bbUpperSeries = chart.addLineSeries({
    color: COLORS.bollinger, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    visible: false, crosshairMarkerVisible: false,
  });
  bbMiddleSeries = chart.addLineSeries({
    color: COLORS.bollinger, lineWidth: 1, visible: false, crosshairMarkerVisible: false,
  });
  bbLowerSeries = chart.addLineSeries({
    color: COLORS.bollinger, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    visible: false, crosshairMarkerVisible: false,
  });

  predSeries = chart.addLineSeries({ color: COLORS.pred, lineWidth: 2 });
  // lastValueVisible/priceLineVisible off: with candle + pred + upper +
  // lower all tagging the axis by default, the labels pile up and overlap -
  // the band's shape is already visible from the dashed lines themselves.
  upperSeries = chart.addLineSeries({
    color: COLORS.pred, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });
  lowerSeries = chart.addLineSeries({
    color: COLORS.pred, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });

  new ResizeObserver(() => {
    if (chart) chart.applyOptions({ width: els.chartWrap.clientWidth, height: els.chartWrap.clientHeight });
  }).observe(els.chartWrap);

  chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
    if (!range || lastTotalBars === null) return;
    els.liveJumpBtn.hidden = lastTotalBars - range.to <= 5;
    if (state.indicators.fibonacci) updateFibonacci();
  });

  els.liveJumpBtn.addEventListener("click", () => {
    if (lastTotalBars === null) return;
    const defaultWindow = Math.min(120, lastTotalBars);
    chart.timeScale().setVisibleLogicalRange({
      from: lastTotalBars - defaultWindow,
      to: lastTotalBars + 2,
    });
  });

  // Floating price readout that tracks the mouse - the axis-edge tag and
  // the fixed anchor marker both sit at one spot, so neither one answers
  // "what's the price under my cursor right now" as you scan the chart.
  chart.subscribeCrosshairMove((param) => {
    if (!param.point) {
      els.chartTooltip.hidden = true;
      return;
    }
    // Read the price straight off the price scale at the cursor's Y pixel,
    // not off a series data point: reading a series' value (e.g. the
    // hovered candle's close) but positioning the label at the mouse's raw
    // Y coordinate showed a number that didn't match where it was drawn
    // relative to the axis - coordinateToPrice() guarantees the two agree,
    // and it also works past "now" (the forecast region), where candle/
    // closePrice series have no data point to look up at all.
    const price = candleSeries.priceScale().coordinateToPrice(param.point.y);
    if (price === null) {
      els.chartTooltip.hidden = true;
      return;
    }
    els.chartTooltip.hidden = false;
    els.chartTooltip.textContent = fmtMoney(price);

    const wrapWidth = els.chartWrap.clientWidth;
    const tooltipWidth = els.chartTooltip.offsetWidth || 90;
    let left = param.point.x + 14;
    if (left + tooltipWidth > wrapWidth) left = param.point.x - tooltipWidth - 14;
    els.chartTooltip.style.left = `${Math.max(4, left)}px`;
    els.chartTooltip.style.top = `${Math.max(4, param.point.y - 14)}px`;
  });
}

function toUnixSeconds(iso) {
  return Math.floor(new Date(iso).getTime() / 1000);
}

// -- Technical indicators ----------------------------------------------

function computeSMA(values, period) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= period) sum -= values[i - period];
    out.push(i >= period - 1 ? sum / period : null);
  }
  return out;
}

function computeEMA(values, period) {
  const k = 2 / (period + 1);
  const out = [];
  let prev = null;
  for (let i = 0; i < values.length; i++) {
    prev = prev === null ? values[i] : values[i] * k + prev * (1 - k);
    out.push(prev);
  }
  return out;
}

function computeBollinger(values, period, mult) {
  const middle = computeSMA(values, period);
  const upper = [];
  const lower = [];
  for (let i = 0; i < values.length; i++) {
    if (middle[i] === null) {
      upper.push(null);
      lower.push(null);
      continue;
    }
    let sumSq = 0;
    for (let j = i - period + 1; j <= i; j++) sumSq += (values[j] - middle[i]) ** 2;
    const std = Math.sqrt(sumSq / period);
    upper.push(middle[i] + mult * std);
    lower.push(middle[i] - mult * std);
  }
  return { upper, middle, lower };
}

function computeRSI(closes, period) {
  const out = new Array(closes.length).fill(null);
  if (closes.length < period + 1) return out;
  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const delta = closes[i] - closes[i - 1];
    if (delta > 0) gainSum += delta;
    else lossSum -= delta;
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;
  out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  for (let i = period + 1; i < closes.length; i++) {
    const delta = closes[i] - closes[i - 1];
    const gain = delta > 0 ? delta : 0;
    const loss = delta < 0 ? -delta : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

function computeMACD(closes, fast, slow, signalPeriod) {
  const emaFast = computeEMA(closes, fast);
  const emaSlow = computeEMA(closes, slow);
  const macdLine = closes.map((_, i) => emaFast[i] - emaSlow[i]);
  const signalLine = computeEMA(macdLine, signalPeriod);
  const histogram = macdLine.map((v, i) => v - signalLine[i]);
  return { macdLine, signalLine, histogram };
}

function getActivePriceSeries() {
  return state.chartType === "line" ? closePriceSeries : candleSeries;
}

function clearFibonacciLines() {
  const series = getActivePriceSeries();
  fibPriceLines.forEach((line) => series.removePriceLine(line));
  fibPriceLines = [];
}

function updateIndicatorSeries(sorted) {
  lastSortedCandles = sorted;
  const times = sorted.map((c) => toUnixSeconds(c.timestamp));
  const closes = sorted.map((c) => c.close);

  const toSeriesData = (values) =>
    values.map((v, i) => (v === null ? null : { time: times[i], value: v })).filter(Boolean);

  smaSeries.applyOptions({ visible: state.indicators.sma });
  smaSeries.setData(state.indicators.sma ? toSeriesData(computeSMA(closes, SMA_PERIOD)) : []);

  emaSeries.applyOptions({ visible: state.indicators.ema });
  emaSeries.setData(state.indicators.ema ? toSeriesData(computeEMA(closes, EMA_PERIOD)) : []);

  const showBB = state.indicators.bollinger;
  bbUpperSeries.applyOptions({ visible: showBB });
  bbMiddleSeries.applyOptions({ visible: showBB });
  bbLowerSeries.applyOptions({ visible: showBB });
  if (showBB) {
    const bb = computeBollinger(closes, BOLLINGER_PERIOD, BOLLINGER_MULT);
    bbUpperSeries.setData(toSeriesData(bb.upper));
    bbMiddleSeries.setData(toSeriesData(bb.middle));
    bbLowerSeries.setData(toSeriesData(bb.lower));
  } else {
    bbUpperSeries.setData([]);
    bbMiddleSeries.setData([]);
    bbLowerSeries.setData([]);
  }

  updateFibonacci();
}

// Fibonacci retracement is only meaningful for the swing you're actually
// looking at, not the whole fetched history - so it's computed from
// whichever candles are currently in view, and recomputed live as you pan
// or zoom (see the subscribeVisibleLogicalRangeChange call in initChart).
function updateFibonacci() {
  clearFibonacciLines();
  if (!state.indicators.fibonacci || !lastSortedCandles.length || !chart) return;

  const range = chart.timeScale().getVisibleLogicalRange();
  const n = lastSortedCandles.length;
  const from = Math.max(0, Math.floor(range ? range.from : 0));
  const to = Math.min(n - 1, Math.ceil(range ? range.to : n - 1));
  const visible = from <= to ? lastSortedCandles.slice(from, to + 1) : lastSortedCandles;
  if (!visible.length) return;

  const high = Math.max(...visible.map((c) => c.high));
  const low = Math.min(...visible.map((c) => c.low));
  const series = getActivePriceSeries();
  fibPriceLines = FIB_LEVELS.map((level) =>
    series.createPriceLine({
      price: high - level * (high - low),
      color: COLORS.fibonacci,
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      axisLabelVisible: true,
      title: `Fib ${(level * 100).toFixed(1)}%`,
    })
  );
}

async function apiGet(path, params) {
  const url = new URL(API_BASE + path, window.location.origin);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const resp = await fetch(url);
  if (!resp.ok) {
    let payload;
    try { payload = await resp.json(); } catch { payload = null; }
    const err = new Error((payload && payload.error) || `${resp.status} ${resp.statusText}`);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

function showError(message) {
  els.errorBanner.hidden = false;
  els.errorBanner.textContent = message;
}
function clearError() {
  els.errorBanner.hidden = true;
  els.errorBanner.textContent = "";
}

function fmtMoney(v) {
  if (v === null || v === undefined) return "—";
  return `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

// Pinned Scout AI assets can be sub-cent meme coins - fmtMoney's fixed 2
// decimals would just show "$0.00" for those, so this scales precision
// the same way scout.js/detail.js already do for their own price columns.
function fmtMoneySmart(v) {
  if (v === null || v === undefined) return "—";
  const abs = Math.abs(v);
  let decimals = 2;
  if (abs > 0 && abs < 0.01) decimals = 6;
  else if (abs < 1) decimals = 4;
  return `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

function directionClass(v) {
  return v >= 0 ? "up" : "down";
}
function arrow(v) {
  return v >= 0 ? "▲" : "▼";
}

// 3-tier quality read for confidence (0-100): red under 40 (weak/
// uncertain), yellow 40-70 (decent), green over 70 (precise/strong).
function qualityClass(confidence) {
  if (confidence < 40) return "quality-poor";
  if (confidence < 70) return "quality-decent";
  return "quality-good";
}

function fmtTime(iso) {
  const d = new Date(iso);
  return d.toLocaleString("ro-RO", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}
function fmtTimeShort(iso) {
  const d = new Date(iso);
  return d.toLocaleString("ro-RO", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fmtRelative(iso) {
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "chiar acum";
  if (mins < 60) return `acum ${mins} min`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `acum ${hours} h`;
  return `acum ${Math.round(hours / 24)} zile`;
}

function sentimentBadge(label) {
  const icons = { positive: "▲", negative: "▼", neutral: "●" };
  const safe = ["positive", "negative", "neutral"].includes(label) ? label : "neutral";
  return `<span class="sentiment-badge ${safe}">${icons[safe]} ${safe}</span>`;
}

function renderCurrentPrice(current) {
  els.price.textContent = fmtMoney(current.price);
  const change = current.change_24h_pct;
  if (change !== null && change !== undefined) {
    const up = change >= 0;
    els.change.textContent = `${up ? "▲" : "▼"} ${Math.abs(change).toFixed(2)}% (24h)`;
    els.change.className = `ticker-change ${up ? "up" : "down"}`;
  } else {
    els.change.textContent = "";
    els.change.className = "ticker-change";
  }
  els.source.textContent = current.source || "";
}

async function loadCurrentPrice() {
  try {
    renderCurrentPrice(await apiGet("/api/price/current"));
  } catch (exc) {
    els.price.textContent = "—";
    els.change.textContent = "";
    showError(`Nu pot obține prețul curent: ${exc.message}`);
  }
}

// Runs on LIVE_TICK_MS (matching the server's price cache TTL) to make the
// ticker and the currently-forming candle move live, instead of only
// updating once per 30s full refresh.
let liveTickInFlight = false;
async function tickLivePrice() {
  if (liveTickInFlight) return;
  liveTickInFlight = true;
  try {
    const current = await apiGet("/api/price/current");
    renderCurrentPrice(current);
    els.updated.textContent = `actualizat ${new Date().toLocaleTimeString("ro-RO")}`;
    if (chart && liveCandleAnchor && current.price != null) {
      liveCandleAnchor.high = Math.max(liveCandleAnchor.high, current.price);
      liveCandleAnchor.low = Math.min(liveCandleAnchor.low, current.price);
      candleSeries.update({
        time: liveCandleAnchor.time,
        open: liveCandleAnchor.open,
        high: liveCandleAnchor.high,
        low: liveCandleAnchor.low,
        close: current.price,
      });
      closePriceSeries.update({ time: liveCandleAnchor.time, value: current.price });
    }
  } catch {
    // silent - this is a best-effort per-second nudge; the 30s full
    // refresh (which does surface errors) remains the source of truth
  } finally {
    liveTickInFlight = false;
  }
}

async function loadChart() {
  // Fire both requests together - predict involves server-side model
  // training and can take a few seconds, and there's no reason the candle
  // fetch (fast) should sit blocked behind it.
  const [historyResult, predictionResult] = await Promise.allSettled([
    apiGet("/api/price/history", { interval: state.interval, limit: state.historyLimit }),
    apiGet("/api/predict", { interval: state.interval, steps: state.steps, use_sentiment: state.useSentiment }),
  ]);

  let candles = [];
  if (historyResult.status === "fulfilled") {
    candles = historyResult.value.candles || [];
  } else {
    showError(`Nu pot obține istoricul de prețuri: ${historyResult.reason.message}`);
  }

  let prediction = null;
  if (predictionResult.status === "fulfilled") {
    prediction = predictionResult.value;
    if (historyResult.status === "fulfilled") clearError();
  } else {
    const exc = predictionResult.reason;
    if (exc.status === 409) {
      showError(exc.message);
    } else {
      showError(`Predicția a eșuat: ${exc.message}`);
    }
  }

  if (chart) {
    if (candles.length) {
      els.chartEmpty.hidden = true;

      // Was the user actually looking at the live edge before this refresh?
      // If so, keep following live data afterwards (never get "stuck" in the
      // past just because a refresh happened) but preserve their zoom level
      // - don't snap back to some fixed bar count every 30s, which felt like
      // the chart "reverting" on its own. Only if they'd deliberately
      // scrolled back into history do we pin their exact time window - by
      // real timestamp, not bar offset, so it doesn't drift as new bars
      // keep appending on the right.
      let savedTimeRange = null;
      let savedWindowWidth = null;
      if (lastTotalBars !== null) {
        const existingLogical = chart.timeScale().getVisibleLogicalRange();
        if (existingLogical) {
          savedWindowWidth = existingLogical.to - existingLogical.from;
          const offsetFromRight = lastTotalBars - existingLogical.to;
          if (offsetFromRight > 5) {
            savedTimeRange = chart.timeScale().getVisibleRange();
          }
        }
      }

      const sorted = [...candles].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      candleSeries.setData(
        sorted.map((c) => ({ time: toUnixSeconds(c.timestamp), open: c.open, high: c.high, low: c.low, close: c.close }))
      );
      volumeSeries.setData(
        sorted.map((c) => ({
          time: toUnixSeconds(c.timestamp),
          value: c.volume || 0,
          color: c.close >= c.open ? "rgba(25,158,112,0.5)" : "rgba(230,103,103,0.5)",
        }))
      );
      closePriceSeries.setData(
        sorted.map((c) => ({ time: toUnixSeconds(c.timestamp), value: c.close }))
      );
      updateIndicatorSeries(sorted);

      // Anchor for the 1s live-price ticker below: it nudges just this
      // last (currently-forming) bar between full 30s refreshes, so the
      // candle visibly breathes with the live price instead of sitting
      // static for half a minute.
      const anchorCandle = sorted[sorted.length - 1];
      liveCandleAnchor = {
        time: toUnixSeconds(anchorCandle.timestamp),
        open: anchorCandle.open,
        high: anchorCandle.high,
        low: anchorCandle.low,
      };

      if (prediction && prediction.predictions && prediction.predictions.length) {
        const last = sorted[sorted.length - 1];
        const anchor = { time: toUnixSeconds(last.timestamp), value: last.close };
        const predPoints = [anchor, ...prediction.predictions.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.predicted_price }))];
        const upperPoints = [anchor, ...prediction.predictions.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.upper }))];
        const lowerPoints = [anchor, ...prediction.predictions.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.lower }))];
        predSeries.setData(predPoints);
        upperSeries.setData(upperPoints);
        lowerSeries.setData(lowerPoints);
      } else {
        predSeries.setData([]);
        upperSeries.setData([]);
        lowerSeries.setData([]);
      }
      const predCount = prediction && prediction.predictions ? prediction.predictions.length : 0;
      const newTotalBars = sorted.length + predCount;

      if (forceFullView) {
        // A range preset (1S/1L/.../Tot) was just picked - show the whole
        // fetched window, not just the last 120 bars.
        chart.timeScale().setVisibleLogicalRange({ from: 0, to: newTotalBars + 2 });
        forceFullView = false;
      } else if (savedTimeRange) {
        // User had deliberately scrolled into history - keep them pinned to
        // that exact time window regardless of how many new bars arrived.
        chart.timeScale().setVisibleRange(savedTimeRange);
      } else {
        // At the live edge (or first render): always show bars ending at
        // "now", so a refresh never leaves you stranded in the past - but
        // keep whatever zoom level (window width) the user already had
        // instead of forcing a fixed bar count every time. `to` is anchored
        // in candles+predictions index space, so `from` must be too (not
        // `sorted.length`, which is candles-only) or the window balloons by
        // the prediction step count on every refresh.
        const to = newTotalBars + 2;
        const defaultWindow = Math.min(savedWindowWidth ?? 120, to);
        chart.timeScale().setVisibleLogicalRange({ from: to - defaultWindow, to });
      }
      lastTotalBars = newTotalBars;
    } else {
      els.chartEmpty.hidden = false;
      candleSeries.setData([]);
      volumeSeries.setData([]);
      closePriceSeries.setData([]);
      predSeries.setData([]);
      upperSeries.setData([]);
      lowerSeries.setData([]);
      updateIndicatorSeries([]);
      liveCandleAnchor = null;
    }
  }

  if (prediction) {
    els.confidence.textContent = `${Math.round(prediction.confidence)}/100`;
    els.confidence.className = `stat-value ${qualityClass(prediction.confidence)}`;
    els.sentimentAvg.textContent = prediction.sentiment_avg.toFixed(2);
    els.backend.textContent = prediction.backend;

    if (prediction.predictions.length) {
      const basePrice = prediction.last_known_price;
      const preds = prediction.predictions;

      els.predictionRows.innerHTML = preds
        .map((p) => {
          const cls = directionClass(p.predicted_price - basePrice);
          const pct = ((p.predicted_price - basePrice) / basePrice) * 100;
          return `<tr>
            <td>${fmtTime(p.timestamp)}</td>
            <td class="price price-${cls}">${fmtMoney(p.predicted_price)}</td>
            <td class="price-${cls}">${arrow(pct)} ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%</td>
          </tr>`;
        })
        .join("");

      const final = preds[preds.length - 1];
      const trendPct = ((final.predicted_price - basePrice) / basePrice) * 100;
      els.trendPct.textContent = `${arrow(trendPct)} ${trendPct >= 0 ? "+" : ""}${trendPct.toFixed(2)}%`;
      els.trendPct.className = `summary-value ${directionClass(trendPct)}`;

      const minPoint = preds.reduce((a, b) => (b.predicted_price < a.predicted_price ? b : a));
      const maxPoint = preds.reduce((a, b) => (b.predicted_price > a.predicted_price ? b : a));
      els.minPredicted.textContent = `${fmtMoney(minPoint.predicted_price)} · ${fmtTimeShort(minPoint.timestamp)}`;
      els.maxPredicted.textContent = `${fmtMoney(maxPoint.predicted_price)} · ${fmtTimeShort(maxPoint.timestamp)}`;

      updateSignalPanel({
        minPoint, maxPoint, trendPct,
        confidence: prediction.confidence,
        sentimentAvg: prediction.sentiment_avg,
      });
    } else {
      els.predictionRows.innerHTML = `<tr class="empty-row"><td colspan="3">—</td></tr>`;
      els.trendPct.textContent = "—";
      els.trendPct.className = "summary-value";
      els.minPredicted.textContent = "—";
      els.maxPredicted.textContent = "—";
      resetSignalPanel();
    }
  } else {
    els.confidence.textContent = "—";
    els.confidence.className = "stat-value";
    els.sentimentAvg.textContent = "—";
    els.backend.textContent = "—";
    els.predictionRows.innerHTML = `<tr class="empty-row"><td colspan="3">—</td></tr>`;
    els.trendPct.textContent = "—";
    els.trendPct.className = "summary-value";
    els.minPredicted.textContent = "—";
    els.maxPredicted.textContent = "—";
    resetSignalPanel();
  }
}

function resetSignalPanel() {
  els.signalBadge.textContent = "—";
  els.signalBadge.className = "signal-badge";
  els.bestBuy.textContent = "—";
  els.bestSell.textContent = "—";
  els.rsiValue.textContent = "—";
  els.macdValue.textContent = "—";
}

function updateSignalPanel({ minPoint, maxPoint, trendPct, confidence, sentimentAvg }) {
  els.bestBuy.textContent = `${fmtMoney(minPoint.predicted_price)} · ${fmtTimeShort(minPoint.timestamp)}`;
  els.bestSell.textContent = `${fmtMoney(maxPoint.predicted_price)} · ${fmtTimeShort(maxPoint.timestamp)}`;

  const closes = lastSortedCandles.map((c) => c.close);
  let rsi = null;
  let macdBullish = null;
  if (closes.length >= RSI_PERIOD + 1) {
    const rsiSeries = computeRSI(closes, RSI_PERIOD);
    rsi = rsiSeries[rsiSeries.length - 1];
  }
  if (closes.length >= MACD_SLOW + MACD_SIGNAL) {
    const macd = computeMACD(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL);
    const last = macd.histogram.length - 1;
    macdBullish = macd.macdLine[last] > macd.signalLine[last];
  }

  if (rsi !== null) {
    const zone = rsi < 30 ? " (supravândut)" : rsi > 70 ? " (supracumpărat)" : "";
    els.rsiValue.textContent = `${rsi.toFixed(1)}${zone}`;
  } else {
    els.rsiValue.textContent = "—";
  }
  els.macdValue.textContent = macdBullish === null ? "—" : macdBullish ? "▲ bullish" : "▼ bearish";

  // Simple, transparent weighted heuristic - not a guarantee, just a
  // summary of the same signals already shown elsewhere on the page.
  let score = 0;
  if (trendPct > 1) score += 1;
  else if (trendPct < -1) score -= 1;
  if (rsi !== null) {
    if (rsi < 30) score += 1;
    else if (rsi > 70) score -= 1;
  }
  if (macdBullish !== null) score += macdBullish ? 1 : -1;
  if (sentimentAvg > 0.1) score += 0.5;
  else if (sentimentAvg < -0.1) score -= 0.5;

  let label = "AȘTEAPTĂ";
  let cls = "wait";
  if (confidence >= 40 && score >= 1.5) {
    label = "CUMPĂRĂ";
    cls = "buy";
  } else if (confidence >= 40 && score <= -1.5) {
    label = "VINDE";
    cls = "sell";
  }
  els.signalBadge.textContent = label;
  els.signalBadge.className = `signal-badge ${cls}`;
}

async function loadOutlook() {
  try {
    const outlook = await apiGet("/api/outlook", { use_sentiment: state.useSentiment });
    for (const [leg, pctEl, targetEl] of [
      [outlook.daily, els.outlookDailyPct, els.outlookDailyTarget],
      [outlook.weekly, els.outlookWeeklyPct, els.outlookWeeklyTarget],
    ]) {
      const cls = directionClass(leg.change_pct);
      pctEl.textContent = `${arrow(leg.change_pct)} ${leg.change_pct >= 0 ? "+" : ""}${leg.change_pct.toFixed(2)}%`;
      pctEl.className = `outlook-pct ${cls}`;
      targetEl.textContent = `țintă ${fmtMoney(leg.predicted_price)}`;
    }
  } catch {
    els.outlookDailyPct.textContent = "—";
    els.outlookDailyPct.className = "outlook-pct";
    els.outlookWeeklyPct.textContent = "—";
    els.outlookWeeklyPct.className = "outlook-pct";
    els.outlookDailyTarget.textContent = "";
    els.outlookWeeklyTarget.textContent = "";
  }
}

async function loadSummary() {
  try {
    const summary = await apiGet("/api/summary", { interval: state.interval });
    els.narrative.textContent = summary.narrative;
  } catch (exc) {
    if (exc.status !== 409) {
      els.narrative.textContent = `Rezumatul a eșuat: ${exc.message}`;
    }
  }
}

async function loadNews() {
  try {
    const { news } = await apiGet("/api/news", { limit: 30 });
    if (!news.length) {
      els.newsList.innerHTML = `<p class="empty-hint">Niciun articol colectat încă. Rulează <code>python run_scheduler.py</code>.</p>`;
      return;
    }
    els.newsList.innerHTML = news
      .map((article) => {
        const sentiment = article.sentiment || {};
        return `<div class="news-item">
          <a class="news-title" href="${article.url}" target="_blank" rel="noopener noreferrer">${article.title}</a>
          <div class="news-meta">
            ${sentimentBadge(sentiment.label)}
            <span>score ${(sentiment.compound || 0).toFixed(2)}</span>
            <span>·</span>
            <span>${article.source || ""}</span>
            <span>·</span>
            <span>${article.published_at ? `${fmtTimeShort(article.published_at)} · ${fmtRelative(article.published_at)}` : ""}</span>
          </div>
        </div>`;
      })
      .join("");
  } catch (exc) {
    els.newsList.innerHTML = `<p class="empty-hint">Nu pot obține știrile: ${exc.message}</p>`;
  }
}

// Pinned assets (Scout AI, see scout.html) shown right next to the ETH
// price so they're visible everywhere, not just on the Scout AI page.
async function loadPinnedTicker() {
  try {
    const { pins } = await apiGet("/api/scout/pins");
    if (!pins || !pins.length) {
      els.pinnedTicker.innerHTML = "";
      return;
    }
    const quotes = await Promise.all(
      pins.map(async (p) => {
        try {
          const url = new URL("/api/scout/price", window.location.origin);
          url.searchParams.set("asset_type", p.asset_type);
          url.searchParams.set("id", p.id);
          const resp = await fetch(url);
          return { ...p, quote: resp.ok ? await resp.json() : null };
        } catch {
          return { ...p, quote: null };
        }
      })
    );
    els.pinnedTicker.innerHTML = quotes
      .map((p) => {
        const change = p.quote ? p.quote.change_24h_pct : null;
        const cls = change === null || change === undefined ? "" : directionClass(change);
        const href = `/detail.html?type=${encodeURIComponent(p.asset_type)}&id=${encodeURIComponent(p.id)}`;
        return `<a class="pinned-chip ${cls}" href="${href}" title="${p.name}">
          <span class="pinned-chip-symbol">${p.symbol}</span>
          <span class="pinned-chip-price">${fmtMoneySmart(p.quote ? p.quote.price : null)}</span>
        </a>`;
      })
      .join("");
  } catch {
    // best-effort - header ticker just stays as-is if this fails
  }
}

async function loadGas() {
  try {
    const gas = await apiGet("/api/gas");
    if (!gas.error) {
      els.gasBadge.hidden = false;
      els.gasBadge.textContent = `⛽ Gas: safe ${gas.safe_gwei} · propose ${gas.propose_gwei} · fast ${gas.fast_gwei} gwei`;
    }
  } catch {
    els.gasBadge.hidden = true;
  }
}

async function loadAll() {
  clearError();
  els.updated.textContent = "se încarcă…";
  await Promise.all([loadCurrentPrice(), loadChart(), loadOutlook(), loadSummary(), loadNews(), loadGas(), loadPinnedTicker()]);
  els.updated.textContent = `actualizat ${new Date().toLocaleTimeString("ro-RO")}`;
}

// Every trigger (buttons, slider, checkbox, manual/auto refresh) funnels
// through this single coalescing queue so two loadChart() runs can never
// overlap - concurrent calls would race on lastTotalBars/forceFullView and
// corrupt the saved chart view (observed as a bogus negative visible range).
let inFlight = null;
let queued = false;

function requestReload() {
  if (inFlight) {
    queued = true;
    return;
  }
  inFlight = loadAll().finally(() => {
    inFlight = null;
    if (queued) {
      queued = false;
      requestReload();
    }
  });
}

els.intervalGroup.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-interval]");
  if (!btn) return;
  state.interval = btn.dataset.interval;
  state.historyLimit = DEFAULT_HISTORY_LIMIT;
  [...els.intervalGroup.children].forEach((b) => b.classList.toggle("active", b === btn));
  [...els.rangeGroup.children].forEach((b) => b.classList.remove("active"));
  requestReload();
});

els.rangeGroup.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-range]");
  if (!btn) return;
  const preset = RANGE_PRESETS[btn.dataset.range];
  if (!preset) return;
  state.interval = preset.interval;
  state.historyLimit = preset.limit;
  forceFullView = true;
  lastTotalBars = null;
  [...els.rangeGroup.children].forEach((b) => b.classList.toggle("active", b === btn));
  [...els.intervalGroup.children].forEach((b) => b.classList.toggle("active", b.dataset.interval === preset.interval));
  requestReload();
});

els.steps.addEventListener("input", () => {
  els.stepsValue.textContent = els.steps.value;
});
els.steps.addEventListener("change", () => {
  state.steps = Number(els.steps.value);
  requestReload();
});

els.useSentiment.addEventListener("change", () => {
  state.useSentiment = els.useSentiment.checked;
  requestReload();
});

els.refresh.addEventListener("click", requestReload);

els.chartTypeGroup.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-chart-type]");
  if (!btn) return;
  clearFibonacciLines(); // tied to the currently-active series - drop before switching

  state.chartType = btn.dataset.chartType;
  candleSeries.applyOptions({ visible: state.chartType === "candles" });
  closePriceSeries.applyOptions({ visible: state.chartType === "line" });
  [...els.chartTypeGroup.children].forEach((b) => b.classList.toggle("active", b === btn));

  updateIndicatorSeries(lastSortedCandles); // re-adds fibonacci lines on the new active series, if on
});

els.indicatorsGroup.addEventListener("change", (e) => {
  const input = e.target.closest("input[data-indicator]");
  if (!input) return;
  state.indicators[input.dataset.indicator] = input.checked;
  updateIndicatorSeries(lastSortedCandles);
});

const AUTO_REFRESH_MS = 30000;
// Matches the server's price cache TTL (api/routes.py) - polling faster
// than that just re-fetches the same cached value. Now that the price
// source is Binance-first (weight-1, ~1200/min limit) rather than
// CoinGecko-first, this can run much faster than it used to without any
// rate-limit risk.
const LIVE_TICK_MS = 3000;

initChart();
requestReload();
setInterval(requestReload, AUTO_REFRESH_MS);
setInterval(tickLivePrice, LIVE_TICK_MS);
