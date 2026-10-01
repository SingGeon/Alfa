"use strict";

const API_BASE = "";

// Matches what the backend actually stores per interval (see jobs.py /
// market_data.py), so panning back isn't artificially capped at a few days.
const DEFAULT_HISTORY_LIMIT = 1000;

const RANGE_PRESETS = {
  "1S": { interval: "1h", limit: 168 },   // 1 week
  "1L": { interval: "4h", limit: 180 },   // 1 month
  "3L": { interval: "4h", limit: 540 },   // 3 months
  "6L": { interval: "1d", limit: 182 },   // 6 months
  "1A": { interval: "1d", limit: 365 },   // 1 year
  TOT: { interval: "1w", limit: 1000 },   // all real history available (Binance ETHUSDT since 2017)
};

const state = {
  interval: "1h",
  steps: 24,
  useSentiment: true,
  historyLimit: DEFAULT_HISTORY_LIMIT,
  chartType: "candles",
  indicators: { sma: false, ema: false, bollinger: false, fibonacci: false, vwap: false, patterns: false, box: false },
  backend: "sklearn",
  // "tuned" (regularized GBR, better average error) vs "legacy" (sklearn's
  // own defaults, the original model) - see ml/price_predictor.py's
  // module-level comment for why both stay selectable rather than one
  // replacing the other.
  modelVariant: "legacy",
};

const els = {
  price: document.getElementById("price"),
  change: document.getElementById("change"),
  source: document.getElementById("source"),
  pinnedTicker: document.getElementById("pinnedTicker"),
  updated: document.getElementById("updated"),
  errorBanner: document.getElementById("errorBanner"),
  intervalGroup: document.getElementById("intervalGroup"),
  rangeGroup: document.getElementById("rangeGroup"),
  chartTypeGroup: document.getElementById("chartTypeGroup"),
  indicatorsGroup: document.getElementById("indicatorsGroup"),
  steps: document.getElementById("steps"),
  stepsValue: document.getElementById("stepsValue"),
  useSentiment: document.getElementById("useSentiment"),
  modelVariantGroup: document.getElementById("modelVariantGroup"),
  chart: document.getElementById("chart"),
  chartWrap: document.querySelector(".chart-wrap"),
  chartEmpty: document.getElementById("chartEmpty"),
  liveJumpBtn: document.getElementById("liveJumpBtn"),
  chartTooltip: document.getElementById("chartTooltip"),
  confidence: document.getElementById("confidence"),
  sentimentAvg: document.getElementById("sentimentAvg"),
  backend: document.getElementById("backend"),
  sentimentContribution: document.getElementById("sentimentContribution"),
  recalibratedHint: document.getElementById("recalibratedHint"),
  evolutionCard: document.getElementById("evolutionCard"),
  evoMood: document.getElementById("evoMood"),
  evoStats: document.getElementById("evoStats"),
  evoTrack: document.getElementById("evoTrack"),
  evoTrackNote: document.getElementById("evoTrackNote"),
  evoLeaders: document.getElementById("evoLeaders"),
  evoUsage: document.getElementById("evoUsage"),
  evoSurvival: document.getElementById("evoSurvival"),
  evoSentiment: document.getElementById("evoSentiment"),
  evoMoodChart: document.getElementById("evoMoodChart"),
  backtestHint: document.getElementById("backtestHint"),
  outlookDailyPct: document.getElementById("outlookDailyPct"),
  outlookDailyTarget: document.getElementById("outlookDailyTarget"),
  outlookDailyActual: document.getElementById("outlookDailyActual"),
  outlookWeeklyPct: document.getElementById("outlookWeeklyPct"),
  outlookWeeklyTarget: document.getElementById("outlookWeeklyTarget"),
  outlookWeeklyActual: document.getElementById("outlookWeeklyActual"),
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
  potentialGain: document.getElementById("potentialGain"),
  rsiValue: document.getElementById("rsiValue"),
  macdValue: document.getElementById("macdValue"),
  volatilityBadge: document.getElementById("volatilityBadge"),
  signalAccuracyBuy: document.getElementById("signalAccuracyBuy"),
  signalAccuracySell: document.getElementById("signalAccuracySell"),
  signalAccuracyWait: document.getElementById("signalAccuracyWait"),
  signalAccuracyEmpty: document.getElementById("signalAccuracyEmpty"),
  signalAccuracyRow: document.getElementById("signalAccuracyRow"),
  accuracyRangeGroup: document.getElementById("accuracyRangeGroup"),
  accuracyIntervalGroup: document.getElementById("accuracyIntervalGroup"),
  accuracyMape: document.getElementById("accuracyMape"),
  accuracyChart: document.getElementById("accuracyChart"),
  accuracyChartEmpty: document.getElementById("accuracyChartEmpty"),
  explainBtn: document.getElementById("explainBtn"),
  explainPanel: document.getElementById("explainPanel"),
  explainText: document.getElementById("explainText"),
  patternAlert: document.getElementById("patternAlert"),
  patternLegendItem: document.getElementById("patternLegendItem"),
  boxPanel: document.getElementById("boxPanel"),
  boxChoice: document.getElementById("boxChoice"),
  boxBoxStats: document.getElementById("boxBoxStats"),
  boxStatus: document.getElementById("boxStatus"),
  boxStats: document.getElementById("boxStats"),
  boxLegendItem: document.getElementById("boxLegendItem"),
  explainClose: document.getElementById("explainClose"),
  l2Rows: document.getElementById("l2Rows"),
};

const COLORS = {
  up: "#2fbf76",
  down: "#f0546b",
  pred: "#6d7cf5",
  sma: "#8b96ff",
  ema: "#e0729a",
  bollinger: "#a3aabb",
  fibonacci: "#676f83",
  vwap: "#f2a93c",
  pattern: "#b98bff",
  grid: "rgba(255, 255, 255, 0.06)",
  text: "#676f83",
  surface: "#12151c",
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

let chart, candleSeries, volumeSeries, predSeries, upperBandSeries, lowerBandSeries;
let closePriceSeries, smaSeries, emaSeries, bbUpperSeries, bbMiddleSeries, bbLowerSeries, vwapSeries;
let patternUpperSeries, patternLowerSeries;
let fibPriceLines = [];
let patternPriceLines = [];
let boxSeries = [];        // one fill (baseline) + one floor line series per drawn box
let boxPriceLines = [];
// Patterns and Box Breakout both mark candles; Lightweight Charts keeps a
// single marker list per series, so each overlay owns its own set here.
const markerSets = { patterns: [], box: [] };

function applyMarkers() {
  if (!candleSeries) return;
  candleSeries.setMarkers([...markerSets.patterns, ...markerSets.box].sort((a, b) => a.time - b.time));
}
let lastSortedCandles = [];
let lastTotalBars = null;
let forceFullView = false;
let liveCandleAnchor = null;

let accuracyPanelSklearn;

// installLeftEdgeClamp (shared with detail.js) lives in chart-utils.js,
// loaded before this script - see that file for the full reasoning
// (loadChart()'s "keep the user's exact scroll position across a 30s
// refresh" logic is what makes an unclamped empty window look frozen).

function initChart() {
  if (!window.LightweightCharts) {
    els.chartEmpty.hidden = false;
    els.chartEmpty.textContent = t("dashboard.chart.libError");
    return;
  }

  // Lightweight Charts formats UTCTimestamp values in UTC by default, not
  // the viewer's local timezone - on this dashboard that showed candles/
  // predictions looking hours "behind" wall-clock time. These formatters
  // render everything in the browser's own local time instead.
  const tickMarkFormatter = (time, tickMarkType) => {
    const d = new Date(time * 1000);
    const TMT = LightweightCharts.TickMarkType;
    if (tickMarkType === TMT.Year) return d.toLocaleDateString("en-US", { year: "numeric" });
    if (tickMarkType === TMT.Month) return d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
    if (tickMarkType === TMT.DayOfMonth) return d.toLocaleDateString("en-US", { day: "2-digit", month: "short" });
    return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  };

  chart = LightweightCharts.createChart(els.chart, {
    layout: {
      background: { color: "transparent" },
      textColor: COLORS.text,
      fontFamily: "Sora, system-ui, -apple-system, 'Segoe UI', sans-serif",
    },
    grid: {
      vertLines: { color: COLORS.grid },
      horzLines: { color: COLORS.grid },
    },
    localization: {
      timeFormatter: (time) =>
        new Date(time * 1000).toLocaleString("en-US", {
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
    ...CHART_PAN_ZOOM_OPTIONS,
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
  vwapSeries = chart.addLineSeries({ color: COLORS.vwap, lineWidth: 2, visible: false, crosshairMarkerVisible: false });

  // Two lines rather than a generic "connect these N points" series - a
  // geometric pattern (triangle/wedge/channel/flag) is always exactly two
  // boundaries (upper, lower), so this covers every pattern type with one
  // pair of series instead of allocating a variable number of them.
  patternUpperSeries = chart.addLineSeries({
    color: COLORS.pattern, lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed,
    visible: false, crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });
  patternLowerSeries = chart.addLineSeries({
    color: COLORS.pattern, lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed,
    visible: false, crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });

  predSeries = chart.addLineSeries({ color: COLORS.pred, lineWidth: 2 });
  // lastValueVisible/priceLineVisible off: with candle + pred + upper +
  // lower all tagging the axis by default, the labels pile up and overlap -
  // the band's shape is already visible from the shaded area itself.
  //
  // Shaded confidence band: an Area series fills from the upper bound down
  // to a `bottomColor` that fades to *fully transparent* rather than a
  // flat semi-opaque fill - a flat fill (an earlier version of this, plus
  // the classic "stack two areas, paint the second in the background
  // color to erase everything below the real lower bound" technique used
  // to fake a fill-between-two-lines) extends, opaque, all the way to the
  // bottom of the price scale, which blanked out the grid/whatever else
  // sat underneath it for the entire future/prediction span - visible
  // live as a stark black rectangle under the band. A gradient that ends
  // in alpha 0 can never paint over anything no matter how far down it
  // notionally extends, so this can't regress the same way. The lower
  // bound is a plain dashed line (marks the boundary, fills nothing).
  upperBandSeries = chart.addAreaSeries({
    lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, lineColor: COLORS.pred,
    topColor: "rgba(57,135,229,0.28)", bottomColor: "rgba(57,135,229,0)",
    crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });
  lowerBandSeries = chart.addLineSeries({
    color: COLORS.pred, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });

  new ResizeObserver(() => {
    if (chart) chart.applyOptions({ width: els.chartWrap.clientWidth, height: els.chartWrap.clientHeight });
  }).observe(els.chartWrap);

  installLeftEdgeClamp(chart);

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
    const price = candleSeries.coordinateToPrice(param.point.y);
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

// Historical predicted-vs-actual comparison (see /api/predict/accuracy):
// how closely the model's one-step-ahead forecast has tracked what
// actually happened, over the last month or year - not just its current
// live forecast. Factored out as its own instance (rather than one fixed
// chart) since this used to also track a second, since-removed LSTM
// backend on its own chart - kept this way in case a second backend
// worth comparing against ever comes back.
function createAccuracyPanel({ chartEl, emptyEl, mapeEl, rangeGroupEl, intervalGroupEl, backend, defaultInterval = "1h", defaultDays = 30 }) {
  const panel = {
    interval: defaultInterval, days: defaultDays, chart: null,
    actualSeries: null, predSeries: null, predSeriesOther: null,
  };

  function init() {
    if (!window.LightweightCharts) return;

    panel.chart = LightweightCharts.createChart(chartEl, {
      layout: {
        background: { color: "transparent" },
        textColor: COLORS.text,
        fontFamily: "Sora, system-ui, -apple-system, 'Segoe UI', sans-serif",
      },
      grid: {
        vertLines: { color: COLORS.grid },
        horzLines: { color: COLORS.grid },
      },
      localization: {
        timeFormatter: (time) =>
          new Date(time * 1000).toLocaleString("en-US", { day: "2-digit", month: "short", year: "numeric" }),
      },
      rightPriceScale: { borderColor: COLORS.grid },
      timeScale: { borderColor: COLORS.grid, timeVisible: false, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      ...CHART_PAN_ZOOM_OPTIONS,
    });
    panel.chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });

    panel.actualSeries = panel.chart.addLineSeries({ color: COLORS.up, lineWidth: 2 });
    panel.predSeries = panel.chart.addLineSeries({ color: COLORS.pred, lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed });
    // The *other* model variant's own predictions, same real backdrop -
    // lets a "new model" toggle selection still show what "old model"
    // would have called here, and vice versa, without switching the toggle
    // back and forth. Red (COLORS.down) specifically so it never gets
    // confused with the primary (accent-blue) prediction line.
    panel.predSeriesOther = panel.chart.addLineSeries({
      color: COLORS.down, lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dotted,
    });

    installLeftEdgeClamp(panel.chart);

    new ResizeObserver(() => {
      if (panel.chart) {
        panel.chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
      }
    }).observe(chartEl);

    rangeGroupEl.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-accuracy-days]");
      if (!btn) return;
      panel.days = Number(btn.dataset.accuracyDays);
      [...rangeGroupEl.children].forEach((b) => b.classList.toggle("active", b === btn));
      load();
    });

    intervalGroupEl.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-accuracy-interval]");
      if (!btn) return;
      panel.interval = btn.dataset.accuracyInterval;
      [...intervalGroupEl.children].forEach((b) => b.classList.toggle("active", b === btn));
      load();
    });
  }

  async function load() {
    if (!panel.chart) return;
    try {
      // Unfiltered, same as before the model-variant toggle existed - the
      // full recorded history, regardless of which model happens to be
      // selected live right now. Filtering this to state.modelVariant
      // seemed appealing (this line "follows the toggle") but broke the
      // whole panel the moment it was tried: "tuned" has no history at all
      // until its first prediction's target time actually elapses, so the
      // panel just went blank instead of showing years of real data it
      // already had. The "tuned" line specifically is what the second
      // (red) series below is for - always that one variant, not whichever
      // is toggled.
      const data = await apiGet("/api/predict/accuracy", { interval: panel.interval, days: panel.days, backend });
      if (!data.points.length) {
        emptyEl.hidden = false;
        panel.actualSeries.setData([]);
        panel.predSeries.setData([]);
        panel.predSeriesOther.setData([]);
        mapeEl.textContent = "—";
        return;
      }
      emptyEl.hidden = true;
      panel.actualSeries.setData(data.points.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.actual_price })));
      panel.predSeries.setData(data.points.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.predicted_price })));
      panel.chart.timeScale().fitContent();
      mapeEl.textContent =
        data.mape === null || data.mape === undefined
          ? "—"
          : tf("dashboard.accuracy.mapeTemplate", { pct: data.mape.toFixed(2), n: data.points.length, interval: panel.interval });

      // Best-effort, never blocks the primary line above: "tuned" only
      // started being tagged today (see repository.get_prediction_accuracy_series),
      // so this can legitimately be empty for a while after a fresh
      // deploy - expected, not an error.
      try {
        const other = await apiGet("/api/predict/accuracy", {
          interval: panel.interval, days: panel.days, backend, model_variant: "tuned",
        });
        panel.predSeriesOther.setData(
          (other.points || []).map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.predicted_price }))
        );
      } catch {
        panel.predSeriesOther.setData([]);
      }
    } catch (exc) {
      emptyEl.hidden = false;
      emptyEl.textContent = tf("dashboard.accuracy.fetchError", { msg: exc.message });
      mapeEl.textContent = "—";
    }
  }

  return { init, load };
}

const BACKTEST_LIMIT = 100;

// Compact "backtesting visible" hint next to the confidence score - the
// confidence score itself only reflects the *current* forecast's interval
// width; this instead says how the model's actual past 1-step-ahead calls
// at the currently selected interval panned out against what really
// happened, so trust doesn't rest on a single static number.
async function loadBacktestHint() {
  try {
    const data = await apiGet("/api/predict/accuracy", { interval: state.interval, limit: BACKTEST_LIMIT });
    if (!data.count) {
      els.backtestHint.textContent = t("dashboard.backtest.accumulating");
      return;
    }
    els.backtestHint.textContent =
      tf("dashboard.backtest.template", { pct: data.mape.toFixed(2), n: data.count, interval: state.interval });
  } catch {
    els.backtestHint.textContent = t("dashboard.backtest.dash");
  }
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

// Cumulative VWAP over the whole currently-loaded window (not anchored to
// a calendar day/session the way an exchange's intraday VWAP usually is -
// simplest honest option given this dashboard's interval ranges from 15m
// to 1w, where a "trading day" anchor doesn't mean the same thing).
// Volume-weighted, so it moves toward wherever the heaviest trading
// actually happened, not just the average price - a real institutional/
// whale reference: price trading above it means buyers have been paying a
// premium over that volume-weighted level, and vice versa below it.
function computeVWAP(candles) {
  const out = [];
  let cumPV = 0;
  let cumVolume = 0;
  for (const c of candles) {
    const typicalPrice = (c.high + c.low + c.close) / 3;
    const volume = c.volume || 0;
    cumPV += typicalPrice * volume;
    cumVolume += volume;
    out.push(cumVolume > 0 ? cumPV / cumVolume : null);
  }
  return out;
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
  let bb = null;
  if (showBB) {
    bb = computeBollinger(closes, BOLLINGER_PERIOD, BOLLINGER_MULT);
    bbUpperSeries.setData(toSeriesData(bb.upper));
    bbMiddleSeries.setData(toSeriesData(bb.middle));
    bbLowerSeries.setData(toSeriesData(bb.lower));
  } else {
    bbUpperSeries.setData([]);
    bbMiddleSeries.setData([]);
    bbLowerSeries.setData([]);
  }

  vwapSeries.applyOptions({ visible: state.indicators.vwap });
  const vwapValues = state.indicators.vwap ? computeVWAP(sorted) : [];
  vwapSeries.setData(state.indicators.vwap ? toSeriesData(vwapValues) : []);

  updateFibonacci();
  updateLastIndicatorValues(closes, bb, vwapValues);
}

// Snapshot of each active indicator's latest value, kept for the "explică
// cu AI" feature (loadExplainContext) - the toggles show these on the
// chart but nowhere summarize what they currently say together.
let lastIndicatorValues = {};
function updateLastIndicatorValues(closes, bb, vwapValues) {
  const lastClose = closes[closes.length - 1];
  lastIndicatorValues = {
    price: lastClose,
    sma: state.indicators.sma ? computeSMA(closes, SMA_PERIOD).at(-1) : null,
    ema: state.indicators.ema ? computeEMA(closes, EMA_PERIOD).at(-1) : null,
    bollinger: state.indicators.bollinger && bb
      ? { upper: bb.upper.at(-1), middle: bb.middle.at(-1), lower: bb.lower.at(-1) }
      : null,
    vwap: state.indicators.vwap ? vwapValues.at(-1) : null,
  };
}

// Fibonacci retracement is only meaningful for the swing you're actually
// looking at, not the whole fetched history - so it's computed from
// whichever candles are currently in view, and recomputed live as you pan
// or zoom (see the subscribeVisibleLogicalRangeChange call in initChart).
let lastFibonacciRange = null;
function updateFibonacci() {
  clearFibonacciLines();
  lastFibonacciRange = null;
  if (!state.indicators.fibonacci || !lastSortedCandles.length || !chart) return;

  const range = chart.timeScale().getVisibleLogicalRange();
  const n = lastSortedCandles.length;
  const from = Math.max(0, Math.floor(range ? range.from : 0));
  const to = Math.min(n - 1, Math.ceil(range ? range.to : n - 1));
  const visible = from <= to ? lastSortedCandles.slice(from, to + 1) : lastSortedCandles;
  if (!visible.length) return;

  const high = Math.max(...visible.map((c) => c.high));
  const low = Math.min(...visible.map((c) => c.low));
  lastFibonacciRange = { high, low };
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

// Rule-based chart pattern recognition (see ml/pattern_recognition.py):
// candlestick shapes get a small arrow marker on their own candle;
// geometric patterns (triangle/wedge/channel etc.) get their defining
// points connected as one or two dashed lines, plus a neckline/target
// price line when the pattern has one. Only these two "two boundary
// lines" pattern families actually have separate upper/lower trendlines
// in the data (see pattern_recognition.py's detect_triangle_or_wedge /
// _detect_channel) - everything else (Double Top, Head & Shoulders,
// Flag...) is a single sequence of defining points, so it goes on one line.
const PATTERN_TWO_LINE_NAMES = new Set([
  "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle", "Expanding Triangle",
  "Rising Wedge", "Falling Wedge", "Ascending Channel", "Descending Channel", "Horizontal Channel",
]);
const PATTERN_MARKER_SHAPE = { bullish: "arrowUp", bearish: "arrowDown", neutral: "circle" };
function patternDirectionLabel(direction) {
  const key = { bullish: "dashboard.pattern.dirBullish", bearish: "dashboard.pattern.dirBearish", neutral: "dashboard.pattern.dirNeutral" }[direction];
  return key ? t(key) : direction;
}

function clearPatternOverlay() {
  if (!chart) return;
  const series = getActivePriceSeries();
  patternPriceLines.forEach((line) => series.removePriceLine(line));
  patternPriceLines = [];
  patternUpperSeries.applyOptions({ visible: false });
  patternLowerSeries.applyOptions({ visible: false });
  patternUpperSeries.setData([]);
  patternLowerSeries.setData([]);
  markerSets.patterns = [];
  applyMarkers();
  els.patternAlert.hidden = true;
  els.patternLegendItem.hidden = true;
}

function renderPatternAlert(candlestickPatterns, topGeometric) {
  // One entry per distinct pattern *type* (Doji, Tweezer Bottom, ...) -
  // the chart can label the same shape on several candles at once, but a
  // reader wants to know what each shape means once, not read the same
  // explanation repeated for every candle it happened to match on.
  const byName = new Map();
  [...candlestickPatterns].sort((a, b) => a.index - b.index).forEach((p) => byName.set(p.name, p));
  const distinctCandleTypes = [...byName.values()].sort((a, b) => b.index - a.index);

  const items = [];
  if (topGeometric) items.push(topGeometric);
  items.push(...distinctCandleTypes);

  if (!items.length) {
    els.patternAlert.hidden = true;
    return;
  }
  const renderItem = (p) => `
    <div class="pattern-alert-item">
      <span class="pattern-alert-badge ${p.direction}">${patternDirectionLabel(p.direction)}</span>
      <span class="pattern-alert-name">${p.name}${p.confirmed === false ? t("dashboard.pattern.forming") : ""}</span>
      <span class="pattern-alert-note">${p.note}</span>
      ${p.target != null ? `<span class="pattern-alert-target">${tf("dashboard.pattern.target", { price: fmtMoney(p.target) })}</span>` : ""}
    </div>
  `;

  els.patternAlert.hidden = false;
  els.patternAlert.className = `pattern-alert ${items[0].direction}`;
  els.patternAlert.innerHTML =
    (topGeometric ? renderItem(topGeometric) : "") +
    (distinctCandleTypes.length
      ? `<div class="pattern-alert-heading">${t("dashboard.pattern.headingCandlestick")}</div>${distinctCandleTypes.map(renderItem).join("")}`
      : "");
}

async function loadPatterns() {
  if (!state.indicators.patterns) {
    clearPatternOverlay();
    return;
  }
  try {
    const data = await apiGet("/api/patterns", { interval: state.interval, lang: getLang() });

    const markers = data.candlestick
      .map((p) => ({
        time: toUnixSeconds(p.timestamp),
        position: p.direction === "bearish" ? "aboveBar" : "belowBar",
        color: p.direction === "bullish" ? COLORS.up : p.direction === "bearish" ? COLORS.down : COLORS.text,
        shape: PATTERN_MARKER_SHAPE[p.direction] || "circle",
        text: p.name,
      }))
      .sort((a, b) => a.time - b.time); // Lightweight Charts requires markers in ascending time order.
    markerSets.patterns = markers;
    applyMarkers();

    const series = getActivePriceSeries();
    patternPriceLines.forEach((line) => series.removePriceLine(line));
    patternPriceLines = [];

    const top = data.geometric[0];
    const toPoint = (lvl) => ({ time: toUnixSeconds(lvl.timestamp), value: lvl.price });
    els.patternLegendItem.hidden = !top;
    if (!top) {
      patternUpperSeries.applyOptions({ visible: false });
      patternLowerSeries.applyOptions({ visible: false });
      patternUpperSeries.setData([]);
      patternLowerSeries.setData([]);
    } else if (PATTERN_TWO_LINE_NAMES.has(top.name)) {
      // upper_count marks where the upper-line pivots end and the lower-line
      // ones begin - highs/lows aren't always equal counts (see
      // ml/pattern_recognition.py's _detect_triangle_or_wedge), so a plain
      // half/half split can put a low on the upper line or vice versa.
      const splitAt = top.upper_count ?? Math.ceil(top.levels.length / 2);
      patternUpperSeries.setData(top.levels.slice(0, splitAt).map(toPoint).sort((a, b) => a.time - b.time));
      patternLowerSeries.setData(top.levels.slice(splitAt).map(toPoint).sort((a, b) => a.time - b.time));
      patternUpperSeries.applyOptions({ visible: true });
      patternLowerSeries.applyOptions({ visible: true });
    } else {
      patternUpperSeries.setData([...top.levels].map(toPoint).sort((a, b) => a.time - b.time));
      patternUpperSeries.applyOptions({ visible: true });
      patternLowerSeries.applyOptions({ visible: false });
      patternLowerSeries.setData([]);
    }

    if (top && top.neckline != null) {
      patternPriceLines.push(series.createPriceLine({
        price: top.neckline, color: COLORS.text, lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: true, title: t("dashboard.pattern.necklineTitle"),
      }));
    }
    if (top && top.target != null) {
      patternPriceLines.push(series.createPriceLine({
        price: top.target, color: top.direction === "bearish" ? COLORS.down : COLORS.up, lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted, axisLabelVisible: true, title: t("dashboard.pattern.targetTitle"),
      }));
    }

    renderPatternAlert(data.candlestick, top);
  } catch (exc) {
    clearPatternOverlay();
  }
}

// -- Box Breakout strategy (ml/box_breakout.py) --------------------------
//
// Each box is a Baseline series whose base is the floor and whose line is
// the ceiling, so the area between them is filled; a separate line marks
// the floor. Green/red: broke out up/down (grey: without volume, no
// trade), blue: still active. Entries/exits are markers, the open trade's
// entry/stop/target are price lines.

const BOX_MAX_DRAWN = 40;
const BOX_EXIT_LABEL = { stop: "SL", target: "TP", trail_stop: "Trail", reverse: "Rev" };

function boxColor(b) {
  if (!b.breakout) return COLORS.pred;
  if (!b.volume_confirmed) return COLORS.text;
  return b.breakout === "up" ? COLORS.up : COLORS.down;
}

function withAlpha(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function clearBoxDrawing() {
  boxSeries.forEach((s) => chart.removeSeries(s));
  boxSeries = [];
  const series = getActivePriceSeries();
  boxPriceLines.forEach((line) => series.removePriceLine(line));
  boxPriceLines = [];
  markerSets.box = [];
  applyMarkers();
}

function clearBoxOverlay() {
  if (!chart) return;
  clearBoxDrawing();
  els.boxPanel.hidden = true;
  els.boxLegendItem.hidden = true;
}

function drawBoxes(data) {
  clearBoxDrawing();
  const hidden = { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
  data.boxes.slice(-BOX_MAX_DRAWN).forEach((b) => {
    const color = boxColor(b);
    const from = toUnixSeconds(b.start_time);
    const to = toUnixSeconds(b.breakout ? b.end_time : data.last_time);
    if (to <= from) return;
    const fill = chart.addBaselineSeries({
      ...hidden, lineWidth: 1, baseValue: { type: "price", price: b.floor },
      topLineColor: color, topFillColor1: withAlpha(color, 0.16), topFillColor2: withAlpha(color, 0.16),
      bottomLineColor: color, bottomFillColor1: "transparent", bottomFillColor2: "transparent",
    });
    fill.setData([{ time: from, value: b.ceiling }, { time: to, value: b.ceiling }]);
    const floor = chart.addLineSeries({ ...hidden, color, lineWidth: 1 });
    floor.setData([{ time: from, value: b.floor }, { time: to, value: b.floor }]);
    boxSeries.push(fill, floor);
  });

  const markers = [];
  data.trades.forEach((tr) => {
    const long = tr.side === "long";
    markers.push({
      time: toUnixSeconds(tr.entry_time), position: long ? "belowBar" : "aboveBar",
      color: long ? COLORS.up : COLORS.down, shape: long ? "arrowUp" : "arrowDown",
      text: long ? "Long" : "Short",
    });
    if (tr.exit_time) {
      markers.push({
        time: toUnixSeconds(tr.exit_time), position: long ? "aboveBar" : "belowBar",
        color: tr.return_pct > 0 ? COLORS.up : COLORS.down, shape: "circle",
        text: `${BOX_EXIT_LABEL[tr.exit_reason] || "Exit"} ${tr.return_pct > 0 ? "+" : ""}${tr.return_pct.toFixed(1)}%`,
      });
    }
  });
  markerSets.box = markers;
  applyMarkers();

  const open = data.open_trade;
  if (open) {
    const series = getActivePriceSeries();
    const line = (price, color, title) => boxPriceLines.push(series.createPriceLine({
      price, color, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: true, title,
    }));
    line(open.entry_price, COLORS.pred, t("dashboard.box.entryTitle"));
    line(open.stop, COLORS.down, t("dashboard.box.stopTitle"));
    if (open.target != null) line(open.target, COLORS.up, t("dashboard.box.targetTitle"));
  }
}

function fmtPct(v, signed = true) {
  if (v === null || v === undefined) return "—";
  return `${signed && v > 0 ? "+" : ""}${Number(v).toFixed(2)}%`;
}

function renderBoxChoice(data) {
  const cur = data.current;
  const s = cur.settings;
  const next = tf("dashboard.box.nextCheck", { n: Math.max(1, data.next_reopt_in_bars) });
  if (!s) {
    els.boxChoice.innerHTML = `<b>${t("dashboard.box.standingAside")}</b> ${tf("dashboard.box.standingAsideWhy", { n: data.train_bars })} ${next}`;
    return;
  }
  const exit = s.exit_mode === "trail" ? t("dashboard.box.exitTrailShort") : tf("dashboard.box.exitRrShort", { rr: s.rr });
  els.boxChoice.innerHTML = `${tf("dashboard.box.chosen", { x: s.lookback, y: s.padding, exit })} ${tf("dashboard.box.chosenWhy", {
    n: data.train_bars, ret: fmtPct(cur.train_return_pct), trades: cur.train_trades, candidates: cur.candidates,
  })} ${next}`;
}

function renderBoxPanel(data) {
  renderBoxChoice(data);
  const st = data.stats;
  const parts = [];
  const open = data.open_trade;
  if (open) {
    parts.push(tf(open.side === "long" ? "dashboard.box.openLong" : "dashboard.box.openShort", {
      entry: fmtMoney(open.entry_price), stop: fmtMoney(open.stop),
      target: open.target != null ? fmtMoney(open.target) : t("dashboard.box.trailing"),
      pnl: fmtPct(open.return_pct),
    }));
  }
  const box = data.active_box;
  if (box) {
    parts.push(tf("dashboard.box.activeBox", { floor: fmtMoney(box.floor), ceiling: fmtMoney(box.ceiling) }));
  } else if (!open) {
    parts.push(t("dashboard.box.noBox"));
  }
  els.boxStatus.innerHTML = parts.map((p) => `<div>${p}</div>`).join("");

  const cell = (label, value, cls = "") => `<div class="box-stat"><span>${label}</span><b class="${cls}">${value}</b></div>`;
  const sign = (v) => (v > 0 ? "up" : v < 0 ? "down" : "");
  els.boxStats.innerHTML = [
    cell(t("dashboard.box.statTrades"), `${st.trades}${st.open_trades ? ` (+${st.open_trades})` : ""}`),
    cell(t("dashboard.box.statWinRate"), st.win_rate_pct == null ? "—" : `${st.win_rate_pct.toFixed(1)}%`),
    cell(t("dashboard.box.statReturn"), fmtPct(st.total_return_pct), sign(st.total_return_pct)),
    cell(t("dashboard.box.statBuyHold"), fmtPct(st.buy_hold_pct), sign(st.buy_hold_pct)),
    cell(t("dashboard.box.statProfitFactor"), st.profit_factor == null ? "—" : st.profit_factor.toFixed(2)),
    cell(t("dashboard.box.statAvgR"), st.avg_r == null ? "—" : `${st.avg_r.toFixed(2)}R`),
    cell(t("dashboard.box.statDrawdown"), st.max_drawdown_pct ? `-${st.max_drawdown_pct.toFixed(2)}%` : "0%", st.max_drawdown_pct ? "down" : ""),
  ].join("");
  const bs = data.box_stats;
  const traded = data.segments.filter((s) => s.settings).length;
  const pct = (v) => (v == null ? "—" : `${v.toFixed(1)}%`);
  els.boxBoxStats.innerHTML = [
    cell(t("dashboard.box.statBoxes"), bs.boxes),
    cell(t("dashboard.box.statHeight"), pct(bs.avg_height_pct)),
    cell(t("dashboard.box.statDuration"), bs.avg_duration_bars == null ? "—" : tf("dashboard.box.candles", { n: Math.round(bs.avg_duration_bars) })),
    cell(t("dashboard.box.statUpDown"), `${bs.breakouts_up} / ${bs.breakouts_down}`),
    cell(t("dashboard.box.statVolume"), pct(bs.volume_confirmed_pct)),
    cell(t("dashboard.box.statFollow"), pct(bs.follow_through_pct)),
    cell(t("dashboard.box.statInMarket"), `${traded} / ${data.segments.length}`),
  ].join("");
  els.boxPanel.hidden = false;
  els.boxLegendItem.hidden = false;
}

async function loadBoxBreakout() {
  if (!state.indicators.box) {
    clearBoxOverlay();
    return;
  }
  try {
    const data = await apiGet("/api/box-breakout", { interval: state.interval });
    drawBoxes(data);
    renderBoxPanel(data);
  } catch (exc) {
    clearBoxDrawing();
    els.boxPanel.hidden = false;
    els.boxStats.innerHTML = "";
    els.boxBoxStats.innerHTML = "";
    els.boxChoice.textContent = "";
    els.boxStatus.textContent = exc.message;
  }
}

// -- Evolving model population ("tuned", ml/evolution.py) -----------------

const EMOTION_EMOJI = { fear: "😨", caution: "🤔", calm: "😐", confidence: "🙂", euphoria: "🤩" };

function emotionLabel(name) {
  return `${EMOTION_EMOJI[name] || ""} ${t(`dashboard.evo.emotion.${name}`)}`;
}

function renderEvolution(evo) {
  if (!evo) {
    els.evolutionCard.hidden = true;
    return;
  }
  els.evolutionCard.hidden = false;
  if (evo.newborn) {
    els.evoMood.textContent = "";
    els.evoStats.innerHTML = `<div class="box-stat"><span>${t("dashboard.evo.newbornTitle")}</span><b>${t("dashboard.evo.newborn")}</b></div>`;
    els.evoLeaders.innerHTML = "";
    els.evoUsage.innerHTML = "";
    els.evoSurvival.innerHTML = "";
    els.evoSentiment.textContent = "";
    els.evoMoodChart.innerHTML = "";
    els.evoTrack.innerHTML = "";
    els.evoTrackNote.textContent = "";
    return;
  }
  els.evoMood.textContent = emotionLabel(evo.emotion);
  els.evoMood.className = `evo-mood ${evo.emotion}`;

  const cell = (label, value) => `<div class="box-stat"><span>${label}</span><b>${value}</b></div>`;
  els.evoStats.innerHTML = [
    cell(t("dashboard.evo.alive"), evo.population),
    cell(t("dashboard.evo.generation"), evo.max_generation),
    cell(t("dashboard.evo.births"), evo.births),
    cell(t("dashboard.evo.deaths"), evo.deaths),
    cell(t("dashboard.evo.deathsRecent"), evo.deaths_last_100),
    cell(t("dashboard.evo.lifespan"), evo.avg_lifespan_of_dead == null ? "—" : tf("dashboard.box.candles", { n: Math.round(evo.avg_lifespan_of_dead) })),
    cell(t("dashboard.evo.band"), t(evo.band_calibrated ? "dashboard.evo.bandYes" : "dashboard.evo.bandNo")),
  ].join("");

  // Skill: 1 - (vote's error / "no change" error); above 0 = it beat a flat line.
  const skillCell = (v) => (v == null ? "—" : `<span class="${v > 0 ? "up" : v < 0 ? "down" : ""}">${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%</span>`);
  const dirCell = (v) => (v == null ? "—" : `<span class="${v > 50 ? "up" : v < 50 ? "down" : ""}">${v.toFixed(1)}%</span>`);
  els.evoTrackNote.textContent = evo.lived_from
    ? tf("dashboard.evo.trackNote", { from: new Date(evo.track_from || evo.lived_from).toLocaleDateString(), n: evo.candles_lived.toLocaleString() })
    : "";
  els.evoTrack.innerHTML = (evo.track_record || []).map((r) => `
    <tr>
      <td>${tf("dashboard.box.candles", { n: r.h })}</td>
      <td>${r.n.toLocaleString()}</td>
      <td>${skillCell(r.skill)}</td>
      <td>${dirCell(r.direction_pct)}</td>
      <td>${skillCell(r.recent_skill)}</td>
      <td>${dirCell(r.recent_direction_pct)}</td>
    </tr>`).join("");

  const maxEnergy = Math.max(...evo.leaders.map((o) => o.energy), 1);
  els.evoLeaders.innerHTML = evo.leaders.map((o) => `
    <tr>
      <td>${o.id}</td>
      <td>${emotionLabel(o.emotion)}</td>
      <td><span class="evo-energy"><i style="width:${Math.max(4, (o.energy / maxEnergy) * 100)}%"></i></span> ${Math.round(o.energy)}</td>
      <td>${o.age}</td>
      <td>${o.generation}</td>
      <td>${o.combos ?? 0}</td>
      <td><span class="${o.news_effect > 0.005 ? "up" : o.news_effect < -0.005 ? "down" : ""}">${o.news_effect > 0 ? "+" : ""}${Math.round((o.news_effect || 0) * 100)}%</span></td>
      <td class="evo-inputs">${o.inputs.map((g) => t(`dashboard.evo.input.${g}`)).join(", ")}</td>
    </tr>`).join("");

  els.evoUsage.innerHTML = Object.entries(evo.input_usage)
    .sort((a, b) => b[1] - a[1])
    .map(([g, share]) => `
      <div class="evo-usage-row">
        <span>${t(`dashboard.evo.input.${g}`)}</span>
        <span class="evo-energy"><i style="width:${share * 100}%"></i></span>
        <b>${Math.round(share * 100)}%</b>
      </div>`).join("");

  const ms = evo.market_sentiment;
  els.evoSentiment.textContent = ms && ms.source
    ? tf("dashboard.evo.sentimentNow", {
      value: `${ms.value > 0 ? "+" : ""}${ms.value.toFixed(2)}`,
      source: t(ms.source === "news" ? "dashboard.evo.sourceNews" : "dashboard.evo.sourceFearGreed"),
      sens: evo.avg_news_sensitivity == null ? "—" : `${evo.avg_news_sensitivity > 0 ? "+" : ""}${evo.avg_news_sensitivity.toFixed(2)}`,
    })
    : t("dashboard.evo.sentimentNone");

  const surv = Object.entries(evo.gene_survival || {}).sort((a, b) => b[1].avg_lifespan - a[1].avg_lifespan);
  const maxLife = Math.max(...surv.map(([, v]) => v.avg_lifespan), 1);
  els.evoSurvival.innerHTML = surv.map(([g, v]) => `
      <div class="evo-usage-row">
        <span>${t(`dashboard.evo.input.${g}`)}</span>
        <span class="evo-energy"><i style="width:${(v.avg_lifespan / maxLife) * 100}%"></i></span>
        <b>${Math.round(v.avg_lifespan)}</b>
      </div>`).join("");

  // Mood timeline: 0 (fear) at the bottom, 1 (euphoria) at the top, the
  // dashed line is 0.5 ("no better than no change").
  const hist = evo.history || [];
  if (hist.length > 1) {
    const pts = hist.map((h, i) => `${(i / (hist.length - 1)) * 300},${60 - h.mood * 60}`).join(" ");
    els.evoMoodChart.innerHTML = `
      <line x1="0" y1="30" x2="300" y2="30" class="evo-mid" />
      <polyline points="${pts}" class="evo-line" />`;
  } else {
    els.evoMoodChart.innerHTML = "";
  }
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
  return d.toLocaleString("en-US", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}
function fmtTimeShort(iso) {
  const d = new Date(iso);
  return d.toLocaleString("en-US", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fmtRelative(iso) {
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return t("common.relative.justNow");
  if (mins < 60) return tf("common.relative.minutesAgo", { n: mins });
  const hours = Math.round(mins / 60);
  if (hours < 24) return tf("common.relative.hoursAgo", { n: hours });
  return tf("common.relative.daysAgo", { n: Math.round(hours / 24) });
}

// Same wording as fmtRelative, but from an elapsed-seconds count (how
// long ago the model was last retrained) rather than an ISO timestamp.
function fmtElapsed(seconds) {
  const mins = Math.round(seconds / 60);
  if (mins < 1) return t("common.relative.justNow");
  if (mins < 60) return tf("common.relative.minutesAgo", { n: mins });
  return tf("common.relative.hoursAgo", { n: Math.round(mins / 60) });
}

function sentimentBadge(label) {
  const icons = { positive: "▲", negative: "▼", neutral: "●" };
  const safe = ["positive", "negative", "neutral"].includes(label) ? label : "neutral";
  return `<span class="sentiment-badge ${safe}">${icons[safe]} ${t(`common.sentiment.${safe}`)}</span>`;
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
    showError(tf("dashboard.error.currentPrice", { msg: exc.message }));
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
    els.updated.textContent = tf("dashboard.updated.template", { time: new Date().toLocaleTimeString("en-US") });
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
    apiGet("/api/predict", { interval: state.interval, steps: state.steps, use_sentiment: state.useSentiment, backend: state.backend, model_variant: state.modelVariant }),
  ]);

  let candles = [];
  if (historyResult.status === "fulfilled") {
    candles = historyResult.value.candles || [];
  } else {
    showError(tf("dashboard.error.priceHistory", { msg: historyResult.reason.message }));
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
      showError(tf("dashboard.error.predictionFailed", { msg: exc.message }));
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
        upperBandSeries.setData(upperPoints);
        lowerBandSeries.setData(lowerPoints);
      } else {
        predSeries.setData([]);
        upperBandSeries.setData([]);
        lowerBandSeries.setData([]);
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
      upperBandSeries.setData([]);
      lowerBandSeries.setData([]);
      updateIndicatorSeries([]);
      liveCandleAnchor = null;
    }
  }

  if (prediction) {
    els.confidence.textContent = `${Math.round(prediction.confidence)}/100`;
    els.confidence.className = `stat-value ${qualityClass(prediction.confidence)}`;
    els.sentimentAvg.textContent = prediction.sentiment_avg.toFixed(2);
    els.backend.textContent = prediction.backend;
    els.sentimentContribution.textContent =
      prediction.sentiment_contribution_pct === null || prediction.sentiment_contribution_pct === undefined
        ? t("dashboard.sentiment.unavailable")
        : tf("dashboard.sentiment.template", { pct: prediction.sentiment_contribution_pct.toFixed(1) });
    // Concrete evidence the "sentiment in model" toggle isn't just a
    // static setting - the model actually retrains from scratch (fresh
    // sentiment + price data) every time the cache expires, not once ever.
    els.recalibratedHint.textContent = tf("dashboard.recalibrated", { time: fmtElapsed(prediction.trained_ago_seconds) });
    renderEvolution(prediction.evolution);

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
    els.sentimentContribution.textContent = t("dashboard.sentiment.dash");
    els.recalibratedHint.textContent = "—";
    renderEvolution(null);
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
  delete els.signalBadge.dataset.signal;
  els.bestBuy.textContent = "—";
  els.bestSell.textContent = "—";
  els.potentialGain.textContent = "—";
  els.potentialGain.className = "stat-value stat-value--small";
  els.rsiValue.textContent = "—";
  els.macdValue.textContent = "—";
  els.volatilityBadge.hidden = true;
}

// How much the model's own predicted low->high swing is worth, as a %,
// spelled out as one number instead of making the reader do the
// subtraction between Best Buy and Best Sell themselves. Not a claim that
// low necessarily comes before high chronologically (see minPoint/maxPoint
// above) - just "the spread the model is currently predicting".
function computePotentialGainPct(minPoint, maxPoint) {
  if (!minPoint.predicted_price) return null;
  return ((maxPoint.predicted_price - minPoint.predicted_price) / minPoint.predicted_price) * 100;
}

// Recent (last 24 candles) return volatility vs. this same asset's own
// longer trailing average, as a ratio - not an absolute number, so "Calm"/
// "Volatile" mean the same thing regardless of which chart interval
// (15m..1w) is selected. A ratio well above 1 means this asset is choppier
// right now than its own normal behavior (lean less on the confidence
// score); well below 1 means calmer than usual.
const VOLATILITY_RECENT_WINDOW = 24;
const VOLATILITY_HIGH_RATIO = 1.3;
const VOLATILITY_LOW_RATIO = 0.7;

function computeVolatilityRegime(closes) {
  if (closes.length < 30) return null;
  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    if (closes[i - 1]) returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
  }
  const stdDev = (values) => {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  };
  const baselineVol = stdDev(returns);
  if (!baselineVol) return null;
  const recentVol = stdDev(returns.slice(-VOLATILITY_RECENT_WINDOW));
  const ratio = recentVol / baselineVol;
  if (ratio >= VOLATILITY_HIGH_RATIO) return "volatile";
  if (ratio <= VOLATILITY_LOW_RATIO) return "calm";
  return "normal";
}

function updateSignalPanel({ minPoint, maxPoint, trendPct, confidence, sentimentAvg }) {
  els.bestBuy.textContent = `${fmtMoney(minPoint.predicted_price)} · ${fmtTimeShort(minPoint.timestamp)}`;
  els.bestSell.textContent = `${fmtMoney(maxPoint.predicted_price)} · ${fmtTimeShort(maxPoint.timestamp)}`;

  const potentialGainPct = computePotentialGainPct(minPoint, maxPoint);
  if (potentialGainPct !== null) {
    els.potentialGain.textContent = `${arrow(potentialGainPct)} +${potentialGainPct.toFixed(2)}%`;
    els.potentialGain.className = `stat-value stat-value--small price-${directionClass(potentialGainPct)}`;
  } else {
    els.potentialGain.textContent = "—";
  }

  const closes = lastSortedCandles.map((c) => c.close);

  const regime = computeVolatilityRegime(closes);
  if (regime) {
    els.volatilityBadge.hidden = false;
    els.volatilityBadge.textContent = t(`dashboard.volatility.${regime}`);
    els.volatilityBadge.className = `regime-badge ${regime}`;
  } else {
    els.volatilityBadge.hidden = true;
  }

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
    const zone = rsi < 30 ? t("dashboard.signal.oversold") : rsi > 70 ? t("dashboard.signal.overbought") : "";
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

  // cls ("wait"/"buy"/"sell") is the language-independent signal state -
  // stored in the badge's class, not derived from its (localized) label -
  // so explainIndicators() below can key off it regardless of language.
  let labelKey = "dashboard.signal.wait";
  let cls = "wait";
  if (confidence >= 40 && score >= 1.5) {
    labelKey = "dashboard.signal.buy";
    cls = "buy";
  } else if (confidence >= 40 && score <= -1.5) {
    labelKey = "dashboard.signal.sell";
    cls = "sell";
  }
  els.signalBadge.textContent = t(labelKey);
  els.signalBadge.className = `signal-badge ${cls}`;
  els.signalBadge.dataset.signal = cls;
}

// The outlook tiles' own %/target (above) are the AI's forecast for the
// NEXT period - a completely different time window than "how much has ETH
// actually already moved", so the two will often disagree (that's expected,
// not staleness: a small predicted 24h move right after a real, already-
// happened rally isn't a contradiction, since that rally is already priced
// into the current baseline the forecast starts from). Fetched and shown
// separately here so the forward-looking prediction and the backward-
// looking fact are never mistaken for one another.
//
// Calendar-aligned, not rolling: "actual today" resets at 00:00 UTC (the
// open of the current 1d candle), "actual this week" resets every Monday
// 00:00 UTC (the open of the current 1w candle) - explicitly requested
// over a rolling "last 24h"/"last 7d" window, and it falls out for free
// from data already being collected, since Binance's own 1d/1w candles are
// already bucketed to UTC calendar days / ISO weeks (Monday start).
function formatActualPct(pct) {
  return `${arrow(pct)} ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

async function loadOutlook() {
  try {
    const [outlook, currentPrice, todayCandles, thisWeekCandles] = await Promise.all([
      apiGet("/api/outlook", { use_sentiment: state.useSentiment, backend: state.backend, model_variant: state.modelVariant }),
      apiGet("/api/price/current", {}),
      apiGet("/api/price/history", { interval: "1d", limit: 1 }),
      apiGet("/api/price/history", { interval: "1w", limit: 1 }),
    ]);
    for (const [leg, pctEl, targetEl] of [
      [outlook.daily, els.outlookDailyPct, els.outlookDailyTarget],
      [outlook.weekly, els.outlookWeeklyPct, els.outlookWeeklyTarget],
    ]) {
      const cls = directionClass(leg.change_pct);
      pctEl.textContent = `${arrow(leg.change_pct)} ${leg.change_pct >= 0 ? "+" : ""}${leg.change_pct.toFixed(2)}%`;
      pctEl.className = `outlook-pct ${cls}`;
      targetEl.textContent = tf("dashboard.outlook.targetTemplate", { price: fmtMoney(leg.predicted_price) });
    }

    const livePrice = currentPrice.price;
    const todayOpen = todayCandles.candles?.[0]?.open;
    if (typeof livePrice === "number" && todayOpen) {
      const actualTodayPct = ((livePrice - todayOpen) / todayOpen) * 100;
      els.outlookDailyActual.textContent = tf("dashboard.outlook.actual24hTemplate", { pct: formatActualPct(actualTodayPct) });
      els.outlookDailyActual.className = `outlook-actual ${directionClass(actualTodayPct)}`;
    } else {
      els.outlookDailyActual.textContent = "";
    }

    const weekOpen = thisWeekCandles.candles?.[0]?.open;
    if (typeof livePrice === "number" && weekOpen) {
      const actualWeekPct = ((livePrice - weekOpen) / weekOpen) * 100;
      els.outlookWeeklyActual.textContent = tf("dashboard.outlook.actual7dTemplate", { pct: formatActualPct(actualWeekPct) });
      els.outlookWeeklyActual.className = `outlook-actual ${directionClass(actualWeekPct)}`;
    } else {
      els.outlookWeeklyActual.textContent = "";
    }
  } catch {
    els.outlookDailyPct.textContent = "—";
    els.outlookDailyPct.className = "outlook-pct";
    els.outlookWeeklyPct.textContent = "—";
    els.outlookWeeklyPct.className = "outlook-pct";
    els.outlookDailyTarget.textContent = "";
    els.outlookWeeklyTarget.textContent = "";
    els.outlookDailyActual.textContent = "";
    els.outlookWeeklyActual.textContent = "";
  }
}

// Renders one "Buy calls" / "Sell calls" tile: count + win rate, colored
// by whether the heuristic has actually been right more often than not.
// "Wait" gets its own simpler renderer just below - it makes no
// directional claim, so there's no correct/incorrect to score.
function renderSignalAccuracyTile(el, stat) {
  if (!stat || !stat.count) {
    el.textContent = t("dashboard.signalAccuracy.noCalls");
    el.className = "stat-value stat-value--small";
    return;
  }
  const pct = Math.round((stat.correct / stat.count) * 100);
  el.textContent = tf("dashboard.signalAccuracy.record", { correct: stat.correct, count: stat.count, pct });
  el.className = `stat-value stat-value--small price-${pct >= 50 ? "up" : "down"}`;
}

async function loadSignalAccuracy() {
  try {
    // Fixed 1h/24-step, matching backfill_signal_history's default - the
    // one combo actually being backtested (see data_collector/jobs.py),
    // regardless of whatever interval the main chart is currently on.
    const data = await apiGet("/api/signal/accuracy", { interval: "1h" });
    const total = (data.buy?.count || 0) + (data.sell?.count || 0) + (data.wait?.count || 0);
    els.signalAccuracyEmpty.hidden = total > 0;
    els.signalAccuracyRow.hidden = total === 0;
    renderSignalAccuracyTile(els.signalAccuracyBuy, data.buy);
    renderSignalAccuracyTile(els.signalAccuracySell, data.sell);
    els.signalAccuracyWait.textContent = data.wait?.count
      ? String(data.wait.count)
      : t("dashboard.signalAccuracy.noCalls");
    els.signalAccuracyWait.className = "stat-value stat-value--small";
  } catch {
    els.signalAccuracyEmpty.hidden = false;
    els.signalAccuracyRow.hidden = true;
  }
}

async function loadSummary() {
  try {
    // No `interval` param: the daily summary is always about tomorrow
    // (matches the "Predicție 24h" outlook tile) regardless of which
    // interval the main chart happens to be showing - see
    // api.services.run_combined_summary's docstring for why.
    const summary = await apiGet("/api/summary", { backend: state.backend, lang: getLang(), model_variant: state.modelVariant });
    els.narrative.textContent = summary.narrative;
  } catch (exc) {
    if (exc.status !== 409) {
      els.narrative.textContent = tf("dashboard.summary.fetchFailed", { msg: exc.message });
    }
  }
}

async function loadNews() {
  try {
    const { news } = await apiGet("/api/news", { limit: 30 });
    if (!news.length) {
      els.newsList.innerHTML = `<p class="empty-hint">${t("dashboard.news.empty")} <code>python run_scheduler.py</code>.</p>`;
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
    els.newsList.innerHTML = `<p class="empty-hint">${tf("dashboard.news.fetchFailed", { msg: exc.message })}</p>`;
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

function fmtTvl(usd) {
  if (usd >= 1e9) return `$${(usd / 1e9).toFixed(2)}B`;
  if (usd >= 1e6) return `$${(usd / 1e6).toFixed(1)}M`;
  return `$${usd.toLocaleString("en-US")}`;
}

function l2TierLabel(tier) {
  const key = { established: "dashboard.l2.tier.established", growing: "dashboard.l2.tier.growing", emerging: "dashboard.l2.tier.emerging" }[tier];
  return key ? t(key) : tier;
}

// Ethereum L2 rollup TVL leaderboard (see /api/l2 / data_collector/l2_registry.py) -
// "layers" of ETH the user asked to see, ranked by real, live TVL.
async function loadL2Panel() {
  try {
    const { chains } = await apiGet("/api/l2");
    if (!chains.length) {
      els.l2Rows.innerHTML = `<tr class="empty-row"><td colspan="4">${t("dashboard.l2.fetchFail")}</td></tr>`;
      return;
    }
    els.l2Rows.innerHTML = chains
      .map(
        (c) => `<tr>
          <td>${c.display_name}</td>
          <td>${fmtTvl(c.tvl_usd)}</td>
          <td>${c.launch_year}</td>
          <td><span class="l2-tier l2-tier-${c.tier}">${l2TierLabel(c.tier)}</span></td>
        </tr>`
      )
      .join("");
  } catch (exc) {
    els.l2Rows.innerHTML = `<tr class="empty-row"><td colspan="4">${tf("dashboard.l2.fetchError", { msg: exc.message })}</td></tr>`;
  }
}

// Rule-based explanation of whatever indicators are currently active -
// no external API, no key, computed instantly from values already sitting
// in lastIndicatorValues/lastFibonacciRange. Same "always works, nothing
// to configure" approach ml/combined_predictor.py already uses for the
// daily narrative's rule-based fallback - this just doesn't have (or
// need) a Claude-backed upgrade path on top of it.
function explainIndicators() {
  const price = lastIndicatorValues.price;
  if (price == null) return t("dashboard.explain.noPrice");

  const active = Object.entries(state.indicators).filter(([, on]) => on).map(([name]) => name);
  if (!active.length) {
    return t("dashboard.explain.noIndicators");
  }

  const lines = [];

  if (state.indicators.sma && lastIndicatorValues.sma != null) {
    const sma = lastIndicatorValues.sma;
    const diffPct = ((price - sma) / sma) * 100;
    if (Math.abs(diffPct) < 0.1) {
      lines.push(tf("dashboard.explain.smaFlat", { price: fmtMoney(price), sma: fmtMoney(sma) }));
    } else {
      lines.push(tf("dashboard.explain.smaTrend", {
        pct: Math.abs(diffPct).toFixed(2),
        dir: t(diffPct > 0 ? "dashboard.explain.above" : "dashboard.explain.below"),
        sma: fmtMoney(sma),
        trend: t(diffPct > 0 ? "dashboard.explain.uptrend" : "dashboard.explain.downtrend"),
      }));
    }
  }

  if (state.indicators.ema && lastIndicatorValues.ema != null) {
    const ema = lastIndicatorValues.ema;
    lines.push(tf(price >= ema ? "dashboard.explain.emaUp" : "dashboard.explain.emaDown", { ema: fmtMoney(ema) }));
  }

  if (state.indicators.bollinger && lastIndicatorValues.bollinger) {
    const { upper, lower } = lastIndicatorValues.bollinger;
    const bandWidth = upper - lower;
    const posPct = bandWidth > 0 ? ((price - lower) / bandWidth) * 100 : 50;
    if (posPct >= 90) {
      lines.push(tf("dashboard.explain.bollUpper", { upper: fmtMoney(upper) }));
    } else if (posPct <= 10) {
      lines.push(tf("dashboard.explain.bollLower", { lower: fmtMoney(lower) }));
    } else {
      lines.push(tf("dashboard.explain.bollMid", { lower: fmtMoney(lower), upper: fmtMoney(upper) }));
    }
  }

  if (state.indicators.vwap && lastIndicatorValues.vwap != null) {
    const vwap = lastIndicatorValues.vwap;
    lines.push(tf(price >= vwap ? "dashboard.explain.vwapAbove" : "dashboard.explain.vwapBelow", { vwap: fmtMoney(vwap) }));
  }

  if (state.indicators.fibonacci && lastFibonacciRange) {
    const { high, low } = lastFibonacciRange;
    const range = high - low;
    if (range > 0) {
      const nearest = FIB_LEVELS
        .map((level) => ({ level, price: high - level * range }))
        .reduce((a, b) => (Math.abs(price - b.price) < Math.abs(price - a.price) ? b : a));
      lines.push(tf("dashboard.explain.fib", {
        low: fmtMoney(low), high: fmtMoney(high),
        level: (nearest.level * 100).toFixed(1), price: fmtMoney(nearest.price),
      }));
    }
  }

  const rsiText = els.rsiValue.textContent;
  if (rsiText && rsiText !== "—") lines.push(tf("dashboard.explain.rsi", { value: rsiText }));
  const macdText = els.macdValue.textContent;
  if (macdText && macdText !== "—") lines.push(tf("dashboard.explain.macd", { value: macdText }));

  const signalState = els.signalBadge.dataset.signal;
  const signalText = els.signalBadge.textContent;
  if (signalState === "wait") {
    lines.push(t("dashboard.explain.signalWait"));
  } else if (signalState && signalText && signalText !== "—") {
    lines.push(tf("dashboard.explain.signalOther", { signal: signalText }));
  }

  return lines.join(" ");
}

function toggleExplainPanel() {
  const willShow = els.explainPanel.hidden;
  if (willShow) els.explainText.textContent = explainIndicators();
  els.explainPanel.hidden = !willShow;
}

async function loadAll() {
  clearError();
  els.updated.textContent = t("dashboard.updated.loading");
  await Promise.all([
    loadCurrentPrice(), loadChart(), loadOutlook(), loadSummary(), loadNews(), loadGas(),
    loadPinnedTicker(), accuracyPanelSklearn.load(), loadL2Panel(), loadBacktestHint(), loadPatterns(),
    loadBoxBreakout(), loadSignalAccuracy(),
  ]);
  els.updated.textContent = tf("dashboard.updated.template", { time: new Date().toLocaleTimeString("en-US") });
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
  // Without this, switching intervals reused lastTotalBars/the saved zoom
  // window width from the *previous* interval's bar density (e.g. going
  // from "1w" - a few hundred bars total - to "15m" - thousands) against
  // the new data, landing on a nonsensical zoom/pan range instead of the
  // clean "last ~120 bars" default a fresh load gets (see loadChart()).
  lastTotalBars = null;
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

els.modelVariantGroup.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-model-variant]");
  if (!btn) return;
  state.modelVariant = btn.dataset.modelVariant;
  [...els.modelVariantGroup.children].forEach((b) => b.classList.toggle("active", b === btn));
  requestReload();
  // Not accuracyPanelSklearn.load() - that chart's two lines (unfiltered
  // history + the "tuned" comparison line) are independent of the live
  // toggle above, so there's nothing there for this click to refresh.
});

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
  if (input.dataset.indicator === "patterns") loadPatterns();
  if (input.dataset.indicator === "box") loadBoxBreakout();
});

els.explainBtn.addEventListener("click", toggleExplainPanel);
els.explainClose.addEventListener("click", () => { els.explainPanel.hidden = true; });

const AUTO_REFRESH_MS = 30000;
// Matches the server's price cache TTL (api/routes.py) - polling faster
// than that just re-fetches the same cached value. Now that the price
// source is Binance-first (weight-1, ~1200/min limit) rather than
// CoinGecko-first, this can run much faster than it used to without any
// rate-limit risk.
const LIVE_TICK_MS = 3000;

initChart();
accuracyPanelSklearn = createAccuracyPanel({
  chartEl: els.accuracyChart, emptyEl: els.accuracyChartEmpty, mapeEl: els.accuracyMape,
  rangeGroupEl: els.accuracyRangeGroup, intervalGroupEl: els.accuracyIntervalGroup, backend: "sklearn",
});
accuracyPanelSklearn.init();
requestReload();
setInterval(requestReload, AUTO_REFRESH_MS);
setInterval(tickLivePrice, LIVE_TICK_MS);

// i18n.js already re-applies static data-i18n markup on its own; this
// re-runs everything *dynamically* rendered (predictions table, signal
// panel, pattern alert, news, L2 table, the fetched summary/pattern
// narrative text) so a language switch updates it immediately instead of
// waiting for the next 30s auto-refresh.
document.addEventListener("langchange", () => {
  requestReload();
  accuracyPanelSklearn.load();
  if (!els.explainPanel.hidden) els.explainText.textContent = explainIndicators();
});
