import { useEffect, useImperativeHandle, useRef, useState, type Ref } from "react";
import {
  createChart,
  CrosshairMode,
  LineStyle,
  TickMarkType,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type SeriesMarker,
  type SeriesType,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { Radio } from "lucide-react";
import type { Candle, GeometricPattern, CandlePattern, PredictionPoint, StrategyBox, StrategyTrade, EvolutionTrade } from "../lib/api";
import {
  bollinger, confirmedTrend, ema, fibEntrySignals, FIB_LEVELS, findFractals, findLastSwing, futurePriceLabels,
  rollingSlope, sma, smaBollingerSignals, supportResistanceLevels, vwap,
} from "../lib/indicators";
import { chartColors, CHART_FONT } from "../lib/theme";
import { fmtMoney, toUnix } from "../lib/format";
import { useI18n } from "../i18n";

export type IndicatorKey =
  | "sma" | "ema" | "bollinger" | "fibonacci" | "vwap" | "regression" | "fractals" | "levels" | "futureLabel";
export type Indicators = Record<IndicatorKey, boolean>;

export interface PriceChartHandle {
  /** Nudge the still-forming last candle with a live price (between refreshes). */
  tick: (price: number) => void;
}

export interface IndicatorSnapshot {
  price: number | null;
  sma: number | null;
  ema: number | null;
  bollinger: { upper: number; middle: number; lower: number } | null;
  vwap: number | null;
  fib: { high: number; low: number } | null;
}

interface Props {
  candles: Candle[];
  predictions?: PredictionPoint[];
  chartType?: "candles" | "line";
  indicators?: Partial<Indicators>;
  patterns?: { candlestick: CandlePattern[]; geometric: GeometricPattern | null; necklineLabel: string; targetLabel: string } | null;
  box?: {
    boxes: StrategyBox[];
    trades: StrategyTrade[];
    openTrade: StrategyTrade | null;
    lastTime: string;
    /** Does the aggregate breakout-up / breakout-down signal (ml.signal_validation) hold up
     * statistically over this whole window? null = not computed yet, false = drawn neutral/grey
     * instead of the usual up/down color (same demotion already used for unconfirmed volume). */
    significantUp?: boolean | null;
    significantDown?: boolean | null;
  } | null;
  /** The evolution fund's own entry/exit history - markers plus a live price line for an open
   * position, never a predicted price line (see memory feedback-chart-entry-exit-not-prediction). */
  evolutionTrades?: { trades: EvolutionTrade[]; lastClose?: number | null } | null;
  /** Changing this resets the view (e.g. interval switch). */
  viewKey?: string;
  /** With a new viewKey: show the whole loaded window instead of the last N bars. */
  fitAll?: boolean;
  defaultWindow?: number;
  timeVisible?: boolean;
  smartPrice?: boolean;
  onIndicatorSnapshot?: (s: IndicatorSnapshot) => void;
  handleRef?: Ref<PriceChartHandle>;
  className?: string;
}

const PAN_ZOOM = {
  handleScroll: { mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
  handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
  kineticScroll: { touch: true, mouse: false },
} as const;

const TWO_LINE_PATTERNS = new Set([
  "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle", "Expanding Triangle",
  "Rising Wedge", "Falling Wedge", "Ascending Channel", "Descending Channel", "Horizontal Channel",
]);

const ts = (iso: string) => toUnix(iso) as UTCTimestamp;

const BOX_MAX_DRAWN = 40;
const BOX_EXIT_LABEL: Record<string, string> = { stop: "SL", target: "TP", trail_stop: "Trail", reverse: "Rev" };

function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

// The evolution fund trades virtual EUR (ml/evolution.py's START_MONEY), not USD like the
// asset's own price - fmtMoney's "$" prefix would be misleading there. Tolerates a missing
// amount (an already-open trade from before this field existed, still on disk) rather than
// showing "€NaN" until that trade rolls over.
function fmtEur(v: number | null | undefined): string {
  return v == null || Number.isNaN(v) ? "" : `€${Math.round(v).toLocaleString("en-US")}`;
}

export function PriceChart({
  candles,
  predictions = [],
  chartType = "candles",
  indicators = {},
  patterns = null,
  box = null,
  evolutionTrades = null,
  viewKey = "",
  fitAll = false,
  defaultWindow = 120,
  timeVisible = true,
  smartPrice = false,
  onIndicatorSnapshot,
  handleRef,
  className,
}: Props) {
  const { t, locale } = useI18n();
  const wrapRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [showLive, setShowLive] = useState(false);

  // Everything imperative lives in one ref so effects can share it.
  const s = useRef<{
    chart: IChartApi;
    candle: ISeriesApi<"Candlestick">;
    line: ISeriesApi<"Line">;
    volume: ISeriesApi<"Histogram">;
    pred: ISeriesApi<"Line">;
    bandUp: ISeriesApi<"Area">;
    bandLo: ISeriesApi<"Line">;
    sma: ISeriesApi<"Line">;
    ema: ISeriesApi<"Line">;
    bbU: ISeriesApi<"Line">;
    bbM: ISeriesApi<"Line">;
    bbL: ISeriesApi<"Line">;
    vwap: ISeriesApi<"Line">;
    patU: ISeriesApi<"Line">;
    patL: ISeriesApi<"Line">;
    fibLines: IPriceLine[];
    srLines: IPriceLine[];
    fibConnector: ISeriesApi<"Line">;
    fibBandSeries: ISeriesApi<SeriesType>[];
    patLines: IPriceLine[];
    boxSeries: ISeriesApi<SeriesType>[];
    boxPriceLines: IPriceLine[];
    evolutionPriceLines: IPriceLine[];
    markerSets: {
      patterns: SeriesMarker<Time>[]; box: SeriesMarker<Time>[]; evolution: SeriesMarker<Time>[];
      smaSignal: SeriesMarker<Time>[]; regression: SeriesMarker<Time>[]; fractals: SeriesMarker<Time>[];
      fibSignal: SeriesMarker<Time>[]; futureLabel: SeriesMarker<Time>[];
    };
    totalBars: number | null;
    viewKey: string | null;
    sorted: Candle[];
    anchor: { time: UTCTimestamp; open: number; high: number; low: number } | null;
  } | null>(null);

  const latest = useRef({ indicators, chartType, onIndicatorSnapshot, smartPrice, locale });
  latest.current = { indicators, chartType, onIndicatorSnapshot, smartPrice, locale };

  function applyMarkers() {
    const st = s.current;
    if (!st) return;
    st.candle.setMarkers(
      [
        ...st.markerSets.patterns, ...st.markerSets.box, ...st.markerSets.evolution,
        ...st.markerSets.smaSignal, ...st.markerSets.regression, ...st.markerSets.fractals, ...st.markerSets.fibSignal,
        ...st.markerSets.futureLabel,
      ].sort((a, b) => (a.time as number) - (b.time as number)),
    );
  }

  // ---------------------------------------------------------------- mount
  useEffect(() => {
    const el = wrapRef.current!;
    const C = chartColors();
    const chart = createChart(el, {
      width: el.clientWidth,
      height: el.clientHeight,
      layout: { background: { color: "transparent" }, textColor: C.text, fontFamily: CHART_FONT, fontSize: 11 },
      grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
      rightPriceScale: { borderColor: C.border },
      timeScale: {
        borderColor: C.border,
        timeVisible,
        secondsVisible: false,
        rightOffset: 8,
        // Lightweight Charts formats in UTC by default; render local time.
        tickMarkFormatter: (time: Time, type: TickMarkType) => {
          const d = new Date((time as number) * 1000);
          const loc = latest.current.locale;
          if (type === TickMarkType.Year) return d.toLocaleDateString(loc, { year: "numeric" });
          if (type === TickMarkType.Month) return d.toLocaleDateString(loc, { month: "short" });
          if (type === TickMarkType.DayOfMonth) return d.toLocaleDateString(loc, { day: "2-digit", month: "short" });
          return d.toLocaleTimeString(loc, { hour: "2-digit", minute: "2-digit" });
        },
      },
      localization: {
        timeFormatter: (time: Time) =>
          new Date((time as number) * 1000).toLocaleString(latest.current.locale, {
            day: "2-digit", month: "short", year: "numeric", ...(timeVisible ? { hour: "2-digit", minute: "2-digit" } : {}),
          }),
        priceFormatter: (p: number) => fmtMoney(p, { smart: latest.current.smartPrice }).slice(1),
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(236,230,216,0.25)", labelBackgroundColor: "#2e2b25" },
        horzLine: { color: "rgba(236,230,216,0.25)", labelBackgroundColor: "#2e2b25" },
      },
      ...PAN_ZOOM,
    });

    const quiet = { crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false };
    const candle = chart.addCandlestickSeries({
      upColor: C.up, downColor: C.down, borderUpColor: C.up, borderDownColor: C.down, wickUpColor: C.up, wickDownColor: C.down,
    });
    const line = chart.addLineSeries({ color: C.bollinger, lineWidth: 2, visible: false });
    const volume = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "volume", lastValueVisible: false, priceLineVisible: false });
    chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.86, bottom: 0 } });
    const smaS = chart.addLineSeries({ color: C.sma, lineWidth: 2, visible: false, ...quiet });
    const emaS = chart.addLineSeries({ color: C.ema, lineWidth: 2, visible: false, ...quiet });
    const bbU = chart.addLineSeries({ color: C.bollinger, lineWidth: 1, lineStyle: LineStyle.Dashed, visible: false, ...quiet });
    const bbM = chart.addLineSeries({ color: C.bollinger, lineWidth: 1, visible: false, ...quiet });
    const bbL = chart.addLineSeries({ color: C.bollinger, lineWidth: 1, lineStyle: LineStyle.Dashed, visible: false, ...quiet });
    const vwapS = chart.addLineSeries({ color: C.vwap, lineWidth: 2, visible: false, ...quiet });
    const patU = chart.addLineSeries({ color: C.pattern, lineWidth: 2, lineStyle: LineStyle.Dashed, visible: false, ...quiet });
    const patL = chart.addLineSeries({ color: C.pattern, lineWidth: 2, lineStyle: LineStyle.Dashed, visible: false, ...quiet });
    // The swing itself (low<->high) that the Fibonacci levels below are measured from, drawn
    // right on the candles - same idea as patU/patL above, just for the Fib swing instead of a
    // chart pattern's boundary.
    const fibConnector = chart.addLineSeries({ color: C.fib, lineWidth: 2, visible: false, ...quiet });
    // Confidence band: an area fading to fully transparent (never an opaque
    // fill that would blank the grid below the band), lower bound dashed.
    const bandUp = chart.addAreaSeries({
      lineWidth: 1, lineStyle: LineStyle.Dashed, lineColor: C.pred, topColor: C.band, bottomColor: C.bandFade, ...quiet,
    });
    const bandLo = chart.addLineSeries({ color: C.pred, lineWidth: 1, lineStyle: LineStyle.Dashed, ...quiet });
    const pred = chart.addLineSeries({ color: C.pred, lineWidth: 2, lastValueVisible: true, priceLineVisible: false });

    s.current = {
      chart, candle, line, volume, pred, bandUp, bandLo, sma: smaS, ema: emaS, bbU, bbM, bbL, vwap: vwapS, patU, patL,
      fibLines: [], srLines: [], fibConnector, fibBandSeries: [], patLines: [], boxSeries: [], boxPriceLines: [], evolutionPriceLines: [],
      markerSets: {
        patterns: [], box: [], evolution: [], smaSignal: [], regression: [], fractals: [], fibSignal: [], futureLabel: [],
      },
      totalBars: null, viewKey: null, sorted: [], anchor: null,
    };

    const ro = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);

    // Never let the window drift entirely past the first bar: a later
    // refresh would faithfully re-pin that empty view forever.
    const MIN_FROM = -10;
    chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      const st = s.current;
      if (!range || !st) return;
      if (range.from < MIN_FROM) {
        chart.timeScale().setVisibleLogicalRange({ from: MIN_FROM, to: MIN_FROM + (range.to - range.from) });
        return;
      }
      if (st.totalBars !== null) setShowLive(st.totalBars - range.to > 5);
    });

    // Price readout under the cursor, read off the price scale so it always
    // matches the axis (and works over the forecast region too).
    chart.subscribeCrosshairMove((param) => {
      const tip = tooltipRef.current;
      if (!tip) return;
      if (!param.point) {
        tip.hidden = true;
        return;
      }
      const price = candle.coordinateToPrice(param.point.y);
      if (price === null) {
        tip.hidden = true;
        return;
      }
      tip.hidden = false;
      tip.textContent = fmtMoney(price, { smart: latest.current.smartPrice });
      const w = tip.offsetWidth || 90;
      let left = param.point.x + 14;
      if (left + w > el.clientWidth - 60) left = param.point.x - w - 14;
      tip.style.transform = `translate(${Math.max(4, left)}px, ${Math.max(4, param.point.y - 14)}px)`;
    });

    return () => {
      ro.disconnect();
      chart.remove();
      s.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Fewer bars on narrow screens so candles stay readable (~6px per bar). */
  function fitWindow() {
    const w = wrapRef.current?.clientWidth ?? 800;
    return Math.min(defaultWindow, Math.max(30, Math.floor(w / 6)));
  }

  function activeSeries() {
    const st = s.current!;
    return latest.current.chartType === "line" ? st.line : st.candle;
  }

  // The 61.8%-78.6% band traders watch most closely for a retracement to stall and reverse -
  // highlighted more strongly than the other bands, same idea as any charting tool's "golden
  // zone" shading.
  const FIB_GOLDEN_ZONE: [number, number] = [0.618, 0.786];

  function drawFib() {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    for (const l of st.fibLines) {
      try {
        st.candle.removePriceLine(l);
      } catch { /* already gone */ }
      try {
        st.line.removePriceLine(l);
      } catch { /* already gone */ }
    }
    st.fibLines = [];
    st.fibBandSeries.forEach((ser) => {
      try {
        st.chart.removeSeries(ser);
      } catch { /* already gone */ }
    });
    st.fibBandSeries = [];

    let fib: IndicatorSnapshot["fib"] = null;
    if (latest.current.indicators.fibonacci && st.sorted.length) {
      // Anchored to the most recent real swing (a confirmed local high/low pair), not to
      // whatever happens to be in the current zoom/scroll window - otherwise the levels would
      // jump around on every pan instead of marking an actual price move.
      const swing = findLastSwing(st.sorted);
      if (swing) {
        const { high, low } = swing;
        fib = { high, low };
        const levelPrice = (lv: number) => high - lv * (high - low);
        st.fibLines = FIB_LEVELS.map((lv) =>
          activeSeries().createPriceLine({
            price: levelPrice(lv), color: C.fib, lineWidth: 1, lineStyle: LineStyle.Dotted,
            axisLabelVisible: true, title: `${(lv * 100).toFixed(1)}%`,
          }),
        );

        // The swing itself, drawn as a line through its two pivots (in chronological order).
        const highTime = ts(st.sorted[swing.highIndex].timestamp);
        const lowTime = ts(st.sorted[swing.lowIndex].timestamp);
        const swingPts = swing.lowIndex < swing.highIndex
          ? [{ time: lowTime, value: low }, { time: highTime, value: high }]
          : [{ time: highTime, value: high }, { time: lowTime, value: low }];
        st.fibConnector.setData(swingPts);
        st.fibConnector.applyOptions({ visible: true });

        // A shaded band between every two consecutive levels, from the swing to "now" - the
        // Golden Zone shaded more strongly than the rest, same visual language as a charting
        // tool's "golden zone" highlight.
        const spanStart = Math.min(highTime, lowTime);
        const spanEnd = ts(st.sorted[st.sorted.length - 1].timestamp);
        if (spanEnd > spanStart) {
          for (let i = 0; i < FIB_LEVELS.length - 1; i++) {
            const a = FIB_LEVELS[i], b = FIB_LEVELS[i + 1];
            const top = levelPrice(Math.min(a, b));
            const bottom = levelPrice(Math.max(a, b));
            const golden = Math.min(a, b) === FIB_GOLDEN_ZONE[0] && Math.max(a, b) === FIB_GOLDEN_ZONE[1];
            const bandColor = golden ? C.pred : C.fib;
            const bandAlpha = golden ? 0.28 : 0.1;
            const band = st.chart.addBaselineSeries({
              priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
              lineWidth: 1, baseValue: { type: "price", price: bottom },
              topLineColor: "transparent", topFillColor1: withAlpha(bandColor, bandAlpha),
              topFillColor2: withAlpha(bandColor, bandAlpha),
              bottomLineColor: "transparent", bottomFillColor1: "transparent", bottomFillColor2: "transparent",
            });
            band.setData([{ time: spanStart as UTCTimestamp, value: top }, { time: spanEnd as UTCTimestamp, value: top }]);
            st.fibBandSeries.push(band);
          }
        }
      }
    }
    if (!fib) {
      st.fibConnector.setData([]);
      st.fibConnector.applyOptions({ visible: false });
    }
    return fib;
  }

  // Support/resistance key levels: horizontal lines at clustered fractal-pivot prices. Not
  // C.up/C.down - those are already the chart's language for live buy/sell direction (candles,
  // trend markers, entry signals), so reusing them here made these structural levels blend
  // straight into every other green/red element on the chart. Support yellow, resistance blue -
  // plus solid vs. dashed - so the two read apart at a glance.
  function drawLevels() {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    for (const l of st.srLines) {
      try {
        st.candle.removePriceLine(l);
      } catch { /* already gone */ }
      try {
        st.line.removePriceLine(l);
      } catch { /* already gone */ }
    }
    st.srLines = [];
    if (!latest.current.indicators.levels || !st.sorted.length) return;
    const { support, resistance } = supportResistanceLevels(st.sorted, st.sorted[st.sorted.length - 1].close);
    const series = activeSeries();
    for (const price of resistance) {
      st.srLines.push(series.createPriceLine({
        price, color: C.resistance, lineWidth: 1, lineStyle: LineStyle.Solid, axisLabelVisible: true, title: "R",
      }));
    }
    for (const price of support) {
      st.srLines.push(series.createPriceLine({
        price, color: C.support, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title: "S",
      }));
    }
  }

  // ---------------------------------------------------------------- data
  useEffect(() => {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    const { chart } = st;

    if (!candles.length) {
      for (const series of [st.candle, st.line, st.volume, st.pred, st.bandUp, st.bandLo]) series.setData([]);
      st.sorted = [];
      st.anchor = null;
      return;
    }

    const freshView = st.viewKey !== viewKey;
    let savedRange: { from: Time; to: Time } | null = null;
    let savedWidth: number | null = null;
    if (!freshView && st.totalBars !== null) {
      const lr = chart.timeScale().getVisibleLogicalRange();
      if (lr) {
        savedWidth = lr.to - lr.from;
        if (st.totalBars - lr.to > 5) savedRange = chart.timeScale().getVisibleRange();
      }
    }

    const sorted = [...candles].sort((a, b) => toUnix(a.timestamp) - toUnix(b.timestamp));
    st.sorted = sorted;
    st.candle.setData(sorted.map((c) => ({ time: ts(c.timestamp), open: c.open, high: c.high, low: c.low, close: c.close })));
    st.line.setData(sorted.map((c) => ({ time: ts(c.timestamp), value: c.close })));
    st.volume.setData(sorted.map((c) => ({ time: ts(c.timestamp), value: c.volume || 0, color: c.close >= c.open ? C.volUp : C.volDown })));
    const last = sorted[sorted.length - 1];
    st.anchor = { time: ts(last.timestamp), open: last.open, high: last.high, low: last.low };

    // Drop any forecast point that doesn't land strictly after the anchor (the last real
    // candle) - lightweight-charts requires strictly ascending time, and a forecast's first step
    // can coincide with the anchor's own timestamp right at a candle boundary, which would
    // otherwise crash setData with a duplicate-time assertion.
    const futurePredictions = predictions.filter((p) => ts(p.timestamp) > ts(last.timestamp));
    if (futurePredictions.length) {
      const a = { time: ts(last.timestamp), value: last.close };
      st.pred.setData([a, ...futurePredictions.map((p) => ({ time: ts(p.timestamp), value: p.predicted_price }))]);
      st.bandUp.setData([a, ...futurePredictions.map((p) => ({ time: ts(p.timestamp), value: p.upper }))]);
      st.bandLo.setData([a, ...futurePredictions.map((p) => ({ time: ts(p.timestamp), value: p.lower }))]);
    } else {
      st.pred.setData([]);
      st.bandUp.setData([]);
      st.bandLo.setData([]);
    }

    const total = sorted.length + predictions.length;
    if (freshView && fitAll) {
      chart.timeScale().setVisibleLogicalRange({ from: 0, to: total + 2 });
    } else if (savedRange) {
      chart.timeScale().setVisibleRange(savedRange);
    } else {
      const to = total + 2;
      const width = Math.min(savedWidth ?? fitWindow(), to);
      chart.timeScale().setVisibleLogicalRange({ from: to - width, to });
    }
    st.totalBars = total;
    st.viewKey = viewKey;
    applyIndicators();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candles, predictions, viewKey]);

  // ---------------------------------------------------------------- indicators / type
  function applyIndicators() {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    const ind = latest.current.indicators;
    const sorted = st.sorted;
    const times = sorted.map((c) => ts(c.timestamp));
    const closes = sorted.map((c) => c.close);
    const toData = (vals: (number | null)[]) =>
      vals.flatMap((v, i) => (v === null || v === undefined ? [] : [{ time: times[i], value: v }]));

    const isLine = latest.current.chartType === "line";
    st.candle.applyOptions({ visible: !isLine });
    st.line.applyOptions({ visible: isLine });

    st.sma.applyOptions({ visible: !!ind.sma });
    st.sma.setData(ind.sma ? toData(sma(closes, 20)) : []);
    st.ema.applyOptions({ visible: !!ind.ema });
    st.ema.setData(ind.ema ? toData(ema(closes, 20)) : []);
    const bb = ind.bollinger ? bollinger(closes, 20, 2) : null;
    for (const [series, key] of [[st.bbU, "upper"], [st.bbM, "middle"], [st.bbL, "lower"]] as const) {
      series.applyOptions({ visible: !!bb });
      series.setData(bb ? toData(bb[key]) : []);
    }
    const vw = ind.vwap ? vwap(sorted) : [];
    st.vwap.applyOptions({ visible: !!ind.vwap });
    st.vwap.setData(ind.vwap ? toData(vw) : []);

    // A trend-filtered Bollinger entry signal only means something with both indicators on
    // (the trend filter needs the SMA, the entry trigger needs the bands) - see
    // lib/indicators.ts::smaBollingerSignals.
    st.markerSets.smaSignal = ind.sma && bb
      ? smaBollingerSignals(sorted.map((c) => c.low), sorted.map((c) => c.high), closes, sma(closes, 20), bb.upper, bb.lower).flatMap(
          (sig, i) => (sig ? [{
            time: times[i], position: sig === "buy" ? ("belowBar" as const) : ("aboveBar" as const),
            color: sig === "buy" ? C.up : C.down, shape: sig === "buy" ? ("arrowUp" as const) : ("arrowDown" as const),
            text: "SMA+BB",
          }] : []),
        )
      : [];

    // Trend confirmation: three EMAs of different lengths, each one's own rolling
    // linear-regression slope, marked only where the state *changes* (a fresh confirmation),
    // not on every candle a trend continues - otherwise it would be marker noise, not a signal.
    if (ind.regression) {
      const slopes = [20, 50, 80].map((p) => rollingSlope(ema(closes, p), 7));
      const trend = confirmedTrend(slopes);
      st.markerSets.regression = trend.flatMap((state, i) => {
        if (!state || state === trend[i - 1]) return [];
        return [{
          time: times[i], position: state === "up" ? ("belowBar" as const) : ("aboveBar" as const),
          color: state === "up" ? C.up : C.down, shape: state === "up" ? ("arrowUp" as const) : ("arrowDown" as const),
          text: "Trend",
        }];
      });
    } else {
      st.markerSets.regression = [];
    }

    // Fractal pivots: raw local high/low candle comparison, distinct "square" shape so it
    // doesn't visually blend with the arrow markers above when both are on.
    st.markerSets.fractals = ind.fractals
      ? findFractals(sorted, 2).flatMap((kind, i) => (kind ? [{
          time: times[i], position: kind === "high" ? ("aboveBar" as const) : ("belowBar" as const),
          color: kind === "high" ? C.down : C.up, shape: "square" as const, text: "Fractal",
        }] : []))
      : [];

    // Fibonacci entry signal: part of the same "Fibonacci" toggle as the levels/bands below, not
    // a separate indicator - a trend-filtered retracement entry, own EMA-slope trend filter (not
    // gated on the separate "Trend (regression)" toggle, so it works whenever Fibonacci is on) -
    // see lib/indicators.ts::fibEntrySignals. Marked only where a fresh signal appears, not on
    // every candle it remains valid.
    if (ind.fibonacci) {
      const slopes = [20, 50, 80].map((p) => rollingSlope(ema(closes, p), 7));
      const trend = confirmedTrend(slopes);
      const signals = fibEntrySignals(sorted, trend);
      st.markerSets.fibSignal = signals.flatMap((sig, i) => {
        if (!sig || signals[i - 1]) return [];
        return [{
          time: times[i], position: sig.side === "long" ? ("belowBar" as const) : ("aboveBar" as const),
          color: sig.side === "long" ? C.up : C.down, shape: sig.side === "long" ? ("arrowUp" as const) : ("arrowDown" as const),
          text: "Fib Entry",
        }];
      });
    } else {
      st.markerSets.fibSignal = [];
    }

    // Forward outcome: a hindsight diagnostic, not a trading signal - marks candles that were
    // followed by a significant move, so a threshold choice can be eyeballed on the chart. See
    // lib/indicators.ts::futurePriceLabels.
    st.markerSets.futureLabel = ind.futureLabel
      ? futurePriceLabels(closes).flatMap((label, i) => (label ? [{
          time: times[i], position: label === "up" ? ("belowBar" as const) : ("aboveBar" as const),
          color: C.pattern, shape: label === "up" ? ("arrowUp" as const) : ("circle" as const),
          text: label === "up" ? "Fwd ↑" : "Fwd ↓",
        }] : []))
      : [];

    applyMarkers();

    const fib = drawFib();
    drawLevels();

    latest.current.onIndicatorSnapshot?.({
      price: closes.at(-1) ?? null,
      sma: ind.sma ? sma(closes, 20).at(-1) ?? null : null,
      ema: ind.ema ? ema(closes, 20).at(-1) ?? null : null,
      bollinger: bb && bb.upper.at(-1) != null ? { upper: bb.upper.at(-1)!, middle: bb.middle.at(-1)!, lower: bb.lower.at(-1)! } : null,
      vwap: ind.vwap ? vw.at(-1) ?? null : null,
      fib: fib ?? null,
    });
  }

  const indKey = JSON.stringify(indicators);
  useEffect(() => {
    applyIndicators();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indKey, chartType]);

  // ---------------------------------------------------------------- patterns
  useEffect(() => {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    for (const l of st.patLines) {
      try {
        st.candle.removePriceLine(l);
      } catch { /* gone */ }
    }
    st.patLines = [];
    if (!patterns) {
      st.markerSets.patterns = [];
      applyMarkers();
      st.patU.setData([]);
      st.patL.setData([]);
      st.patU.applyOptions({ visible: false });
      st.patL.applyOptions({ visible: false });
      return;
    }
    const markers: SeriesMarker<Time>[] = patterns.candlestick
      .map((p) => ({
        time: ts(p.timestamp),
        position: p.direction === "bearish" ? ("aboveBar" as const) : ("belowBar" as const),
        color: p.direction === "bullish" ? C.up : p.direction === "bearish" ? C.down : C.text,
        shape: p.direction === "bullish" ? ("arrowUp" as const) : p.direction === "bearish" ? ("arrowDown" as const) : ("circle" as const),
        text: p.name,
      }))
      .sort((a, b) => (a.time as number) - (b.time as number));
    st.markerSets.patterns = markers;
    applyMarkers();

    const top = patterns.geometric;
    const pt = (l: { timestamp: string; price: number }) => ({ time: ts(l.timestamp), value: l.price });
    const byTime = (a: { time: UTCTimestamp }, b: { time: UTCTimestamp }) => a.time - b.time;
    // A boundary line (triangle/wedge/channel side) stops dead at its last detected pivot
    // otherwise - extend it forward along its own slope (first pivot -> last pivot) so the
    // chart shows where the two lines are headed (e.g. a triangle's converging apex), not just
    // where they've already been. Horizon = how long the line took to form, a common and simple
    // projection convention, capped by that same span so it never runs away unbounded.
    const projectSlope = (points: { time: UTCTimestamp; value: number }[]) => {
      if (points.length < 2) return points;
      const first = points[0], last = points[points.length - 1];
      const span = (last.time as number) - (first.time as number);
      if (span <= 0) return points;
      const slope = (last.value - first.value) / span;
      const projTime = (last.time as number) + span;
      return [...points, { time: projTime as UTCTimestamp, value: last.value + slope * span }];
    };
    if (!top) {
      st.patU.setData([]);
      st.patL.setData([]);
      st.patU.applyOptions({ visible: false });
      st.patL.applyOptions({ visible: false });
    } else if (TWO_LINE_PATTERNS.has(top.name)) {
      const split = top.upper_count ?? Math.ceil(top.levels.length / 2);
      st.patU.setData(projectSlope(top.levels.slice(0, split).map(pt).sort(byTime)));
      st.patL.setData(projectSlope(top.levels.slice(split).map(pt).sort(byTime)));
      st.patU.applyOptions({ visible: true });
      st.patL.applyOptions({ visible: true });
    } else {
      st.patU.setData(top.levels.map(pt).sort(byTime));
      st.patU.applyOptions({ visible: true });
      st.patL.setData([]);
      st.patL.applyOptions({ visible: false });
    }
    if (top?.neckline != null) {
      st.patLines.push(st.candle.createPriceLine({ price: top.neckline, color: C.text, lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: patterns.necklineLabel }));
    }
    if (top?.target != null) {
      st.patLines.push(st.candle.createPriceLine({ price: top.target, color: top.direction === "bearish" ? C.down : C.up, lineWidth: 1, lineStyle: LineStyle.Dotted, axisLabelVisible: true, title: patterns.targetLabel }));
    }
  }, [patterns]);

  // ---------------------------------------------------------------- box breakout
  // Each box is a Baseline series whose base is the floor and whose line is
  // the ceiling, so the area between them fills; a separate line marks the
  // floor. Green/red: broke out up/down (grey: without volume, no trade),
  // accent: still active. Entries/exits are markers, the open trade's
  // entry/stop/target are price lines.
  useEffect(() => {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    const priceSeries = activeSeries();

    st.boxSeries.forEach((ser) => {
      try {
        st.chart.removeSeries(ser);
      } catch { /* already gone */ }
    });
    st.boxSeries = [];
    for (const l of st.boxPriceLines) {
      try {
        priceSeries.removePriceLine(l);
      } catch { /* already gone */ }
    }
    st.boxPriceLines = [];

    if (!box) {
      st.markerSets.box = [];
      applyMarkers();
      return;
    }

    const hidden = { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false };
    box.boxes.slice(-BOX_MAX_DRAWN).forEach((b) => {
      const color = b.breakout == null ? C.pred
        : !b.volume_confirmed ? C.text
        : b.breakout === "up" ? (box.significantUp === false ? C.text : C.up)
        : (box.significantDown === false ? C.text : C.down);
      const from = ts(b.start_time);
      const to = ts(b.breakout ? b.end_time : box.lastTime);
      if (to <= from) return;
      const fill = st.chart.addBaselineSeries({
        ...hidden, lineWidth: 1, baseValue: { type: "price", price: b.floor },
        topLineColor: color, topFillColor1: withAlpha(color, 0.16), topFillColor2: withAlpha(color, 0.16),
        bottomLineColor: color, bottomFillColor1: "transparent", bottomFillColor2: "transparent",
      });
      fill.setData([{ time: from, value: b.ceiling }, { time: to, value: b.ceiling }]);
      const floorLine = st.chart.addLineSeries({ ...hidden, color, lineWidth: 1 });
      floorLine.setData([{ time: from, value: b.floor }, { time: to, value: b.floor }]);
      st.boxSeries.push(fill, floorLine);
    });

    const markers: SeriesMarker<Time>[] = [];
    box.trades.forEach((tr) => {
      const long = tr.side === "long";
      markers.push({
        time: ts(tr.entry_time), position: long ? "belowBar" : "aboveBar",
        color: long ? C.up : C.down, shape: long ? "arrowUp" : "arrowDown", text: long ? "Long" : "Short",
      });
      if (tr.exit_time) {
        markers.push({
          time: ts(tr.exit_time), position: long ? "aboveBar" : "belowBar",
          color: tr.return_pct > 0 ? C.up : C.down, shape: "circle",
          text: `${BOX_EXIT_LABEL[tr.exit_reason ?? ""] ?? "Exit"} ${tr.return_pct > 0 ? "+" : ""}${tr.return_pct.toFixed(1)}%`,
        });
      }
    });
    st.markerSets.box = markers.sort((a, b) => (a.time as number) - (b.time as number));
    applyMarkers();

    const open = box.openTrade;
    if (open) {
      const line = (price: number, color: string, title: string) =>
        st.boxPriceLines.push(priceSeries.createPriceLine({ price, color, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title }));
      line(open.entry_price, C.pred, t("dashboard.box.entryTitle"));
      line(open.stop, C.down, t("dashboard.box.stopTitle"));
      if (open.target != null) line(open.target, C.up, t("dashboard.box.targetTitle"));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [box, chartType]);

  // ---------------------------------------------------------------- evolution fund trades
  // When the fund opened/closed a position - never a predicted price line. A still-open
  // position gets a live dashed price line at its entry, labeled with its running % (entry vs
  // the latest close), so "how is the current trade doing" is visible at a glance, not just
  // each closed trade's final result.
  useEffect(() => {
    const st = s.current;
    if (!st) return;
    const C = chartColors();
    const priceSeries = activeSeries();

    for (const l of st.evolutionPriceLines) {
      try {
        priceSeries.removePriceLine(l);
      } catch { /* already gone */ }
    }
    st.evolutionPriceLines = [];

    if (!evolutionTrades) {
      st.markerSets.evolution = [];
      applyMarkers();
      return;
    }

    const markers: SeriesMarker<Time>[] = [];
    evolutionTrades.trades.forEach((tr, i) => {
      const long = tr.side === "long";
      const open = !tr.exit_time;
      const livePct = open && evolutionTrades.lastClose != null
        ? (evolutionTrades.lastClose / tr.entry_price - 1) * (long ? 1 : -1) * 100 : null;
      // Entries are always blue (side is already shown by the arrow direction) and numbered -
      // several can land close together, the number says which is which. The invested amount
      // is what makes "how big a bet was this" visible, not just the direction.
      markers.push({
        time: ts(tr.entry_time), position: long ? "belowBar" : "aboveBar",
        color: C.sky, shape: long ? "arrowUp" : "arrowDown",
        text: `#${i + 1} ${long ? "Long" : "Short"} ${fmtEur(tr.entry_amount)}${livePct != null ? ` ${livePct >= 0 ? "+" : ""}${livePct.toFixed(1)}%` : ""}`,
      });
      if (tr.exit_time && tr.return_pct != null) {
        markers.push({
          time: ts(tr.exit_time), position: long ? "aboveBar" : "belowBar",
          color: tr.return_pct > 0 ? C.up : C.down, shape: "circle",
          text: `Exit ${tr.return_pct > 0 ? "+" : ""}${tr.return_pct.toFixed(1)}%`,
        });
      } else if (open) {
        const color = livePct == null ? C.pred : livePct >= 0 ? C.up : C.down;
        st.evolutionPriceLines.push(priceSeries.createPriceLine({
          price: tr.entry_price, color, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true,
          title: `#${i + 1} ${fmtEur(tr.entry_amount)}${livePct != null ? ` ${livePct >= 0 ? "+" : ""}${livePct.toFixed(1)}%` : ""}`,
        }));
      }
    });
    st.markerSets.evolution = markers.sort((a, b) => (a.time as number) - (b.time as number));
    applyMarkers();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evolutionTrades, chartType]);

  // ---------------------------------------------------------------- live tick
  useImperativeHandle(handleRef, () => ({
    tick(price: number) {
      const st = s.current;
      if (!st?.anchor) return;
      st.anchor.high = Math.max(st.anchor.high, price);
      st.anchor.low = Math.min(st.anchor.low, price);
      st.candle.update({ time: st.anchor.time, open: st.anchor.open, high: st.anchor.high, low: st.anchor.low, close: price });
      st.line.update({ time: st.anchor.time, value: price });
    },
  }));

  const jumpToLive = () => {
    const st = s.current;
    if (!st || st.totalBars === null) return;
    const w = Math.min(fitWindow(), st.totalBars);
    st.chart.timeScale().setVisibleLogicalRange({ from: st.totalBars - w, to: st.totalBars + 2 });
  };

  return (
    <div className={className ?? "relative h-full w-full"}>
      <div ref={wrapRef} className="absolute inset-0" />
      <div
        ref={tooltipRef}
        hidden
        className="num pointer-events-none absolute left-0 top-0 z-10 whitespace-nowrap rounded-xs border border-line-strong bg-raised/95 px-2 py-1 text-xs font-medium text-bone shadow-lg"
      />
      {showLive && (
        <button
          type="button"
          onClick={jumpToLive}
          className="absolute bottom-10 right-20 z-10 inline-flex items-center gap-1.5 rounded-full bg-accent px-3 py-1.5 text-xs font-semibold text-accent-ink shadow-lg shadow-accent/20 transition-transform hover:-translate-y-px animate-fade"
        >
          <Radio className="size-3.5" /> {t("dashboard.chart.liveJump")}
        </button>
      )}
    </div>
  );
}
