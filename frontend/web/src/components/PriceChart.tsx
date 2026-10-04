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
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { Radio } from "lucide-react";
import type { Candle, GeometricPattern, CandlePattern, PredictionPoint } from "../lib/api";
import { bollinger, ema, FIB_LEVELS, sma, vwap } from "../lib/indicators";
import { chartColors, CHART_FONT } from "../lib/theme";
import { fmtMoney, toUnix } from "../lib/format";
import { useI18n } from "../i18n";

export type IndicatorKey = "sma" | "ema" | "bollinger" | "fibonacci" | "vwap";
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

export function PriceChart({
  candles,
  predictions = [],
  chartType = "candles",
  indicators = {},
  patterns = null,
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
    patLines: IPriceLine[];
    totalBars: number | null;
    viewKey: string | null;
    sorted: Candle[];
    anchor: { time: UTCTimestamp; open: number; high: number; low: number } | null;
  } | null>(null);

  const latest = useRef({ indicators, chartType, onIndicatorSnapshot, smartPrice, locale });
  latest.current = { indicators, chartType, onIndicatorSnapshot, smartPrice, locale };

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
    // Confidence band: an area fading to fully transparent (never an opaque
    // fill that would blank the grid below the band), lower bound dashed.
    const bandUp = chart.addAreaSeries({
      lineWidth: 1, lineStyle: LineStyle.Dashed, lineColor: C.pred, topColor: C.band, bottomColor: C.bandFade, ...quiet,
    });
    const bandLo = chart.addLineSeries({ color: C.pred, lineWidth: 1, lineStyle: LineStyle.Dashed, ...quiet });
    const pred = chart.addLineSeries({ color: C.pred, lineWidth: 2, lastValueVisible: true, priceLineVisible: false });

    s.current = {
      chart, candle, line, volume, pred, bandUp, bandLo, sma: smaS, ema: emaS, bbU, bbM, bbL, vwap: vwapS, patU, patL,
      fibLines: [], patLines: [], totalBars: null, viewKey: null, sorted: [], anchor: null,
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
      if (latest.current.indicators.fibonacci) drawFib();
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
    let fib: IndicatorSnapshot["fib"] = null;
    if (latest.current.indicators.fibonacci && st.sorted.length) {
      const range = st.chart.timeScale().getVisibleLogicalRange();
      const n = st.sorted.length;
      const from = Math.max(0, Math.floor(range ? range.from : 0));
      const to = Math.min(n - 1, Math.ceil(range ? range.to : n - 1));
      const visible = from <= to ? st.sorted.slice(from, to + 1) : st.sorted;
      if (visible.length) {
        const high = Math.max(...visible.map((c) => c.high));
        const low = Math.min(...visible.map((c) => c.low));
        fib = { high, low };
        st.fibLines = FIB_LEVELS.map((lv) =>
          activeSeries().createPriceLine({
            price: high - lv * (high - low), color: C.fib, lineWidth: 1, lineStyle: LineStyle.Dotted,
            axisLabelVisible: true, title: `${(lv * 100).toFixed(1)}%`,
          }),
        );
      }
    }
    return fib;
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

    if (predictions.length) {
      const a = { time: ts(last.timestamp), value: last.close };
      st.pred.setData([a, ...predictions.map((p) => ({ time: ts(p.timestamp), value: p.predicted_price }))]);
      st.bandUp.setData([a, ...predictions.map((p) => ({ time: ts(p.timestamp), value: p.upper }))]);
      st.bandLo.setData([a, ...predictions.map((p) => ({ time: ts(p.timestamp), value: p.lower }))]);
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
    const fib = drawFib();

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
      st.candle.setMarkers([]);
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
    st.candle.setMarkers(markers);

    const top = patterns.geometric;
    const pt = (l: { timestamp: string; price: number }) => ({ time: ts(l.timestamp), value: l.price });
    const byTime = (a: { time: UTCTimestamp }, b: { time: UTCTimestamp }) => a.time - b.time;
    if (!top) {
      st.patU.setData([]);
      st.patL.setData([]);
      st.patU.applyOptions({ visible: false });
      st.patL.applyOptions({ visible: false });
    } else if (TWO_LINE_PATTERNS.has(top.name)) {
      const split = top.upper_count ?? Math.ceil(top.levels.length / 2);
      st.patU.setData(top.levels.slice(0, split).map(pt).sort(byTime));
      st.patL.setData(top.levels.slice(split).map(pt).sort(byTime));
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
