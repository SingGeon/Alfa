import { useEffect, useRef } from "react";
import { createChart, LineStyle, type IChartApi, type ISeriesApi, type Time, type UTCTimestamp } from "lightweight-charts";
import { chartColors, CHART_FONT } from "../lib/theme";
import { toUnix } from "../lib/format";
import { useI18n } from "../i18n";

type Pt = { timestamp: string; value: number };

/** Predicted-vs-actual history: real price, this model's predictions and
 *  the "tuned" variant's own predictions for comparison. */
export function AccuracyChart({ actual, predicted, other }: { actual: Pt[]; predicted: Pt[]; other: Pt[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const { locale } = useI18n();
  const st = useRef<{ chart: IChartApi; a: ISeriesApi<"Line">; p: ISeriesApi<"Line">; o: ISeriesApi<"Line"> } | null>(null);
  const localeRef = useRef(locale);
  localeRef.current = locale;

  useEffect(() => {
    const el = ref.current!;
    const C = chartColors();
    const chart = createChart(el, {
      width: el.clientWidth,
      height: el.clientHeight,
      layout: { background: { color: "transparent" }, textColor: C.text, fontFamily: CHART_FONT, fontSize: 11 },
      grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
      rightPriceScale: { borderColor: C.border },
      timeScale: { borderColor: C.border, timeVisible: false },
      localization: {
        timeFormatter: (t: Time) => new Date((t as number) * 1000).toLocaleString(localeRef.current, { day: "2-digit", month: "short", year: "numeric" }),
      },
      handleScroll: { mouseWheel: false, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
      handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: true },
    });
    const a = chart.addLineSeries({ color: C.bollinger, lineWidth: 2, priceLineVisible: false });
    const p = chart.addLineSeries({ color: C.pred, lineWidth: 2, lineStyle: LineStyle.Dashed, priceLineVisible: false });
    const o = chart.addLineSeries({ color: C.sky, lineWidth: 1, lineStyle: LineStyle.Dotted, priceLineVisible: false, lastValueVisible: false });
    st.current = { chart, a, p, o };
    const ro = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      chart.remove();
      st.current = null;
    };
  }, []);

  useEffect(() => {
    const s = st.current;
    if (!s) return;
    const map = (pts: Pt[]) => {
      // de-dup by second (setData requires strictly ascending times)
      const seen = new Map<number, number>();
      for (const p of pts) seen.set(toUnix(p.timestamp), p.value);
      return [...seen.entries()].sort((x, y) => x[0] - y[0]).map(([time, value]) => ({ time: time as UTCTimestamp, value }));
    };
    s.a.setData(map(actual));
    s.p.setData(map(predicted));
    s.o.setData(map(other));
    s.chart.timeScale().fitContent();
  }, [actual, predicted, other]);

  return <div ref={ref} className="h-full w-full" />;
}
