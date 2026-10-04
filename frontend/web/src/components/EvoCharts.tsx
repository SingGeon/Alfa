// Shared chart plumbing for the Population and Fund pages (ml/evolution.py,
// ml/strategy_fund.py): small lightweight-charts panes that stay in sync
// (time range + crosshair), and the little hover sparklines used for gene
// and trait history. Kept separate from PriceChart.tsx, which is tuned for
// the candlestick + indicators case.
import { useEffect, useRef, useState } from "react";
import { CrosshairMode, createChart, type IChartApi } from "lightweight-charts";
import { chartColors, CHART_FONT } from "../lib/theme";

export function evoChartOptions(el: HTMLElement) {
  const C = chartColors();
  return {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: { background: { color: "transparent" }, textColor: C.text, fontFamily: CHART_FONT, fontSize: 11 },
    grid: { vertLines: { visible: false }, horzLines: { color: C.grid } },
    rightPriceScale: { borderVisible: false, minimumWidth: 70 },
    timeScale: { borderVisible: false, timeVisible: true, secondsVisible: false },
    crosshair: { mode: CrosshairMode.Magnet },
    handleScroll: true,
    handleScale: true,
  };
}

/** Keeps N chart panes' visible range and crosshair in lockstep. Returns a
 *  cleanup function. `onCrosshair` fires with the hovered unix time (or the
 *  last point's time when the cursor leaves the chart). */
export function syncPanes(charts: IChartApi[], onCrosshair: (time: number | null) => void) {
  let syncing = false;
  const unsubs = charts.map((c) => {
    const onRange = (range: { from: number; to: number } | null) => {
      if (syncing || !range) return;
      syncing = true;
      for (const o of charts) if (o !== c) o.timeScale().setVisibleLogicalRange(range);
      syncing = false;
    };
    c.timeScale().subscribeVisibleLogicalRangeChange(onRange);
    const onMove = (param: { time?: unknown }) => onCrosshair((param.time as number | undefined) ?? null);
    c.subscribeCrosshairMove(onMove);
    return () => {
      c.timeScale().unsubscribeVisibleLogicalRangeChange(onRange);
      c.unsubscribeCrosshairMove(onMove);
    };
  });
  return () => unsubs.forEach((u) => u());
}

export function useChartResize(ref: React.RefObject<HTMLDivElement | null>, chart: IChartApi | null) {
  useEffect(() => {
    const el = ref.current;
    if (!el || !chart) return;
    const ro = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, chart]);
}

// --------------------------------------------------------------- Sparkline

interface SparklineProps {
  title: string;
  values: (number | null | undefined)[];
  times: string[];
  range?: [number, number] | null;
  mid?: number | null;
  format?: (v: number | null) => string;
  fmtDay: (iso: string) => string;
}

/** A tiny hoverable line chart (not lightweight-charts - plain SVG is
 *  plenty for ~100 points and it's far cheaper to mount N of these). */
export function Sparkline({ title, values, times, range = null, mid = null, format, fmtDay }: SparklineProps) {
  const finite = values.filter((v): v is number => v != null && Number.isFinite(v));
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  if (!finite.length) return null;

  const lo = range ? range[0] : Math.min(...finite);
  const hi = range ? range[1] : Math.max(...finite);
  const span = hi - lo || 1;
  const W = 300, H = 56;
  const x = (i: number) => (values.length > 1 ? (i / (values.length - 1)) * W : 0);
  const y = (v: number) => H - ((v - lo) / span) * (H - 4) - 2;
  const fmt = format ?? ((v: number | null) => (v == null ? "—" : v.toFixed(2)));
  const points = values.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean).join(" ");
  const last = finite[finite.length - 1];
  const shown = hover != null ? values[hover] ?? null : last;
  const when = hover != null ? times[hover] : times[times.length - 1];

  const onMove: React.MouseEventHandler<SVGSVGElement> = (e) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const i = Math.round(((e.clientX - rect.left) / rect.width) * (values.length - 1));
    setHover(Math.max(0, Math.min(values.length - 1, i)));
  };

  return (
    <div className="panel flex flex-col gap-1.5 p-3.5">
      <div className="flex items-baseline justify-between gap-2 text-[12px]">
        <span className="min-w-0 truncate text-faint">{title}</span>
        <b className="num shrink-0 text-bone">{fmt(shown)}</b>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={title}
        className="h-14 w-full cursor-crosshair overflow-visible"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {range && <polygon points={`0,${H} ${points} ${W},${H}`} fill="var(--color-accent)" opacity={0.08} />}
        {mid != null && (
          <line x1={0} x2={W} y1={y(mid)} y2={y(mid)} stroke="var(--color-line-strong)" strokeWidth={1} strokeDasharray="3 3" />
        )}
        <polyline points={points} fill="none" stroke="var(--color-accent-hi)" strokeWidth={1.75} />
        {hover != null && (
          <line x1={x(hover)} x2={x(hover)} y1={0} y2={H} stroke="var(--color-line-strong)" strokeWidth={1} />
        )}
      </svg>
      <div className="flex justify-between text-[10.5px] text-faint">
        <span>{fmtDay(times[0])}</span>
        <span>{fmtDay(when)}</span>
      </div>
    </div>
  );
}

export { createChart };
