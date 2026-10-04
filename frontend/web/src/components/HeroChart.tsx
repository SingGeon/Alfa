import { useMemo } from "react";
import type { Candle, PredictionPoint } from "../lib/api";

const W = 640;
const H = 300;
const PAD = { t: 18, r: 64, b: 22, l: 4 };

/** Synthetic random walk, used only when the API has no data yet (fresh
 *  install, backend still seeding) so the hero never renders empty. */
function demoSeries(): { closes: number[]; preds: { p: number; lo: number; hi: number }[] } {
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) - 0.5;
  const closes: number[] = [];
  let v = 3200;
  for (let i = 0; i < 96; i++) {
    v *= 1 + rnd() * 0.012 + 0.0004;
    closes.push(v);
  }
  const preds = Array.from({ length: 24 }, (_, i) => {
    const p = v * (1 + 0.0009 * (i + 1));
    const w = v * 0.0025 * Math.sqrt(i + 1);
    return { p, lo: p - w, hi: p + w };
  });
  return { closes, preds };
}

export function HeroChart({ candles, predictions }: { candles: Candle[] | undefined; predictions: PredictionPoint[] | undefined }) {
  const geo = useMemo(() => {
    const live = candles && candles.length > 10;
    const closes = live ? [...candles!].sort((a, b) => +new Date(a.timestamp) - +new Date(b.timestamp)).map((c) => c.close) : demoSeries().closes;
    const preds = live
      ? (predictions ?? []).map((p) => ({ p: p.predicted_price, lo: p.lower, hi: p.upper }))
      : demoSeries().preds;
    const all = [...closes, ...preds.flatMap((p) => [p.lo, p.hi])];
    const min = Math.min(...all);
    const max = Math.max(...all);
    const span = max - min || 1;
    const n = closes.length + preds.length;
    const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
    const y = (v: number) => PAD.t + (1 - (v - min) / span) * (H - PAD.t - PAD.b);
    const histPath = closes.map((c, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(c).toFixed(1)}`).join("");
    const lastI = closes.length - 1;
    const last = closes[lastI];
    const areaPath = `${histPath}L${x(lastI).toFixed(1)},${H - PAD.b}L${x(0).toFixed(1)},${H - PAD.b}Z`;
    const pAt = (i: number) => x(lastI + 1 + i);
    const predPath = preds.length ? `M${x(lastI)},${y(last)}` + preds.map((p, i) => `L${pAt(i).toFixed(1)},${y(p.p).toFixed(1)}`).join("") : "";
    const cone = preds.length
      ? `M${x(lastI)},${y(last)}` +
        preds.map((p, i) => `L${pAt(i).toFixed(1)},${y(p.hi).toFixed(1)}`).join("") +
        [...preds].reverse().map((p, ri) => `L${pAt(preds.length - 1 - ri).toFixed(1)},${y(p.lo).toFixed(1)}`).join("") +
        "Z"
      : "";
    const gridY = Array.from({ length: 4 }, (_, i) => min + (span * (i + 0.5)) / 4);
    const final = preds.at(-1);
    return { live, histPath, areaPath, predPath, cone, nowX: x(lastI), nowY: y(last), last, gridY, y, final, endX: final ? pAt(preds.length - 1) : 0 };
  }, [candles, predictions]);

  const fmt = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full overflow-visible" role="img" aria-label="ETH price with AI forecast">
      <defs>
        <linearGradient id="hc-area" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="var(--color-bone)" stopOpacity="0.10" />
          <stop offset="1" stopColor="var(--color-bone)" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="hc-cone" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor="var(--color-accent)" stopOpacity="0.35" />
          <stop offset="1" stopColor="var(--color-accent)" stopOpacity="0.08" />
        </linearGradient>
        <filter id="hc-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>

      {geo.gridY.map((g) => (
        <g key={g}>
          <line x1={PAD.l} x2={W - PAD.r} y1={geo.y(g)} y2={geo.y(g)} stroke="var(--color-line)" strokeDasharray="2 4" />
          <text x={W - PAD.r + 10} y={geo.y(g) + 3.5} className="num" fontSize="10" fill="var(--color-faint)">{fmt(g)}</text>
        </g>
      ))}

      {/* "now" divider */}
      <line x1={geo.nowX} x2={geo.nowX} y1={PAD.t - 6} y2={H - PAD.b} stroke="var(--color-line-strong)" strokeDasharray="3 3" />
      <text x={geo.nowX + 6} y={PAD.t - 2} fontSize="9.5" letterSpacing="1.2" fill="var(--color-faint)" className="num">FORECAST →</text>

      <path d={geo.areaPath} fill="url(#hc-area)" className="hc-fade" />
      <path d={geo.histPath} fill="none" stroke="var(--color-bone)" strokeWidth="1.6" strokeLinejoin="round" pathLength={1} className="hc-draw" />

      {geo.cone && <path d={geo.cone} fill="url(#hc-cone)" className="hc-cone" />}
      {geo.predPath && (
        <>
          <path d={geo.predPath} fill="none" stroke="var(--color-accent)" strokeWidth="5" opacity="0.35" filter="url(#hc-glow)" className="hc-cone" />
          <path d={geo.predPath} fill="none" stroke="var(--color-accent)" strokeWidth="2" strokeDasharray="5 4" className="hc-cone" />
        </>
      )}

      <circle cx={geo.nowX} cy={geo.nowY} r="9" fill="var(--color-bone)" opacity="0.12" className="hc-ping" />
      <circle cx={geo.nowX} cy={geo.nowY} r="3.5" fill="var(--color-bone)" className="hc-fade" />

      {geo.final && (
        <g className="hc-cone">
          <circle cx={geo.endX} cy={geo.y(geo.final.p)} r="3.5" fill="var(--color-accent)" />
          <rect x={geo.endX + 8} y={geo.y(geo.final.p) - 10} width="52" height="20" rx="4" fill="var(--color-accent)" />
          <text x={geo.endX + 34} y={geo.y(geo.final.p) + 3.5} textAnchor="middle" fontSize="10.5" fontWeight="600" fill="var(--color-accent-ink)" className="num">
            {fmt(geo.final.p)}
          </text>
        </g>
      )}

      <style>{`
        .hc-draw { stroke-dasharray: 1; stroke-dashoffset: 1; animation: hc-draw 1.8s cubic-bezier(.22,1,.36,1) .2s forwards; }
        .hc-fade { opacity: 0; animation: hc-fadein .8s ease-out 1.1s forwards; }
        .hc-cone { opacity: 0; animation: hc-fadein .9s ease-out 1.7s forwards; }
        .hc-ping { transform-origin: ${geo.nowX}px ${geo.nowY}px; animation: hc-ping 2.4s ease-out 2s infinite; opacity: 0; }
        @keyframes hc-draw { to { stroke-dashoffset: 0; } }
        @keyframes hc-fadein { to { opacity: 1; } }
        @keyframes hc-ping { 0% { transform: scale(.6); opacity: .5 } 100% { transform: scale(2.4); opacity: 0 } }
        @media (prefers-reduced-motion: reduce) { .hc-draw { stroke-dashoffset: 0 } .hc-fade, .hc-cone { opacity: 1 } }
      `}</style>
    </svg>
  );
}
