import clsx from "clsx";
import { useEffect, useId, useRef, useState } from "react";
import type { AssetType } from "../lib/api";
import { EthGlyph } from "./Shell";

/** Small area sparkline that draws itself in. */
export function Sparkline({ values, className, tone }: { values: number[]; className?: string; tone?: "up" | "down" | "accent" }) {
  const id = useId();
  if (values.length < 2) return <span className={clsx("block", className)} />;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const W = 100;
  const H = 32;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * W, H - 2 - ((v - min) / span) * (H - 4)]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join("");
  const t = tone ?? (values.at(-1)! >= values[0] ? "up" : "down");
  const color = t === "up" ? "var(--color-up)" : t === "down" ? "var(--color-down)" : "var(--color-accent-hi)";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className={clsx("block overflow-visible", className)} aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.28" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={`${line}L${W},${H}L0,${H}Z`} fill={`url(#${id})`} className="animate-fade" />
      <path d={line} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" pathLength={1} className="spark-draw" />
    </svg>
  );
}

const TILE_TONES = [
  "from-sky/30 to-sky/5 text-sky",
  "from-rose/30 to-rose/5 text-rose",
  "from-warn/30 to-warn/5 text-warn",
  "from-up/30 to-up/5 text-up",
  "from-violet/30 to-violet/5 text-violet",
];

/** Rounded gradient tile with the asset's initials (no third-party logos). */
export function TokenIcon({ symbol, type, size = "md", eth = false }: { symbol: string; type?: AssetType; size?: "sm" | "md" | "lg"; eth?: boolean }) {
  const hash = [...symbol].reduce((a, c) => a + c.charCodeAt(0), 0);
  const tone = eth ? "from-accent/50 to-accent/10 text-white" : TILE_TONES[(hash + (type === "stock" ? 1 : 0)) % TILE_TONES.length];
  return (
    <span
      className={clsx(
        "flex shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br font-semibold shadow-[inset_0_1px_0_rgb(255_255_255/0.15)]",
        size === "sm" && "size-8 rounded-xl text-[10px]",
        size === "md" && "size-11 text-xs",
        size === "lg" && "size-14 text-sm",
        tone,
      )}
    >
      {eth ? <EthGlyph className={size === "lg" ? "size-7" : size === "sm" ? "size-4" : "size-5"} /> : symbol.slice(0, 3)}
    </span>
  );
}

/** Large price with dimmed decimals ($110,973.64) that flashes green/red on change. */
export function BigPrice({ value, className, smart = false }: { value: number | null | undefined; className?: string; smart?: boolean }) {
  const prev = useRef<number | null>(null);
  const [flash, setFlash] = useState<"up" | "down" | null>(null);
  useEffect(() => {
    if (value == null) return;
    if (prev.current != null && value !== prev.current) {
      setFlash(value > prev.current ? "up" : "down");
      const id = setTimeout(() => setFlash(null), 1000);
      prev.current = value;
      return () => clearTimeout(id);
    }
    prev.current = value;
  }, [value]);
  if (value == null) return <span className={clsx("num text-faint", className)}>—</span>;
  const abs = Math.abs(value);
  const decimals = smart ? (abs > 0 && abs < 0.01 ? 6 : abs < 1 ? 4 : 2) : 2;
  const [int, dec] = value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).split(".");
  return (
    <span key={flash ? `${flash}-${value}` : "idle"} className={clsx("num", flash === "up" && "animate-flash-up", flash === "down" && "animate-flash-down", className)}>
      ${int}
      <span className="opacity-40">.{dec}</span>
    </span>
  );
}

/** Round arrow button used on token cards. */
export function ArrowCircle({ className }: { className?: string }) {
  return (
    <span className={clsx("flex size-9 shrink-0 items-center justify-center rounded-full border border-line bg-white/[0.04] text-dim transition-all duration-300 group-hover:rotate-45 group-hover:border-accent/50 group-hover:bg-accent/20 group-hover:text-bone", className)}>
      <svg viewBox="0 0 24 24" className="size-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        <path d="M7 17 17 7M8 7h9v9" />
      </svg>
    </span>
  );
}
