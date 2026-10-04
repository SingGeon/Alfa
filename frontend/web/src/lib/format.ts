// Shared number/time formatting. Previously copy-pasted (with drift)
// across app.js, scout.js, detail.js and the evaluation pages.

export function fmtMoney(v: number | null | undefined, opts: { smart?: boolean } = {}): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  let decimals = 2;
  if (opts.smart) {
    const abs = Math.abs(v);
    // Sub-cent assets (meme coins) need more decimals to show anything at all.
    if (abs > 0 && abs < 0.01) decimals = 6;
    else if (abs < 1) decimals = 4;
  }
  return `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

export function fmtCompactUsd(v: number): string {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(1)}M`;
  return `$${Math.round(v).toLocaleString("en-US")}`;
}

export function fmtPct(v: number | null | undefined, { signed = true, digits = 2 } = {}): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${signed && v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export const dir = (v: number | null | undefined): "up" | "down" | "flat" =>
  v === null || v === undefined || v === 0 ? "flat" : v > 0 ? "up" : "down";

/** Text color class for a signed value. */
export const dirText = (v: number | null | undefined) =>
  ({ up: "text-up", down: "text-down", flat: "text-dim" })[dir(v)];

/** 3-tier read for a 0-100 confidence score: <40 weak, <70 decent, else strong. */
export function quality(conf: number | null | undefined): "poor" | "decent" | "good" | null {
  if (conf === null || conf === undefined) return null;
  if (conf < 40) return "poor";
  if (conf < 70) return "decent";
  return "good";
}
export const qualityText = (conf: number | null | undefined) =>
  ({ poor: "text-down", decent: "text-warn", good: "text-up", none: "text-dim" })[quality(conf) ?? "none"];
export const qualityBg = (conf: number | null | undefined) =>
  ({ poor: "bg-down", decent: "bg-warn", good: "bg-up", none: "bg-faint" })[quality(conf) ?? "none"];

export function fmtDateTime(iso: string, locale: string, opts: Intl.DateTimeFormatOptions = {}): string {
  return new Date(iso).toLocaleString(locale, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", ...opts });
}

export function fmtDay(iso: string, locale: string): string {
  return new Date(iso).toLocaleDateString(locale, { day: "2-digit", month: "short" });
}

/** "YYYY-MM-DD" (a UTC day) -> localized date without shifting it across timezones. */
export function fmtUtcDay(day: string, locale: string, opts: Intl.DateTimeFormatOptions = {}): string {
  const [y, m, d] = day.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString(locale, {
    timeZone: "UTC", weekday: "short", day: "2-digit", month: "short", year: "numeric", ...opts,
  });
}

type T = (key: string, vars?: Record<string, string | number>) => string;

export function fmtRelative(iso: string, t: T): string {
  return fmtElapsed((Date.now() - new Date(iso).getTime()) / 1000, t);
}

export function fmtElapsed(seconds: number, t: T): string {
  const mins = Math.round(seconds / 60);
  if (mins < 1) return t("common.relative.justNow");
  if (mins < 60) return t("common.relative.minutesAgo", { n: mins });
  const hours = Math.round(mins / 60);
  if (hours < 24) return t("common.relative.hoursAgo", { n: hours });
  return t("common.relative.daysAgo", { n: Math.round(hours / 24) });
}

export const toUnix = (iso: string) => Math.floor(new Date(iso).getTime() / 1000);

/** Only ever render http(s) links from scraped content (RSS feeds). */
export function safeHref(url: string | undefined): string | undefined {
  if (!url) return undefined;
  try {
    const u = new URL(url);
    return u.protocol === "http:" || u.protocol === "https:" ? u.href : undefined;
  } catch {
    return undefined;
  }
}

export const userTimeZone = () => Intl.DateTimeFormat().resolvedOptions().timeZone;
