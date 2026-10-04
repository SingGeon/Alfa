import clsx from "clsx";
import { ArrowUpRight } from "lucide-react";
import type { NewsItem, PredictionPoint } from "../lib/api";
import { dirText, fmtDateTime, fmtMoney, fmtRelative, safeHref } from "../lib/format";
import { useI18n } from "../i18n";
import { Delta, Empty, SentimentTag, Skeleton } from "./ui";

export function NewsList({ news, loading, emptyText, className }: { news: NewsItem[] | undefined; loading?: boolean; emptyText: string; className?: string }) {
  const { t, locale } = useI18n();
  if (loading && !news) {
    return (
      <div className="space-y-3 p-1">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="space-y-2">
            <Skeleton className="h-3.5 w-11/12" />
            <Skeleton className="h-3 w-1/3" />
          </div>
        ))}
      </div>
    );
  }
  if (!news?.length) return <Empty>{emptyText}</Empty>;
  return (
    <ul className={clsx("divide-y divide-line", className)}>
      {news.map((a, i) => {
        const href = safeHref(a.url);
        const label = a.sentiment?.label;
        return (
          <li key={`${a.url}-${i}`} className="group relative py-3 first:pt-0 last:pb-0">
            <span
              className={clsx(
                "absolute -left-3 top-3 bottom-3 w-px group-first:top-0 sm:-left-4",
                label === "positive" ? "bg-up/60" : label === "negative" ? "bg-down/60" : "bg-line-strong",
              )}
              aria-hidden
            />
            {href ? (
              <a href={href} target="_blank" rel="noopener noreferrer" className="flex items-start gap-1.5 text-[13.5px] font-medium leading-snug text-bone hover:text-accent-hi">
                <span className="flex-1">{a.title}</span>
                <ArrowUpRight className="mt-0.5 size-3.5 shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100" />
              </a>
            ) : (
              <span className="text-[13.5px] font-medium leading-snug text-bone">{a.title}</span>
            )}
            <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-faint">
              <SentimentTag label={label} compound={a.sentiment?.compound} />
              {a.source && <span>{a.source}</span>}
              {a.published_at && (
                <span className="num" title={fmtDateTime(a.published_at, locale)}>
                  {fmtRelative(a.published_at, t)}
                </span>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/** Forecast path table: one row per predicted step vs. the base price. */
export function ForecastTable({
  predictions,
  basePrice,
  formatTime,
  timeHeader,
  smart = false,
}: {
  predictions: PredictionPoint[];
  basePrice: number;
  formatTime: (iso: string) => string;
  timeHeader: string;
  smart?: boolean;
}) {
  const { t } = useI18n();
  if (!predictions.length) return <Empty>—</Empty>;
  return (
    <table className="w-full border-collapse text-[12.5px]">
      <thead className="sticky top-0 z-[1] bg-panel">
        <tr className="text-left text-[11px] text-faint">
          <th className="py-2 pl-4 pr-2 font-medium sm:pl-5">{timeHeader}</th>
          <th className="px-2 py-2 text-right font-medium">{t("common.prediction.colPrice")}</th>
          <th className="py-2 pl-2 pr-4 text-right font-medium sm:pr-5">{t("common.prediction.colVsNow")}</th>
        </tr>
      </thead>
      <tbody>
        {predictions.map((p) => {
          const pct = ((p.predicted_price - basePrice) / basePrice) * 100;
          return (
            <tr key={p.timestamp} className="border-t border-line transition-colors hover:bg-raised">
              <td className="num py-1.5 pl-4 pr-2 text-dim sm:pl-5">{formatTime(p.timestamp)}</td>
              <td className={clsx("num px-2 py-1.5 text-right font-medium", dirText(pct))}>{fmtMoney(p.predicted_price, { smart })}</td>
              <td className="py-1.5 pl-2 pr-4 text-right sm:pr-5">
                <Delta value={pct} className="text-[12px]" />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export function Legend({ items }: { items: { color: string; label: string; dashed?: boolean; soft?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11.5px] text-faint">
      {items.map((it) => (
        <span key={it.label} className="inline-flex items-center gap-1.5">
          {it.soft ? (
            <span className="h-2.5 w-3.5 rounded-[2px]" style={{ background: it.color, opacity: 0.35 }} />
          ) : (
            <span className="w-3.5 border-t-2" style={{ borderColor: it.color, borderStyle: it.dashed ? "dashed" : "solid" }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  );
}
