import clsx from "clsx";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { apiGet, type EvolutionTrade, type EvolutionTradesResponse } from "../../lib/api";
import { userTimeZone } from "../../lib/format";
import { useI18n } from "../../i18n";
import { Page } from "../../components/Shell";
import { Badge, Empty, ErrorBanner, ImageViewer, Pager, Panel, Skeleton } from "../../components/ui";
import { EvalTabs } from "./common";

const INTERVALS = ["15m", "1h", "4h", "1d", "1w"] as const;
type IntervalT = (typeof INTERVALS)[number];
const PAGE_SIZE = 24;

function fmtEur(v: number | null | undefined): string {
  return v == null ? "—" : `${Math.round(v).toLocaleString("en-US")} €`;
}
function fmtSignedPct(v: number | null | undefined, d = 2): string {
  return v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(d)}%`;
}

export default function Trades() {
  const { t, locale } = useI18n();
  const [interval, setInterval_] = useState<IntervalT>("1h");
  const [page, setPage] = useState(1);
  const [viewer, setViewer] = useState<{ src: string; alt: string } | null>(null);

  const pick = (i: IntervalT) => {
    setInterval_(i);
    setPage(1);
  };

  const query = useQuery({
    queryKey: ["evolution-trades", interval],
    queryFn: () => apiGet<EvolutionTradesResponse>("/api/evolution/trades", { interval }),
    refetchInterval: 60_000,
  });

  const trades = query.data?.trades ?? [];
  const closed = useMemo(() => trades.filter((tr) => tr.exit_time), [trades]);
  const wins = useMemo(() => closed.filter((tr) => (tr.return_pct ?? 0) > 0), [closed]);
  const avgReturn = closed.length ? closed.reduce((a, tr) => a + (tr.return_pct ?? 0), 0) / closed.length : null;
  const winRate = closed.length ? (wins.length / closed.length) * 100 : null;

  const open = query.data?.open_trade ?? null;
  const lastClose = query.data?.last_close ?? null;
  const openLivePct = open && lastClose != null ? (lastClose / open.entry_price - 1) * (open.side === "long" ? 1 : -1) * 100 : null;

  const newestFirst = useMemo(() => [...trades].reverse(), [trades]);
  const pages = Math.max(1, Math.ceil(newestFirst.length / PAGE_SIZE));
  const fmtTime = (iso: string) => new Date(iso).toLocaleString(locale, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });

  const groups = useMemo(() => {
    const slice = newestFirst.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
    const out: { key: string; title: string; rows: EvolutionTrade[] }[] = [];
    for (const r of slice) {
      const d = new Date(r.entry_time);
      const key = d.toDateString();
      const last = out[out.length - 1];
      if (last && last.key === key) last.rows.push(r);
      else out.push({ key, title: d.toLocaleDateString(locale, { weekday: "long", day: "2-digit", month: "long", year: "numeric" }), rows: [r] });
    }
    return out;
  }, [newestFirst, page, locale]);

  return (
    <Page kicker={t("nav.evaluation")} title={t("trades.title")} subtitle={t("trades.subtitle", { tz: userTimeZone() })}>
      <EvalTabs />
      {query.error && (
        <div className="mb-4">
          <ErrorBanner message={t("eval.error.load", { msg: query.error.message })} />
        </div>
      )}

      <section className="mb-6">
        <div className="eyebrow mb-3">{t("trades.filter.interval")}</div>
        <div className="flex gap-1.5 overflow-x-auto pb-1">
          {INTERVALS.map((i) => (
            <button
              key={i}
              type="button"
              onClick={() => pick(i)}
              className={clsx(
                "inline-flex shrink-0 items-center rounded-xl border px-3 py-1.5 text-[12.5px] font-medium transition-colors",
                interval === i ? "border-bone bg-bone text-page" : "border-line text-dim hover:border-line-strong hover:text-bone",
              )}
            >
              {i}
            </button>
          ))}
        </div>
      </section>

      <section className="mb-6">
        {!query.data ? (
          <Skeleton className="h-24" />
        ) : (
          <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[20px] border border-line bg-line sm:grid-cols-3 xl:grid-cols-5">
            <Tile label={t("trades.summary.total")} value={trades.length} />
            <Tile label={t("trades.summary.closed")} value={closed.length} />
            <Tile label={t("trades.summary.winRate")} value={winRate == null ? "—" : `${winRate.toFixed(1)}%`} valueClass={winRate == null ? "" : winRate >= 50 ? "text-up" : "text-down"} />
            <Tile label={t("trades.summary.avgReturn")} value={fmtSignedPct(avgReturn)} valueClass={avgReturn == null ? "" : avgReturn >= 0 ? "text-up" : "text-down"} />
            <Tile
              label={t("trades.summary.openPosition")}
              value={open ? <>{t(open.side === "long" ? "trades.card.long" : "trades.card.short")} {fmtEur(open.entry_amount)}</> : "—"}
              sub={open ? fmtSignedPct(openLivePct) : undefined}
              valueClass={openLivePct == null ? "" : openLivePct >= 0 ? "text-up" : "text-down"}
            />
          </div>
        )}
      </section>

      <Panel className="overflow-hidden">
        <div className="p-4 sm:p-5">
          {!query.data ? (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="aspect-[10/7]" />)}</div>
          ) : !groups.length ? (
            <Empty className="py-16">{t("trades.empty")}</Empty>
          ) : (
            <div className="space-y-8">
              {groups.map((g) => (
                <section key={g.key}>
                  <h3 className="mb-3 flex items-baseline gap-2 font-display text-[15px] font-semibold capitalize text-bone">
                    {g.title}
                    <span className="num text-xs font-normal text-faint">{g.rows.length}</span>
                  </h3>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {g.rows.map((tr) => (
                      <TradeCard key={tr.entry_time} tr={tr} fmtTime={fmtTime} onOpen={(src, alt) => setViewer({ src, alt })} />
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
        {trades.length > 0 && (
          <Pager page={page} pages={pages} info={t("eval.page", { page, pages, total: trades.length })} onPage={setPage} />
        )}
      </Panel>

      <ImageViewer src={viewer?.src ?? null} alt={viewer?.alt} onClose={() => setViewer(null)} />
    </Page>
  );
}

function Tile({ label, value, sub, valueClass }: { label: React.ReactNode; value: React.ReactNode; sub?: string; valueClass?: string }) {
  return (
    <div className="flex flex-col gap-1 bg-panel/90 px-5 py-4 backdrop-blur-md">
      <span className="eyebrow">{label}</span>
      <span className={clsx("num text-[22px] font-semibold leading-tight tracking-[-0.03em] text-bone", valueClass)}>{value}</span>
      {sub && <span className={clsx("num text-[11px] leading-snug", valueClass)}>{sub}</span>}
    </div>
  );
}

function TradeCard({ tr, fmtTime, onOpen }: { tr: EvolutionTrade; fmtTime: (iso: string) => string; onOpen: (src: string, alt: string) => void }) {
  const { t } = useI18n();
  const long = tr.side === "long";
  const closedTrade = !!tr.exit_time;
  const img = closedTrade ? tr.exit_snapshot_url : tr.entry_snapshot_url;
  const alt = `${long ? "Long" : "Short"} · ${fmtTime(tr.entry_time)}`;
  return (
    <figure
      onClick={() => img && onOpen(img, alt)}
      className={clsx(
        "group m-0 overflow-hidden rounded-xl border border-line bg-raised transition-all duration-200",
        img ? "cursor-pointer hover:-translate-y-0.5 hover:border-line-strong" : "cursor-default",
      )}
    >
      <div className="overflow-hidden bg-[#19171d]">
        {img ? (
          <img loading="lazy" src={img} alt="" className="block aspect-[10/6] w-full object-cover transition-transform duration-500 group-hover:scale-[1.02]" />
        ) : (
          <div className="flex aspect-[10/6] w-full items-center justify-center px-4 text-center text-[11px] text-faint">{t("trades.noSnapshotYet")}</div>
        )}
      </div>
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2.5 text-xs">
        <Badge tone={long ? "up" : "down"}>
          {long ? <ArrowUpRight className="size-3" /> : <ArrowDownRight className="size-3" />}
          {t(long ? "trades.card.long" : "trades.card.short")}
        </Badge>
        <span className="num text-faint">{fmtTime(tr.entry_time)}</span>
        {closedTrade ? (
          <span className={clsx("num ml-auto", (tr.return_pct ?? 0) >= 0 ? "text-up" : "text-down")}>{fmtSignedPct(tr.return_pct)}</span>
        ) : (
          <span className="num ml-auto text-warn">{t("trades.card.open")}</span>
        )}
      </figcaption>
    </figure>
  );
}
