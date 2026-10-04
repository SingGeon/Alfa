// Model population page: everything about the evolving "tuned" population
// (ml/evolution.py, served by /api/evolution/*). Ported from the vanilla
// frontend's population.html/js - same data, same interactions, redrawn
// with the design-system primitives.
import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { X } from "lucide-react";
import { LineStyle, PriceScaleMode, type UTCTimestamp } from "lightweight-charts";
import {
  apiGet, type AncestorNode, type EvolutionDeathsResponse, type EvolutionOrganismResponse,
  type EvolutionReport, type EvolutionTimelinePoint,
} from "../lib/api";
import { toUnix } from "../lib/format";
import { useI18n, type TFn } from "../i18n";
import { Page } from "../components/Shell";
import { Badge, Button, Empty, ErrorBanner, Pager, Panel, PanelHeader, Segmented, Skeleton } from "../components/ui";
import { createChart, evoChartOptions, Sparkline, syncPanes } from "../components/EvoCharts";

type Interval = "15m" | "1h" | "4h" | "1d" | "1w";
const INTERVALS: Interval[] = ["15m", "1h", "4h", "1d", "1w"];
const PAGE_SIZE = 50;
const GENE_ORDER = ["momentum", "trend", "long", "volatility", "volume", "oscillators", "btc", "sentiment", "news", "defi", "futures"];
const TRAITS: { key: keyof EvolutionTimelinePoint; digits: number; range: [number, number] | null; mid?: number }[] = [
  { key: "boldness", digits: 2, range: [0, 2] },
  { key: "temperament", digits: 2, range: [0, 0.6] },
  { key: "news_sensitivity", digits: 2, range: [-1, 1], mid: 0 },
  { key: "window", digits: 0, range: null },
  { key: "patience", digits: 2, range: [0, 5] },
  { key: "avg_generation", digits: 0, range: null },
];
const EMOJI: Record<string, string> = { fear: "😨", caution: "🤔", calm: "😐", confidence: "🙂", euphoria: "🤩" };
const CAUSE_TONE: Record<string, "neutral" | "down" | "warn"> = { outcompeted: "neutral", big_loss: "down", fees: "warn", bad_trades: "down", taxes: "warn" };

type Filters = { cause: string; generation: string; generationTo: string; organism: string; order: "asc" | "desc" };
const EMPTY_FILTERS: Filters = { cause: "", generation: "", generationTo: "", organism: "", order: "desc" };

function makeFmt(locale: string) {
  const num = (v: number | null | undefined, d = 0) => (v == null ? "—" : v.toLocaleString(locale, { minimumFractionDigits: d, maximumFractionDigits: d }));
  const eur = (v: number | null | undefined) => (v == null ? "—" : `${num(v, 2)} €`);
  const pct = (v: number | null | undefined, d = 2) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}%`);
  const date = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString(locale, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
  const day = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" }) : "—");
  return { num, eur, pct, date, day };
}
type Fmt = ReturnType<typeof makeFmt>;

const moodName = (m: number) => (m < 0.3 ? "fear" : m < 0.45 ? "caution" : m < 0.6 ? "calm" : m < 0.75 ? "confidence" : "euphoria");
const emotionLabel = (t: TFn, e: string) => `${EMOJI[e] ?? ""} ${t(`dashboard.evo.emotion.${e}`)}`;
const geneName = (t: TFn, g: string) => t(`dashboard.evo.input.${g}`);

export default function Population() {
  const { t, locale } = useI18n();
  const fmt = useMemo(() => makeFmt(locale), [locale]);
  const [interval, setInterval_] = useState<Interval>(() => {
    try {
      const saved = localStorage.getItem("pop.interval");
      if (saved && (INTERVALS as string[]).includes(saved)) return saved as Interval;
    } catch { /* storage unavailable */ }
    return "1h";
  });
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [page, setPage] = useState(1);
  const [organismId, setOrganismId] = useState<number | null>(null);
  const graveyardRef = useRef<HTMLDivElement>(null);

  const pickInterval = (i: Interval) => {
    setInterval_(i);
    try { localStorage.setItem("pop.interval", i); } catch { /* ignore */ }
  };

  const report = useQuery({
    queryKey: ["evolution-report", interval],
    queryFn: () => apiGet<EvolutionReport>("/api/evolution/report", { interval }),
    retry: false,
  });

  const deathParams = useMemo(
    () => ({
      interval,
      cause: filters.cause || undefined,
      generation: filters.generation || undefined,
      generation_to: filters.generationTo || undefined,
      organism: filters.organism || undefined,
      order: filters.order,
    }),
    [interval, filters],
  );
  const deaths = useQuery({
    queryKey: ["evolution-deaths", deathParams, page],
    queryFn: () => apiGet<EvolutionDeathsResponse>("/api/evolution/deaths", { ...deathParams, page, page_size: PAGE_SIZE }),
    enabled: !!report.data,
    placeholderData: keepPreviousData,
  });

  const organism = useQuery({
    queryKey: ["evolution-organism", interval, organismId],
    queryFn: () => apiGet<EvolutionOrganismResponse>(`/api/evolution/organism/${organismId}`, { interval }),
    enabled: organismId != null,
  });

  const setFilter = (patch: Partial<Filters>) => {
    setFilters((f) => ({ ...f, ...patch }));
    setPage(1);
  };
  const openGeneration = (from: number, to: number) => {
    setFilters({ ...EMPTY_FILTERS, generation: String(from), generationTo: to !== from ? String(to) : "" });
    setPage(1);
    graveyardRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const dirty = JSON.stringify(filters) !== JSON.stringify(EMPTY_FILTERS);

  const r = report.data;
  const notFound = report.error instanceof Error ? report.error.message : null;

  return (
    <Page
      kicker={t("nav.population")}
      title={t("pop.title")}
      subtitle={t("pop.subtitle")}
      actions={<Segmented ariaLabel={t("nav.population")} options={INTERVALS.map((i) => ({ value: i, label: i }))} value={interval} onChange={pickInterval} size="sm" />}
    >
      {notFound && (
        <div className="mb-4">
          <ErrorBanner message={notFound} />
        </div>
      )}

      {!r ? (
        <div className="space-y-4">
          <Skeleton className="h-28" />
          <Skeleton className="h-64" />
        </div>
      ) : (
        <>
          <Panel className="mb-4 p-4 sm:p-5">
            <PanelHeader title={t("pop.summary.header")} />
            <p className="mt-2 text-[13px] text-faint">{t("pop.summary.lived", { from: fmt.day(r.summary.lived_from), to: fmt.date(r.summary.lived_to), n: fmt.num(r.summary.candles_lived) })}</p>
            <SummaryTiles r={r} fmt={fmt} t={t} />
          </Panel>

          <MoneyPanel r={r} fmt={fmt} t={t} />
          <TimelinePanel r={r} fmt={fmt} t={t} />

          <div className="mb-4 grid gap-4 lg:grid-cols-2">
            <Panel className="p-4 sm:p-5">
              <PanelHeader title={t("pop.causes.header")} />
              <Bars className="mt-4" rows={r.causes.map((c) => ({
                label: <Badge tone={CAUSE_TONE[c.cause] ?? "neutral"}>{t(`pop.cause.${c.cause}`)}</Badge>,
                value: c.deaths,
                text: `${fmt.num(c.deaths)} (${c.share_pct}%${c.avg_age != null ? ` · ${t("pop.causes.avgAge", { n: fmt.num(c.avg_age) })}` : ""})`,
              }))} />
              <dl className="mt-5 grid gap-2.5 text-[12px] sm:grid-cols-2">
                {["outcompeted", "big_loss", "fees", "bad_trades", "taxes"].map((c) => (
                  <div key={c}>
                    <dt className="font-medium text-bone">{t(`pop.cause.${c}`)}</dt>
                    <dd className="text-faint">{t(`pop.causeHelp.${c}`)}</dd>
                  </div>
                ))}
              </dl>
            </Panel>
            <Panel className="p-4 sm:p-5">
              <PanelHeader title={t("pop.survival.header")} hint={t("pop.survival.hint")} />
              <Bars className="mt-4" rows={Object.entries(r.gene_survival).sort((a, b) => b[1].avg_lifespan - a[1].avg_lifespan).map(([g, v]) => ({
                label: geneName(t, g),
                value: v.avg_lifespan,
                text: `${fmt.num(v.avg_lifespan)} (${t("pop.survival.deaths", { n: fmt.num(v.deaths) })})`,
              }))} />
            </Panel>
          </div>

          <Panel className="mb-4 p-4 sm:p-5">
            <PanelHeader title={t("pop.genes.header")} hint={t("pop.genes.hint")} />
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {GENE_ORDER.filter((g) => r.genes.includes(g)).concat(r.genes.filter((g) => !GENE_ORDER.includes(g))).map((g) => (
                <Sparkline
                  key={g}
                  title={geneName(t, g)}
                  times={r.timeline.map((p) => p.time)}
                  range={[0, 1]}
                  format={(v) => (v == null ? "—" : `${Math.round(v * 100)}%`)}
                  values={r.timeline.map((p) => (g in p.shares ? p.shares[g] : null))}
                  fmtDay={fmt.day}
                />
              ))}
            </div>
          </Panel>

          <Panel className="mb-4 p-4 sm:p-5">
            <PanelHeader title={t("pop.traits.header")} hint={t("pop.traits.hint")} />
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {TRAITS.map((tr) => (
                <Sparkline
                  key={tr.key}
                  title={t(`pop.trait.${tr.key}`)}
                  times={r.timeline.map((p) => p.time)}
                  range={tr.range}
                  mid={tr.mid}
                  format={(v) => (v == null ? "—" : v.toFixed(tr.digits))}
                  values={r.timeline.map((p) => p[tr.key] as number)}
                  fmtDay={fmt.day}
                />
              ))}
            </div>
          </Panel>

          <Panel className="mb-4 overflow-hidden">
            <div className="p-4 sm:p-5">
              <PanelHeader title={t("pop.generations.header")} hint={t("pop.generations.hint")} />
            </div>
            <div className="overflow-x-auto border-t border-line">
              <table className="w-full min-w-[980px] border-collapse text-[12.5px]">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] text-faint">
                    {[t("pop.col.generations"), t("pop.col.models"), t("pop.col.alive"), t("pop.col.avgLife"), t("pop.col.causes"), t("pop.col.topInputs"), t("pop.col.boldness"), t("pop.col.temperament"), t("pop.col.news"), t("pop.col.window"), t("pop.col.patience")].map((h, i) => (
                      <th key={h} className={clsx("py-2.5 font-medium", i >= 2 && i <= 3 ? "px-3 text-right" : i >= 6 ? "px-3 text-right" : "px-3 text-left", i === 0 && "pl-4 sm:pl-5")}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {r.generations.map((g) => {
                    const top = Object.entries(g.input_shares).slice(0, 3).map(([k, v]) => `${geneName(t, k)} ${Math.round(v * 100)}%`).join(", ");
                    const dead = Object.values(g.causes).reduce((a, b) => a + b, 0);
                    return (
                      <tr key={`${g.from}-${g.to}`} onClick={() => openGeneration(g.from, g.to)} className="cursor-pointer border-b border-line last:border-0 transition-colors hover:bg-raised">
                        <td className="py-2.5 pl-4 pr-3 text-bone sm:pl-5">{g.from === g.to ? g.from : `${g.from}–${g.to}`}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.organisms)}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.alive)}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{g.avg_lifespan == null ? "—" : fmt.num(g.avg_lifespan)}</td>
                        <td className="max-w-[220px] px-3 py-2.5 text-dim">
                          {dead ? Object.entries(g.causes).filter(([, n]) => n).map(([c, n]) => (
                            <span key={c} className="mr-1.5 inline-flex items-center gap-1"><Badge tone={CAUSE_TONE[c] ?? "neutral"}>{t(`pop.cause.${c}`)}</Badge><span className="text-[11px] text-faint">{Math.round((n / dead) * 100)}%</span></span>
                          )) : "—"}
                        </td>
                        <td className="max-w-[260px] px-3 py-2.5 text-dim">{top}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.boldness, 2)}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.temperament, 2)}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.news_sensitivity, 2)}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.window)}</td>
                        <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(g.patience, 2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Panel>

          <div ref={graveyardRef} className="scroll-mt-24">
            <Panel className="mb-4 overflow-hidden">
              <div className="flex flex-wrap items-end justify-between gap-3 p-4 sm:p-5">
                <PanelHeader title={t("pop.graveyard.header")} hint={deaths.data ? t("pop.graveyard.total", { n: fmt.num(deaths.data.total) }) : undefined} />
              </div>
              <div className="flex flex-wrap items-end gap-3 border-t border-line p-4 sm:p-5">
                <GField label={t("pop.filter.cause")}>
                  <select className="field" value={filters.cause} onChange={(e) => setFilter({ cause: e.target.value })}>
                    <option value="">{t("eval.filter.all")}</option>
                    {["outcompeted", "big_loss", "fees", "bad_trades", "taxes"].map((c) => <option key={c} value={c}>{t(`pop.cause.${c}`)}</option>)}
                  </select>
                </GField>
                <GField label={t("pop.filter.generation")}>
                  <input type="number" min={0} step={1} className="field" value={filters.generation} onChange={(e) => setFilter({ generation: e.target.value, generationTo: "" })} />
                </GField>
                <GField label={t("pop.filter.organism")}>
                  <input type="number" min={0} step={1} className="field" value={filters.organism} onChange={(e) => setFilter({ organism: e.target.value })} />
                </GField>
                <GField label={t("pop.filter.order")}>
                  <select className="field" value={filters.order} onChange={(e) => setFilter({ order: e.target.value as "asc" | "desc" })}>
                    <option value="desc">{t("pop.filter.newest")}</option>
                    <option value="asc">{t("pop.filter.oldest")}</option>
                  </select>
                </GField>
                {dirty && (
                  <Button size="md" onClick={() => { setFilters(EMPTY_FILTERS); setPage(1); }}>{t("eval.filter.reset")}</Button>
                )}
              </div>
              <p className="border-t border-line px-4 py-2.5 text-[12px] text-faint sm:px-5">{t("pop.graveyard.hint")}</p>
              <div className="overflow-x-auto border-t border-line">
                <table className="w-full min-w-[1100px] border-collapse text-[12.5px]">
                  <thead>
                    <tr className="border-b border-line text-left text-[11px] text-faint">
                      {["#", t("pop.col.died"), t("pop.col.age"), t("pop.col.gen"), t("pop.col.cause"), t("pop.col.money"), t("pop.col.tradesFees"), t("pop.col.market"), t("pop.col.mood"), t("pop.col.record"), t("pop.col.inputs")].map((h, i) => (
                        <th key={h} className={clsx("py-2.5 font-medium", i === 2 || i === 5 || i === 6 ? "px-3 text-right" : "px-3", i === 0 && "pl-4 sm:pl-5")}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className={clsx(deaths.isPlaceholderData && "opacity-60 transition-opacity")}>
                    {!deaths.data ? (
                      <tr><td colSpan={11}><div className="space-y-2 p-5">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-5" />)}</div></td></tr>
                    ) : !deaths.data.rows.length ? (
                      <tr><td colSpan={11}><Empty>{t("pop.graveyard.none")}</Empty></td></tr>
                    ) : (
                      deaths.data.rows.map((d) => (
                        <tr key={d.id} onClick={() => setOrganismId(d.id)} className="cursor-pointer border-b border-line last:border-0 transition-colors hover:bg-raised">
                          <td className="py-2.5 pl-4 pr-3 text-bone sm:pl-5">{d.id}</td>
                          <td className="num px-3 py-2.5 text-dim">{fmt.date(d.died)}</td>
                          <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(d.age)}</td>
                          <td className="px-3 py-2.5 text-dim">{d.generation}</td>
                          <td className="px-3 py-2.5"><Badge tone={CAUSE_TONE[d.cause] ?? "neutral"}>{t(`pop.cause.${d.cause}`)}</Badge></td>
                          <td className="num px-3 py-2.5 text-right text-dim">{fmt.eur(d.money)} / {fmt.eur(d.peak_money)}</td>
                          <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(d.trades)} / {fmt.eur(d.fees_paid)}</td>
                          <td className="max-w-[200px] px-3 py-2.5 text-dim">
                            ${fmt.num(d.price, 2)} · <span className={d.move_24_pct > 0 ? "text-up" : d.move_24_pct < 0 ? "text-down" : ""}>{fmt.pct(d.move_24_pct)}</span>
                            {d.volatility_vs_month != null && ` · ${d.volatility_vs_month > 1.5 ? t("pop.market.wild") : d.volatility_vs_month < 0.7 ? t("pop.market.quiet") : t("pop.market.normal")}`}
                          </td>
                          <td className="px-3 py-2.5 text-dim">{emotionLabel(t, d.emotion)}</td>
                          <td className="num px-3 py-2.5 text-right text-dim">{d.wins} / {d.losses} / {d.combos}</td>
                          <td className="max-w-[220px] px-3 py-2.5 text-dim">{d.inputs.map((g) => geneName(t, g)).join(", ")}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              {deaths.data && (
                <Pager page={deaths.data.page} pages={deaths.data.pages} info={t("pop.page", { page: deaths.data.page, pages: deaths.data.pages })} onPage={setPage} />
              )}
            </Panel>
          </div>

          <Panel className="overflow-hidden">
            <div className="p-4 sm:p-5">
              <PanelHeader title={t("pop.alive.header")} />
            </div>
            <div className="overflow-x-auto border-t border-line">
              <table className="w-full min-w-[980px] border-collapse text-[12.5px]">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] text-faint">
                    {["#", t("pop.col.money2"), t("pop.col.position"), t("pop.col.tradesFees"), t("pop.col.mood"), t("pop.col.age"), t("pop.col.gen"), t("pop.col.record"), t("pop.col.inputs"), t("pop.col.voting")].map((h, i) => (
                      <th key={h} className={clsx("py-2.5 font-medium", i === 1 || i === 3 ? "px-3 text-right" : "px-3", i === 0 && "pl-4 sm:pl-5")}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {r.alive.map((o) => (
                    <tr key={o.id} onClick={() => setOrganismId(o.id)} className="cursor-pointer border-b border-line last:border-0 transition-colors hover:bg-raised">
                      <td className="py-2.5 pl-4 pr-3 text-bone sm:pl-5">{o.id}</td>
                      <td className="num px-3 py-2.5 text-right text-dim">{fmt.eur(o.money)}</td>
                      <td className="px-3 py-2.5 text-dim">{positionLabel(t, o.position)}</td>
                      <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(o.trades)} / {fmt.eur(o.fees_paid)}</td>
                      <td className="px-3 py-2.5 text-dim">{emotionLabel(t, o.emotion)}</td>
                      <td className="num px-3 py-2.5 text-dim">{fmt.num(o.age)}</td>
                      <td className="px-3 py-2.5 text-dim">{o.generation}</td>
                      <td className="num px-3 py-2.5 text-dim">{o.wins} / {o.losses} / {o.combos}</td>
                      <td className="max-w-[220px] px-3 py-2.5 text-dim">{o.inputs.map((g) => geneName(t, g)).join(", ")}</td>
                      <td className="px-3 py-2.5 text-up">{o.voting ? "✓" : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      <OrganismDialog data={organism.data} loading={organismId != null && organism.isLoading} onOpenAnother={setOrganismId} onClose={() => setOrganismId(null)} fmt={fmt} t={t} />
    </Page>
  );
}

function GField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex min-w-[130px] flex-1 flex-col gap-1.5 text-[11px] text-faint sm:flex-none">
      {label}
      {children}
    </label>
  );
}

function positionLabel(t: TFn, p: number) {
  if (Math.abs(p) < 0.005) return t("pop.money.out");
  return `${t(p > 0 ? "pop.money.long" : "pop.money.short")} ${Math.round(Math.abs(p) * 100)}%`;
}

// --------------------------------------------------------------- tiles/bars

function Tile({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 bg-panel/90 px-4 py-3">
      <span className="text-[11px] text-faint">{label}</span>
      <span className="num text-[15px] font-medium text-bone">{value}</span>
    </div>
  );
}

function SummaryTiles({ r, fmt, t }: { r: EvolutionReport; fmt: Fmt; t: TFn }) {
  const tr = Object.fromEntries(r.track_record.map((x) => [x.h, x]));
  return (
    <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-[20px] border border-line bg-line sm:grid-cols-3 xl:grid-cols-5">
      <Tile label={t("dashboard.evo.alive")} value={fmt.num(r.summary.alive)} />
      <Tile label={t("dashboard.evo.births")} value={fmt.num(r.summary.births)} />
      <Tile label={t("dashboard.evo.deaths")} value={fmt.num(r.summary.deaths)} />
      <Tile label={t("dashboard.evo.generation")} value={fmt.num(r.summary.max_generation)} />
      <Tile label={t("dashboard.evo.lifespan")} value={r.summary.avg_lifespan_of_dead == null ? "—" : t("dashboard.box.candles", { n: fmt.num(r.summary.avg_lifespan_of_dead) })} />
      <Tile label={t("pop.summary.oldest")} value={t("dashboard.box.candles", { n: fmt.num(r.summary.oldest_alive) })} />
      <Tile label={t("pop.summary.mood")} value={emotionLabel(t, r.summary.emotion)} />
      <Tile label={t("pop.summary.dir1")} value={tr[1] ? `${tr[1].direction_pct}%` : "—"} />
      <Tile label={t("pop.summary.dir24")} value={tr[24] ? `${tr[24].direction_pct}%` : "—"} />
    </div>
  );
}

function Bars({ rows, className }: { rows: { label: React.ReactNode; value: number; text: string }[]; className?: string }) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  return (
    <div className={clsx("space-y-2.5", className)}>
      {rows.map((row, i) => (
        <div key={i} className="grid grid-cols-[minmax(0,1fr)_minmax(0,2fr)_auto] items-center gap-3 text-[12.5px]">
          <span className="min-w-0 truncate text-dim">{row.label}</span>
          <span className="h-1.5 overflow-hidden rounded-full bg-press"><span className="block h-full rounded-full bg-accent/70" style={{ width: `${(row.value / max) * 100}%` }} /></span>
          <b className="num whitespace-nowrap text-right text-bone">{row.text}</b>
        </div>
      ))}
    </div>
  );
}

// --------------------------------------------------------------- money chart

function MoneyPanel({ r, fmt, t }: { r: EvolutionReport; fmt: Fmt; t: TFn }) {
  const ref = useRef<HTMLDivElement>(null);
  const [readout, setReadout] = useState("");
  const m = r.money;

  useEffect(() => {
    const el = ref.current;
    if (!el || !m) return;
    const chart = createChart(el, { ...evoChartOptions(el), rightPriceScale: { borderVisible: false, minimumWidth: 70, mode: PriceScaleMode.Logarithmic } });
    const pred = getComputedStyle(document.documentElement).getPropertyValue("--color-accent-hi").trim() || "#ff8a4c";
    const muted = getComputedStyle(document.documentElement).getPropertyValue("--color-faint").trim() || "#66666e";
    const pts = r.timeline.filter((p) => p.fund != null).map((p) => ({ ...p, t: toUnix(p.time) as UTCTimestamp }))
      .filter((p, i, a) => i === 0 || p.t > a[i - 1].t);
    const eurFmt = { type: "custom" as const, formatter: (v: number) => `${v.toFixed(0)} €` };
    const hold = chart.addLineSeries({ color: muted, lineWidth: 2, priceLineVisible: false, lastValueVisible: true, priceFormat: eurFmt, title: t("pop.money.hold") });
    hold.setData(pts.map((p) => ({ time: p.t, value: p.buy_hold ?? undefined })).filter((p) => p.value != null) as { time: UTCTimestamp; value: number }[]);
    const fund = chart.addLineSeries({ color: pred, lineWidth: 2, priceLineVisible: false, lastValueVisible: true, priceFormat: eurFmt, title: t("pop.money.fund") });
    fund.setData(pts.map((p) => ({ time: p.t, value: p.fund! })));
    fund.createPriceLine({ price: 100, color: muted, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false });

    const byTime = new Map(pts.map((p) => [p.t, p]));
    const readoutFor = (point?: typeof pts[number]) => {
      if (!point) { setReadout(""); return; }
      setReadout(`${fmt.eur(point.fund)} · ${fmt.eur(point.buy_hold)} · ${fmt.day(point.time)}`);
    };
    chart.subscribeCrosshairMove((param) => readoutFor(param.time ? byTime.get(param.time as UTCTimestamp) : pts[pts.length - 1]));
    chart.timeScale().fitContent();
    readoutFor(pts[pts.length - 1]);

    const ro = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
    ro.observe(el);
    return () => { ro.disconnect(); chart.remove(); };
  }, [r, m, fmt, t]);

  if (!m) return null;
  const sign = (v: number) => (v > 0 ? "text-up" : v < 0 ? "text-down" : "");
  return (
    <Panel className="mb-4 p-4 sm:p-5">
      <PanelHeader title={t("pop.money.header")} hint={t("pop.money.hint")} />
      <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-[20px] border border-line bg-line sm:grid-cols-3">
        <Tile label={t("pop.money.fund")} value={<>{fmt.eur(m.fund)} <small className={sign(m.fund_return_pct)}>{fmt.pct(m.fund_return_pct, 1)}</small></>} />
        <Tile label={t("pop.money.hold")} value={<>{fmt.eur(m.buy_hold)} <small className={sign(m.buy_hold_return_pct)}>{fmt.pct(m.buy_hold_return_pct, 1)}</small></>} />
        <Tile label={t("pop.money.fundTrades")} value={`${fmt.num(m.fund_trades)} · ${fmt.eur(m.fund_fees)}`} />
        <Tile label={t("pop.money.fundNow")} value={positionLabel(t, m.fund_position)} />
        <Tile label={t("pop.money.richest")} value={m.richest ? `#${m.richest.id} · ${fmt.eur(m.richest.money)}` : "—"} />
        <Tile label={t("pop.money.positions")} value={t("pop.money.positionsValue", { long: m.long, short: m.short, out: m.out })} />
      </div>
      <div className="mt-4 flex items-baseline justify-between text-[12px]">
        <span className="text-faint">{t("pop.money.fund")} · {t("pop.money.hold")}</span>
        <b className="num text-bone">{readout}</b>
      </div>
      <div ref={ref} className="mt-1.5 h-56" />
    </Panel>
  );
}

// --------------------------------------------------------------- timeline (3 synced panes)

function TimelinePanel({ r, fmt, t }: { r: EvolutionReport; fmt: Fmt; t: TFn }) {
  const priceRef = useRef<HTMLDivElement>(null);
  const deathsRef = useRef<HTMLDivElement>(null);
  const moodRef = useRef<HTMLDivElement>(null);
  const [readout, setReadout] = useState({ price: "", deaths: "", mood: "" });

  useEffect(() => {
    const priceEl = priceRef.current, deathsEl = deathsRef.current, moodEl = moodRef.current;
    if (!priceEl || !deathsEl || !moodEl) return;
    const C = getComputedStyle(document.documentElement);
    const pred = C.getPropertyValue("--color-accent-hi").trim() || "#ff8a4c";
    const muted = C.getPropertyValue("--color-faint").trim() || "#66666e";
    const pts = r.timeline.map((p) => ({ ...p, t: toUnix(p.time) as UTCTimestamp }))
      .filter((p, i, a) => i === 0 || p.t > a[i - 1].t);

    const priceChart = createChart(priceEl, evoChartOptions(priceEl));
    const deathsChart = createChart(deathsEl, evoChartOptions(deathsEl));
    const moodChart = createChart(moodEl, evoChartOptions(moodEl));

    const price = priceChart.addLineSeries({ color: pred, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
    price.setData(pts.filter((p) => p.price != null).map((p) => ({ time: p.t, value: p.price! })));
    const deathsS = deathsChart.addHistogramSeries({ color: pred, priceLineVisible: false, lastValueVisible: false, priceFormat: { type: "volume" } });
    deathsS.setData(pts.map((p) => ({ time: p.t, value: p.deaths })));
    const mood = moodChart.addLineSeries({
      color: pred, lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
      autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }),
    });
    mood.setData(pts.map((p) => ({ time: p.t, value: p.mood })));
    mood.createPriceLine({ price: 0.5, color: muted, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false });

    const charts = [priceChart, deathsChart, moodChart];
    const byTime = new Map(pts.map((p) => [p.t, p]));
    const readoutFor = (point?: typeof pts[number]) => {
      if (!point) { setReadout({ price: "", deaths: "", mood: "" }); return; }
      setReadout({
        price: point.price != null ? `$${fmt.num(point.price, 2)} · ${fmt.date(point.time)}` : "",
        deaths: t("pop.timeline.deathsValue", { n: fmt.num(point.deaths) }),
        mood: `${emotionLabel(t, moodName(point.mood))} (${Math.round(point.mood * 100)}%)`,
      });
    };
    const unsync = syncPanes(charts, (time) => readoutFor(time != null ? byTime.get(time as UTCTimestamp) : pts[pts.length - 1]));
    charts.forEach((c) => c.timeScale().fitContent());
    readoutFor(pts[pts.length - 1]);

    const ro = new ResizeObserver(() => charts.forEach((c, i) => c.applyOptions({ width: [priceEl, deathsEl, moodEl][i].clientWidth, height: [priceEl, deathsEl, moodEl][i].clientHeight })));
    [priceEl, deathsEl, moodEl].forEach((el) => ro.observe(el));
    return () => { ro.disconnect(); unsync(); charts.forEach((c) => c.remove()); };
  }, [r, fmt, t]);

  return (
    <Panel className="mb-4 p-4 sm:p-5">
      <PanelHeader title={t("pop.timeline.header")} hint={t("pop.timeline.hint")} />
      <div className="mt-4 space-y-4">
        <div>
          <div className="flex items-baseline justify-between text-[12px]"><span className="text-faint">{t("pop.timeline.price")}</span><b className="num text-bone">{readout.price}</b></div>
          <div ref={priceRef} className="mt-1 h-48" />
        </div>
        <div>
          <div className="flex items-baseline justify-between text-[12px]"><span className="text-faint">{t("pop.timeline.deaths")}</span><b className="num text-bone">{readout.deaths}</b></div>
          <div ref={deathsRef} className="mt-1 h-28" />
        </div>
        <div>
          <div className="flex items-baseline justify-between text-[12px]"><span className="text-faint">{t("pop.timeline.mood")}</span><b className="num text-bone">{readout.mood}</b></div>
          <div ref={moodRef} className="mt-1 h-28" />
        </div>
      </div>
    </Panel>
  );
}

// --------------------------------------------------------------- organism dialog

function OrganismDialog({
  data, loading, onOpenAnother, onClose, fmt, t,
}: {
  data: EvolutionOrganismResponse | undefined;
  loading: boolean;
  onOpenAnother: (id: number) => void;
  onClose: () => void;
  fmt: Fmt;
  t: TFn;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const open = loading || !!data;
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const chip = (a: { id: number; generation?: number; dead: boolean }) => (
    <button
      key={a.id}
      type="button"
      onClick={() => onOpenAnother(a.id)}
      className={clsx(
        "rounded-full border px-2.5 py-1 text-[12px] transition-colors",
        a.dead ? "border-line text-faint hover:border-line-strong hover:text-dim" : "border-up/30 bg-up/10 text-up hover:bg-up/15",
      )}
    >
      #{a.id}{a.generation != null ? ` · g${a.generation}` : ""}{a.dead ? " †" : ""}
    </button>
  );

  const o = data?.organism;
  const levels: Record<number, AncestorNode[]> = {};
  (data?.ancestors ?? []).forEach((a) => { (levels[a.level] = levels[a.level] || []).push(a); });

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className="m-auto max-h-[88dvh] w-[min(760px,calc(100vw-24px))] overflow-y-auto rounded-3xl border border-line-strong bg-panel p-0 text-bone"
    >
      {o && (
        <div className="p-5 sm:p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold">#{o.id} {o.dead ? "†" : "❤"}</h2>
            <Button size="sm" onClick={onClose}><X className="size-4" /> {t("eval.close")}</Button>
          </div>
          <p className="text-[13.5px] leading-relaxed text-dim">
            {o.dead
              ? t("pop.story.dead", { id: o.id, born: fmt.date(o.born), died: fmt.date(o.died), age: fmt.num(o.age), gen: o.generation, cause: t(`pop.cause.${o.cause}`), why: t(`pop.causeHelp.${o.cause}`) })
              : t("pop.story.alive", { id: o.id, born: fmt.date(o.born), age: fmt.num(o.age), gen: o.generation, energy: fmt.num(o.energy, 1) })}
          </p>
          {o.dead && (
            <p className="mt-1 text-[13.5px] leading-relaxed text-dim">
              {t("pop.story.market", { price: fmt.num(o.price, 2), move: fmt.pct(o.move_24_pct), mood: t(`dashboard.evo.emotion.${o.emotion}`) })}
            </p>
          )}
          <p className="mt-3 text-[13.5px] leading-relaxed text-dim">
            {t("pop.story.money", { money: fmt.eur(o.money), peak: fmt.eur(o.peak_money), trades: fmt.num(o.trades), fees: fmt.eur(o.fees_paid), pnl: fmt.eur(o.trading_pnl) })}
          </p>
          <h3 className="mt-5 text-[13px] font-semibold text-bone">{t("pop.story.genes")}</h3>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {o.inputs.map((g) => <Badge key={g}>{geneName(t, g)}</Badge>)}
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5 text-[12px] text-dim">
            {[
              [t("pop.trait.boldness"), fmt.num(o.boldness, 2)],
              [t("pop.trait.temperament"), fmt.num(o.temperament, 2)],
              [t("pop.trait.news_sensitivity"), fmt.num(o.news_sensitivity, 2)],
              [t("pop.trait.window"), fmt.num(o.window)],
              [t("pop.trait.patience"), fmt.num(o.patience, 2)],
            ].map(([k, v]) => <span key={k} className="rounded-full border border-line px-2.5 py-1">{k}: <b className="text-bone">{v}</b></span>)}
          </div>
          <p className="mt-3 text-[13.5px] text-dim">{t("pop.story.record", { wins: o.wins, losses: o.losses, combos: o.combos })}</p>
          <h3 className="mt-5 text-[13px] font-semibold text-bone">{t("pop.story.family")}</h3>
          {Object.keys(levels).length ? (
            <div className="mt-2 space-y-2">
              {Object.entries(levels).map(([lvl, list]) => (
                <div key={lvl} className="flex flex-wrap items-center gap-1.5">
                  <span className="mr-1 text-[11px] text-faint">{t("pop.story.level", { n: lvl })}</span>
                  {list!.map(chip)}
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-2 text-[13px] text-faint">{t("pop.story.founder")}</p>
          )}
          {!!data.children.length && (
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <span className="mr-1 text-[11px] text-faint">{t("pop.story.children")}</span>
              {data.children.map(chip)}
            </div>
          )}
        </div>
      )}
    </dialog>
  );
}
