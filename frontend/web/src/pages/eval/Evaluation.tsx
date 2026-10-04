import clsx from "clsx";
import { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, Check, Download, RotateCcw, X } from "lucide-react";
import { apiGet, buildUrl, type EvalPredictionsResponse } from "../../lib/api";
import { fmtMoney, fmtPct, userTimeZone } from "../../lib/format";
import { useI18n } from "../../i18n";
import { Page } from "../../components/Shell";
import { buttonClass, Button, Empty, ErrorBanner, ImageViewer, Pager, Panel, PanelHeader, Skeleton } from "../../components/ui";
import { EvalTabs, StatTiles, StatusPill } from "./common";

const PAGE_SIZE = 50;
const EMPTY_FILTERS = { interval: "", model: "", status: "", date_from: "", date_to: "" };

export default function Evaluation() {
  const { t, locale } = useI18n();
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [sort, setSort] = useState({ key: "created_at", order: "desc" as "asc" | "desc" });
  const [page, setPage] = useState(1);
  const [viewer, setViewer] = useState<{ src: string; alt: string } | null>(null);

  const params = useMemo(() => ({ ...filters, sort: sort.key, order: sort.order }), [filters, sort]);
  const data = useQuery({
    queryKey: ["eval-predictions", params, page],
    queryFn: () => apiGet<EvalPredictionsResponse>("/api/evaluation/predictions", { ...params, page, page_size: PAGE_SIZE }),
    refetchInterval: 60_000,
    placeholderData: keepPreviousData,
  });
  const models = useQuery({ queryKey: ["eval-models"], queryFn: () => apiGet<{ models: string[] }>("/api/evaluation/models") });

  const setFilter = (k: keyof typeof EMPTY_FILTERS, v: string) => {
    setFilters((f) => ({ ...f, [k]: v }));
    setPage(1);
  };
  const toggleSort = (key: string) => {
    setSort((s) => (s.key === key ? { key, order: s.order === "asc" ? "desc" : "asc" } : { key, order: key === "created_at" ? "desc" : "asc" }));
    setPage(1);
  };
  const fmtTime = (iso: string) => new Date(iso).toLocaleString(locale, { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  const dirty = JSON.stringify(filters) !== JSON.stringify(EMPTY_FILTERS);

  const cols: { key: string; label: string; num?: boolean }[] = [
    { key: "created_at", label: t("eval.col.created") },
    { key: "interval", label: t("eval.col.interval") },
    { key: "model_name", label: t("eval.col.model") },
    { key: "price_at_prediction", label: t("eval.col.priceAt"), num: true },
    { key: "predicted_final_price", label: t("eval.col.predicted"), num: true },
    { key: "actual_final_price", label: t("eval.col.actual"), num: true },
    { key: "pct_error", label: t("eval.col.error"), num: true },
    { key: "direction_correct", label: t("eval.col.direction") },
    { key: "status", label: t("eval.col.status") },
  ];

  return (
    <Page kicker={t("nav.evaluation")} title={t("eval.title")} subtitle={t("eval.subtitle", { tz: userTimeZone() })}>
      <EvalTabs />
      {data.error && (
        <div className="mb-4">
          <ErrorBanner message={t("eval.error.load", { msg: data.error.message })} />
        </div>
      )}

      <section className="mb-8">
        <div className="eyebrow mb-3">{t("eval.summary.header")}</div>
        {!data.data ? (
          <Skeleton className="h-24" />
        ) : !data.data.summary.length ? (
          <p className="text-[13px] text-faint">{t("eval.summary.empty")}</p>
        ) : (
          <div className="grid gap-4 xl:grid-cols-2">
            {data.data.summary.map((m) => (
              <div key={m.model_name}>
                <div className="num mb-2 text-[13px] font-semibold text-accent">{m.model_name}</div>
                <StatTiles s={{ ...m, expired: 0 }} className="!grid-cols-3 xl:!grid-cols-3" />
              </div>
            ))}
          </div>
        )}
      </section>

      <Panel className="overflow-hidden">
        <div className="p-4 sm:p-5">
          <PanelHeader
            title={t("eval.table.header")}
            hint={t("eval.chartHint")}
            actions={
              <a href={buildUrl("/api/evaluation/predictions.csv", params)} className={buttonClass("outline", "sm")}>
                <Download className="size-3.5" /> {t("eval.csv")}
              </a>
            }
          />
          <div className="mt-4 flex flex-wrap items-end gap-3">
            <Field label={t("eval.filter.interval")}>
              <select className="field" value={filters.interval} onChange={(e) => setFilter("interval", e.target.value)}>
                <option value="">{t("eval.filter.all")}</option>
                {["15m", "1h", "4h", "1d", "1w"].map((i) => <option key={i}>{i}</option>)}
              </select>
            </Field>
            <Field label={t("eval.filter.model")}>
              <select className="field" value={filters.model} onChange={(e) => setFilter("model", e.target.value)}>
                <option value="">{t("eval.filter.all")}</option>
                {(models.data?.models ?? []).map((m) => <option key={m}>{m}</option>)}
              </select>
            </Field>
            <Field label={t("eval.filter.status")}>
              <select className="field" value={filters.status} onChange={(e) => setFilter("status", e.target.value)}>
                <option value="">{t("eval.filter.all")}</option>
                {["pending", "completed", "expired"].map((s) => <option key={s} value={s}>{t(`eval.status.${s}`)}</option>)}
              </select>
            </Field>
            <Field label={t("eval.filter.from")}>
              <input type="date" className="field" value={filters.date_from} onChange={(e) => setFilter("date_from", e.target.value)} />
            </Field>
            <Field label={t("eval.filter.to")}>
              <input type="date" className="field" value={filters.date_to} onChange={(e) => setFilter("date_to", e.target.value)} />
            </Field>
            {dirty && (
              <Button size="md" onClick={() => { setFilters(EMPTY_FILTERS); setPage(1); }}>
                <RotateCcw className="size-3.5" /> {t("eval.filter.reset")}
              </Button>
            )}
          </div>
        </div>

        <div className="overflow-x-auto border-t border-line">
          <table className="w-full min-w-[980px] border-collapse text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-[11px] text-faint">
                {cols.map((c, i) => (
                  <th key={c.key} className={clsx("py-2.5 font-medium", c.num ? "px-3 text-right" : "px-3 text-left", i === 0 && "pl-4 sm:pl-5")} aria-sort={sort.key === c.key ? (sort.order === "asc" ? "ascending" : "descending") : "none"}>
                    <button type="button" onClick={() => toggleSort(c.key)} className={clsx("inline-flex items-center gap-1 hover:text-bone", sort.key === c.key && "text-bone")}>
                      {c.label}
                      {sort.key === c.key && (sort.order === "asc" ? <ArrowUp className="size-3 text-accent" /> : <ArrowDown className="size-3 text-accent" />)}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className={clsx(data.isPlaceholderData && "opacity-60 transition-opacity")}>
              {!data.data ? (
                <tr><td colSpan={9}><div className="space-y-2 p-5">{Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-5" />)}</div></td></tr>
              ) : !data.data.rows.length ? (
                <tr><td colSpan={9}><Empty>{t("eval.empty")}</Empty></td></tr>
              ) : (
                data.data.rows.map((r) => {
                  const dc = r.direction_correct === null || r.direction_correct === undefined ? null : !!r.direction_correct;
                  return (
                    <tr key={r.id} onClick={() => setViewer({ src: `${r.snapshot_url}?t=${Date.now()}`, alt: `${r.model_name} · ${r.interval} · ${fmtTime(r.created_at)}` })} className="cursor-pointer border-b border-line last:border-0 transition-colors hover:bg-raised">
                      <td className="num py-2.5 pl-4 pr-3 text-dim sm:pl-5">{fmtTime(r.created_at)}</td>
                      <td className="num px-3 py-2.5 text-bone">{r.interval}</td>
                      <td className="num px-3 py-2.5 text-dim">{r.model_name}</td>
                      <td className="num px-3 py-2.5 text-right text-dim">{fmtMoney(r.price_at_prediction)}</td>
                      <td className="num px-3 py-2.5 text-right text-accent">{fmtMoney(r.predicted_final_price)}</td>
                      <td className="num px-3 py-2.5 text-right text-bone">{fmtMoney(r.actual_final_price)}</td>
                      <td className={clsx("num px-3 py-2.5 text-right", r.pct_error == null ? "text-faint" : Math.abs(r.pct_error) < 1 ? "text-up" : "text-down")}>{fmtPct(r.pct_error)}</td>
                      <td className="px-3 py-2.5">
                        {dc === null ? <span className="text-faint">—</span> : (
                          <span className={clsx("inline-flex items-center gap-1", dc ? "text-up" : "text-down")}>
                            {dc ? <Check className="size-3.5" /> : <X className="size-3.5" />} {t(dc ? "eval.yes" : "eval.no")}
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2.5"><StatusPill status={r.status} /></td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        {data.data && (
          <Pager page={data.data.page} pages={data.data.pages} info={t("eval.page", { page: data.data.page, pages: data.data.pages, total: data.data.total })} onPage={setPage} />
        )}
      </Panel>

      <ImageViewer
        src={viewer?.src ?? null}
        alt={viewer?.alt}
        onClose={() => setViewer(null)}
      />
    </Page>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex min-w-[130px] flex-1 flex-col gap-1.5 text-[11px] text-faint sm:flex-none">
      {label}
      {children}
    </label>
  );
}
