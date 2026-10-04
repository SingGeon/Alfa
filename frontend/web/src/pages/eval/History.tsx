import { useEffect } from "react";
import { useSearchParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, FileDown, Film, ImageIcon } from "lucide-react";
import { apiGet, type EvalDayDetail } from "../../lib/api";
import { fmtPct, fmtUtcDay } from "../../lib/format";
import { useI18n } from "../../i18n";
import { Page } from "../../components/Shell";
import { buttonClass, Empty, ErrorBanner, Panel, Skeleton } from "../../components/ui";
import { EvalTabs, StatTiles } from "./common";

export default function History() {
  const { t, locale } = useI18n();
  const [params, setParams] = useSearchParams();
  const days = useQuery({ queryKey: ["eval-days"], queryFn: () => apiGet<{ days: string[] }>("/api/evaluation/days") });
  const list = days.data?.days ?? [];
  const wanted = params.get("date");
  const day = wanted && list.includes(wanted) ? wanted : list[0];
  const idx = day ? list.indexOf(day) : -1;

  const detail = useQuery({
    queryKey: ["eval-day", day],
    queryFn: () => apiGet<EvalDayDetail>(`/api/evaluation/days/${day}`),
    enabled: !!day,
  });

  const go = (d: string) => setParams({ date: d }, { replace: true });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "SELECT") return;
      if (e.key === "ArrowLeft" && idx < list.length - 1) go(list[idx + 1]);
      if (e.key === "ArrowRight" && idx > 0) go(list[idx - 1]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const base = day ? `/api/evaluation/days/${day}/overview` : "";
  const d = detail.data;

  return (
    <Page kicker={t("nav.evaluation")} title={t("history.title")} subtitle={t("history.subtitle")}>
      <EvalTabs />
      {(days.error || detail.error) && (
        <div className="mb-4">
          <ErrorBanner message={t("history.error.load", { msg: (days.error || detail.error)!.message })} />
        </div>
      )}

      {days.isLoading ? (
        <Skeleton className="h-96" />
      ) : !list.length ? (
        <Panel><Empty className="py-20">{t("history.empty")}</Empty></Panel>
      ) : (
        <>
          <div className="mb-5 flex flex-wrap items-center gap-3">
            <div className="inline-flex items-center rounded-xl border border-line bg-raised">
              <button type="button" className="px-2.5 py-2 text-dim hover:text-bone disabled:opacity-30" disabled={idx >= list.length - 1} onClick={() => go(list[idx + 1])} aria-label="Previous day">
                <ChevronLeft className="size-4" />
              </button>
              <select value={day} onChange={(e) => go(e.target.value)} className="num cursor-pointer appearance-none border-x border-line bg-transparent px-3 py-2 text-[13px] text-bone focus:outline-none" aria-label={t("history.date")}>
                {list.map((x) => <option key={x} value={x} className="bg-raised">{fmtUtcDay(x, locale)}</option>)}
              </select>
              <button type="button" className="px-2.5 py-2 text-dim hover:text-bone disabled:opacity-30" disabled={idx <= 0} onClick={() => go(list[idx - 1])} aria-label="Next day">
                <ChevronRight className="size-4" />
              </button>
            </div>
            <div className="ml-auto flex flex-wrap gap-2">
              <a className={buttonClass("outline", "sm")} href={`${base}.png`} target="_blank" rel="noopener"><ImageIcon className="size-3.5" />{t("history.png")}</a>
              {d?.gif && <a className={buttonClass("outline", "sm")} href={`${base}.gif`} target="_blank" rel="noopener"><Film className="size-3.5" />{t("history.gif")}</a>}
              {d?.pdf && <a className={buttonClass("outline", "sm")} href={`${base}.pdf`} target="_blank" rel="noopener"><FileDown className="size-3.5" />{t("history.pdf")}</a>}
            </div>
          </div>

          {d ? (
            <div className="mb-5 space-y-3">
              <StatTiles s={{ predictions: d.predictions, completed: d.predictions, pending: 0, mean_abs_pct_error: d.mean_abs_pct_error, direction_accuracy_pct: d.direction_accuracy_pct }} className="!grid-cols-3" />
              {(d.best_model || d.worst_model) && (
                <div className="flex flex-wrap gap-x-6 gap-y-1 text-[13px]">
                  {d.best_model && (
                    <span className="text-dim">{t("history.best")}: <span className="num text-up">{d.best_model.model_name} ({fmtPct(d.best_model.mean_abs_pct_error, { signed: false })})</span></span>
                  )}
                  {d.worst_model && d.worst_model.model_name !== d.best_model?.model_name && (
                    <span className="text-dim">{t("history.worst")}: <span className="num text-down">{d.worst_model.model_name} ({fmtPct(d.worst_model.mean_abs_pct_error, { signed: false })})</span></span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <Skeleton className="mb-5 h-24" />
          )}

          <Panel className="overflow-hidden p-2">
            <a href={`${base}.png`} target="_blank" rel="noopener" className="block">
              <img key={day} src={`${base}.png`} alt={`${t("history.title")} ${day}`} className="block h-auto w-full rounded-xl animate-fade" />
            </a>
          </Panel>
        </>
      )}
    </Page>
  );
}
