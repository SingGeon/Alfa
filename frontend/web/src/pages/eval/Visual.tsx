import clsx from "clsx";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Check, X } from "lucide-react";
import { apiGet, type EvalPredictionsResponse, type EvalRow, type EvalStat, type EvalStatsResponse } from "../../lib/api";
import { fmtPct, fmtUtcDay, userTimeZone } from "../../lib/format";
import { useI18n } from "../../i18n";
import { Page } from "../../components/Shell";
import { Empty, ErrorBanner, ImageViewer, Pager, Panel, Segmented, Skeleton } from "../../components/ui";
import { EvalTabs, StatTiles, StatusPill } from "./common";

const PAGE_SIZE = 24;
const DAY_PAGE_SIZE = 500;
const INTERVALS = ["15m", "1h", "4h", "1d", "1w"];

interface VState {
  view: "model" | "day";
  model: string;
  interval: string;
  day: string;
  horizon: string; // "" = all
}

function readHash(hash: string): Partial<VState> {
  const h = new URLSearchParams(hash.replace(/^#/, ""));
  const out: Partial<VState> = {};
  const v = h.get("view");
  if (v === "day" || v === "model") out.view = v;
  if (h.has("model")) out.model = h.get("model")!;
  if (h.has("interval")) out.interval = INTERVALS.includes(h.get("interval")!) ? h.get("interval")! : "";
  if (h.has("day")) out.day = h.get("day")!;
  if (h.has("h")) out.horizon = h.get("h") === "all" ? "" : h.get("h")!;
  return out;
}

export default function Visual() {
  const { t, locale } = useI18n();
  const location = useLocation();
  const navigate = useNavigate();
  const [st, setSt] = useState<VState>(() => ({ view: "model", model: "", interval: "", day: "", horizon: "24", ...readHash(location.hash) }));
  const [page, setPage] = useState(1);
  const [viewer, setViewer] = useState<{ src: string; alt: string } | null>(null);
  const [stamp, setStamp] = useState(() => Date.now());

  const go = (patch: Partial<VState>) => {
    setSt((s) => ({ ...s, ...patch }));
    setPage(1);
  };

  const hq = st.horizon ? { horizon_steps: st.horizon } : {};
  const models = useQuery({ queryKey: ["eval-models"], queryFn: () => apiGet<{ models: string[] }>("/api/evaluation/models") });
  const days = useQuery({
    queryKey: ["eval-pred-days", st.horizon],
    queryFn: () => apiGet<{ days: { day: string; predictions: number }[] }>("/api/evaluation/prediction-days", hq),
    refetchInterval: 60_000,
  });
  const general = useQuery({
    queryKey: ["eval-stats", st.horizon],
    queryFn: () => apiGet<EvalStatsResponse>("/api/evaluation/stats", hq),
    refetchInterval: 60_000,
  });

  // Resolve defaults once lists arrive (same rules as before: prefer sklearn-tuned, newest day).
  const modelList = models.data?.models ?? [];
  const dayList = days.data?.days ?? [];
  const model = modelList.includes(st.model) ? st.model : modelList.includes("sklearn-tuned") ? "sklearn-tuned" : modelList[0] ?? "";
  const day = dayList.some((d) => d.day === st.day) ? st.day : dayList[0]?.day ?? "";

  useEffect(() => {
    const h = new URLSearchParams({ view: st.view, h: st.horizon || "all" });
    if (st.view === "model") {
      if (model) h.set("model", model);
      if (st.interval) h.set("interval", st.interval);
    } else if (day) h.set("day", day);
    navigate({ hash: h.toString() }, { replace: true });
  }, [st.view, st.horizon, st.interval, model, day, navigate]);

  const selFilters = st.view === "model" ? { model, interval: st.interval } : { date_from: day, date_to: day };
  const selEnabled = st.view === "model" ? !!model : !!day;
  const sel = useQuery({
    queryKey: ["eval-stats", st.horizon, selFilters],
    queryFn: () => apiGet<EvalStatsResponse>("/api/evaluation/stats", { ...hq, ...selFilters }),
    enabled: selEnabled,
    refetchInterval: 60_000,
  });
  const gallery = useQuery({
    queryKey: ["eval-gallery", st.view, st.horizon, selFilters, page],
    queryFn: () =>
      apiGet<EvalPredictionsResponse>("/api/evaluation/predictions", {
        ...hq,
        ...selFilters,
        sort: "created_at",
        order: "desc",
        page: st.view === "model" ? page : 1,
        page_size: st.view === "model" ? PAGE_SIZE : DAY_PAGE_SIZE,
      }),
    enabled: selEnabled,
    refetchInterval: 60_000,
    placeholderData: keepPreviousData,
  });
  useEffect(() => {
    if (gallery.dataUpdatedAt) setStamp(gallery.dataUpdatedAt);
  }, [gallery.dataUpdatedAt]);

  const fmtTime = (iso: string) => new Date(iso).toLocaleString(locale, { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });

  // Groups: by local day (model view) or by model · interval (day view).
  const groups = useMemo(() => {
    const rows = gallery.data?.rows ?? [];
    const sorted =
      st.view === "day"
        ? [...rows].sort((a, b) => a.model_name.localeCompare(b.model_name) || INTERVALS.indexOf(a.interval) - INTERVALS.indexOf(b.interval) || b.created_at.localeCompare(a.created_at))
        : rows;
    const out: { key: string; title: string; rows: EvalRow[] }[] = [];
    for (const r of sorted) {
      const key = st.view === "day" ? `${r.model_name}|${r.interval}` : new Date(r.created_at).toDateString();
      const title =
        st.view === "day"
          ? `${r.model_name} · ${r.interval}`
          : new Date(r.created_at).toLocaleDateString(locale, { weekday: "long", day: "2-digit", month: "long", year: "numeric" });
      const last = out[out.length - 1];
      if (last && last.key === key) last.rows.push(r);
      else out.push({ key, title, rows: [r] });
    }
    return out;
  }, [gallery.data, st.view, locale]);

  const breakdown = useMemo(() => {
    const s = sel.data;
    if (!s) return { first: "", rows: [] as { label: string; s: EvalStat; jump: Partial<VState> }[] };
    if (st.view === "model") {
      if (st.interval) {
        return {
          first: t("visual.col.day"),
          rows: [...s.by_created_day].reverse().map((r) => ({ label: fmtUtcDay(r.created_day, locale), s: r, jump: { view: "day" as const, day: r.created_day } })),
        };
      }
      return {
        first: t("visual.col.interval"),
        rows: INTERVALS.flatMap((i) => {
          const r = s.by_interval.find((x) => x.interval === i);
          return r ? [{ label: i, s: r, jump: { interval: i } }] : [];
        }),
      };
    }
    return {
      first: t("visual.col.modelInterval"),
      rows: [...s.by_model_interval]
        .sort((a, b) => a.model.localeCompare(b.model) || INTERVALS.indexOf(a.interval) - INTERVALS.indexOf(b.interval))
        .map((r) => ({ label: `${r.model} · ${r.interval}`, s: r, jump: { view: "model" as const, model: r.model, interval: r.interval } })),
    };
  }, [sel.data, st.view, st.interval, t, locale]);

  const modelCounts = Object.fromEntries((general.data?.by_model ?? []).map((r) => [r.model, r.predictions]));
  const intervalCounts = Object.fromEntries((general.data?.by_model_interval ?? []).filter((r) => r.model === model).map((r) => [r.interval, r.predictions]));
  const err = models.error || days.error || general.error || sel.error || gallery.error;

  const Chip = ({ active, onClick, label, count }: { active: boolean; onClick: () => void; label: string; count?: number }) => (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        "inline-flex shrink-0 items-center gap-2 rounded-xl border px-3 py-1.5 text-[12.5px] font-medium transition-colors",
        active ? "border-bone bg-bone text-page" : "border-line text-dim hover:border-line-strong hover:text-bone",
      )}
    >
      {label}
      {count != null && <span className={clsx("num text-[11px]", active ? "text-page/60" : "text-faint")}>{count}</span>}
    </button>
  );

  return (
    <Page kicker={t("nav.evaluation")} title={t("visual.title")} subtitle={t("visual.subtitle", { tz: userTimeZone() })}>
      <EvalTabs />
      {err && <div className="mb-4"><ErrorBanner message={t("eval.error.load", { msg: err.message })} /></div>}

      <section className="mb-6">
        <div className="eyebrow mb-3">{t("visual.stats.overall")}</div>
        {general.data ? <StatTiles s={general.data.overall} /> : <Skeleton className="h-24" />}
      </section>

      <Panel className="overflow-hidden">
        <div className="space-y-4 border-b border-line p-4 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Segmented
              options={[
                { value: "model", label: t("visual.view.model") },
                { value: "day", label: t("visual.view.day") },
              ]}
              value={st.view}
              onChange={(v) => go({ view: v })}
            />
            <label className="flex items-center gap-2 text-[13px] text-dim">
              {t("visual.filter.horizon")}
              <select className="field" value={st.horizon} onChange={(e) => go({ horizon: e.target.value })}>
                <option value="24">{t("visual.filter.steps", { n: 24 })}</option>
                <option value="">{t("eval.filter.all")}</option>
              </select>
            </label>
          </div>

          {st.view === "model" ? (
            <>
              <NavRow label={t("eval.filter.model")}>
                {modelList.map((m) => <Chip key={m} active={m === model} onClick={() => go({ model: m, interval: "" })} label={m} count={modelCounts[m] ?? 0} />)}
              </NavRow>
              <NavRow label={t("eval.filter.interval")}>
                <Chip active={!st.interval} onClick={() => go({ interval: "" })} label={t("visual.allIntervals")} count={Object.values(intervalCounts).reduce((a, b) => a + b, 0)} />
                {INTERVALS.map((i) => <Chip key={i} active={st.interval === i} onClick={() => go({ interval: i })} label={i} count={intervalCounts[i] ?? 0} />)}
              </NavRow>
            </>
          ) : (
            <NavRow label={t("visual.day")} hint={t("visual.dayHint")}>
              {dayList.map((d) => <Chip key={d.day} active={d.day === day} onClick={() => go({ day: d.day })} label={fmtUtcDay(d.day, locale, { year: undefined })} count={d.predictions} />)}
            </NavRow>
          )}
        </div>

        {/* selection summary */}
        <div className="border-b border-line bg-raised/40 p-4 sm:p-5">
          <div className="num mb-3 text-[13px] font-semibold text-accent">
            {st.view === "model" ? `${model} · ${st.interval || t("visual.allIntervals")}` : day ? fmtUtcDay(day, locale) : ""}
          </div>
          {sel.data ? <StatTiles s={sel.data.overall} /> : selEnabled ? <Skeleton className="h-24" /> : <StatTiles s={null} />}
          {breakdown.rows.length > 0 && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[480px] border-collapse text-[12.5px]">
                <thead>
                  <tr className="text-[11px] text-faint">
                    {[breakdown.first, t("visual.col.predictions"), t("visual.col.completed"), t("visual.col.error"), t("visual.col.direction")].map((h, i) => (
                      <th key={h} className={clsx("py-2 font-medium", i === 0 ? "text-left" : "text-right")}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.rows.map((r) => (
                    <tr key={r.label} onClick={() => go(r.jump)} className="cursor-pointer border-t border-line text-dim transition-colors hover:bg-hover hover:text-bone">
                      <td className="py-2 pr-3">{r.label}</td>
                      <td className="num py-2 text-right">{r.s.predictions}</td>
                      <td className="num py-2 text-right">{r.s.completed}</td>
                      <td className="num py-2 text-right">{fmtPct(r.s.mean_abs_pct_error, { signed: false })}</td>
                      <td className="num py-2 text-right">{r.s.direction_accuracy_pct == null ? "—" : `${r.s.direction_accuracy_pct.toFixed(1)}%`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-[11px] text-faint">{t("visual.jumpHint")}</p>
            </div>
          )}
        </div>

        {/* gallery */}
        <div className="p-4 sm:p-5">
          {!gallery.data && selEnabled ? (
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="aspect-[10/7]" />)}</div>
          ) : !groups.length ? (
            <Empty className="py-16">{t("visual.empty")}</Empty>
          ) : (
            <div className="space-y-8">
              {groups.map((g) => (
                <section key={g.key}>
                  <h3 className="mb-3 flex items-baseline gap-2 font-display text-[15px] font-semibold capitalize text-bone">
                    {g.title}
                    <span className="num text-xs font-normal text-faint">{g.rows.length}</span>
                  </h3>
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {g.rows.map((r) => (
                      <SnapshotCard
                        key={r.id}
                        r={r}
                        stamp={stamp}
                        time={fmtTime(r.created_at)}
                        onOpen={() => setViewer({ src: `${r.snapshot_url}?t=${Date.now()}`, alt: `${r.model_name} · ${r.interval} · ${fmtTime(r.created_at)}` })}
                      />
                    ))}
                  </div>
                </section>
              ))}
            </div>
          )}
        </div>
        {st.view === "model" && gallery.data && (
          <Pager page={gallery.data.page} pages={gallery.data.pages} info={t("eval.page", { page: gallery.data.page, pages: gallery.data.pages, total: gallery.data.total })} onPage={setPage} />
        )}
      </Panel>

      <ImageViewer src={viewer?.src ?? null} alt={viewer?.alt} onClose={() => setViewer(null)} />
    </Page>
  );
}

function NavRow({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:gap-4">
      <div className="eyebrow shrink-0 pt-2 sm:w-20">{label}</div>
      <div className="min-w-0 flex-1">
        <div className="flex gap-1.5 overflow-x-auto pb-1 sm:flex-wrap">{children}</div>
        {hint && <p className="mt-1 text-[11px] text-faint">{hint}</p>}
      </div>
    </div>
  );
}

function SnapshotCard({ r, stamp, time, onOpen }: { r: EvalRow; stamp: number; time: string; onOpen: () => void }) {
  const { t } = useI18n();
  const actualLen = Array.isArray(r.actual_path) ? r.actual_path.length : 0;
  const dc = r.direction_correct == null ? null : !!r.direction_correct;
  return (
    <figure
      onClick={onOpen}
      className="group m-0 cursor-pointer overflow-hidden rounded-xl border border-line bg-raised transition-all duration-200 hover:-translate-y-0.5 hover:border-line-strong"
    >
      <div className="overflow-hidden bg-[#19171d]">
        <img loading="lazy" src={`${r.snapshot_url}?t=${stamp}`} alt="" className="block aspect-[10/6] w-full object-cover transition-transform duration-500 group-hover:scale-[1.02]" />
      </div>
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2.5 text-xs">
        <span className="num font-medium text-bone">{time}</span>
        <span className="num flex-1 text-faint">{r.interval} · {t("visual.filter.steps", { n: r.horizon_steps })}</span>
        {r.status === "completed" ? (
          <span className={clsx("num inline-flex items-center gap-1", dc ? "text-up" : "text-down")}>
            {fmtPct(r.pct_error)} {dc ? <Check className="size-3.5" /> : <X className="size-3.5" />}
          </span>
        ) : r.status === "expired" ? (
          <StatusPill status="expired" />
        ) : (
          <span className="num inline-flex items-center gap-1.5 text-warn">
            <span className="h-1 w-10 overflow-hidden rounded-full bg-press">
              <span className="block h-full bg-warn" style={{ width: `${(actualLen / Math.max(1, r.horizon_steps)) * 100}%` }} />
            </span>
            {actualLen}/{r.horizon_steps}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
