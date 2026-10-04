import clsx from "clsx";
import type { ReactNode } from "react";
import { NavLink } from "react-router";
import { CalendarDays, Images, Table2 } from "lucide-react";
import type { EvalStat } from "../../lib/api";
import { fmtPct } from "../../lib/format";
import { useI18n } from "../../i18n";

export function EvalTabs() {
  const { t } = useI18n();
  const tabs = [
    { to: "/evaluation", label: t("ui.eval.tabPredictions"), Icon: Table2 },
    { to: "/history", label: t("nav.history"), Icon: CalendarDays },
    { to: "/visual", label: t("nav.visual"), Icon: Images },
  ];
  return (
    <nav className="mb-8 flex gap-1.5 overflow-x-auto" aria-label="Evaluation">
      {tabs.map(({ to, label, Icon }) => (
        <NavLink
          key={to}
          to={to}
          end
          className={({ isActive }) =>
            clsx(
              "inline-flex h-10 shrink-0 items-center gap-2 rounded-full border px-4 text-[13px] font-medium transition-all",
              isActive
                ? "border-accent/40 bg-accent/15 text-bone shadow-[0_0_24px_-8px_rgb(255_106_31/0.9)]"
                : "border-line bg-white/[0.03] text-dim hover:border-line-strong hover:text-bone",
            )
          }
        >
          <Icon className="size-3.5" /> {label}
        </NavLink>
      ))}
    </nav>
  );
}

/** Big-number stat row used by every evaluation view. */
export function StatTiles({ s, className }: { s: EvalStat | null | undefined; className?: string }) {
  const { t } = useI18n();
  if (!s) return <p className="text-[13px] text-faint">{t("eval.summary.empty")}</p>;
  const dirOk = s.direction_accuracy_pct == null ? null : s.direction_accuracy_pct >= 50;
  return (
    <div className={clsx("grid grid-cols-2 gap-px overflow-hidden rounded-[20px] border border-line bg-line sm:grid-cols-3 xl:grid-cols-5", className)}>
      <Tile label={t("eval.summary.predictions")} value={s.predictions} sub={`${s.completed} ${t("eval.summary.completed")} · ${s.pending} ${t("eval.summary.pending")}${s.expired ? ` · ${s.expired} ${t("eval.status.expired")}` : ""}`} />
      <Tile label={t("eval.summary.meanError")} value={fmtPct(s.mean_abs_pct_error, { signed: false })} sub={s.completed ? undefined : t("visual.stats.noneYet")} />
      {s.mean_abs_error_usd !== undefined && <Tile label={t("visual.stats.usd")} value={s.mean_abs_error_usd == null ? "—" : `$${s.mean_abs_error_usd.toFixed(2)}`} />}
      <Tile
        label={t("eval.summary.direction")}
        value={s.direction_accuracy_pct == null ? "—" : `${s.direction_accuracy_pct.toFixed(1)}%`}
        valueClass={dirOk == null ? "" : dirOk ? "text-up" : "text-down"}
        bar={s.direction_accuracy_pct}
      />
      {s.mean_pct_error !== undefined && <Tile label={t("visual.stats.bias")} value={fmtPct(s.mean_pct_error)} sub={t("visual.stats.biasHint")} />}
    </div>
  );
}

function Tile({ label, value, sub, valueClass, bar }: { label: string; value: ReactNode; sub?: string; valueClass?: string; bar?: number | null }) {
  return (
    <div className="flex flex-col gap-1 bg-panel/90 px-5 py-4 backdrop-blur-md">
      <span className="eyebrow">{label}</span>
      <span className={clsx("num text-[26px] font-semibold leading-tight tracking-[-0.03em] text-bone", valueClass)}>{value}</span>
      {bar != null && (
        <span className="mt-1 h-1 overflow-hidden rounded-full bg-press">
          <span className={clsx("block h-full rounded-full", bar >= 50 ? "bg-up" : "bg-down")} style={{ width: `${Math.min(100, bar)}%` }} />
        </span>
      )}
      {sub && <span className="text-[11px] leading-snug text-faint">{sub}</span>}
    </div>
  );
}

export function StatusPill({ status }: { status: string }) {
  const { t } = useI18n();
  const tone = status === "completed" ? "bg-up/10 text-up" : status === "pending" ? "bg-warn/12 text-warn" : "bg-press text-faint";
  return <span className={clsx("inline-flex items-center gap-1.5 rounded-xs px-2 py-0.5 text-[11px] font-medium", tone)}>
    <span className={clsx("size-1.5 rounded-full bg-current", status === "pending" && "animate-pulse")} />
    {t(`eval.status.${status}`)}
  </span>;
}
