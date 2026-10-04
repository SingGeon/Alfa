import clsx from "clsx";
import { useEffect, useRef, type ReactNode } from "react";
import { AlertTriangle, ArrowDownRight, ArrowUpRight, ChevronLeft, ChevronRight, Minus, X } from "lucide-react";
import { dir, dirText, fmtPct, qualityBg, qualityText } from "../lib/format";
import { useI18n } from "../i18n";

// ------------------------------------------------------------- Panel

export function Panel({
  children,
  className,
  as: Tag = "section",
  ...rest
}: { children: ReactNode; className?: string; as?: "section" | "div" | "aside" | "article" } & React.HTMLAttributes<HTMLElement>) {
  return (
    <Tag className={clsx("panel", className)} {...rest}>
      {children}
    </Tag>
  );
}

/** Mono, uppercase panel heading with an optional index ("01") and actions. */
export function PanelHeader({
  index,
  title,
  hint,
  actions,
  className,
}: {
  index?: string;
  title: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <header className={clsx("flex flex-wrap items-start justify-between gap-x-4 gap-y-2", className)}>
      <div className="min-w-0">
        <h2 className="flex items-center gap-2.5 text-[15px] font-medium tracking-[-0.01em] text-bone">
          {index && <span className="num rounded-full border border-line px-2 py-0.5 text-[11px] text-faint">{index}</span>}
          <span>{title}</span>
        </h2>
        {hint && <p className="mt-1.5 max-w-prose text-xs leading-relaxed text-faint">{hint}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

// ------------------------------------------------------------- Segmented

export interface SegOption<V extends string> {
  value: V;
  label: ReactNode;
  title?: string;
}

export function Segmented<V extends string>({
  options,
  value,
  onChange,
  size = "md",
  ariaLabel,
  className,
}: {
  options: SegOption<V>[];
  value: V | null;
  onChange: (v: V) => void;
  size?: "sm" | "md";
  ariaLabel?: string;
  className?: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className={clsx("inline-flex rounded-full border border-line bg-white/[0.03] p-1", className)}>
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={o.title}
            onClick={() => onChange(o.value)}
            className={clsx(
              "relative whitespace-nowrap rounded-full font-medium transition-all duration-200",
              size === "sm" ? "px-3 py-1 text-xs" : "px-3.5 py-1.5 text-[13px]",
              active
                ? "bg-accent/20 text-bone shadow-[inset_0_0_0_1px_rgb(255_138_76/0.45),0_0_18px_-4px_rgb(255_106_31/0.7)] after:absolute after:inset-x-3 after:-bottom-px after:h-px after:bg-gradient-to-r after:from-transparent after:via-accent-hi after:to-transparent"
                : "text-dim hover:bg-white/5 hover:text-bone",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// ------------------------------------------------------------- Switch

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer select-none items-center gap-2 text-[13px] text-dim hover:text-bone">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={clsx(
          "relative h-[18px] w-8 shrink-0 rounded-full border transition-colors duration-200",
          checked ? "border-accent bg-accent" : "border-line-strong bg-press",
        )}
      >
        <span
          className={clsx(
            "absolute top-1/2 size-3 -translate-y-1/2 rounded-full transition-all duration-200 ease-out-quint",
            checked ? "left-[15px] bg-accent-ink" : "left-[2px] bg-dim",
          )}
        />
      </button>
      {label}
    </label>
  );
}

// ------------------------------------------------------------- Button

export function Button({
  children,
  variant = "ghost",
  size = "md",
  className,
  ...rest
}: { variant?: "primary" | "ghost" | "outline"; size?: "sm" | "md" | "lg" } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button type="button" className={clsx(buttonClass(variant, size), className)} {...rest}>
      {children}
    </button>
  );
}

export function buttonClass(variant: "primary" | "ghost" | "outline" = "ghost", size: "sm" | "md" | "lg" = "md") {
  return clsx(
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-full font-medium transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-40 active:scale-[0.98]",
    size === "sm" && "h-8 px-3 text-xs",
    size === "md" && "h-9 px-4 text-[13px]",
    size === "lg" && "h-12 px-6 text-[15px]",
    variant === "primary" &&
      "bg-gradient-to-b from-accent-hi to-accent text-accent-ink shadow-[inset_0_1px_0_rgb(255_255_255/0.35),0_10px_30px_-8px_rgb(255_106_31/0.85)] hover:shadow-[inset_0_1px_0_rgb(255_255_255/0.45),0_14px_40px_-8px_rgb(255_106_31/1)] hover:brightness-110",
    variant === "ghost" && "text-dim hover:bg-white/5 hover:text-bone",
    variant === "outline" && "border border-line-strong bg-white/[0.03] text-bone backdrop-blur-md hover:border-white/25 hover:bg-white/[0.07]",
  );
}

// ------------------------------------------------------------- values

export function Delta({ value, className, digits = 2, arrow = true }: { value: number | null | undefined; className?: string; digits?: number; arrow?: boolean }) {
  const d = dir(value);
  const Icon = d === "up" ? ArrowUpRight : d === "down" ? ArrowDownRight : Minus;
  return (
    <span className={clsx("num inline-flex items-center gap-0.5", dirText(value), className)}>
      {arrow && value != null && <Icon className="size-[1em] shrink-0" strokeWidth={2.5} aria-hidden />}
      {fmtPct(value, { digits })}
    </span>
  );
}

export function Stat({
  label,
  value,
  hint,
  className,
  valueClass,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  className?: string;
  valueClass?: string;
}) {
  return (
    <div className={clsx("flex min-w-0 flex-col gap-1", className)}>
      <span className="text-xs text-faint">{label}</span>
      <span className={clsx("num text-lg font-medium leading-tight text-bone", valueClass)}>{value}</span>
      {hint && <span className="text-[11px] leading-snug text-faint">{hint}</span>}
    </div>
  );
}

export function ConfidenceMeter({ value, compact = false }: { value: number | null | undefined; compact?: boolean }) {
  if (value == null) return <span className="num text-faint">—</span>;
  const pct = Math.max(0, Math.min(100, value));
  return (
    <span className={clsx("inline-flex items-center gap-2", qualityText(value))}>
      <span className={clsx("num font-medium", compact ? "w-6 text-right" : "")}>{Math.round(value)}</span>
      <span className={clsx("relative h-1 overflow-hidden rounded-full bg-press", compact ? "w-10" : "w-16")}>
        <span className={clsx("absolute inset-y-0 left-0 rounded-full", qualityBg(value))} style={{ width: `${pct}%` }} />
      </span>
    </span>
  );
}

export function Badge({ children, tone = "neutral", className }: { children: ReactNode; tone?: "neutral" | "up" | "down" | "warn" | "accent" | "violet" | "rose" | "sky"; className?: string }) {
  const tones: Record<string, string> = {
    neutral: "bg-press text-dim",
    up: "bg-up/12 text-up",
    down: "bg-down/12 text-down",
    warn: "bg-warn/14 text-warn",
    accent: "bg-accent/14 text-accent",
    violet: "bg-violet/14 text-violet",
    rose: "bg-rose/14 text-rose",
    sky: "bg-sky/14 text-sky",
  };
  return (
    <span className={clsx("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] font-medium", tones[tone], className)}>
      {children}
    </span>
  );
}

export function SentimentTag({ label, compound }: { label?: string; compound?: number }) {
  const { t } = useI18n();
  const safe = label === "positive" || label === "negative" ? label : "neutral";
  return (
    <span className={clsx("inline-flex items-center gap-1.5 text-xs font-medium", safe === "positive" ? "text-up" : safe === "negative" ? "text-down" : "text-faint")}>
      <span className={clsx("size-1.5 rounded-full", safe === "positive" ? "bg-up" : safe === "negative" ? "bg-down" : "bg-faint")} />
      {t(`common.sentiment.${safe}`)}
      {compound !== undefined && <span className="num text-faint">{compound >= 0 ? "+" : ""}{compound.toFixed(2)}</span>}
    </span>
  );
}

// ------------------------------------------------------------- states

export function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden className={clsx("skeleton block", className)} />;
}

export function ErrorBanner({ message, onClose }: { message: string | null | undefined; onClose?: () => void }) {
  if (!message) return null;
  return (
    <div role="alert" className="flex items-start gap-3 rounded-2xl border border-down/30 bg-down/8 px-4 py-3 text-[13px] text-[#ffc2c7] backdrop-blur-md animate-fade">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-down" aria-hidden />
      <span className="flex-1">{message}</span>
      {onClose && (
        <button type="button" onClick={onClose} className="text-down/70 hover:text-down" aria-label="Close">
          <X className="size-4" />
        </button>
      )}
    </div>
  );
}

export function Empty({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={clsx("flex items-center justify-center px-4 py-10 text-center text-[13px] text-faint", className)}>{children}</div>;
}

// ------------------------------------------------------------- pager

export function Pager({ page, pages, info, onPage }: { page: number; pages: number; info: ReactNode; onPage: (p: number) => void }) {
  const { t } = useI18n();
  return (
    <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3 text-[13px] text-dim sm:px-5">
      <Button size="sm" variant="outline" disabled={page <= 1} onClick={() => onPage(page - 1)}>
        <ChevronLeft className="size-3.5" /> <span className="hidden sm:inline">{t("eval.prev")}</span>
      </Button>
      <span className="num text-xs text-faint">{info}</span>
      <Button size="sm" variant="outline" disabled={page >= pages} onClick={() => onPage(page + 1)}>
        <span className="hidden sm:inline">{t("eval.next")}</span> <ChevronRight className="size-3.5" />
      </Button>
    </div>
  );
}

// ------------------------------------------------------------- image viewer

export function ImageViewer({ src, alt, onClose }: { src: string | null; alt?: string; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const { t } = useI18n();
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (src && !d.open) d.showModal();
    if (!src && d.open) d.close();
  }, [src]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className="viewer m-auto max-h-[92dvh] w-[min(1100px,calc(100vw-24px))] overflow-hidden rounded-3xl border border-line-strong bg-panel p-0 text-bone"
    >
      <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
        <span className="eyebrow truncate">{alt}</span>
        <Button size="sm" onClick={onClose} aria-label={t("eval.close")}>
          <X className="size-4" /> {t("eval.close")}
        </Button>
      </div>
      {src && <img src={src} alt={alt ?? ""} className="block h-auto w-full" />}
    </dialog>
  );
}
