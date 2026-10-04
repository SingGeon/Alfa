import clsx from "clsx";
import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { Dna, House, LineChart, Menu, Radar, Target, Wallet, X, type LucideIcon } from "lucide-react";
import { apiGet, type CurrentPrice } from "../lib/api";
import { fmtMoney } from "../lib/format";
import { useI18n, type Lang } from "../i18n";
import { Delta } from "./ui";

/** The original "ETH Price Predictor" logo (public/logo.png, 769x160). */
export function Logo({ className }: { className?: string }) {
  return <img src="/logo.png" alt="ETH Price Predictor" width={154} height={32} className={clsx("block h-8 w-auto", className)} />;
}

/** Generic Ethereum diamond glyph (not a brand logo file). */
export function EthGlyph({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden>
      <path d="M12 2 5.5 12.3 12 16l6.5-3.7L12 2Z" fill="currentColor" opacity="0.9" />
      <path d="M12 17.3 5.5 13.6 12 22l6.5-8.4-6.5 3.7Z" fill="currentColor" opacity="0.55" />
    </svg>
  );
}

const NAV: { to: string; key: string; match: string[]; Icon: LucideIcon; exact?: boolean }[] = [
  { to: "/", key: "nav.home", match: ["/"], Icon: House, exact: true },
  { to: "/dashboard", key: "nav.dashboard", match: ["/dashboard"], Icon: LineChart },
  { to: "/scout", key: "nav.scout", match: ["/scout", "/asset"], Icon: Radar },
  { to: "/evaluation", key: "nav.evaluation", match: ["/evaluation", "/history", "/visual"], Icon: Target },
  { to: "/population", key: "nav.population", match: ["/population"], Icon: Dna },
  { to: "/fund", key: "nav.fund", match: ["/fund"], Icon: Wallet },
];

function LangToggle() {
  const { lang, setLang, t } = useI18n();
  return (
    <div role="group" aria-label={t("nav.langToggleLabel")} className="inline-flex rounded-full border border-line bg-white/[0.03] p-1 backdrop-blur-md">
      {(["en", "ro"] as Lang[]).map((l) => (
        <button
          key={l}
          type="button"
          onClick={() => setLang(l)}
          aria-pressed={lang === l}
          className={clsx(
            "rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider transition-colors",
            lang === l ? "bg-white/10 text-bone" : "text-faint hover:text-bone",
          )}
        >
          {l}
        </button>
      ))}
    </div>
  );
}

/** Always-on ETH quote, like the network selector in the reference. Shares
 *  its query key with the dashboard ticker (one request per tick). */
function EthChip() {
  const { data } = useQuery({
    queryKey: ["price-current"],
    queryFn: () => apiGet<CurrentPrice>("/api/price/current"),
    refetchInterval: 5000,
  });
  return (
    <Link
      to="/dashboard"
      className="hidden items-center gap-2.5 rounded-full border border-line bg-white/[0.03] py-1 pl-1 pr-3.5 text-xs backdrop-blur-md transition-colors hover:border-line-strong lg:inline-flex"
    >
      <span className="flex size-7 items-center justify-center rounded-full bg-accent/20 text-accent-hi">
        <EthGlyph className="size-4" />
      </span>
      <span className="num font-medium text-bone">{data ? fmtMoney(data.price) : "—"}</span>
      {data?.change_24h_pct != null && <Delta value={data.change_24h_pct} arrow={false} className="text-[11px]" />}
    </Link>
  );
}

function useHideOnScroll() {
  const [hidden, setHidden] = useState(false);
  useEffect(() => {
    let last = window.scrollY;
    let ticking = false;
    const onScroll = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const y = window.scrollY;
        setHidden(y > last && y > 140);
        last = y;
        ticking = false;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  return hidden;
}

/** One pointer listener drives the cursor-following ring on every .panel. */
function usePanelSpotlight() {
  useEffect(() => {
    let lastPanel: HTMLElement | null = null;
    const onMove = (e: PointerEvent) => {
      const panel = (e.target as Element | null)?.closest?.(".panel") as HTMLElement | null;
      if (lastPanel && lastPanel !== panel) {
        lastPanel.style.removeProperty("--mx");
        lastPanel.style.removeProperty("--my");
      }
      lastPanel = panel;
      if (!panel) return;
      const r = panel.getBoundingClientRect();
      panel.style.setProperty("--mx", `${e.clientX - r.left}px`);
      panel.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);
}

export function TopNav() {
  const { t } = useI18n();
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const hidden = useHideOnScroll();

  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  const isActive = (n: (typeof NAV)[number]) => (n.exact ? pathname === n.to : n.match.some((m) => pathname.startsWith(m)));

  return (
    <header className={clsx("sticky top-0 z-40 px-3 pt-3 transition-transform duration-500 ease-out-quint sm:px-5", hidden && !open && "-translate-y-[120%]")}>
      <div className="relative mx-auto flex h-14 max-w-[1440px] items-center gap-4 rounded-full border border-line bg-page/55 pl-4 pr-2 shadow-[0_20px_50px_-25px_rgb(0_0_0/0.9)] backdrop-blur-xl sm:pl-5">
        <Link to="/" aria-label="ETH Price Predictor — home" className="shrink-0 transition-opacity hover:opacity-85">
          <Logo className="h-7" />
        </Link>

        {/* center icon pills: the active one expands with its label */}
        <nav className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-1.5 md:flex" aria-label="Main">
          {NAV.map((n) => {
            const active = isActive(n);
            return (
              <NavLink
                key={n.to}
                to={n.to}
                title={t(n.key)}
                aria-label={t(n.key)}
                className={clsx(
                  "group flex h-10 items-center gap-2 rounded-full border transition-all duration-300 ease-out-quint",
                  active
                    ? "border-accent/40 bg-accent/15 pl-3 pr-4 text-bone shadow-[0_0_24px_-6px_rgb(255_106_31/0.8)]"
                    : "w-10 justify-center border-line bg-white/[0.03] text-dim hover:border-line-strong hover:text-bone",
                )}
              >
                <n.Icon className={clsx("size-4 shrink-0", active && "text-accent-hi")} strokeWidth={1.8} />
                {active && <span className="text-[13px] font-medium animate-fade">{t(n.key)}</span>}
              </NavLink>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <EthChip />
          <LangToggle />
          <button
            type="button"
            className="inline-flex size-9 items-center justify-center rounded-full border border-line text-dim md:hidden"
            aria-expanded={open}
            aria-label="Menu"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="size-4" /> : <Menu className="size-4" />}
          </button>
        </div>
      </div>

      {open && (
        <nav className="panel mx-auto mt-2 max-w-[1440px] p-2 md:hidden animate-fade" aria-label="Main">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              className={clsx(
                "flex items-center gap-3 rounded-2xl px-4 py-3 text-[15px]",
                isActive(n) ? "bg-accent/15 text-bone" : "text-dim",
              )}
            >
              <n.Icon className={clsx("size-4", isActive(n) && "text-accent-hi")} />
              {t(n.key)}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  );
}

export function Footer() {
  const { t } = useI18n();
  return (
    <footer className="relative z-[1] mt-20 border-t border-line">
      <div className="mx-auto flex max-w-[1440px] flex-col gap-4 px-4 py-10 text-xs text-faint sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <Logo className="h-7 opacity-80" />
        <p className="max-w-[70ch] leading-relaxed">{t("landing.footer.disclaimer")}</p>
      </div>
    </footer>
  );
}

export function AppShell() {
  const { pathname } = useLocation();
  usePanelSpotlight();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return (
    <div className="flex min-h-dvh flex-col">
      <TopNav />
      <main className="relative z-[1] flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

/** Standard page frame: pill kicker, gradient title, consistent gutters. */
export function Page({ title, kicker, subtitle, actions, children, wide = true }: { title?: ReactNode; kicker?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; wide?: boolean }) {
  return (
    <div className={clsx("mx-auto px-4 pb-6 pt-8 sm:px-6 sm:pt-12", wide ? "max-w-[1440px]" : "max-w-[1100px]")}>
      {(title || actions) && (
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4 animate-rise">
          <div className="min-w-0">
            {kicker && <div className="badge-pill mb-4">{kicker}</div>}
            {title && <h1 className="text-gradient pb-1 text-[32px] font-semibold leading-[1.05] tracking-[-0.035em] sm:text-[44px]">{title}</h1>}
            {subtitle && <p className="mt-3 max-w-[80ch] text-[15px] leading-relaxed text-dim">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
}
