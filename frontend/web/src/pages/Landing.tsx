import clsx from "clsx";
import { lazy, Suspense, useMemo } from "react";
import { Link } from "react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  BrainCircuit,
  CandlestickChart,
  Gauge,
  MousePointerClick,
  Newspaper,
  Radar,
  Radio,
  ShieldCheck,
  Sigma,
  Target,
  Timer,
} from "lucide-react";
import { apiGet, type Candle, type CurrentPrice, type EvalStatsResponse, type PredictResponse, type ScoutResult } from "../lib/api";
import { computeSignal } from "../lib/indicators";
import { fmtMoney, fmtPct } from "../lib/format";
import { useI18n } from "../i18n";
import { HeroChart } from "../components/HeroChart";
import { ArrowCircle, BigPrice, Sparkline, TokenIcon } from "../components/market";
import { Badge, buttonClass, ConfidenceMeter, Delta } from "../components/ui";

// three.js is heavy: load the beams only on this page, after first paint.
const Beams = lazy(() => import("../components/Beams"));

const closesOf = (c: Candle[] | undefined) =>
  (c ?? []).slice().sort((a, b) => +new Date(a.timestamp) - +new Date(b.timestamp)).map((x) => x.close);

// Real sources the backend reads from (data_collector/, config.NEWS_RSS_FEEDS).
const SOURCES = ["Binance", "CoinGecko", "DeFiLlama", "Etherscan", "Yahoo Finance", "CoinDesk", "Cointelegraph", "Decrypt", "The Block", "CryptoSlate", "BeInCrypto", "U.Today"];

export default function Landing() {
  const { t } = useI18n();

  const price = useQuery({ queryKey: ["price-current"], queryFn: () => apiGet<CurrentPrice>("/api/price/current"), refetchInterval: 5000 });
  const hist = useQuery({ queryKey: ["landing-hist"], queryFn: () => apiGet<{ candles: Candle[] }>("/api/price/history", { interval: "1h", limit: 96 }), retry: false, refetchInterval: 60_000 });
  const hist30 = useQuery({ queryKey: ["landing-hist-1d"], queryFn: () => apiGet<{ candles: Candle[] }>("/api/price/history", { interval: "1d", limit: 30 }), retry: false });
  const hist1y = useQuery({ queryKey: ["landing-hist-1w"], queryFn: () => apiGet<{ candles: Candle[] }>("/api/price/history", { interval: "1w", limit: 52 }), retry: false });
  // Same parameters as the dashboard's first load, so this request also
  // warms the exact server-side prediction cache entry the dashboard needs.
  const pred = useQuery({
    queryKey: ["predict", "1h", 24, true, "legacy"],
    queryFn: () => apiGet<PredictResponse>("/api/predict", { interval: "1h", steps: 24, use_sentiment: true, backend: "sklearn", model_variant: "legacy" }),
    retry: false,
    refetchInterval: 60_000,
  });
  const stats = useQuery({ queryKey: ["eval-stats", "24"], queryFn: () => apiGet<EvalStatsResponse>("/api/evaluation/stats", { horizon_steps: 24 }), retry: false });
  const scout = useQuery({ queryKey: ["scout", ""], queryFn: () => apiGet<{ results: ScoutResult[] }>("/api/scout"), retry: false });

  const forecast = pred.data?.predictions.at(-1);
  const forecastPct = forecast && pred.data ? ((forecast.predicted_price - pred.data.last_known_price) / pred.data.last_known_price) * 100 : null;
  const overall = stats.data?.overall;
  const topPick = scout.data?.results?.[0];

  const signal = useMemo(() => {
    const c = hist.data?.candles;
    if (!c?.length || !pred.data) return null;
    const sorted = [...c].sort((a, b) => +new Date(a.timestamp) - +new Date(b.timestamp));
    return computeSignal(sorted, pred.data.predictions, pred.data.last_known_price, pred.data.confidence, pred.data.sentiment_avg);
  }, [hist.data, pred.data]);

  const c24 = closesOf(hist.data?.candles).slice(-24);
  const c30 = closesOf(hist30.data?.candles);
  const c1y = closesOf(hist1y.data?.candles);
  const pctOf = (xs: number[]) => (xs.length > 1 ? ((xs.at(-1)! - xs[0]) / xs[0]) * 100 : null);

  const steps = [
    { Icon: Radio, title: t("landing.step1.title"), body: t("landing.step1.body") },
    { Icon: Sigma, title: t("landing.step2.title"), body: t("landing.step2.body") },
    { Icon: BrainCircuit, title: t("landing.step3.title"), body: t("landing.step3.body") },
    { Icon: Target, title: t("landing.step4.title"), body: t("landing.step4.body") },
  ];

  return (
    <div className="relative overflow-x-clip">
      {/* ============================================================ background: React Bits "Beams" */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 -top-[76px] z-0 h-[1180px] animate-fade [mask-image:linear-gradient(180deg,#000_55%,transparent_100%)]"
      >
        <Suspense fallback={null}>
          <Beams beamWidth={2.4} beamHeight={18} beamNumber={14} lightColor="#ff7a2e" backgroundColor="#050505" speed={1.6} noiseIntensity={1.6} scale={0.18} rotation={32} />
        </Suspense>
        {/* soften the beams behind the copy so the headline stays crisp */}
        <div className="absolute inset-0 bg-[radial-gradient(60%_45%_at_50%_32%,rgb(5_5_5/0.72),transparent_75%)]" />
      </div>

      {/* ============================================================ hero */}
      <section className="relative z-[1] mx-auto max-w-[1440px] px-4 pt-14 text-center sm:px-6 sm:pt-20">
        <Link
          to="/dashboard"
          className="group inline-flex items-center gap-2.5 rounded-full border border-line-strong bg-white/[0.04] py-1.5 pl-1.5 pr-4 text-[13px] text-dim backdrop-blur-md transition-colors hover:border-accent/50 hover:text-bone animate-rise"
        >
          <span className="inline-flex items-center gap-1.5 rounded-full bg-up/15 px-2 py-0.5 text-[11px] font-medium text-up">
            <span className="size-1.5 rounded-full bg-up animate-pulse-dot" /> {t("ui.landing.live")}
          </span>
          {t("ui.hero.pill")}
          <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
        </Link>

        <h1 className="mx-auto mt-7 max-w-[16ch] pb-2 text-[clamp(40px,6.6vw,88px)] font-semibold leading-[1] tracking-[-0.045em] animate-rise [animation-delay:80ms]">
          <span className="text-gradient">{t("ui.hero.title1")}</span>{" "}
          <span className="text-gradient-accent">{t("ui.hero.title2")}</span>
        </h1>
        <p className="mx-auto mt-6 max-w-[58ch] text-[16px] leading-relaxed text-dim sm:text-[17px] animate-rise [animation-delay:160ms]">{t("ui.hero.sub")}</p>
        <div className="mt-9 flex flex-wrap items-center justify-center gap-3 animate-rise [animation-delay:240ms]">
          <Link to="/dashboard" className={buttonClass("primary", "lg")}>
            {t("landing.cta.primary")} <ArrowRight className="size-4" />
          </Link>
          <Link to="/scout" className={buttonClass("outline", "lg")}>
            <Radar className="size-4" /> {t("landing.cta.secondary")}
          </Link>
        </div>
        <p className="mt-5 inline-flex items-center gap-2 text-xs text-faint animate-rise [animation-delay:300ms]">
          <ShieldCheck className="size-3.5" /> {t("landing.disclaimer")}
        </p>

        {/* ---------------------------------------------- floating cards over the ETH gem */}
        <div className="relative mt-14 pb-10 sm:mt-16">
          <div className="relative z-[1] grid items-start gap-5 text-left lg:grid-cols-[1fr_1.25fr_1fr]">
            {/* markets */}
            <div className="panel p-5 lg:mt-16 animate-rise [animation-delay:320ms]">
              <div className="animate-float-slow">
                <div className="mb-4 flex items-center justify-between">
                  <span className="text-[15px] font-medium text-bone">{t("ui.hero.markets")}</span>
                  <Link to="/dashboard" className="text-xs text-accent-hi hover:underline">{t("ui.hero.seeAll")}</Link>
                </div>
                <div className="divide-y divide-line">
                  {[
                    { label: "24H", xs: c24 },
                    { label: "30D", xs: c30 },
                    { label: "1Y", xs: c1y },
                  ].map((r) => (
                    <div key={r.label} className="flex items-center gap-3 py-3">
                      <TokenIcon symbol="ETH" eth size="sm" />
                      <div className="w-[76px]">
                        <div className="text-[13px] font-medium text-bone">ETH · {r.label}</div>
                        <div className="num text-[11px] text-faint">{r.xs.length ? fmtMoney(r.xs[0]) : "—"}</div>
                      </div>
                      <Sparkline values={r.xs} className="h-8 flex-1" />
                      <Delta value={pctOf(r.xs)} className="w-[72px] justify-end text-[13px] font-medium" />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* device with the live forecast chart */}
            <Link to="/dashboard" className="group block animate-rise [animation-delay:240ms]">
              <div className="ring-gradient panel overflow-hidden rounded-[28px] !bg-[rgb(10_11_16/0.82)] shadow-[0_40px_120px_-30px_rgb(255_106_31/0.55)]">
                <div className="flex items-start justify-between gap-4 px-6 pt-6">
                  <div>
                    <div className="eyebrow flex items-center gap-2">
                      <TokenIcon symbol="ETH" eth size="sm" /> ETH / USD · 1H
                    </div>
                    <div className="mt-3 flex items-baseline gap-3">
                      <BigPrice value={price.data?.price} className="text-[34px] font-semibold tracking-[-0.03em] text-bone" />
                      {price.data?.change_24h_pct != null && <Delta value={price.data.change_24h_pct} className="text-sm" />}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="eyebrow">{t("ui.landing.forecast24")}</div>
                    <div className="mt-3">{forecastPct != null ? <Delta value={forecastPct} className="text-xl font-medium" /> : <span className="num text-xl text-faint">—</span>}</div>
                  </div>
                </div>
                <div className="px-4 pb-2 pt-4">
                  <HeroChart candles={hist.data?.candles} predictions={pred.data?.predictions} />
                </div>
                <div className="flex items-center justify-between border-t border-line px-6 py-4 text-[13px]">
                  <span className="flex items-center gap-3 text-faint">
                    {t("common.stat.confidence")} <ConfidenceMeter value={pred.data?.confidence} />
                  </span>
                  <span className="flex items-center gap-2 text-dim group-hover:text-bone">
                    {t("ui.landing.open")} <ArrowCircle className="size-8" />
                  </span>
                </div>
              </div>
            </Link>

            {/* signal + top scout pick */}
            <div className="flex flex-col gap-5 lg:mt-10">
              <div className="panel p-5 animate-rise [animation-delay:380ms]">
                <div className="animate-float">
                  <div className="eyebrow">{t("ui.hero.signal")}</div>
                  <div className="mt-3 flex items-center justify-between gap-3">
                    <span
                      className={clsx(
                        "text-[30px] font-semibold tracking-[-0.03em]",
                        signal?.kind === "buy" ? "text-up" : signal?.kind === "sell" ? "text-down" : "text-bone",
                      )}
                    >
                      {signal ? t(`dashboard.signal.${signal.kind}`) : "—"}
                    </span>
                    {signal && <Badge tone={signal.trendPct >= 0 ? "up" : "down"}>{fmtPct(signal.trendPct)}</Badge>}
                  </div>
                  <Sparkline values={(pred.data?.predictions ?? []).map((p) => p.predicted_price)} tone="accent" className="mt-4 h-10 w-full" />
                </div>
              </div>
              <Link to={topPick ? `/asset/${topPick.asset_type}/${encodeURIComponent(topPick.id)}` : "/scout"} className="panel group block p-5 animate-rise [animation-delay:440ms]">
                <div className="animate-float-slow">
                  <div className="flex items-start justify-between gap-3">
                    <div className="eyebrow">{t("ui.hero.topPick")}</div>
                    <ArrowCircle className="size-8" />
                  </div>
                  {topPick ? (
                    <div className="mt-2 flex items-center gap-3">
                      <TokenIcon symbol={topPick.symbol} type={topPick.asset_type} />
                      <div className="min-w-0 flex-1">
                        <div className="font-medium text-bone">{topPick.symbol}</div>
                        <div className="truncate text-xs text-faint">{topPick.name}</div>
                      </div>
                      <div className="text-right">
                        <Delta value={topPick.predicted_change_pct} className="text-lg font-medium" />
                        <div className="mt-1 flex justify-end"><ConfidenceMeter value={topPick.confidence} compact /></div>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-3 text-[13px] text-faint">{t("ui.hero.scanning")}</p>
                  )}
                </div>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================ sources marquee */}
      <section className="relative border-y border-line bg-white/[0.015] py-10">
        <p className="mb-7 text-center text-[15px] font-medium text-bone">{t("ui.hero.sources")}</p>
        <div className="relative overflow-hidden [mask-image:linear-gradient(90deg,transparent,#000_15%,#000_85%,transparent)]">
          <div className="flex w-max animate-marquee gap-14 pr-14 hover:[animation-play-state:paused]">
            {[...SOURCES, ...SOURCES].map((s, i) => (
              <span key={i} className="whitespace-nowrap text-[22px] font-semibold tracking-[-0.02em] text-faint/80 transition-colors hover:text-dim">
                {s}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ============================================================ track record */}
      <section className="mx-auto max-w-[1440px] px-4 pt-24 sm:px-6">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            { k: t("ui.landing.proof.logged"), v: overall ? overall.predictions.toLocaleString("en-US") : "—", sub: t("ui.landing.proof.loggedSub") },
            { k: t("eval.summary.meanError"), v: overall?.mean_abs_pct_error != null ? fmtPct(overall.mean_abs_pct_error, { signed: false }) : "—", sub: t("ui.landing.proof.errorSub") },
            { k: t("eval.summary.direction"), v: overall?.direction_accuracy_pct != null ? `${overall.direction_accuracy_pct.toFixed(1)}%` : "—", sub: t("ui.landing.proof.dirSub") },
            scout.data?.results.length
              ? { k: t("ui.landing.proof.scanned"), v: String(scout.data.results.length), sub: t("ui.landing.proof.scannedSub") }
              : { k: t("ui.landing.proof.universe"), v: "~135", sub: t("ui.landing.proof.universeSub") },
          ].map((s) => (
            <Link to="/evaluation" key={s.k} className="panel group p-6">
              <div className="flex items-start justify-between">
                <span className="eyebrow">{s.k}</span>
                <ArrowCircle className="size-8" />
              </div>
              <div className="num mt-6 text-[40px] font-semibold leading-none tracking-[-0.04em] text-bone">{s.v}</div>
              <div className="mt-3 text-xs text-faint">{s.sub}</div>
            </Link>
          ))}
        </div>
      </section>

      {/* ============================================================ how it works */}
      <section className="mx-auto max-w-[1440px] px-4 py-28 text-center sm:px-6">
        <span className="badge-pill">{t("landing.howItWorks.title")}</span>
        <h2 className="text-gradient mx-auto mt-6 max-w-[18ch] pb-1 text-[clamp(32px,4.2vw,56px)] font-semibold leading-[1.02] tracking-[-0.04em]">{t("landing.howItWorks.subtitle")}</h2>
        <div className="relative mt-16 grid gap-5 text-left md:grid-cols-2 xl:grid-cols-4">
          {/* connector line with a travelling light */}
          <div aria-hidden className="absolute left-[10%] right-[10%] top-[52px] hidden h-px overflow-hidden bg-line xl:block">
            <div className="flex h-full w-[200%] animate-marquee">
              <span className="h-full w-1/4 bg-gradient-to-r from-transparent via-accent-hi to-transparent" />
            </div>
          </div>
          {steps.map(({ Icon, title, body }, i) => (
            <div key={title} className="panel group relative p-7">
              <div className="mb-8 flex items-center justify-between">
                <span className="relative flex size-12 items-center justify-center rounded-2xl border border-accent/30 bg-accent/10 text-accent-hi shadow-[0_0_30px_-8px_rgb(255_106_31/0.8)] transition-transform duration-500 group-hover:scale-110">
                  <Icon className="size-5" strokeWidth={1.7} />
                </span>
                <span className="num text-sm text-faint">0{i + 1}</span>
              </div>
              <h3 className="text-lg font-semibold tracking-[-0.02em] text-bone">{title}</h3>
              <p className="mt-3 text-[14px] leading-relaxed text-dim">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ============================================================ features bento */}
      <section className="mx-auto max-w-[1440px] px-4 pb-28 sm:px-6">
        <div className="mb-14 text-center">
          <span className="badge-pill">{t("landing.features.title")}</span>
          <h2 className="text-gradient mx-auto mt-6 max-w-[22ch] pb-1 text-[clamp(32px,4.2vw,56px)] font-semibold leading-[1.02] tracking-[-0.04em]">{t("landing.features.subtitle")}</h2>
        </div>

        <div className="grid gap-5 md:grid-cols-6">
          <Link to="/scout" className="panel group relative flex flex-col overflow-hidden p-7 sm:p-9 md:col-span-6 lg:col-span-4 lg:row-span-2">
            <div className="flex items-start justify-between">
              <span className="flex size-12 items-center justify-center rounded-2xl border border-accent/30 bg-accent/10 text-accent-hi">
                <Radar className="size-5" strokeWidth={1.7} />
              </span>
              <ArrowCircle />
            </div>
            <h3 className="mt-7 text-[30px] font-semibold tracking-[-0.035em] text-bone">{t("landing.feature4.title")}</h3>
            <p className="mt-3 max-w-[60ch] text-[14.5px] leading-relaxed text-dim">{t("landing.feature4.body")}</p>
            <div className="mt-8 flex-1 space-y-2.5">
              {(scout.data?.results ?? []).slice(0, 5).map((r) => (
                <div key={r.id} className="flex items-center gap-4 rounded-2xl border border-line bg-white/[0.02] px-4 py-3 text-[13px] transition-colors group-hover:border-line-strong">
                  <TokenIcon symbol={r.symbol} type={r.asset_type} size="sm" />
                  <div className="min-w-0 flex-1">
                    <div className="font-medium text-bone">{r.symbol}</div>
                    <div className="truncate text-xs text-faint">{r.name}</div>
                  </div>
                  <ConfidenceMeter value={r.confidence} compact />
                  <Delta value={r.predicted_change_pct} className="w-20 justify-end font-medium" />
                </div>
              ))}
              {!scout.data?.results?.length && <p className="rounded-2xl border border-dashed border-line px-4 py-6 text-center text-[13px] text-faint">{t("ui.hero.scanning")}</p>}
            </div>
          </Link>

          {[
            { Icon: Timer, title: t("landing.feature1.title"), body: t("landing.feature1.body") },
            { Icon: Gauge, title: t("landing.feature5.title"), body: t("landing.feature5.body") },
            { Icon: CandlestickChart, title: t("landing.feature2.title"), body: t("landing.feature2.body") },
            { Icon: Newspaper, title: t("landing.feature3.title"), body: t("landing.feature3.body") },
            { Icon: MousePointerClick, title: t("landing.feature6.title"), body: t("landing.feature6.body") },
          ].map(({ Icon, title, body }) => (
            <div key={title} className="panel group p-7 md:col-span-3 lg:col-span-2">
              <span className="flex size-11 items-center justify-center rounded-2xl border border-line bg-white/[0.04] text-accent-hi transition-colors group-hover:border-accent/40 group-hover:bg-accent/10">
                <Icon className="size-5" strokeWidth={1.7} />
              </span>
              <h3 className="mt-6 text-lg font-semibold tracking-[-0.02em] text-bone">{title}</h3>
              <p className="mt-2 text-[14px] leading-relaxed text-dim">{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ============================================================ closing CTA */}
      <section className="mx-auto max-w-[1440px] px-4 sm:px-6">
        <div className="panel relative isolate overflow-hidden rounded-[32px] px-6 py-20 text-center sm:px-12 sm:py-24">
          {/* same React Bits "Beams" as the hero, contained in the card */}
          {/* The scene is rendered much taller than the card (and clipped by it):
              the beams' lit area scales with the canvas, so it spans the whole card. */}
          <div aria-hidden className="pointer-events-none absolute inset-x-0 top-1/2 -z-10 h-[720px] -translate-y-1/2 animate-fade">
            <Suspense fallback={null}>
              <Beams beamWidth={2} beamHeight={22} beamNumber={26} lightColor="#ff7a2e" backgroundColor="#050505" speed={1.4} noiseIntensity={1.5} scale={0.2} rotation={-30} />
            </Suspense>
          </div>
          <div className="relative [filter:drop-shadow(0_2px_24px_rgb(0_0_0/0.85))]">
            <h2 className="text-gradient mx-auto max-w-[20ch] pb-1 text-[clamp(30px,3.8vw,52px)] font-semibold leading-[1.02] tracking-[-0.04em]">{t("ui.landing.closing")}</h2>
            <div className="mt-10 flex flex-wrap justify-center gap-3">
              <Link to="/dashboard" className={buttonClass("primary", "lg")}>
                {t("landing.cta.primary")} <ArrowRight className="size-4" />
              </Link>
              <Link to="/evaluation" className={buttonClass("outline", "lg")}>
                {t("ui.landing.seeTrack")}
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
