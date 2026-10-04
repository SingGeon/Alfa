import clsx from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router";
import { keepPreviousData, useQueries, useQuery } from "@tanstack/react-query";
import { Activity, ArrowRight, CandlestickChart, ChevronDown, Fuel, ImageIcon, Layers, LineChart, Lightbulb, Radar, ScanSearch, Target, X } from "lucide-react";
import {
  apiGet,
  ApiError,
  type AccuracyResponse,
  type Candle,
  type CurrentPrice,
  type L2Chain,
  type NewsItem,
  type OutlookResponse,
  type PatternsResponse,
  type Pin,
  type PredictResponse,
  type Quote,
  type SignalAccuracyResponse,
  type SignalStat,
} from "../lib/api";
import { computeSignal, FIB_LEVELS, type SignalSummary } from "../lib/indicators";
import { dirText, fmtCompactUsd, fmtDateTime, fmtElapsed, fmtMoney, qualityText } from "../lib/format";
import { useI18n, type TFn } from "../i18n";
import { PriceChart, type IndicatorKey, type Indicators, type IndicatorSnapshot, type PriceChartHandle } from "../components/PriceChart";
import { AccuracyChart } from "../components/AccuracyChart";
import { ArrowCircle, BigPrice, TokenIcon } from "../components/market";
import { ForecastTable, Legend, NewsList } from "../components/shared";
import { Badge, Button, ConfidenceMeter, Delta, Empty, ErrorBanner, Panel, PanelHeader, Segmented, Skeleton, Stat, Switch } from "../components/ui";

type Interval = "15m" | "1h" | "4h" | "1d" | "1w";
type RangeKey = "1S" | "1L" | "3L" | "6L" | "1A" | "TOT";
type Variant = "legacy" | "tuned";

const INTERVALS: Interval[] = ["15m", "1h", "4h", "1d", "1w"];
const DEFAULT_LIMIT = 1000;
const RANGE_PRESETS: Record<RangeKey, { interval: Interval; limit: number; key: string }> = {
  "1S": { interval: "1h", limit: 168, key: "dashboard.chart.range.week" },
  "1L": { interval: "4h", limit: 180, key: "dashboard.chart.range.month" },
  "3L": { interval: "4h", limit: 540, key: "dashboard.chart.range.3month" },
  "6L": { interval: "1d", limit: 182, key: "dashboard.chart.range.6month" },
  "1A": { interval: "1d", limit: 365, key: "dashboard.chart.range.year" },
  TOT: { interval: "1w", limit: 1000, key: "dashboard.chart.range.all" },
};
const INDICATORS: { key: IndicatorKey; label: string; color: string }[] = [
  { key: "sma", label: "SMA 20", color: "var(--color-sky)" },
  { key: "ema", label: "EMA 20", color: "var(--color-rose)" },
  { key: "bollinger", label: "Bollinger", color: "var(--color-dim)" },
  { key: "fibonacci", label: "Fibonacci", color: "var(--color-faint)" },
  { key: "vwap", label: "VWAP", color: "var(--color-violet)" },
];

const REFRESH = 30_000;

export default function Dashboard() {
  const { t, lang, locale } = useI18n();

  // ------------------------------------------------------------ state
  const [interval, setIntervalState] = useState<Interval>("1h");
  const [range, setRange] = useState<RangeKey | null>(null);
  const [limit, setLimit] = useState(DEFAULT_LIMIT);
  const [steps, setSteps] = useState(24);
  const [stepsDraft, setStepsDraft] = useState(24);
  const [useSentiment, setUseSentiment] = useState(true);
  const [variant, setVariant] = useState<Variant>("legacy");
  const [chartType, setChartType] = useState<"candles" | "line">("candles");
  const [indicators, setIndicators] = useState<Indicators>({ sma: false, ema: false, bollinger: false, fibonacci: false, vwap: false });
  const [showPatterns, setShowPatterns] = useState(false);
  const [explainOpen, setExplainOpen] = useState(false);
  const [snapshot, setSnapshot] = useState<IndicatorSnapshot | null>(null);
  const [dismissedError, setDismissedError] = useState<string | null>(null);
  const chartRef = useRef<PriceChartHandle>(null);

  const pickInterval = (i: Interval) => {
    setIntervalState(i);
    setLimit(DEFAULT_LIMIT);
    setRange(null);
  };
  const pickRange = (r: RangeKey) => {
    setRange(r);
    setIntervalState(RANGE_PRESETS[r].interval);
    setLimit(RANGE_PRESETS[r].limit);
  };

  // ------------------------------------------------------------ data
  const price = useQuery({ queryKey: ["price-current"], queryFn: () => apiGet<CurrentPrice>("/api/price/current"), refetchInterval: 3000 });
  const history = useQuery({
    queryKey: ["history", interval, limit],
    queryFn: () => apiGet<{ interval: string; candles: Candle[] }>("/api/price/history", { interval, limit }),
    refetchInterval: REFRESH,
    placeholderData: keepPreviousData,
  });
  const predict = useQuery({
    queryKey: ["predict", interval, steps, useSentiment, variant],
    queryFn: () => apiGet<PredictResponse>("/api/predict", { interval, steps, use_sentiment: useSentiment, backend: "sklearn", model_variant: variant }),
    refetchInterval: REFRESH,
    placeholderData: keepPreviousData,
    retry: (n, e) => !(e instanceof ApiError && e.status === 409) && n < 2,
  });
  const outlook = useQuery({
    queryKey: ["outlook", useSentiment, variant],
    queryFn: () => apiGet<OutlookResponse>("/api/outlook", { use_sentiment: useSentiment, backend: "sklearn", model_variant: variant }),
    refetchInterval: REFRESH,
    placeholderData: keepPreviousData,
  });
  const [todayQ, weekQ] = useQueries({
    queries: (["1d", "1w"] as const).map((iv) => ({
      queryKey: ["history", iv, 1],
      queryFn: () => apiGet<{ candles: Candle[] }>("/api/price/history", { interval: iv, limit: 1 }),
      refetchInterval: REFRESH,
    })),
  });
  const summary = useQuery({
    queryKey: ["summary", variant, lang],
    queryFn: () => apiGet<{ narrative: string }>("/api/summary", { backend: "sklearn", lang, model_variant: variant }),
    refetchInterval: REFRESH,
    placeholderData: keepPreviousData,
  });
  const news = useQuery({ queryKey: ["news"], queryFn: () => apiGet<{ news: NewsItem[] }>("/api/news", { limit: 30 }), refetchInterval: 60_000 });
  // 501 = no ETHERSCAN_API_KEY configured: stop polling instead of erroring every minute.
  const gas = useQuery({
    queryKey: ["gas"],
    queryFn: () => apiGet<{ safe_gwei: number; propose_gwei: number; fast_gwei: number }>("/api/gas"),
    refetchInterval: (q) => (q.state.error instanceof ApiError && q.state.error.status === 501 ? false : 60_000),
    retry: false,
  });
  const l2 = useQuery({ queryKey: ["l2"], queryFn: () => apiGet<{ chains: L2Chain[] }>("/api/l2"), refetchInterval: 300_000 });
  const backtest = useQuery({ queryKey: ["accuracy-hint", interval], queryFn: () => apiGet<AccuracyResponse>("/api/predict/accuracy", { interval, limit: 100 }), refetchInterval: REFRESH });
  const signalAcc = useQuery({ queryKey: ["signal-accuracy"], queryFn: () => apiGet<SignalAccuracyResponse>("/api/signal/accuracy", { interval: "1h" }), refetchInterval: 60_000 });
  const patterns = useQuery({
    queryKey: ["patterns", interval, lang],
    queryFn: () => apiGet<PatternsResponse>("/api/patterns", { interval, lang }),
    enabled: showPatterns,
    refetchInterval: REFRESH,
  });
  const pins = useQuery({ queryKey: ["pins"], queryFn: () => apiGet<{ pins: Pin[] }>("/api/scout/pins"), refetchInterval: REFRESH });
  const pinQuotes = useQueries({
    queries: (pins.data?.pins ?? []).map((p) => ({
      queryKey: ["quote", p.asset_type, p.id],
      queryFn: () => apiGet<Quote>("/api/scout/price", { asset_type: p.asset_type, id: p.id }),
      refetchInterval: REFRESH,
      retry: false,
    })),
  });

  // Only render data that actually belongs to the selected interval (stale
  // placeholder data from the previous interval would mis-draw the forecast).
  const candles = history.data && history.data.interval === interval ? history.data.candles : null;
  const prediction = predict.data && predict.data.interval === interval ? predict.data : null;
  const sortedCandles = useMemo(
    () => (candles ? [...candles].sort((a, b) => +new Date(a.timestamp) - +new Date(b.timestamp)) : []),
    [candles],
  );

  useEffect(() => {
    if (price.data?.price != null && candles) chartRef.current?.tick(price.data.price);
  }, [price.data, candles]);

  const signal: SignalSummary | null = useMemo(
    () =>
      prediction && sortedCandles.length
        ? computeSignal(sortedCandles, prediction.predictions, prediction.last_known_price, prediction.confidence, prediction.sentiment_avg)
        : null,
    [prediction, sortedCandles],
  );

  const patternOverlay = useMemo(() => {
    if (!showPatterns || !patterns.data) return null;
    return {
      candlestick: patterns.data.candlestick,
      geometric: patterns.data.geometric[0] ?? null,
      necklineLabel: t("dashboard.pattern.necklineTitle"),
      targetLabel: t("dashboard.pattern.targetTitle"),
    };
  }, [showPatterns, patterns.data, t]);

  const errorMsg = (() => {
    if (history.error) return t("dashboard.error.priceHistory", { msg: history.error.message });
    if (predict.error) {
      const e = predict.error;
      return e instanceof ApiError && e.status === 409 ? e.message : t("dashboard.error.predictionFailed", { msg: e.message });
    }
    if (price.error && !price.data) return t("dashboard.error.currentPrice", { msg: price.error.message });
    return null;
  })();

  const liveChange = price.data?.change_24h_pct;
  const updatedAt = price.dataUpdatedAt ? new Date(price.dataUpdatedAt).toLocaleTimeString(locale) : null;
  const activeIndicatorCount = Object.values(indicators).filter(Boolean).length;
  const viewKey = `${interval}|${limit}|${range ?? ""}`;

  return (
    <div className="mx-auto max-w-[1440px] px-4 pb-4 pt-5 sm:px-6 sm:pt-7">
      {/* ------------------------------------------------ header: price + outlook */}
      <section className="relative mb-8 grid items-end gap-8 lg:grid-cols-[1fr_1.1fr] lg:gap-10">
        <div className="relative z-[1] min-w-0 animate-rise">
          <div className="eyebrow mb-3 flex items-center gap-2">
            <span className="size-1.5 rounded-full bg-up animate-pulse-dot" />
            {t("ui.dash.price")}
            {price.data?.source && <span className="text-faint/70">· {price.data.source}</span>}
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            {price.data ? (
              <BigPrice value={price.data.price} className="text-[48px] font-semibold leading-none tracking-[-0.045em] text-bone sm:text-[68px]" />
            ) : (
              <Skeleton className="h-14 w-72" />
            )}
            {liveChange != null && (
              <span className="inline-flex items-center gap-2 rounded-full border border-line bg-white/[0.03] px-3 py-1.5 text-[13px] backdrop-blur-md">
                <Delta value={liveChange} className="font-medium" />
                <span className="text-faint">{t("ui.dash.last24")}</span>
              </span>
            )}
          </div>
          <div className="my-6 h-px max-w-[560px] bg-gradient-to-r from-line-strong to-transparent" />
          <div className="flex flex-wrap items-center gap-2.5">
            <a
              href="#signal"
              className={clsx(
                "inline-flex h-11 items-center gap-2.5 rounded-full pl-1.5 pr-5 text-[14px] font-semibold transition-all",
                signal?.kind === "buy"
                  ? "bg-up text-[#04140b] shadow-[0_10px_30px_-8px_rgb(52_211_153/0.8)]"
                  : signal?.kind === "sell"
                    ? "bg-down text-white shadow-[0_10px_30px_-8px_rgb(255_61_110/0.8)]"
                    : "bg-gradient-to-b from-accent-hi to-accent text-white shadow-[0_10px_30px_-8px_rgb(255_106_31/0.85)]",
              )}
            >
              <span className="flex size-8 items-center justify-center rounded-full bg-black/20">
                <Activity className="size-4" />
              </span>
              {signal ? t(`dashboard.signal.${signal.kind}`) : t("dashboard.signal.header")}
            </a>
            {[
              { to: "/scout", label: t("nav.scout"), Icon: Radar },
              { to: "/evaluation", label: t("nav.evaluation"), Icon: Target },
              { to: "/visual", label: t("nav.visual"), Icon: ImageIcon },
            ].map(({ to, label, Icon }) => (
              <Link key={to} to={to} className="inline-flex h-11 items-center gap-2.5 rounded-full border border-line bg-white/[0.03] pl-1.5 pr-4 text-[13.5px] text-bone backdrop-blur-md transition-colors hover:border-line-strong hover:bg-white/[0.06]">
                <span className="flex size-8 items-center justify-center rounded-full bg-white/[0.06] text-dim">
                  <Icon className="size-4" />
                </span>
                {label}
              </Link>
            ))}
          </div>
          {updatedAt && <div className="mt-4 text-xs text-faint">{t("dashboard.updated.template", { time: updatedAt })}</div>}
          <div className="mt-4">
            <PinnedStrip pins={pins.data?.pins ?? []} quotes={pinQuotes.map((q) => q.data)} />
          </div>
        </div>

        <div className="relative z-[1] min-w-0 animate-rise [animation-delay:120ms]">
          <div className="mb-4 flex items-end justify-between">
            <h2 className="text-[22px] font-semibold tracking-[-0.03em] text-bone">{t("ui.dash.outlook")}</h2>
            <Link to="/evaluation" className="group inline-flex items-center gap-1.5 text-[13px] text-dim hover:text-bone">
              {t("ui.landing.seeTrack")} <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
            </Link>
          </div>
          <OutlookCards
            outlook={outlook.data}
            livePrice={price.data?.price}
            todayOpen={todayQ.data?.candles?.[0]?.open}
            weekOpen={weekQ.data?.candles?.[0]?.open}
          />
        </div>
      </section>

      {errorMsg && errorMsg !== dismissedError && (
        <div className="mb-4">
          <ErrorBanner message={errorMsg} onClose={() => setDismissedError(errorMsg)} />
        </div>
      )}

      {/* ------------------------------------------------ main grid */}
      <div className="grid gap-4 lg:grid-cols-12">
        <Panel className="flex min-w-0 flex-col lg:col-span-8 xl:col-span-9 animate-rise [animation-delay:60ms]">
          {/* toolbar */}
          <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2.5 sm:px-4">
            <Segmented
              ariaLabel={t("dashboard.chart.intervalAria")}
              options={INTERVALS.map((i) => ({ value: i, label: i }))}
              value={interval}
              onChange={pickInterval}
              size="sm"
            />
            <Segmented
              ariaLabel={t("dashboard.chart.rangeAria")}
              options={(Object.keys(RANGE_PRESETS) as RangeKey[]).map((r) => ({
                value: r,
                label: t(RANGE_PRESETS[r].key),
                title: r === "TOT" ? t("dashboard.chart.allHistoryTitle") : undefined,
              }))}
              value={range}
              onChange={pickRange}
              size="sm"
            />
            <div className="ml-auto flex items-center gap-1.5">
              <Segmented
                ariaLabel={t("dashboard.chart.typeAria")}
                options={[
                  { value: "candles", label: <CandlestickChart className="size-3.5" />, title: t("dashboard.chart.candlesTitle") },
                  { value: "line", label: <LineChart className="size-3.5" />, title: t("dashboard.chart.lineTitle") },
                ]}
                value={chartType}
                onChange={setChartType}
                size="sm"
              />
              <IndicatorMenu value={indicators} onChange={setIndicators} count={activeIndicatorCount} />
              <Button
                size="sm"
                variant={showPatterns ? "outline" : "ghost"}
                className={clsx(showPatterns && "!border-violet/50 !text-violet")}
                onClick={() => setShowPatterns((v) => !v)}
                aria-pressed={showPatterns}
              >
                <ScanSearch className="size-3.5" />
                <span className="hidden sm:inline">{t("dashboard.chart.patternsLabel")}</span>
              </Button>
              <Button size="sm" variant={explainOpen ? "outline" : "ghost"} onClick={() => setExplainOpen((v) => !v)} aria-pressed={explainOpen}>
                <Lightbulb className="size-3.5" />
                <span className="hidden xl:inline">{t("dashboard.chart.explainBtn")}</span>
              </Button>
            </div>
          </div>

          {explainOpen && (
            <div className="relative border-b border-line bg-raised/60 px-4 py-3.5 pr-10 animate-fade sm:px-5">
              <div className="eyebrow mb-1.5 !text-accent/80">{t("dashboard.chart.explainHeader")}</div>
              <p className="max-w-[110ch] text-[13px] leading-relaxed text-bone/90">{explainIndicators(t, indicators, snapshot, signal)}</p>
              <button type="button" onClick={() => setExplainOpen(false)} className="absolute right-3 top-3 text-faint hover:text-bone" aria-label={t("dashboard.chart.explainClose")}>
                <X className="size-4" />
              </button>
            </div>
          )}

          {showPatterns && patterns.data && <PatternAlert data={patterns.data} t={t} />}

          {/* chart */}
          <div className="relative h-[440px] sm:h-[520px] xl:h-[580px]">
            <PriceChart
              candles={candles ?? []}
              predictions={prediction?.predictions ?? []}
              chartType={chartType}
              indicators={indicators}
              patterns={patternOverlay}
              viewKey={viewKey}
              fitAll={range !== null}
              onIndicatorSnapshot={setSnapshot}
              handleRef={chartRef}
            />
            {!candles && !history.error && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="size-6 animate-spin rounded-full border-2 border-line-strong border-t-accent" />
              </div>
            )}
            {candles && !candles.length && <Empty className="absolute inset-0">{t("dashboard.chart.empty")}</Empty>}
          </div>

          {/* forecast controls + legend */}
          <div className="flex flex-col gap-3 border-t border-line px-3 py-3 sm:px-4 xl:flex-row xl:items-center xl:justify-between">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
              <label className="flex min-w-[220px] items-center gap-3 text-[13px] text-dim">
                <span className="whitespace-nowrap">{t("dashboard.chart.horizonPrefix")}</span>
                <input
                  type="range"
                  min={1}
                  max={72}
                  value={stepsDraft}
                  className="range w-28"
                  style={{ ["--fill" as string]: `${((stepsDraft - 1) / 71) * 100}%` }}
                  onChange={(e) => setStepsDraft(Number(e.target.value))}
                  onPointerUp={() => setSteps(stepsDraft)}
                  onKeyUp={() => setSteps(stepsDraft)}
                  aria-label={t("dashboard.chart.horizonPrefix")}
                />
                <span className="num w-16 text-bone">
                  {stepsDraft} <span className="text-faint">× {interval}</span>
                </span>
              </label>
              <Switch checked={useSentiment} onChange={setUseSentiment} label={t("dashboard.chart.sentimentToggle")} />
              <Segmented
                ariaLabel={t("dashboard.chart.modelVariantAria")}
                options={[
                  { value: "legacy", label: t("dashboard.chart.modelVariant.legacy"), title: t("dashboard.chart.modelVariantTitle") },
                  { value: "tuned", label: t("dashboard.chart.modelVariant.tuned"), title: t("dashboard.chart.modelVariantTitle") },
                ]}
                value={variant}
                onChange={setVariant}
                size="sm"
              />
            </div>
            <Legend
              items={[
                { color: "var(--color-up)", label: t("common.legend.up") },
                { color: "var(--color-down)", label: t("common.legend.down") },
                { color: "var(--color-accent)", label: t("common.legend.pred") },
                { color: "var(--color-accent)", label: t("common.legend.band"), soft: true },
                ...(showPatterns && patternOverlay?.geometric ? [{ color: "var(--color-violet)", label: t("dashboard.legend.pattern"), dashed: true }] : []),
              ]}
            />
          </div>
        </Panel>

        {/* ------------------------------------------------ side column */}
        <div className="flex min-w-0 flex-col gap-4 lg:col-span-4 xl:col-span-3">
          <SignalCard signal={signal} confidence={prediction?.confidence ?? null} />
          <Panel className="p-4 sm:p-5 animate-rise [animation-delay:180ms]">
            <PanelHeader title={t("dashboard.stat.model")} />
            <div className="mt-4 grid gap-4">
              <Stat
                label={t("common.stat.confidence")}
                value={
                  <span className="flex flex-wrap items-center gap-2">
                    <span className={qualityText(prediction?.confidence)}>{prediction ? `${Math.round(prediction.confidence)}` : "—"}</span>
                    <span className="text-sm text-faint">/100</span>
                    {signal?.regime && (
                      <Badge tone={signal.regime === "volatile" ? "warn" : signal.regime === "calm" ? "up" : "neutral"}>{t(`dashboard.volatility.${signal.regime}`)}</Badge>
                    )}
                  </span>
                }
                hint={
                  backtest.data
                    ? backtest.data.count && backtest.data.mape != null
                      ? t("dashboard.backtest.template", { pct: backtest.data.mape.toFixed(2), n: backtest.data.count, interval })
                      : t("dashboard.backtest.accumulating")
                    : t("dashboard.backtest.dash")
                }
              />
              <Stat
                label={t("common.stat.sentimentAvg")}
                value={<span className={dirText(prediction?.sentiment_avg)}>{prediction ? prediction.sentiment_avg.toFixed(2) : "—"}</span>}
                hint={
                  !prediction
                    ? t("dashboard.sentiment.dash")
                    : prediction.sentiment_contribution_pct == null
                      ? t("dashboard.sentiment.unavailable")
                      : t("dashboard.sentiment.template", { pct: prediction.sentiment_contribution_pct.toFixed(1) })
                }
              />
              <Stat
                label={t("dashboard.stat.model")}
                value={<span className="text-[15px]">{prediction ? `${prediction.backend} · ${prediction.model_variant}` : "—"}</span>}
                hint={prediction ? t("dashboard.recalibrated", { time: fmtElapsed(prediction.trained_ago_seconds, t) }) : "—"}
              />
            </div>
          </Panel>
        </div>
      </div>

      {/* ------------------------------------------------ second row */}
      <div className="mt-4 grid gap-4 lg:grid-cols-12">
        <Panel className="flex min-w-0 flex-col lg:col-span-4">
          <div className="p-4 sm:p-5">
            <PanelHeader index="01" title={t("dashboard.prediction.header")} hint={t("dashboard.prediction.disclaimer")} />
            <div className="mt-4 grid grid-cols-3 gap-3 border-t border-line pt-4">
              <Stat label={t("dashboard.prediction.trend")} value={signal ? <Delta value={signal.trendPct} /> : "—"} valueClass="text-base" />
              <Stat
                label={t("dashboard.prediction.low")}
                value={signal ? fmtMoney(signal.low.predicted_price) : "—"}
                hint={signal ? fmtDateTime(signal.low.timestamp, locale) : undefined}
                valueClass="text-sm"
              />
              <Stat
                label={t("dashboard.prediction.high")}
                value={signal ? fmtMoney(signal.high.predicted_price) : "—"}
                hint={signal ? fmtDateTime(signal.high.timestamp, locale) : undefined}
                valueClass="text-sm"
              />
            </div>
          </div>
          <div className="max-h-[340px] overflow-y-auto border-t border-line">
            {prediction ? (
              <ForecastTable
                predictions={prediction.predictions}
                basePrice={prediction.last_known_price}
                formatTime={(iso) => fmtDateTime(iso, locale)}
                timeHeader={t("dashboard.prediction.colTime")}
              />
            ) : (
              <div className="space-y-2 p-5">
                {Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-4" />)}
              </div>
            )}
          </div>
        </Panel>

        <Panel className="flex min-w-0 flex-col lg:col-span-8">
          <div className="border-b border-line p-4 sm:p-5">
            <PanelHeader index="02" title={t("dashboard.summary.header")} />
            {summary.data ? (
              <p className="mt-3 max-w-[85ch] font-display text-[17px] leading-[1.55] tracking-[-0.005em] text-bone/95 sm:text-lg">{summary.data.narrative}</p>
            ) : summary.error && !(summary.error instanceof ApiError && summary.error.status === 409) ? (
              <p className="mt-3 text-sm text-faint">{t("dashboard.summary.fetchFailed", { msg: summary.error.message })}</p>
            ) : (
              <div className="mt-4 space-y-2.5">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-11/12" />
                <Skeleton className="h-4 w-2/3" />
              </div>
            )}
          </div>
          <div className="flex min-h-0 flex-1 flex-col p-4 sm:p-5">
            <PanelHeader title={t("dashboard.news.header")} />
            <div className="mt-4 max-h-[360px] overflow-y-auto pl-3 pr-1 sm:pl-4">
              {news.error ? (
                <Empty>{t("dashboard.news.fetchFailed", { msg: news.error.message })}</Empty>
              ) : (
                <NewsList news={news.data?.news} loading={news.isLoading} emptyText={t("dashboard.news.empty")} />
              )}
            </div>
          </div>
        </Panel>
      </div>

      {/* ------------------------------------------------ third row */}
      <div className="mt-4 grid gap-4 lg:grid-cols-12">
        <AccuracyPanel className="lg:col-span-8" />
        <SignalAccuracyPanel data={signalAcc.data} error={!!signalAcc.error} className="lg:col-span-4" />
      </div>

      {/* ------------------------------------------------ L2 */}
      <Panel className="mt-4">
        <div className="p-4 sm:p-5">
          <PanelHeader
            index="05"
            title={t("dashboard.l2.header")}
            hint={t("dashboard.l2.disclaimer")}
            actions={
              gas.data && (
                <span className="num inline-flex items-center gap-1.5 rounded-xl border border-line px-2.5 py-1 text-xs text-dim">
                  <Fuel className="size-3.5 text-faint" />
                  {gas.data.safe_gwei} / {gas.data.propose_gwei} / {gas.data.fast_gwei} gwei
                </span>
              )
            }
          />
        </div>
        {l2.error ? (
          <Empty>{t("dashboard.l2.fetchError", { msg: l2.error.message })}</Empty>
        ) : !l2.data ? (
          <div className="space-y-2 px-5 pb-5">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-5" />)}</div>
        ) : !l2.data.chains.length ? (
          <Empty>{t("dashboard.l2.fetchFail")}</Empty>
        ) : (
          <L2Grid chains={l2.data.chains} />
        )}
      </Panel>
    </div>
  );
}

// =================================================================== parts

function PinnedStrip({ pins, quotes }: { pins: Pin[]; quotes: (Quote | undefined)[] }) {
  if (!pins.length) return null;
  return (
    <div className="flex max-w-full gap-2 overflow-x-auto pb-1 animate-fade">
      {pins.map((p, i) => {
        const q = quotes[i];
        return (
          <Link
            key={`${p.asset_type}:${p.id}`}
            to={`/asset/${p.asset_type}/${encodeURIComponent(p.id)}`}
            title={p.name}
            className="group flex shrink-0 flex-col gap-0.5 rounded-xl border border-line bg-panel px-3 py-2 transition-colors hover:border-line-strong hover:bg-raised"
          >
            <span className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wide text-dim group-hover:text-bone">
              <span className={clsx("size-1 rounded-full", p.asset_type === "crypto" ? "bg-sky" : "bg-rose")} />
              {p.symbol}
            </span>
            <span className="flex items-baseline gap-2">
              <span className="num text-[13px] text-bone">{q ? fmtMoney(q.price, { smart: true }) : "—"}</span>
              {q?.change_24h_pct != null && <Delta value={q.change_24h_pct} arrow={false} className="text-[11px]" />}
            </span>
          </Link>
        );
      })}
    </div>
  );
}

function IndicatorMenu({ value, onChange, count }: { value: Indicators; onChange: (v: Indicators) => void; count: number }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <Button size="sm" variant={open || count ? "outline" : "ghost"} onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-label={t("dashboard.chart.indicatorsAria")}>
        <Layers className="size-3.5" />
        <span className="hidden sm:inline">{t("ui.indicators")}</span>
        {count > 0 && <span className="num rounded-xs bg-accent px-1 text-[10px] font-bold text-accent-ink">{count}</span>}
        <ChevronDown className={clsx("size-3 transition-transform", open && "rotate-180")} />
      </Button>
      {open && (
        <div className="absolute right-0 top-full z-30 mt-1.5 w-52 rounded-xl border border-line-strong bg-raised p-1.5 shadow-2xl shadow-black/50 animate-fade">
          {INDICATORS.map((ind) => (
            <label key={ind.key} className="flex cursor-pointer items-center gap-2.5 rounded-xs px-2.5 py-2 text-[13px] text-dim hover:bg-hover hover:text-bone">
              <input
                type="checkbox"
                checked={value[ind.key]}
                onChange={(e) => onChange({ ...value, [ind.key]: e.target.checked })}
                className="size-3.5 accent-[var(--color-accent)]"
              />
              <span className="w-3.5 border-t-2" style={{ borderColor: ind.color }} />
              {ind.label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function PatternAlert({ data, t }: { data: PatternsResponse; t: TFn }) {
  const top = data.geometric[0];
  const byName = new Map<string, (typeof data.candlestick)[number]>();
  [...data.candlestick].sort((a, b) => a.index - b.index).forEach((p) => byName.set(p.name, p));
  const candleTypes = [...byName.values()].sort((a, b) => b.index - a.index);
  if (!top && !candleTypes.length) return null;
  const dirTone = (d: string) => (d === "bullish" ? "up" : d === "bearish" ? "down" : "neutral") as "up" | "down" | "neutral";
  const dirLabel = (d: string) => t(d === "bullish" ? "dashboard.pattern.dirBullish" : d === "bearish" ? "dashboard.pattern.dirBearish" : "dashboard.pattern.dirNeutral");
  const Item = ({ p }: { p: { name: string; direction: string; note: string; target?: number | null; confirmed?: boolean } }) => (
    <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1 text-[13px]">
      <Badge tone={dirTone(p.direction)}>{dirLabel(p.direction)}</Badge>
      <span className="font-medium text-bone">
        {p.name}
        {p.confirmed === false && <span className="text-faint"> {t("dashboard.pattern.forming")}</span>}
      </span>
      <span className="text-dim">{p.note}</span>
      {p.target != null && <span className="num text-accent">{t("dashboard.pattern.target", { price: fmtMoney(p.target) })}</span>}
    </div>
  );
  return (
    <div className="max-h-56 space-y-2 overflow-y-auto border-b border-line border-l-2 border-l-violet/60 bg-violet/[0.04] px-4 py-3 animate-fade sm:px-5">
      {top && <Item p={top} />}
      {candleTypes.length > 0 && (
        <>
          <div className="eyebrow pt-1">{t("dashboard.pattern.headingCandlestick")}</div>
          {candleTypes.map((p) => <Item key={p.name} p={p} />)}
        </>
      )}
    </div>
  );
}

function SignalCard({ signal, confidence }: { signal: SignalSummary | null; confidence: number | null }) {
  const { t, locale } = useI18n();
  const kind = signal?.kind;
  const tone =
    kind === "buy" ? "from-up/20 text-up border-up/30" : kind === "sell" ? "from-down/20 text-down border-down/30" : "from-press/60 text-dim border-line-strong";
  const rsiZone = signal?.rsi == null ? "" : signal.rsi < 30 ? ` ${t("dashboard.signal.oversold")}` : signal.rsi > 70 ? ` ${t("dashboard.signal.overbought")}` : "";
  return (
    <Panel id="signal" className="scroll-mt-24 overflow-hidden animate-rise [animation-delay:100ms]">
      <div className={clsx("border-b bg-gradient-to-br to-transparent p-4 sm:p-5", tone)}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="eyebrow !text-current opacity-70">{t("dashboard.signal.header")}</div>
            <div className="mt-1.5 text-[40px] font-bold leading-none tracking-[-0.04em]">
              {!kind ? (
                <Skeleton className="h-10 w-28" />
              ) : kind === "wait" ? (
                t("dashboard.signal.wait")
              ) : (
                // A firm call gets a slow light sweep in its own color.
                <span
                  key={kind}
                  className="text-shimmer"
                  style={{ ["--shimmer-base" as string]: kind === "buy" ? "var(--color-up)" : "var(--color-down)" }}
                >
                  {t(`dashboard.signal.${kind}`)}
                </span>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[11px] text-faint">{t("common.stat.confidence")}</div>
            <div className="mt-1">
              <ConfidenceMeter value={confidence} />
            </div>
          </div>
        </div>
        <p className="mt-3 text-[11px] leading-relaxed text-faint">{t("dashboard.signal.disclaimer")}</p>
      </div>
      <dl className="grid grid-cols-2 gap-px bg-line text-[13px]">
        {[
          [t("dashboard.signal.bestBuy"), signal ? fmtMoney(signal.low.predicted_price) : "—", signal ? fmtDateTime(signal.low.timestamp, locale) : ""],
          [t("dashboard.signal.bestSell"), signal ? fmtMoney(signal.high.predicted_price) : "—", signal ? fmtDateTime(signal.high.timestamp, locale) : ""],
        ].map(([label, value, sub]) => (
          <div key={label} className="bg-panel px-4 py-3 sm:px-5">
            <dt className="text-[11px] text-faint">{label}</dt>
            <dd className="num mt-1 text-bone">{value}</dd>
            {sub && <dd className="num mt-0.5 text-[11px] text-faint">{sub}</dd>}
          </div>
        ))}
        <div className="bg-panel px-4 py-3 sm:px-5">
          <dt className="text-[11px] text-faint">{t("dashboard.signal.potential")}</dt>
          <dd className="mt-1">{signal?.spreadPct != null ? <Delta value={signal.spreadPct} /> : "—"}</dd>
        </div>
        <div className="bg-panel px-4 py-3 sm:px-5">
          <dt className="text-[11px] text-faint">
            {t("dashboard.signal.rsi")} · {t("dashboard.signal.macd")}
          </dt>
          <dd className="num mt-1 text-bone">
            {signal?.rsi != null ? signal.rsi.toFixed(1) : "—"}
            <span className="text-faint">{rsiZone}</span>
          </dd>
          <dd className={clsx("mt-0.5 text-[11px]", signal?.macdBullish == null ? "text-faint" : signal.macdBullish ? "text-up" : "text-down")}>
            MACD {signal?.macdBullish == null ? "—" : signal.macdBullish ? t("ui.bullish") : t("ui.bearish")}
          </dd>
        </div>
      </dl>
    </Panel>
  );
}

function OutlookCards({ outlook, livePrice, todayOpen, weekOpen }: { outlook?: OutlookResponse; livePrice?: number; todayOpen?: number; weekOpen?: number }) {
  const { t } = useI18n();
  const legs = [
    { key: "daily" as const, label: t("dashboard.outlook.daily"), tag: "1D", open: todayOpen, tmpl: "dashboard.outlook.actual24hTemplate" },
    { key: "weekly" as const, label: t("dashboard.outlook.weekly"), tag: "1W", open: weekOpen, tmpl: "dashboard.outlook.actual7dTemplate" },
  ];
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {legs.map((l) => {
        const leg = outlook?.[l.key];
        const actual = livePrice != null && l.open ? ((livePrice - l.open) / l.open) * 100 : null;
        return (
          <Link to="/evaluation" key={l.key} className="panel group flex flex-col p-5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <TokenIcon symbol="ETH" eth />
                <div>
                  <div className="text-[15px] font-medium text-bone">
                    {l.label}
                  </div>
                  <div className="num text-xs text-faint">{leg ? t("dashboard.outlook.targetTemplate", { price: fmtMoney(leg.predicted_price) }) : "—"}</div>
                </div>
              </div>
              <ArrowCircle />
            </div>
            <div className="mt-6 flex items-end justify-between gap-3">
              {leg ? <BigPrice value={leg.predicted_price} className="text-[26px] font-semibold tracking-[-0.03em] text-bone" /> : <Skeleton className="h-7 w-28" />}
              {leg && <Delta value={leg.change_pct} className="text-[15px] font-medium" />}
            </div>
            {actual != null && (
              <div className="mt-4 flex items-center justify-between border-t border-line pt-3 text-[11.5px] text-faint">
                <span>{t(l.tmpl, { pct: "" }).replace(/[:：]\s*$/, "")}</span>
                <Delta value={actual} arrow={false} className="text-[11.5px]" />
              </div>
            )}
          </Link>
        );
      })}
    </div>
  );
}

function AccuracyPanel({ className }: { className?: string }) {
  const { t } = useI18n();
  const [iv, setIv] = useState<"15m" | "1h" | "1d">("1h");
  const [days, setDays] = useState<"30" | "365">("30");
  const main = useQuery({
    queryKey: ["accuracy-series", iv, days],
    queryFn: () => apiGet<AccuracyResponse>("/api/predict/accuracy", { interval: iv, days, backend: "sklearn" }),
    refetchInterval: 60_000,
    placeholderData: keepPreviousData,
  });
  const other = useQuery({
    queryKey: ["accuracy-series", iv, days, "tuned"],
    queryFn: () => apiGet<AccuracyResponse>("/api/predict/accuracy", { interval: iv, days, backend: "sklearn", model_variant: "tuned" }),
    refetchInterval: 60_000,
    retry: false,
  });
  const pts = main.data?.points ?? [];
  const actual = useMemo(() => pts.map((p) => ({ timestamp: p.timestamp, value: p.actual_price })), [pts]);
  const predicted = useMemo(() => pts.map((p) => ({ timestamp: p.timestamp, value: p.predicted_price })), [pts]);
  const otherPts = useMemo(() => (other.data?.points ?? []).map((p) => ({ timestamp: p.timestamp, value: p.predicted_price })), [other.data]);

  return (
    <Panel className={clsx("flex min-w-0 flex-col", className)}>
      <div className="p-4 sm:p-5">
        <PanelHeader
          index="03"
          title={t("dashboard.accuracy.header")}
          hint={main.data?.mape != null ? t("dashboard.accuracy.mapeTemplate", { pct: main.data.mape.toFixed(2), n: pts.length, interval: iv }) : undefined}
          actions={
            <>
              <Segmented size="sm" ariaLabel={t("dashboard.accuracy.intervalAria")} options={(["15m", "1h", "1d"] as const).map((v) => ({ value: v, label: v }))} value={iv} onChange={setIv} />
              <Segmented
                size="sm"
                ariaLabel={t("dashboard.accuracy.rangeAria")}
                options={[
                  { value: "30", label: t("dashboard.accuracy.month") },
                  { value: "365", label: t("dashboard.accuracy.year") },
                ]}
                value={days}
                onChange={setDays}
              />
            </>
          }
        />
      </div>
      <div className="relative h-[280px] border-y border-line">
        <AccuracyChart actual={actual} predicted={predicted} other={otherPts} />
        {main.error ? (
          <Empty className="absolute inset-0 z-10 bg-panel">{t("dashboard.accuracy.fetchError", { msg: main.error.message })}</Empty>
        ) : main.data && !pts.length ? (
          <Empty className="absolute inset-0 z-10 bg-panel">{t("dashboard.accuracy.empty")}</Empty>
        ) : null}
      </div>
      <div className="px-4 py-3 sm:px-5">
        <Legend
          items={[
            { color: "var(--color-dim)", label: t("dashboard.accuracy.legendActual") },
            { color: "var(--color-accent)", label: t("dashboard.accuracy.legendPred"), dashed: true },
            { color: "var(--color-sky)", label: t("dashboard.accuracy.legendOther"), dashed: true },
          ]}
        />
      </div>
    </Panel>
  );
}

function SignalAccuracyPanel({ data, error, className }: { data?: SignalAccuracyResponse; error: boolean; className?: string }) {
  const { t } = useI18n();
  const total = (data?.buy?.count || 0) + (data?.sell?.count || 0) + (data?.wait?.count || 0);
  const Row = ({ label, stat, scored, tone }: { label: string; stat?: SignalStat; scored: boolean; tone: string }) => {
    const pct = scored && stat?.count ? Math.round(((stat.correct ?? 0) / stat.count) * 100) : null;
    return (
      <div className="py-3.5">
        <div className="flex items-baseline justify-between gap-3">
          <span className="flex items-center gap-2 text-[13px] text-dim">
            <span className={clsx("size-2 rounded-[2px]", tone)} />
            {label}
          </span>
          <span className={clsx("num text-[13px]", pct == null ? "text-bone" : pct >= 50 ? "text-up" : "text-down")}>
            {!stat?.count
              ? t("dashboard.signalAccuracy.noCalls")
              : scored
                ? t("dashboard.signalAccuracy.record", { correct: stat.correct ?? 0, count: stat.count, pct: pct ?? 0 })
                : stat.count}
          </span>
        </div>
        {pct != null && (
          <div className="mt-2 h-1 overflow-hidden rounded-full bg-press">
            <div className={clsx("h-full rounded-full", pct >= 50 ? "bg-up" : "bg-down")} style={{ width: `${pct}%` }} />
          </div>
        )}
      </div>
    );
  };
  return (
    <Panel className={clsx("p-4 sm:p-5", className)}>
      <PanelHeader index="04" title={t("dashboard.signalAccuracy.header")} hint={t("dashboard.signalAccuracy.disclaimer")} />
      {error || (data && total === 0) ? (
        <p className="mt-6 text-[13px] leading-relaxed text-faint">{t("dashboard.signalAccuracy.empty")}</p>
      ) : !data ? (
        <div className="mt-5 space-y-4">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-5" />)}</div>
      ) : (
        <div className="mt-2 divide-y divide-line">
          <Row label={t("dashboard.signalAccuracy.buy")} stat={data.buy} scored tone="bg-up" />
          <Row label={t("dashboard.signalAccuracy.sell")} stat={data.sell} scored tone="bg-down" />
          <Row label={t("dashboard.signalAccuracy.wait")} stat={data.wait} scored={false} tone="bg-faint" />
        </div>
      )}
    </Panel>
  );
}

function L2Grid({ chains }: { chains: L2Chain[] }) {
  const { t } = useI18n();
  const max = Math.max(...chains.map((c) => c.tvl_usd));
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        <thead>
          <tr className="border-t border-line text-left text-[11px] text-faint">
            <th className="py-2.5 pl-4 pr-3 font-medium sm:pl-5">{t("dashboard.l2.colNetwork")}</th>
            <th className="w-[45%] px-3 py-2.5 font-medium">{t("dashboard.l2.colTvl")}</th>
            <th className="px-3 py-2.5 text-right font-medium">{t("dashboard.l2.colSince")}</th>
            <th className="py-2.5 pl-3 pr-4 text-right font-medium sm:pr-5">{t("dashboard.l2.colTier")}</th>
          </tr>
        </thead>
        <tbody>
          {chains.map((c) => (
            <tr key={c.name} className="border-t border-line transition-colors hover:bg-raised">
              <td className="py-2.5 pl-4 pr-3 font-medium text-bone sm:pl-5">{c.display_name}</td>
              <td className="px-3 py-2.5">
                <div className="flex items-center gap-3">
                  <span className="num w-20 shrink-0 text-dim">{fmtCompactUsd(c.tvl_usd)}</span>
                  <span className="h-1 flex-1 overflow-hidden rounded-full bg-press">
                    <span className="block h-full rounded-full bg-accent/70" style={{ width: `${(c.tvl_usd / max) * 100}%` }} />
                  </span>
                </div>
              </td>
              <td className="num px-3 py-2.5 text-right text-dim">{c.launch_year}</td>
              <td className="py-2.5 pl-3 pr-4 text-right sm:pr-5">
                <Badge tone={c.tier === "established" ? "up" : c.tier === "growing" ? "warn" : "neutral"}>{t(`dashboard.l2.tier.${c.tier}`)}</Badge>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// =================================================================== explain

/** Rule-based reading of the active indicators (no external API). */
function explainIndicators(t: TFn, ind: Indicators, s: IndicatorSnapshot | null, signal: SignalSummary | null): string {
  const price = s?.price;
  if (price == null) return t("dashboard.explain.noPrice");
  if (!Object.values(ind).some(Boolean)) return t("dashboard.explain.noIndicators");
  const lines: string[] = [];
  if (ind.sma && s?.sma != null) {
    const diff = ((price - s.sma) / s.sma) * 100;
    lines.push(
      Math.abs(diff) < 0.1
        ? t("dashboard.explain.smaFlat", { price: fmtMoney(price), sma: fmtMoney(s.sma) })
        : t("dashboard.explain.smaTrend", {
            pct: Math.abs(diff).toFixed(2),
            dir: t(diff > 0 ? "dashboard.explain.above" : "dashboard.explain.below"),
            sma: fmtMoney(s.sma),
            trend: t(diff > 0 ? "dashboard.explain.uptrend" : "dashboard.explain.downtrend"),
          }),
    );
  }
  if (ind.ema && s?.ema != null) lines.push(t(price >= s.ema ? "dashboard.explain.emaUp" : "dashboard.explain.emaDown", { ema: fmtMoney(s.ema) }));
  if (ind.bollinger && s?.bollinger) {
    const { upper, lower } = s.bollinger;
    const pos = upper - lower > 0 ? ((price - lower) / (upper - lower)) * 100 : 50;
    if (pos >= 90) lines.push(t("dashboard.explain.bollUpper", { upper: fmtMoney(upper) }));
    else if (pos <= 10) lines.push(t("dashboard.explain.bollLower", { lower: fmtMoney(lower) }));
    else lines.push(t("dashboard.explain.bollMid", { lower: fmtMoney(lower), upper: fmtMoney(upper) }));
  }
  if (ind.vwap && s?.vwap != null) lines.push(t(price >= s.vwap ? "dashboard.explain.vwapAbove" : "dashboard.explain.vwapBelow", { vwap: fmtMoney(s.vwap) }));
  if (ind.fibonacci && s?.fib && s.fib.high > s.fib.low) {
    const { high, low } = s.fib;
    const nearest = FIB_LEVELS.map((lv) => ({ lv, p: high - lv * (high - low) })).reduce((a, b) => (Math.abs(price - b.p) < Math.abs(price - a.p) ? b : a));
    lines.push(t("dashboard.explain.fib", { low: fmtMoney(low), high: fmtMoney(high), level: (nearest.lv * 100).toFixed(1), price: fmtMoney(nearest.p) }));
  }
  if (signal?.rsi != null) lines.push(t("dashboard.explain.rsi", { value: signal.rsi.toFixed(1) }));
  if (signal?.macdBullish != null) lines.push(t("dashboard.explain.macd", { value: signal.macdBullish ? t("ui.bullish") : t("ui.bearish") }));
  if (signal?.kind === "wait") lines.push(t("dashboard.explain.signalWait"));
  else if (signal) lines.push(t("dashboard.explain.signalOther", { signal: t(`dashboard.signal.${signal.kind}`) }));
  return lines.join(" ");
}
