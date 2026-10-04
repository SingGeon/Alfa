import clsx from "clsx";
import { useEffect } from "react";
import { Link, useParams } from "react-router";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowLeft, Star } from "lucide-react";
import { apiGet, type AssetType, type Pin, type Quote, type ScoutDetail } from "../lib/api";
import { dirText, fmtDay, fmtMoney } from "../lib/format";
import { useI18n } from "../i18n";
import { PriceChart } from "../components/PriceChart";
import { ForecastTable, Legend, NewsList } from "../components/shared";
import { Badge, Button, ConfidenceMeter, Delta, ErrorBanner, Panel, PanelHeader, Skeleton, Stat } from "../components/ui";
import { usePinToggle } from "../components/PinButton";
import { BigPrice, TokenIcon } from "../components/market";

export default function AssetDetail() {
  const { type, id = "" } = useParams<{ type: AssetType; id: string }>();
  const { t, lang, locale } = useI18n();
  const valid = (type === "crypto" || type === "stock") && !!id;

  // The server caches this per (asset, lang); keepPreviousData keeps the
  // chart on screen while a language switch fetches the new narrative.
  const detail = useQuery({
    queryKey: ["scout-detail", type, id, lang],
    queryFn: () => apiGet<ScoutDetail>("/api/scout/detail", { asset_type: type, id, lang }),
    enabled: valid,
    placeholderData: keepPreviousData,
    retry: 1,
    staleTime: 5 * 60_000,
  });
  const quote = useQuery({
    queryKey: ["quote", type, id],
    queryFn: () => apiGet<Quote>("/api/scout/price", { asset_type: type, id }),
    enabled: valid,
    refetchInterval: 5000,
    retry: false,
  });
  const pins = useQuery({ queryKey: ["pins"], queryFn: () => apiGet<{ pins: Pin[] }>("/api/scout/pins") });
  const toggle = usePinToggle();

  const d = detail.data && detail.data.id === id ? detail.data : undefined;
  const isPinned = (pins.data?.pins ?? []).some((p) => p.asset_type === type && p.id === id);
  const livePrice = quote.data?.price ?? d?.current_price;
  // The detail endpoint returns the forecast path only; derive its end point.
  const target = d?.predictions.at(-1)?.predicted_price ?? null;
  const changePct = d && target != null ? ((target - d.current_price) / d.current_price) * 100 : null;

  useEffect(() => {
    if (!d) return;
    const prev = document.title;
    document.title = `${d.symbol} — Scout AI · ETH Price Predictor`;
    return () => {
      document.title = prev;
    };
  }, [d]);

  return (
    <div className="mx-auto max-w-[1440px] px-4 pb-4 pt-5 sm:px-6 sm:pt-7">
      <Link to="/scout" className="mb-6 inline-flex items-center gap-1.5 rounded-full border border-line bg-white/[0.03] px-3.5 py-1.5 text-[13px] text-dim backdrop-blur-md transition-colors hover:border-line-strong hover:text-bone">
        <ArrowLeft className="size-3.5" /> Scout AI
      </Link>

      {/* header */}
      <div className="mb-6 flex flex-wrap items-end justify-between gap-x-8 gap-y-4 animate-rise">
        <div className="min-w-0">
          <div className="eyebrow mb-2 flex items-center gap-2">
            <span className={clsx("size-1.5 rounded-full", type === "crypto" ? "bg-sky" : "bg-rose")} />
            {type ? t(type === "crypto" ? "common.assetType.crypto" : "common.assetType.stock") : ""}
            {d && <span className="text-faint/70">· {d.symbol}</span>}
          </div>
          <h1 className="flex items-center gap-4 text-[34px] font-semibold leading-none tracking-[-0.035em] sm:text-[48px]">
            {d && <TokenIcon symbol={d.symbol} type={d.asset_type} size="lg" />}
            <span className="text-gradient pb-1">{d?.name ?? (detail.isLoading ? <Skeleton className="h-10 w-72" /> : id)}</span>
          </h1>
          <div className="mt-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <BigPrice value={livePrice} smart className="text-[30px] font-semibold tracking-[-0.03em] text-bone" />
            {quote.data?.change_24h_pct != null && (
              <>
                <Delta value={quote.data.change_24h_pct} />
                <span className="text-xs text-faint">24h</span>
              </>
            )}
          </div>
        </div>
        {d && (
          <Button
            variant={isPinned ? "outline" : "ghost"}
            className={clsx("border border-line", isPinned && "!border-warn/40 !bg-warn/10 !text-warn")}
            onClick={() => toggle.mutate({ pin: { asset_type: d.asset_type, id: d.id, symbol: d.symbol, name: d.name }, pinned: isPinned })}
          >
            <Star className="size-3.5" fill={isPinned ? "currentColor" : "none"} />
            {t(isPinned ? "detail.pin.pinned" : "detail.pin.pin")}
          </Button>
        )}
      </div>

      {!valid && <ErrorBanner message={t("detail.error.missingParams")} />}
      {detail.error && <ErrorBanner message={t("detail.error.analysisFailed", { msg: detail.error.message })} />}

      <div className="mt-4 grid gap-4 lg:grid-cols-12">
        <Panel className="flex min-w-0 flex-col lg:col-span-8 xl:col-span-9">
          <div className="relative h-[420px] sm:h-[520px]">
            <PriceChart
              candles={d?.candles ?? []}
              predictions={d?.predictions ?? []}
              viewKey={id}
              defaultWindow={45}
              timeVisible={false}
              smartPrice
            />
            {!d && !detail.error && valid && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-panel text-[13px] text-faint">
                <div className="size-6 animate-spin rounded-full border-2 border-line-strong border-t-accent" />
                {t("detail.chart.analyzing")}
              </div>
            )}
          </div>
          <div className="border-t border-line px-4 py-3">
            <Legend
              items={[
                { color: "var(--color-up)", label: t("common.legend.up") },
                { color: "var(--color-down)", label: t("common.legend.down") },
                { color: "var(--color-accent)", label: t("common.legend.pred") },
                { color: "var(--color-accent)", label: t("common.legend.band"), soft: true },
              ]}
            />
          </div>
        </Panel>

        <div className="flex min-w-0 flex-col gap-4 lg:col-span-4 xl:col-span-3">
          <Panel className="p-4 sm:p-5">
            <div className="text-xs text-faint">{t("ui.detail.target", { n: d?.horizon_days ?? 7 })}</div>
            {d ? (
              <>
                <Delta value={changePct} className="mt-1 text-[32px] font-semibold tracking-tight" />
                <div className="num mt-1 text-[13px] text-dim">
                  {fmtMoney(d.current_price, { smart: true })} → <span className={dirText(changePct)}>{fmtMoney(target, { smart: true })}</span>
                </div>
              </>
            ) : (
              <Skeleton className="mt-2 h-9 w-32" />
            )}
            <div className="mt-5 grid grid-cols-2 gap-4 border-t border-line pt-4">
              <Stat label={t("common.stat.confidence")} value={<ConfidenceMeter value={d?.confidence} />} />
              <Stat label={t("common.stat.sentimentAvg")} value={<span className={dirText(d?.sentiment_avg)}>{d ? d.sentiment_avg.toFixed(2) : "—"}</span>} />
              <Stat label={t("scout.table.news")} value={d ? d.news.length : "—"} />
              <Stat label={t("dashboard.stat.model")} value={<Badge>sklearn · ensemble</Badge>} />
            </div>
          </Panel>

          <Panel className="flex min-h-0 flex-1 flex-col">
            <div className="px-4 pb-3 pt-4 sm:px-5">
              <PanelHeader title={t("detail.prediction.header")} />
            </div>
            <div className="max-h-[300px] overflow-y-auto border-t border-line">
              {d ? (
                <ForecastTable predictions={d.predictions} basePrice={d.current_price} formatTime={(iso) => fmtDay(iso, locale)} timeHeader={t("detail.prediction.colDay")} smart />
              ) : (
                <div className="space-y-2 p-5">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-4" />)}</div>
              )}
            </div>
          </Panel>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-12">
        <Panel className="p-4 sm:p-6 lg:col-span-7">
          <PanelHeader index="01" title={t("detail.narrative.header")} />
          {d ? (
            <p className="mt-4 whitespace-pre-line font-display text-[17px] leading-[1.6] text-bone/95">{d.narrative}</p>
          ) : (
            <div className="mt-4 space-y-2.5">
              <Skeleton className="h-4" />
              <Skeleton className="h-4 w-11/12" />
              <Skeleton className="h-4 w-3/4" />
            </div>
          )}
        </Panel>
        <Panel className="p-4 sm:p-6 lg:col-span-5">
          <PanelHeader index="02" title={t("detail.news.header")} />
          <div className="mt-4 max-h-[440px] overflow-y-auto pl-3 sm:pl-4">
            <NewsList news={d?.news} loading={!d && !detail.error} emptyText={t("detail.news.empty")} />
          </div>
        </Panel>
      </div>
    </div>
  );
}
