import clsx from "clsx";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, ChevronRight, Search, Sparkles, Star } from "lucide-react";
import { apiGet, type AssetType, type Pin, type ScoutResult } from "../lib/api";
import { dirText, fmtMoney, fmtRelative } from "../lib/format";
import { useI18n } from "../i18n";
import { Page } from "../components/Shell";
import { Badge, ConfidenceMeter, Delta, Empty, ErrorBanner, Panel, Segmented, Skeleton } from "../components/ui";
import { PinStar } from "../components/PinButton";
import { ArrowCircle, BigPrice, TokenIcon } from "../components/market";

type SortKey = "score" | "predicted_change_pct" | "confidence" | "sentiment_avg" | "articles_analyzed" | "current_price";
type Row = ScoutResult & { _orphan?: boolean; _rank?: number };

const pinKey = (t: string, id: string) => `${t}:${id}`;
const assetHref = (r: { asset_type: string; id: string }) => `/asset/${r.asset_type}/${encodeURIComponent(r.id)}`;

export default function Scout() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const [type, setType] = useState<"" | AssetType>("");
  const [pinnedOnly, setPinnedOnly] = useState(false);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });

  const results = useQuery({
    queryKey: ["scout", type],
    queryFn: () => apiGet<{ results: ScoutResult[] }>("/api/scout", { asset_type: type || undefined }),
    refetchInterval: 60_000,
  });
  const pins = useQuery({ queryKey: ["pins"], queryFn: () => apiGet<{ pins: Pin[] }>("/api/scout/pins") });
  const pinned = useMemo(() => new Set((pins.data?.pins ?? []).map((p) => pinKey(p.asset_type, p.id))), [pins.data]);

  const all = results.data?.results ?? [];
  const ranked: Row[] = useMemo(() => all.map((r, i) => ({ ...r, _rank: i + 1 })), [all]);

  const rows: Row[] = useMemo(() => {
    let list: Row[] = ranked;
    if (pinnedOnly) {
      const inScan = ranked.filter((r) => pinned.has(pinKey(r.asset_type, r.id)));
      const inKeys = new Set(inScan.map((r) => pinKey(r.asset_type, r.id)));
      const orphans = (pins.data?.pins ?? [])
        .filter((p) => !inKeys.has(pinKey(p.asset_type, p.id)) && (!type || p.asset_type === type))
        .map((p) => ({ ...p, _orphan: true }) as unknown as Row);
      list = [...inScan, ...orphans];
    }
    const q = query.trim().toLowerCase();
    if (q) list = list.filter((r) => r.symbol.toLowerCase().includes(q) || r.name.toLowerCase().includes(q));
    const { key, desc } = sort;
    return [...list].sort((a, b) => {
      if (a._orphan !== b._orphan) return a._orphan ? 1 : -1;
      const d = ((a[key] as number) ?? 0) - ((b[key] as number) ?? 0);
      return desc ? -d : d;
    });
  }, [ranked, pinnedOnly, pinned, pins.data, type, query, sort]);

  const latestScan = useMemo(() => {
    const times = all.map((r) => r.scanned_at).filter(Boolean) as string[];
    return times.length ? times.reduce((a, b) => (new Date(a) > new Date(b) ? a : b)) : null;
  }, [all]);

  const top3 = !pinnedOnly && !query ? ranked.slice(0, 4) : [];
  const toggleSort = (key: SortKey) => setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: true }));

  return (
    <Page
      kicker={
        <span className="flex items-center gap-2">
          <Sparkles className="size-3 text-accent" /> Scout AI
          {latestScan && <span className="normal-case tracking-normal text-faint">· {t("scout.scannedAt", { time: fmtRelative(latestScan, t) })}</span>}
        </span>
      }
      title={t("ui.scout.title")}
      subtitle={t("scout.header.note")}
    >
      {results.error && (
        <div className="mb-4">
          <ErrorBanner message={t("scout.error.fetchResults", { msg: results.error.message })} />
        </div>
      )}

      {/* top picks as token cards */}
      {top3.length > 0 && (
        <div className="mb-8">
          <h2 className="mb-4 text-[20px] font-semibold tracking-[-0.03em] text-bone">{t("ui.scout.topPicks")}</h2>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {top3.map((r, i) => (
              <Link
                key={pinKey(r.asset_type, r.id)}
                to={assetHref(r)}
                className="panel group flex flex-col p-5 transition-transform duration-300 hover:-translate-y-1 animate-rise"
                style={{ animationDelay: `${80 + i * 70}ms` }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <TokenIcon symbol={r.symbol} type={r.asset_type} />
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 text-[15px] font-medium text-bone">
                        <span className="truncate">{r.name}</span>
                        <span className="shrink-0 text-faint">({r.symbol})</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-faint">
                        {t(r.asset_type === "crypto" ? "common.assetType.crypto" : "common.assetType.stock")}
                        {r.tier === "mega_cap" && <Badge tone="accent">{t("scout.megaCap.label")}</Badge>}
                      </div>
                    </div>
                  </div>
                  <ArrowCircle />
                </div>
                <div className="mt-7 flex items-end justify-between gap-3">
                  <div>
                    <div className="mb-1 text-xs text-faint">{t("scout.table.pricePredicted")}</div>
                    <BigPrice value={r.predicted_price} smart className="text-[26px] font-semibold tracking-[-0.03em] text-bone" />
                    <div className="num mt-1 text-xs text-faint">{t("scout.table.priceCurrent")} {fmtMoney(r.current_price, { smart: true })}</div>
                  </div>
                  <div className="flex flex-col items-end gap-1.5">
                    <Delta value={r.predicted_change_pct} className="text-[15px] font-medium" />
                    <ConfidenceMeter value={r.confidence} compact />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* toolbar */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Segmented
          ariaLabel={t("scout.toolbar.typeAria")}
          options={[
            { value: "", label: t("scout.toolbar.all") },
            { value: "crypto", label: t("scout.toolbar.crypto") },
            { value: "stock", label: t("scout.toolbar.stocks") },
          ]}
          value={type}
          onChange={setType}
        />
        <button
          type="button"
          onClick={() => setPinnedOnly((v) => !v)}
          aria-pressed={pinnedOnly}
          className={clsx(
            "inline-flex h-[34px] items-center gap-1.5 rounded-xl border px-3 text-[13px] font-medium transition-colors",
            pinnedOnly ? "border-warn/40 bg-warn/10 text-warn" : "border-line text-dim hover:border-line-strong hover:text-bone",
          )}
        >
          <Star className="size-3.5" fill={pinnedOnly ? "currentColor" : "none"} />
          {t("scout.toolbar.pinned")}
          {pins.data && <span className="num text-[11px] opacity-70">{pins.data.pins.length}</span>}
        </button>
        <label className="relative ml-auto w-full sm:w-64">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-faint" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("ui.scout.search")}
            className="field w-full !pl-8"
          />
        </label>
      </div>

      <Panel className="overflow-hidden">
        {results.isLoading ? (
          <div className="space-y-px">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 px-5 py-4">
                <Skeleton className="h-4 w-6" />
                <Skeleton className="h-4 w-40" />
                <Skeleton className="ml-auto h-4 w-24" />
              </div>
            ))}
          </div>
        ) : !rows.length ? (
          <Empty className="py-16">{pinnedOnly ? t("scout.empty.pinned") : query ? t("ui.scout.noMatch") : t("scout.empty.noScan")}</Empty>
        ) : (
          <>
            {/* desktop table */}
            <div className="hidden overflow-x-auto lg:block">
              <table className="w-full border-collapse text-[13px]">
                <thead>
                  <tr className="border-b border-line text-[11px] text-faint">
                    <th className="w-12 py-3 pl-4" />
                    <th className="w-10 py-3 text-left font-medium">#</th>
                    <th className="py-3 pr-3 text-left font-medium">{t("scout.table.asset")}</th>
                    <SortTh k="current_price" sort={sort} onSort={toggleSort}>{t("scout.table.priceCurrent")}</SortTh>
                    <th className="px-3 py-3 text-right font-medium">{t("scout.table.pricePredicted")}</th>
                    <SortTh k="predicted_change_pct" sort={sort} onSort={toggleSort}>{t("scout.table.change")}</SortTh>
                    <SortTh k="confidence" sort={sort} onSort={toggleSort}>{t("scout.table.confidence")}</SortTh>
                    <SortTh k="sentiment_avg" sort={sort} onSort={toggleSort}>{t("scout.table.sentiment")}</SortTh>
                    <SortTh k="articles_analyzed" sort={sort} onSort={toggleSort}>{t("scout.table.news")}</SortTh>
                    <SortTh k="score" sort={sort} onSort={toggleSort} className="pr-5">{t("scout.table.score")}</SortTh>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const isPinned = pinned.has(pinKey(r.asset_type, r.id));
                    return (
                      <tr
                        key={pinKey(r.asset_type, r.id)}
                        onClick={() => navigate(assetHref(r))}
                        className="group cursor-pointer border-b border-line last:border-0 transition-colors hover:bg-raised"
                      >
                        <td className="py-3 pl-4">
                          <PinStar pin={r} pinned={isPinned} />
                        </td>
                        <td className="num py-3 text-faint">{r._orphan ? "—" : r._rank}</td>
                        <td className="py-3 pr-3">
                          <AssetCell r={r} />
                        </td>
                        {r._orphan ? (
                          <td colSpan={7} className="py-3 pr-5 text-right text-xs italic text-faint">
                            {t("scout.row.notInLatestScan")} <span className="not-italic text-accent">{t("scout.row.viewDetails")}</span>
                          </td>
                        ) : (
                          <>
                            <td className="num px-3 py-3 text-right text-dim">{fmtMoney(r.current_price, { smart: true })}</td>
                            <td className={clsx("num px-3 py-3 text-right", dirText(r.predicted_change_pct))}>{fmtMoney(r.predicted_price, { smart: true })}</td>
                            <td className="px-3 py-3 text-right">
                              <Delta value={r.predicted_change_pct} className="font-medium" />
                            </td>
                            <td className="px-3 py-3 text-right">
                              <ConfidenceMeter value={r.confidence} compact />
                            </td>
                            <td className={clsx("num px-3 py-3 text-right", dirText(r.sentiment_avg))}>{r.sentiment_avg.toFixed(2)}</td>
                            <td className="num px-3 py-3 text-right text-dim">{r.articles_analyzed}</td>
                            <td className="py-3 pl-3 pr-5 text-right">
                              <span className={clsx("num inline-block min-w-14 rounded-full px-2.5 py-1 text-center font-semibold", r.score >= 0 ? "bg-up/10 text-up" : "bg-down/10 text-down")}>
                                {r.score.toFixed(2)}
                              </span>
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* mobile / tablet cards */}
            <ul className="divide-y divide-line lg:hidden">
              {rows.map((r) => {
                const isPinned = pinned.has(pinKey(r.asset_type, r.id));
                return (
                  <li key={pinKey(r.asset_type, r.id)}>
                    <Link to={assetHref(r)} className="flex items-center gap-3 px-3 py-3.5 active:bg-raised sm:px-4">
                      <PinStar pin={r} pinned={isPinned} />
                      <div className="min-w-0 flex-1">
                        <AssetCell r={r} />
                      </div>
                      {r._orphan ? (
                        <span className="text-xs italic text-faint">{t("scout.row.viewDetails")}</span>
                      ) : (
                        <div className="flex flex-col items-end gap-1">
                          <Delta value={r.predicted_change_pct} className="text-[15px] font-medium" />
                          <ConfidenceMeter value={r.confidence} compact />
                        </div>
                      )}
                      <ChevronRight className="size-4 shrink-0 text-faint" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          </>
        )}
      </Panel>
      <p className="mt-3 text-xs leading-relaxed text-faint">{t("scout.toolbar.disclaimer")}</p>
    </Page>
  );
}

function AssetCell({ r }: { r: Row }) {
  const { t } = useI18n();
  return (
    <div className="flex min-w-0 items-center gap-3">
      <TokenIcon symbol={r.symbol} type={r.asset_type} size="sm" />
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-bone">{r.symbol}</span>
          {r.tier === "mega_cap" && (
            <span title={t("scout.megaCap.title")}>
              <Badge tone="accent">{t("scout.megaCap.label")}</Badge>
            </span>
          )}
        </div>
        <div className="truncate text-xs text-faint">
          {r.name} · {t(r.asset_type === "crypto" ? "common.assetType.crypto" : "common.assetType.stock")}
        </div>
      </div>
    </div>
  );
}

function SortTh({ k, sort, onSort, children, className }: { k: SortKey; sort: { key: SortKey; desc: boolean }; onSort: (k: SortKey) => void; children: React.ReactNode; className?: string }) {
  const active = sort.key === k;
  return (
    <th className={clsx("px-3 py-3 text-right font-medium", className)} aria-sort={active ? (sort.desc ? "descending" : "ascending") : "none"}>
      <button type="button" onClick={() => onSort(k)} className={clsx("inline-flex items-center gap-1 transition-colors hover:text-bone", active && "text-bone")}>
        {children}
        {active ? (sort.desc ? <ArrowDown className="size-3 text-accent" /> : <ArrowUp className="size-3 text-accent" />) : <span className="size-3" />}
      </button>
    </th>
  );
}
