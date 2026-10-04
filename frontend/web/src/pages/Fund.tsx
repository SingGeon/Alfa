// Strategy fund page: the evolving strategy population of ml/strategy_fund.py,
// served by /api/strategy/fund. Ported from the vanilla frontend's fund.html/js.
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { LineStyle, PriceScaleMode, type UTCTimestamp } from "lightweight-charts";
import { apiGet, type FundReport } from "../lib/api";
import { toUnix } from "../lib/format";
import { useI18n, type TFn } from "../i18n";
import { Page } from "../components/Shell";
import { ErrorBanner, Panel, PanelHeader, Segmented, Skeleton } from "../components/ui";
import { createChart, evoChartOptions, Sparkline, syncPanes } from "../components/EvoCharts";

type Interval = "4h" | "1d";
const INTERVALS: Interval[] = ["4h", "1d"];
const GENES: { key: "core" | "target" | "box_w" | "range_w"; range: [number, number] }[] = [
  { key: "core", range: [0, 1] },
  { key: "target", range: [0.1, 1.2] },
  { key: "box_w", range: [0, 1] },
  { key: "range_w", range: [0, 1] },
];

function makeFmt(locale: string) {
  const num = (v: number | null | undefined, d = 0) => (v == null ? "—" : v.toLocaleString(locale, { minimumFractionDigits: d, maximumFractionDigits: d }));
  const eur = (v: number | null | undefined) => (v == null ? "—" : `${num(v, 2)} €`);
  const pct = (v: number | null | undefined, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}%`);
  const day = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" }) : "—");
  const date = (iso: string | null | undefined) => (iso ? new Date(iso).toLocaleString(locale, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
  return { num, eur, pct, day, date };
}
type Fmt = ReturnType<typeof makeFmt>;

function positionLabel(t: TFn, p: number) {
  if (Math.abs(p) < 0.005) return t("pop.money.out");
  return `${t(p > 0 ? "pop.money.long" : "pop.money.short")} ${Math.round(Math.abs(p) * 100)}%`;
}
const sign = (v: number) => (v > 0 ? "text-up" : v < 0 ? "text-down" : "");

export default function Fund() {
  const { t, locale } = useI18n();
  const fmt = useMemo(() => makeFmt(locale), [locale]);
  const [interval, setInterval_] = useState<Interval>(() => {
    try {
      const saved = localStorage.getItem("fund.interval");
      if (saved && (INTERVALS as string[]).includes(saved)) return saved as Interval;
    } catch { /* storage unavailable */ }
    return "4h";
  });
  const pickInterval = (i: Interval) => {
    setInterval_(i);
    try { localStorage.setItem("fund.interval", i); } catch { /* ignore */ }
  };

  const report = useQuery({
    queryKey: ["strategy-fund", interval],
    queryFn: () => apiGet<FundReport>("/api/strategy/fund", { interval }),
    refetchInterval: 10 * 60_000,
    retry: false,
  });
  const r = report.data;
  const errMsg = report.error instanceof Error ? report.error.message : null;

  return (
    <Page
      kicker={t("nav.fund")}
      title={t("fund.title")}
      subtitle={t("fund.subtitle")}
      actions={<Segmented ariaLabel={t("nav.fund")} options={INTERVALS.map((i) => ({ value: i, label: i }))} value={interval} onChange={pickInterval} size="sm" />}
    >
      <p className="mb-4 -mt-4 text-[12.5px] text-faint">{t("fund.intervalsNote")}</p>

      {errMsg && (
        <div className="mb-4">
          <ErrorBanner message={errMsg} />
        </div>
      )}

      {!r ? (
        <div className="space-y-4">
          <Skeleton className="h-28" />
          <Skeleton className="h-64" />
        </div>
      ) : (
        <>
          <NowPanel r={r} fmt={fmt} t={t} />
          <MoneyPanel r={r} fmt={fmt} t={t} />
          <YearsPanel r={r} fmt={fmt} t={t} />
          <GenesPanel r={r} fmt={fmt} t={t} />
        </>
      )}
    </Page>
  );
}

function Tile({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1 bg-panel/90 px-4 py-3">
      <span className="text-[11px] text-faint">{label}</span>
      <span className="num text-[15px] font-medium text-bone">{value}</span>
    </div>
  );
}

function NowPanel({ r, fmt, t }: { r: FundReport; fmt: Fmt; t: TFn }) {
  const n = r.now;
  const box = n.box > 0 ? t("fund.now.boxLong") : n.box < 0 ? t("fund.now.boxShort") : t("fund.now.boxNone");
  const max = Math.max(0.01, ...Object.values(n.pieces).map((v) => Math.abs(v)));
  return (
    <Panel className="mb-4 p-4 sm:p-5">
      <PanelHeader title={t("fund.now.header")} />
      <p className="mt-2 text-[13px] text-faint">{t("fund.now.note", { candle: fmt.date(r.last_candle), computed: fmt.date(r.computed_at), price: fmt.num(r.price, 2) })}</p>
      <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-[20px] border border-line bg-line sm:grid-cols-3 lg:grid-cols-5">
        <Tile label={t("fund.now.position")} value={<span className={sign(n.position)}>{positionLabel(t, n.position)}</span>} />
        <Tile label={t("fund.now.inEur")} value={t("fund.now.inEurValue", { eur: fmt.eur(Math.abs(n.position) * r.whole_life.fund.money) })} />
        <Tile label={t("fund.now.vol")} value={n.vol_forecast_annual_pct == null ? "—" : `${fmt.num(n.vol_forecast_annual_pct, 0)}%`} />
        <Tile label={t("fund.now.box")} value={box} />
        <Tile label={t("fund.now.next")} value={fmt.day(n.next_reselection)} />
      </div>
      <div className="mt-5 text-[12px] font-medium text-bone">{t("fund.now.piecesHeading")}</div>
      <div className="mt-3 space-y-2.5">
        {(["core", "box", "range"] as const).map((k) => {
          const v = n.pieces[k];
          return (
            <div key={k} className="grid grid-cols-[minmax(0,1fr)_minmax(0,2fr)_auto] items-center gap-3 text-[12.5px]">
              <span className="text-dim">{t(`fund.piece.${k}`)}</span>
              <span className="h-1.5 overflow-hidden rounded-full bg-press">
                <span className={`block h-full rounded-full ${v < 0 ? "bg-down/70" : "bg-accent/70"}`} style={{ width: `${(Math.abs(v) / max) * 100}%` }} />
              </span>
              <b className={`num whitespace-nowrap text-right ${sign(v)}`}>{v > 0 ? "+" : ""}{Math.round(v * 100)}%</b>
            </div>
          );
        })}
      </div>
      <dl className="mt-5 grid gap-2.5 text-[12px] sm:grid-cols-3">
        {(["core", "box", "range"] as const).map((k) => (
          <div key={k}>
            <dt className="font-medium text-bone">{t(`fund.piece.${k}`)}</dt>
            <dd className="text-faint">{t(`fund.pieceHelp.${k}`)}</dd>
          </div>
        ))}
      </dl>
    </Panel>
  );
}

function MoneyPanel({ r, fmt, t }: { r: FundReport; fmt: Fmt; t: TFn }) {
  const moneyRef = useRef<HTMLDivElement>(null);
  const posRef = useRef<HTMLDivElement>(null);
  const [readout, setReadout] = useState({ money: "", position: "" });
  const w = r.whole_life, te = r.test;

  useEffect(() => {
    const moneyEl = moneyRef.current, posEl = posRef.current;
    if (!moneyEl || !posEl) return;
    const C = getComputedStyle(document.documentElement);
    const pred = C.getPropertyValue("--color-accent-hi").trim() || "#ff8a4c";
    const muted = C.getPropertyValue("--color-faint").trim() || "#66666e";
    const up = C.getPropertyValue("--color-up").trim() || "#34d399";
    const down = C.getPropertyValue("--color-down").trim() || "#ff3d6e";
    const pts = r.curve.map((p) => ({ ...p, t: toUnix(p.time) as UTCTimestamp })).filter((p, i, a) => i === 0 || p.t > a[i - 1].t);

    const moneyChart = createChart(moneyEl, { ...evoChartOptions(moneyEl), rightPriceScale: { borderVisible: false, minimumWidth: 70, mode: PriceScaleMode.Logarithmic } });
    const eurFmt = { type: "custom" as const, formatter: (v: number) => `${v.toFixed(0)} €` };
    moneyChart.addLineSeries({ color: muted, lineWidth: 2, priceLineVisible: false, priceFormat: eurFmt, title: t("pop.money.hold") }).setData(pts.map((p) => ({ time: p.t, value: p.buy_hold })));
    const fund = moneyChart.addLineSeries({ color: pred, lineWidth: 2, priceLineVisible: false, priceFormat: eurFmt, title: t("fund.money.fund") });
    fund.setData(pts.map((p) => ({ time: p.t, value: p.fund })));
    fund.createPriceLine({ price: 100, color: muted, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false });

    const posChart = createChart(posEl, evoChartOptions(posEl));
    const posSeries = posChart.addBaselineSeries({
      baseValue: { type: "price", price: 0 }, lineWidth: 1, priceLineVisible: false,
      topLineColor: up, bottomLineColor: down,
      topFillColor1: "rgba(52,211,153,0.25)", topFillColor2: "rgba(52,211,153,0.05)",
      bottomFillColor1: "rgba(255,61,110,0.05)", bottomFillColor2: "rgba(255,61,110,0.25)",
      priceFormat: { type: "custom", formatter: (v: number) => v.toFixed(2) },
    });
    posSeries.setData(pts.map((p) => ({ time: p.t, value: p.position })));

    const charts = [moneyChart, posChart];
    const byTime = new Map(pts.map((p) => [p.t, p]));
    const readoutFor = (point?: typeof pts[number]) => {
      if (!point) { setReadout({ money: "", position: "" }); return; }
      setReadout({ money: `${fmt.eur(point.fund)} · ${fmt.eur(point.buy_hold)} · ${fmt.day(point.time)}`, position: positionLabel(t, point.position) });
    };
    const unsync = syncPanes(charts, (time) => readoutFor(time != null ? byTime.get(time as UTCTimestamp) : pts[pts.length - 1]));
    moneyChart.timeScale().fitContent();
    readoutFor(pts[pts.length - 1]);

    const ro = new ResizeObserver(() => { moneyChart.applyOptions({ width: moneyEl.clientWidth, height: moneyEl.clientHeight }); posChart.applyOptions({ width: posEl.clientWidth, height: posEl.clientHeight }); });
    ro.observe(moneyEl); ro.observe(posEl);
    return () => { ro.disconnect(); unsync(); charts.forEach((c) => c.remove()); };
  }, [r, fmt, t]);

  return (
    <Panel className="mb-4 p-4 sm:p-5">
      <PanelHeader title={t("fund.money.header")} />
      <p className="mt-2 text-[13px] text-faint">{t("fund.money.note", { from: fmt.day(r.trading_from), test: fmt.day(r.test_from) })}</p>
      <div className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-[20px] border border-line bg-line sm:grid-cols-4">
        <Tile label={t("fund.money.since", { from: fmt.day(r.trading_from) })} value={<>{fmt.eur(w.fund.money)} <small className={sign(w.fund.ret_pct)}>{fmt.pct(w.fund.ret_pct)}</small></>} />
        <Tile label={t("pop.money.hold")} value={<>{fmt.eur(w.buy_hold.money)} <small className={sign(w.buy_hold.ret_pct)}>{fmt.pct(w.buy_hold.ret_pct)}</small></>} />
        <Tile label={t("fund.money.test")} value={<>{fmt.eur(te.fund.money)} <small className={sign(te.fund.ret_pct)}>{fmt.pct(te.fund.ret_pct)}</small></>} />
        <Tile label={t("fund.money.testHold")} value={<>{fmt.eur(te.buy_hold.money)} <small className={sign(te.buy_hold.ret_pct)}>{fmt.pct(te.buy_hold.ret_pct)}</small></>} />
        <Tile label={t("fund.money.dd")} value={`${fmt.num(w.fund.max_dd_pct, 0)}% · ${t("fund.money.ddHold", { dd: fmt.num(w.buy_hold.max_dd_pct, 0) })}`} />
        <Tile label={t("fund.money.sharpe")} value={`${fmt.num(w.fund.sharpe, 2)} · ${t("fund.money.ddHold", { dd: fmt.num(w.buy_hold.sharpe, 2) })}`} />
        <Tile label={t("pop.money.fundTrades")} value={`${fmt.num(w.fund.trades)} · ${fmt.eur(w.fund.fees)}`} />
      </div>
      <div className="mt-4 flex items-baseline justify-between text-[12px]">
        <span className="text-faint">{t("fund.money.fund")} · {t("pop.money.hold")}</span>
        <b className="num text-bone">{readout.money}</b>
      </div>
      <div ref={moneyRef} className="mt-1.5 h-56" />
      <div className="mt-4 flex items-baseline justify-between text-[12px]">
        <span className="text-faint">{t("fund.money.position")}</span>
        <b className="num text-bone">{readout.position}</b>
      </div>
      <div ref={posRef} className="mt-1.5 h-28" />
    </Panel>
  );
}

function YearsPanel({ r, fmt, t }: { r: FundReport; fmt: Fmt; t: TFn }) {
  const testYear = new Date(r.test_from).getUTCFullYear();
  const cell = (v: number) => <span className={sign(v)}>{fmt.pct(v)}</span>;
  return (
    <Panel className="mb-4 overflow-hidden">
      <div className="p-4 sm:p-5">
        <PanelHeader title={t("fund.years.header")} hint={t("fund.years.hint")} />
      </div>
      <div className="overflow-x-auto border-t border-line">
        <table className="w-full min-w-[560px] border-collapse text-[12.5px]">
          <thead>
            <tr className="border-b border-line text-left text-[11px] text-faint">
              {[t("fund.col.year"), t("fund.col.fund"), t("fund.col.fundDd"), t("fund.col.hold"), t("fund.col.holdDd"), t("fund.col.trades")].map((h, i) => (
                <th key={h} className={`py-2.5 font-medium ${i > 0 ? "px-3 text-right" : "px-3"} ${i === 0 ? "pl-4 sm:pl-5" : ""}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {r.years.map((y) => (
              <tr key={y.year} className="border-b border-line last:border-0">
                <td className="py-2.5 pl-4 pr-3 text-bone sm:pl-5">{y.year}{testYear <= y.year && <span className="ml-1.5 text-[11px] text-faint">{t("fund.years.unseen")}</span>}</td>
                <td className="num px-3 py-2.5 text-right">{cell(y.fund_pct)}</td>
                <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(y.fund_dd, 0)}%</td>
                <td className="num px-3 py-2.5 text-right">{cell(y.buy_hold_pct)}</td>
                <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(y.buy_hold_dd, 0)}%</td>
                <td className="num px-3 py-2.5 text-right text-dim">{fmt.num(y.trades)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function GenesPanel({ r, fmt, t }: { r: FundReport; fmt: Fmt; t: TFn }) {
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const times = r.genes_over_time.map((g) => g.time);
  const st = r.settings;
  return (
    <Panel className="p-4 sm:p-5">
      <PanelHeader title={t("fund.genes.header")} hint={t("fund.genes.hint")} />
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {GENES.map((g) => (
          <Sparkline
            key={g.key}
            title={t(`fund.gene.${g.key}`)}
            times={times}
            range={g.range}
            format={(v) => (v == null ? "—" : pct(v))}
            values={r.genes_over_time.map((p) => p[g.key] as number)}
            fmtDay={fmt.day}
          />
        ))}
        <Sparkline
          title={t("fund.gene.long_short")}
          times={times}
          range={[0, 1]}
          format={(v) => (v == null ? "—" : pct(v))}
          values={r.genes_over_time.map((p) => (p.long_short ? 1 : 0))}
          fmtDay={fmt.day}
        />
      </div>

      <div className="mt-6 text-[12px] font-medium text-bone">{t("fund.genes.topHeading")}</div>
      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-[12.5px]">
          <thead>
            <tr className="border-b border-line text-left text-[11px] text-faint">
              {[t("fund.col.core"), t("fund.col.target"), t("fund.col.box"), t("fund.col.range"), t("fund.col.short"), t("fund.col.position"), t("fund.col.score")].map((h, i) => (
                <th key={h} className={`py-2.5 font-medium ${i < 6 ? "px-3 text-right" : "px-3 text-right"} ${i === 0 ? "pl-4 sm:pl-5" : ""}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {r.strategies.map((s, i) => (
              <tr key={i} className="border-b border-line last:border-0">
                <td className="num px-3 py-2.5 pl-4 text-right text-dim sm:pl-5">{pct(s.genes.core)}</td>
                <td className="num px-3 py-2.5 text-right text-dim">{pct(s.genes.target)}</td>
                <td className="num px-3 py-2.5 text-right text-dim">{pct(s.genes.box_w)}</td>
                <td className="num px-3 py-2.5 text-right text-dim">{pct(s.genes.range_w)}</td>
                <td className="px-3 py-2.5 text-right text-dim">{t(s.genes.long_short ? "fund.yes" : "fund.no")}</td>
                <td className={`num px-3 py-2.5 text-right ${sign(s.position)}`}>{positionLabel(t, s.position)}</td>
                <td className="num px-3 py-2.5 text-right text-bone">{fmt.num(s.score, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-[12px] text-faint">
        {t("fund.genes.settings", { pop: st.population, top: st.top_k, replace: st.replace, every: st.reselect_days, years: Math.round(st.lookback_days / 365), seeds: st.seeds.length, fee: st.fee * 100 })}
      </p>
    </Panel>
  );
}
