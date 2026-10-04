"use strict";

// Strategy fund page: the evolving strategy population of
// ml/strategy_fund.py, served by /api/strategy/fund.

const INTERVALS = ["4h", "1d"];
const GENES = [
  { key: "core", range: [0, 1], pct: true },
  { key: "target", range: [0.1, 1.2], pct: true },
  { key: "box_w", range: [0, 1], pct: true },
  { key: "range_w", range: [0, 1], pct: true },
  { key: "long_short", range: [0, 1], pct: true },
];

const state = { interval: "4h", report: null };
const charts = {};

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtNum = (v, d = 0) => (v == null ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const fmtEur = (v) => (v == null ? "—" : `${fmtNum(v, 2)} €`);
const fmtPct = (v, d = 1) => (v == null ? "—" : `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}%`);
const fmtDay = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "—");
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
const sign = (v) => (v > 0 ? "pop-up" : v < 0 ? "pop-down" : "");
const tile = (label, value, cls = "") => `<div class="box-stat"><span>${esc(label)}</span><b class="${cls}">${value}</b></div>`;

function positionLabel(p) {
  if (Math.abs(p) < 0.005) return t("pop.money.out");
  return `${t(p > 0 ? "pop.money.long" : "pop.money.short")} ${Math.round(Math.abs(p) * 100)}%`;
}

function showError(msg) {
  $("errorBanner").hidden = !msg;
  $("errorBanner").textContent = msg || "";
}

// -- now -------------------------------------------------------------------

function renderNow(r) {
  const n = r.now;
  $("nowNote").textContent = tf("fund.now.note", { candle: fmtDate(r.last_candle), computed: fmtDate(r.computed_at), price: fmtNum(r.price, 2) });
  const box = n.box > 0 ? t("fund.now.boxLong") : n.box < 0 ? t("fund.now.boxShort") : t("fund.now.boxNone");
  $("nowTiles").innerHTML = [
    tile(t("fund.now.position"), positionLabel(n.position), sign(n.position)),
    tile(t("fund.now.inEur"), tf("fund.now.inEurValue", { eur: fmtEur(Math.abs(n.position) * r.whole_life.fund.money) })),
    tile(t("fund.now.vol"), n.vol_forecast_annual_pct == null ? "—" : `${fmtNum(n.vol_forecast_annual_pct, 0)}%`),
    tile(t("fund.now.box"), box),
    tile(t("fund.now.next"), fmtDay(n.next_reselection)),
  ].join("");
  const max = Math.max(0.01, ...Object.values(n.pieces).map(Math.abs));
  $("pieces").innerHTML = ["core", "box", "range"].map((k) => {
    const v = n.pieces[k];
    return `<div class="pop-bar-row"><span>${esc(t(`fund.piece.${k}`))}</span>
      <div class="pop-bar-track"><i class="${v < 0 ? "fund-neg" : ""}" style="width:${(Math.abs(v) / max) * 100}%"></i></div>
      <b class="${sign(v)}">${v > 0 ? "+" : ""}${Math.round(v * 100)}%</b></div>`;
  }).join("");
}

// -- money -----------------------------------------------------------------

function chartOptions(el, extra = {}) {
  const css = getComputedStyle(document.documentElement);
  return {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: { background: { color: "transparent" }, textColor: css.getPropertyValue("--text-muted").trim() || "#676f83", fontSize: 11 },
    grid: { vertLines: { visible: false }, horzLines: { color: "rgba(255,255,255,0.05)" } },
    rightPriceScale: { borderVisible: false, minimumWidth: 70 },
    timeScale: { borderVisible: false, timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
    ...extra,
  };
}

function renderMoney(r) {
  const w = r.whole_life, te = r.test;
  $("moneyNote").textContent = tf("fund.money.note", { from: fmtDay(r.trading_from), test: fmtDay(r.test_from) });
  $("moneyTiles").innerHTML = [
    tile(tf("fund.money.since", { from: fmtDay(r.trading_from) }), `${fmtEur(w.fund.money)} <small class="${sign(w.fund.ret_pct)}">${fmtPct(w.fund.ret_pct)}</small>`),
    tile(t("pop.money.hold"), `${fmtEur(w.buy_hold.money)} <small class="${sign(w.buy_hold.ret_pct)}">${fmtPct(w.buy_hold.ret_pct)}</small>`),
    tile(t("fund.money.test"), `${fmtEur(te.fund.money)} <small class="${sign(te.fund.ret_pct)}">${fmtPct(te.fund.ret_pct)}</small>`),
    tile(t("fund.money.testHold"), `${fmtEur(te.buy_hold.money)} <small class="${sign(te.buy_hold.ret_pct)}">${fmtPct(te.buy_hold.ret_pct)}</small>`),
    tile(t("fund.money.dd"), `${fmtNum(w.fund.max_dd_pct, 0)}% <small>${tf("fund.money.ddHold", { dd: fmtNum(w.buy_hold.max_dd_pct, 0) })}</small>`),
    tile(t("fund.money.sharpe"), `${fmtNum(w.fund.sharpe, 2)} <small>${tf("fund.money.ddHold", { dd: fmtNum(w.buy_hold.sharpe, 2) })}</small>`),
    tile(t("pop.money.fundTrades"), `${fmtNum(w.fund.trades)} · ${fmtEur(w.fund.fees)}`),
  ].join("");

  Object.values(charts).forEach((c) => c.remove());
  const css = getComputedStyle(document.documentElement);
  const pred = css.getPropertyValue("--pred").trim() || "#6d7cf5";
  const muted = css.getPropertyValue("--text-muted").trim() || "#676f83";
  const pts = r.curve.map((p) => ({ ...p, time: Math.floor(new Date(p.time).getTime() / 1000) }))
    .filter((p, i, a) => i === 0 || p.time > a[i - 1].time);

  const mel = $("moneyChart");
  charts.money = LightweightCharts.createChart(mel, chartOptions(mel, { rightPriceScale: { borderVisible: false, minimumWidth: 70, mode: LightweightCharts.PriceScaleMode.Logarithmic } }));
  const eur = { type: "custom", formatter: (v) => `${v.toFixed(0)} €` };
  charts.money.addLineSeries({ color: muted, lineWidth: 2, priceLineVisible: false, priceFormat: eur, title: t("pop.money.hold") })
    .setData(pts.map((p) => ({ time: p.time, value: p.buy_hold })));
  const fund = charts.money.addLineSeries({ color: pred, lineWidth: 2, priceLineVisible: false, priceFormat: eur, title: t("fund.money.fund") });
  fund.setData(pts.map((p) => ({ time: p.time, value: p.fund })));
  fund.createPriceLine({ price: 100, color: muted, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false });

  const pel = $("positionChart");
  charts.position = LightweightCharts.createChart(pel, chartOptions(pel));
  const posSeries = charts.position.addBaselineSeries({
    baseValue: { type: "price", price: 0 }, lineWidth: 1, priceLineVisible: false,
    topLineColor: css.getPropertyValue("--up").trim() || "#2fbf76", bottomLineColor: css.getPropertyValue("--down").trim() || "#f0546b",
    topFillColor1: "rgba(47,191,118,0.25)", topFillColor2: "rgba(47,191,118,0.05)",
    bottomFillColor1: "rgba(240,84,107,0.05)", bottomFillColor2: "rgba(240,84,107,0.25)",
    priceFormat: { type: "custom", formatter: (v) => v.toFixed(2) },
  });
  posSeries.setData(pts.map((p) => ({ time: p.time, value: p.position })));

  const byTime = new Map(pts.map((p) => [p.time, p]));
  const readout = (p) => {
    $("moneyReadout").textContent = p ? `${fmtEur(p.fund)} · ${fmtEur(p.buy_hold)} · ${fmtDay(new Date(p.time * 1000).toISOString())}` : "";
    $("positionReadout").textContent = p ? positionLabel(p.position) : "";
  };
  [charts.money, charts.position].forEach((c, i, all) => {
    c.subscribeCrosshairMove((param) => readout(param.time ? byTime.get(param.time) : pts[pts.length - 1]));
    c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (range) all.forEach((o) => o !== c && o.timeScale().setVisibleLogicalRange(range));
    });
  });
  charts.money.timeScale().fitContent();
  readout(pts[pts.length - 1]);
}

// -- years -----------------------------------------------------------------

function renderYears(r) {
  const cell = (v) => `<span class="${sign(v)}">${fmtPct(v)}</span>`;
  $("yearRows").innerHTML = r.years.map((y) => `<tr>
    <td>${y.year}${new Date(r.test_from).getUTCFullYear() <= y.year ? ` <span class="pop-small">${esc(t("fund.years.unseen"))}</span>` : ""}</td>
    <td class="num">${cell(y.fund_pct)}</td><td class="num">${fmtNum(y.fund_dd, 0)}%</td>
    <td class="num">${cell(y.buy_hold_pct)}</td><td class="num">${fmtNum(y.buy_hold_dd, 0)}%</td>
    <td class="num">${fmtNum(y.trades)}</td></tr>`).join("");
}

// -- genes -----------------------------------------------------------------

function sparkline(container, { title, values, times, range }) {
  const W = 300, H = 56;
  const lo = range[0], span = range[1] - range[0] || 1;
  const x = (i) => (values.length > 1 ? (i / (values.length - 1)) * W : 0);
  const y = (v) => H - ((v - lo) / span) * (H - 4) - 2;
  const line = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const fmt = (v) => `${Math.round(v * 100)}%`;
  const last = values[values.length - 1];
  const card = document.createElement("div");
  card.className = "pop-spark";
  card.innerHTML = `
    <div class="pop-spark-head"><span>${esc(title)}</span><b data-value>${fmt(last)}</b></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(title)}">
      <polygon class="spark-area" points="0,${H} ${line} ${W},${H}" />
      <polyline class="spark-line" points="${line}" />
      <line class="spark-cursor" x1="0" x2="0" y1="0" y2="${H}" visibility="hidden" />
    </svg>
    <div class="pop-spark-foot"><span>${fmtDay(times[0])}</span><span data-when>${fmtDay(times[times.length - 1])}</span></div>`;
  const svg = card.querySelector("svg"), cursor = card.querySelector(".spark-cursor");
  const out = card.querySelector("[data-value]"), when = card.querySelector("[data-when]");
  svg.addEventListener("mousemove", (e) => {
    const rect = svg.getBoundingClientRect();
    const k = Math.max(0, Math.min(values.length - 1, Math.round(((e.clientX - rect.left) / rect.width) * (values.length - 1))));
    cursor.setAttribute("x1", x(k));
    cursor.setAttribute("x2", x(k));
    cursor.setAttribute("visibility", "visible");
    out.textContent = fmt(values[k]);
    when.textContent = fmtDay(times[k]);
  });
  svg.addEventListener("mouseleave", () => {
    cursor.setAttribute("visibility", "hidden");
    out.textContent = fmt(last);
    when.textContent = fmtDay(times[times.length - 1]);
  });
  container.appendChild(card);
}

function renderGenes(r) {
  const box = $("geneSparks");
  box.innerHTML = "";
  const times = r.genes_over_time.map((g) => g.time);
  GENES.forEach((g) => sparkline(box, { title: t(`fund.gene.${g.key}`), times, range: g.range, values: r.genes_over_time.map((p) => p[g.key]) }));
  const pct = (v) => `${Math.round(v * 100)}%`;
  $("strategyRows").innerHTML = r.strategies.map((s) => `<tr>
    <td class="num">${pct(s.genes.core)}</td><td class="num">${pct(s.genes.target)}</td>
    <td class="num">${pct(s.genes.box_w)}</td><td class="num">${pct(s.genes.range_w)}</td>
    <td>${esc(t(s.genes.long_short ? "fund.yes" : "fund.no"))}</td>
    <td class="num ${sign(s.position)}">${esc(positionLabel(s.position))}</td>
    <td class="num">${fmtNum(s.score, 3)}</td></tr>`).join("");
  const st = r.settings;
  $("settingsNote").textContent = tf("fund.genes.settings", {
    pop: st.population, top: st.top_k, replace: st.replace, every: st.reselect_days,
    years: Math.round(st.lookback_days / 365), seeds: st.seeds.length, fee: st.fee * 100,
  });
}

// -- load ------------------------------------------------------------------

function renderAll() {
  const r = state.report;
  if (!r) return;
  renderNow(r);
  renderMoney(r);
  renderYears(r);
  renderGenes(r);
}

async function load() {
  [...$("intervalTabs").children].forEach((b) => b.classList.toggle("active", b.dataset.interval === state.interval));
  try {
    const resp = await fetch(`/api/strategy/fund?interval=${state.interval}`);
    const body = await resp.json().catch(() => null);
    // No JSON at all: a server started before this page existed (restart it).
    if (!body) throw new Error(t("fund.error.oldServer"));
    if (!resp.ok) throw new Error(body.error || `${resp.status}`);
    state.report = body;
    showError("");
    renderAll();
  } catch (exc) {
    state.report = null;
    showError(exc.message);
  }
}

$("intervalTabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-interval]");
  if (!b || b.dataset.interval === state.interval) return;
  state.interval = b.dataset.interval;
  try { localStorage.setItem("fund.interval", state.interval); } catch { /* storage unavailable */ }
  load();
});
window.addEventListener("resize", () => {
  if (charts.money) charts.money.resize($("moneyChart").clientWidth, $("moneyChart").clientHeight);
  if (charts.position) charts.position.resize($("positionChart").clientWidth, $("positionChart").clientHeight);
});
document.addEventListener("langchange", renderAll);

try {
  const saved = localStorage.getItem("fund.interval");
  if (INTERVALS.includes(saved)) state.interval = saved;
} catch { /* storage unavailable */ }
const fromUrl = new URLSearchParams(window.location.search).get("interval");
if (INTERVALS.includes(fromUrl)) state.interval = fromUrl;
load();
setInterval(load, 10 * 60 * 1000);
