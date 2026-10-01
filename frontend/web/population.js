"use strict";

// Model population page: everything about the evolving "tuned" population
// (ml/evolution.py, served by /api/evolution/*).

const GENE_ORDER = ["momentum", "trend", "long", "volatility", "volume", "oscillators", "btc", "sentiment", "news", "defi", "futures"];
const TRAITS = [
  { key: "boldness", digits: 2, range: [0, 2] },
  { key: "temperament", digits: 2, range: [0, 0.6] },
  { key: "news_sensitivity", digits: 2, range: [-1, 1], mid: 0 },
  { key: "window", digits: 0, range: null },
  { key: "avg_generation", digits: 0, range: null },
];
const EMOJI = { fear: "😨", caution: "🤔", calm: "😐", confidence: "🙂", euphoria: "🤩" };
const PAGE_SIZE = 50;

const state = { interval: "1h", page: 1, report: null };
let charts = [];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtNum = (v, d = 0) => (v == null ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const fmtPct = (v, d = 2) => (v == null ? "—" : `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}%`);
const fmtDate = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "—");
const fmtDay = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "—");
const geneName = (g) => t(`dashboard.evo.input.${g}`);
const emotion = (e) => `${EMOJI[e] || ""} ${t(`dashboard.evo.emotion.${e}`)}`;
const causeChip = (c) => `<span class="pop-cause ${c}">${esc(t(`pop.cause.${c}`))}</span>`;

async function api(path, params = {}) {
  const url = new URL(path, window.location.origin);
  Object.entries({ interval: state.interval, ...params }).forEach(([k, v]) => {
    if (v !== "" && v != null) url.searchParams.set(k, v);
  });
  const resp = await fetch(url);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `${resp.status}`);
  return body;
}

function showError(msg) {
  $("errorBanner").hidden = !msg;
  $("errorBanner").textContent = msg || "";
}

// -- summary -------------------------------------------------------------

function renderSummary(r) {
  const s = r.summary;
  $("livedNote").textContent = tf("pop.summary.lived", { from: fmtDay(s.lived_from), to: fmtDate(s.lived_to), n: fmtNum(s.candles_lived) });
  const tr = Object.fromEntries((r.track_record || []).map((x) => [x.h, x]));
  const tile = (label, value) => `<div class="box-stat"><span>${esc(label)}</span><b>${value}</b></div>`;
  $("summaryTiles").innerHTML = [
    tile(t("dashboard.evo.alive"), fmtNum(s.alive)),
    tile(t("dashboard.evo.births"), fmtNum(s.births)),
    tile(t("dashboard.evo.deaths"), fmtNum(s.deaths)),
    tile(t("dashboard.evo.generation"), fmtNum(s.max_generation)),
    tile(t("dashboard.evo.lifespan"), s.avg_lifespan_of_dead == null ? "—" : tf("dashboard.box.candles", { n: fmtNum(s.avg_lifespan_of_dead) })),
    tile(t("pop.summary.oldest"), tf("dashboard.box.candles", { n: fmtNum(s.oldest_alive) })),
    tile(t("pop.summary.mood"), emotion(s.emotion)),
    tile(t("pop.summary.dir1"), tr[1] ? `${tr[1].direction_pct}%` : "—"),
    tile(t("pop.summary.dir24"), tr[24] ? `${tr[24].direction_pct}%` : "—"),
  ].join("");
}

// -- time line (three synced charts) ------------------------------------

function chartOptions(el) {
  const css = getComputedStyle(document.documentElement);
  return {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: { background: { color: "transparent" }, textColor: css.getPropertyValue("--text-muted").trim() || "#676f83", fontSize: 11 },
    grid: { vertLines: { visible: false }, horzLines: { color: "rgba(255,255,255,0.05)" } },
    rightPriceScale: { borderVisible: false, minimumWidth: 70 },
    timeScale: { borderVisible: false, timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
    handleScroll: true,
    handleScale: true,
  };
}

function renderTimeline(r) {
  charts.forEach((c) => c.remove());
  charts = [];
  const css = getComputedStyle(document.documentElement);
  const pred = css.getPropertyValue("--pred").trim() || "#6d7cf5";
  const muted = css.getPropertyValue("--text-muted").trim() || "#676f83";
  const pts = r.timeline.map((p) => ({ ...p, time: Math.floor(new Date(p.time).getTime() / 1000) }))
    .filter((p, i, a) => i === 0 || p.time > a[i - 1].time);

  const make = (id) => {
    const el = $(id);
    const c = LightweightCharts.createChart(el, chartOptions(el));
    charts.push(c);
    return c;
  };
  const price = make("priceChart").addLineSeries({ color: pred, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  price.setData(pts.filter((p) => p.price != null).map((p) => ({ time: p.time, value: p.price })));
  const deaths = make("deathsChart").addHistogramSeries({ color: pred, priceLineVisible: false, lastValueVisible: false, priceFormat: { type: "volume" } });
  deaths.setData(pts.map((p) => ({ time: p.time, value: p.deaths })));
  const mood = make("moodChart").addLineSeries({
    color: pred, lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
    autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }),
  });
  mood.setData(pts.map((p) => ({ time: p.time, value: p.mood })));
  mood.createPriceLine({ price: 0.5, color: muted, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, axisLabelVisible: false });

  const byTime = new Map(pts.map((p) => [p.time, p]));
  const readout = (p) => {
    $("priceReadout").textContent = p ? `$${fmtNum(p.price, 2)} · ${fmtDate(new Date(p.time * 1000).toISOString())}` : "";
    $("deathsReadout").textContent = p ? tf("pop.timeline.deathsValue", { n: fmtNum(p.deaths) }) : "";
    $("moodReadout").textContent = p ? `${emotion(moodName(p.mood))} (${Math.round(p.mood * 100)}%)` : "";
  };
  // Keep the three time scales and crosshair readouts in step.
  let syncing = false;
  charts.forEach((c) => {
    c.timeScale().subscribeVisibleLogicalRangeChange((range) => {
      if (syncing || !range) return;
      syncing = true;
      charts.forEach((o) => o !== c && o.timeScale().setVisibleLogicalRange(range));
      syncing = false;
    });
    c.subscribeCrosshairMove((param) => readout(param.time ? byTime.get(param.time) : pts[pts.length - 1]));
  });
  charts.forEach((c) => c.timeScale().fitContent());
  readout(pts[pts.length - 1]);
}

function moodName(m) {
  return m < 0.3 ? "fear" : m < 0.45 ? "caution" : m < 0.6 ? "calm" : m < 0.75 ? "confidence" : "euphoria";
}

// -- bars ------------------------------------------------------------------

function bars(el, rows) {
  const max = Math.max(...rows.map((r) => r.value), 1);
  el.innerHTML = rows.map((r) => `
    <div class="pop-bar-row">
      <span>${r.label}</span>
      <span class="pop-bar-track"><i style="width:${(r.value / max) * 100}%"></i></span>
      <b>${r.text}</b>
    </div>`).join("");
}

function renderCauses(r) {
  bars($("causes"), r.causes.map((c) => ({
    label: causeChip(c.cause),
    value: c.deaths,
    text: `${fmtNum(c.deaths)} <small>(${c.share_pct}%${c.avg_age != null ? ` · ${tf("pop.causes.avgAge", { n: fmtNum(c.avg_age) })}` : ""})</small>`,
  })));
}

function renderSurvival(r) {
  const rows = Object.entries(r.gene_survival || {}).sort((a, b) => b[1].avg_lifespan - a[1].avg_lifespan);
  bars($("survival"), rows.map(([g, v]) => ({
    label: esc(geneName(g)),
    value: v.avg_lifespan,
    text: `${fmtNum(v.avg_lifespan)} <small>(${tf("pop.survival.deaths", { n: fmtNum(v.deaths) })})</small>`,
  })));
}

// -- small multiples ---------------------------------------------------------

function sparkline(container, { title, values, times, digits, range, mid, format }) {
  const finite = values.filter((v) => v != null && Number.isFinite(v));
  if (!finite.length) return;
  const lo = range ? range[0] : Math.min(...finite);
  const hi = range ? range[1] : Math.max(...finite);
  const span = hi - lo || 1;
  const W = 300, H = 56;
  const x = (i) => (values.length > 1 ? (i / (values.length - 1)) * W : 0);
  const y = (v) => H - ((v - lo) / span) * (H - 4) - 2;
  const line = values.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean).join(" ");
  const area = `0,${H} ${line} ${W},${H}`;
  const last = finite[finite.length - 1];
  const fmt = format || ((v) => fmtNum(v, digits));
  const card = document.createElement("div");
  card.className = "pop-spark";
  card.innerHTML = `
    <div class="pop-spark-head"><span>${esc(title)}</span><b data-value>${fmt(last)}</b></div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${esc(title)}">
      ${range ? `<polygon class="spark-area" points="${area}" />` : ""}
      ${mid != null ? `<line class="spark-mid" x1="0" x2="${W}" y1="${y(mid)}" y2="${y(mid)}" />` : ""}
      <polyline class="spark-line" points="${line}" />
      <line class="spark-cursor" x1="0" x2="0" y1="0" y2="${H}" visibility="hidden" />
    </svg>
    <div class="pop-spark-foot"><span>${fmtDay(times[0])}</span><span data-when>${fmtDay(times[times.length - 1])}</span></div>`;
  const svg = card.querySelector("svg");
  const cursor = card.querySelector(".spark-cursor");
  const out = card.querySelector("[data-value]");
  const when = card.querySelector("[data-when]");
  svg.addEventListener("mousemove", (e) => {
    const rect = svg.getBoundingClientRect();
    const i = Math.round(((e.clientX - rect.left) / rect.width) * (values.length - 1));
    const k = Math.max(0, Math.min(values.length - 1, i));
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

function renderEvolution(r) {
  const times = r.timeline.map((p) => p.time);
  const shares = $("geneShares");
  shares.innerHTML = "";
  const genes = GENE_ORDER.filter((g) => r.genes.includes(g)).concat(r.genes.filter((g) => !GENE_ORDER.includes(g)));
  genes.forEach((g) => sparkline(shares, {
    title: geneName(g), times, range: [0, 1],
    values: r.timeline.map((p) => (p.shares && g in p.shares ? p.shares[g] : null)),
    format: (v) => (v == null ? "—" : `${Math.round(v * 100)}%`),
  }));
  const traits = $("traits");
  traits.innerHTML = "";
  TRAITS.forEach((tr) => sparkline(traits, {
    title: t(`pop.trait.${tr.key}`), times, digits: tr.digits, range: tr.range, mid: tr.mid,
    values: r.timeline.map((p) => p[tr.key]),
  }));
}

// -- generations ---------------------------------------------------------------

function renderGenerations(r) {
  $("generationRows").innerHTML = r.generations.map((g) => {
    const top = Object.entries(g.input_shares).slice(0, 3).map(([k, v]) => `${esc(geneName(k))} ${Math.round(v * 100)}%`).join(", ");
    const dead = Object.values(g.causes).reduce((a, b) => a + b, 0);
    const causes = dead ? Object.entries(g.causes).filter(([, n]) => n).map(([c, n]) => `${causeChip(c)} <span class="pop-small">${Math.round((n / dead) * 100)}%</span>`).join(" ") : "—";
    return `<tr class="eval-clickable" data-generation="${g.from}">
      <td>${g.from === g.to ? g.from : `${g.from}–${g.to}`}</td>
      <td class="num">${fmtNum(g.organisms)}</td>
      <td class="num">${fmtNum(g.alive)}</td>
      <td class="num">${g.avg_lifespan == null ? "—" : fmtNum(g.avg_lifespan)}</td>
      <td class="pop-wrap">${causes}</td>
      <td class="pop-wrap">${top}</td>
      <td class="num">${fmtNum(g.boldness, 2)}</td>
      <td class="num">${fmtNum(g.temperament, 2)}</td>
      <td class="num">${fmtNum(g.news_sensitivity, 2)}</td>
      <td class="num">${fmtNum(g.window)}</td>
    </tr>`;
  }).join("");
}

// -- graveyard -------------------------------------------------------------

function filters() {
  return Object.fromEntries(new FormData($("deathFilters")).entries());
}

function killerText(k) {
  if (!k) return "—";
  return tf("pop.killer", { h: k.h, pred: fmtPct(k.predicted_pct), actual: fmtPct(k.actual_pct) });
}

function marketText(d) {
  const vol = d.volatility_vs_month == null ? "" : d.volatility_vs_month > 1.5 ? t("pop.market.wild") : d.volatility_vs_month < 0.7 ? t("pop.market.quiet") : t("pop.market.normal");
  return `$${fmtNum(d.price, 2)} · <span class="${d.move_24_pct > 0 ? "pop-up" : d.move_24_pct < 0 ? "pop-down" : ""}">${fmtPct(d.move_24_pct)}</span>${vol ? ` · ${vol}` : ""}`;
}

async function loadDeaths() {
  try {
    const data = await api("/api/evolution/deaths", { ...filters(), page: state.page, page_size: PAGE_SIZE });
    state.page = data.page;
    $("deathsTotal").textContent = tf("pop.graveyard.total", { n: fmtNum(data.total) });
    $("deathRows").innerHTML = data.rows.map((d) => `
      <tr class="eval-clickable" data-id="${d.id}">
        <td>${d.id}</td>
        <td>${fmtDate(d.died)}</td>
        <td class="num">${fmtNum(d.age)}</td>
        <td class="num">${d.generation}</td>
        <td>${causeChip(d.cause)}</td>
        <td class="pop-wrap">${killerText(d.killing_prediction)}</td>
        <td class="pop-wrap">${marketText(d)}</td>
        <td>${emotion(d.emotion)}</td>
        <td class="num">${d.wins} / ${d.losses} / ${d.combos}</td>
        <td class="pop-wrap">${d.inputs.map((g) => esc(geneName(g))).join(", ")}</td>
      </tr>`).join("") || `<tr class="empty-row"><td colspan="10">${esc(t("pop.graveyard.none"))}</td></tr>`;
    $("pageInfo").textContent = tf("pop.page", { page: data.page, pages: data.pages });
    $("prevPage").disabled = data.page <= 1;
    $("nextPage").disabled = data.page >= data.pages;
  } catch (exc) {
    $("deathRows").innerHTML = `<tr class="empty-row"><td colspan="10">${esc(exc.message)}</td></tr>`;
  }
}

function renderAlive(r) {
  $("aliveRows").innerHTML = r.alive.map((o) => `
    <tr class="eval-clickable" data-id="${o.id}">
      <td>${o.id}</td>
      <td class="num">${fmtNum(o.energy, 1)}</td>
      <td>${emotion(o.emotion)}</td>
      <td class="num">${fmtNum(o.age)}</td>
      <td class="num">${o.generation}</td>
      <td class="num">${o.wins} / ${o.losses} / ${o.combos}</td>
      <td class="pop-wrap">${o.inputs.map((g) => esc(geneName(g))).join(", ")}</td>
      <td>${o.voting ? "✓" : ""}</td>
    </tr>`).join("");
}

// -- one organism's story ------------------------------------------------------

async function openOrganism(id) {
  try {
    const data = await api(`/api/evolution/organism/${id}`);
    const o = data.organism;
    const genes = [
      [t("pop.trait.boldness"), fmtNum(o.boldness, 2)],
      [t("pop.trait.temperament"), fmtNum(o.temperament, 2)],
      [t("pop.trait.news_sensitivity"), fmtNum(o.news_sensitivity, 2)],
      [t("pop.trait.window"), fmtNum(o.window)],
    ].map(([k, v]) => `<span class="pop-chip">${esc(k)}: <b>${v}</b></span>`).join(" ");
    const chip = (a) => `<button type="button" class="pop-chip pop-link ${a.dead ? "dead" : "alive"}" data-id="${a.id}">#${a.id}${a.generation != null ? ` · g${a.generation}` : ""}${a.dead ? " †" : ""}</button>`;
    const levels = {};
    data.ancestors.forEach((a) => (levels[a.level] = levels[a.level] || []).push(a));
    const tree = Object.entries(levels).map(([lvl, list]) => `
      <div class="pop-tree-level"><span>${esc(tf("pop.story.level", { n: lvl }))}</span>${list.map(chip).join("")}</div>`).join("");
    const story = o.dead
      ? tf("pop.story.dead", {
        id: o.id, born: fmtDate(o.born), died: fmtDate(o.died), age: fmtNum(o.age), gen: o.generation,
        cause: t(`pop.cause.${o.cause}`), why: t(`pop.causeHelp.${o.cause}`),
      })
      : tf("pop.story.alive", { id: o.id, born: fmtDate(o.born), age: fmtNum(o.age), gen: o.generation, energy: fmtNum(o.energy, 1) });
    $("organismBody").innerHTML = `
      <h2>#${o.id} ${o.dead ? "†" : "❤"}</h2>
      <p>${esc(story)}</p>
      ${o.dead ? `<p>${esc(tf("pop.story.market", { price: fmtNum(o.price, 2), move: fmtPct(o.move_24_pct), mood: t(`dashboard.evo.emotion.${o.emotion}`) }))}</p>
      ${o.killing_prediction ? `<p>${esc(killerText(o.killing_prediction))}</p>` : ""}
      <p>${esc(tf("pop.story.energy", { gained: fmtNum(o.lifetime_gained, 1), lost: fmtNum(o.lifetime_lost_to_misses, 1) }))}</p>` : ""}
      <h3>${esc(t("pop.story.genes"))}</h3>
      <p>${o.inputs.map((g) => `<span class="pop-chip">${esc(geneName(g))}</span>`).join(" ")}</p>
      <p>${genes}</p>
      <p>${esc(tf("pop.story.record", { wins: o.wins, losses: o.losses, combos: o.combos }))}</p>
      <h3>${esc(t("pop.story.family"))}</h3>
      ${tree || `<p>${esc(t("pop.story.founder"))}</p>`}
      ${data.children.length ? `<div class="pop-tree-level"><span>${esc(t("pop.story.children"))}</span>${data.children.map(chip).join("")}</div>` : ""}`;
    const dialog = $("organismDialog");
    if (!dialog.open) dialog.showModal();
  } catch (exc) {
    showError(exc.message);
  }
}

// -- wiring ------------------------------------------------------------------

async function load() {
  showError("");
  [...$("intervalTabs").children].forEach((b) => b.classList.toggle("active", b.dataset.interval === state.interval));
  try {
    const r = await api("/api/evolution/report");
    state.report = r;
    renderSummary(r);
    renderTimeline(r);
    renderCauses(r);
    renderSurvival(r);
    renderEvolution(r);
    renderGenerations(r);
    renderAlive(r);
  } catch (exc) {
    state.report = null;
    showError(exc.message);
    charts.forEach((c) => c.remove());
    charts = [];
    ["summaryTiles", "causes", "survival", "geneShares", "traits", "generationRows", "aliveRows", "deathRows"].forEach((id) => ($(id).innerHTML = ""));
    return;
  }
  state.page = 1;
  loadDeaths();
}

$("intervalTabs").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-interval]");
  if (!b || b.dataset.interval === state.interval) return;
  state.interval = b.dataset.interval;
  try { localStorage.setItem("pop.interval", state.interval); } catch { /* storage unavailable */ }
  load();
});
$("deathFilters").addEventListener("input", () => { state.page = 1; loadDeaths(); });
$("deathFilters").addEventListener("reset", () => setTimeout(() => { state.page = 1; loadDeaths(); }));
$("prevPage").addEventListener("click", () => { state.page -= 1; loadDeaths(); });
$("nextPage").addEventListener("click", () => { state.page += 1; loadDeaths(); });
document.addEventListener("click", (e) => {
  const link = e.target.closest("[data-id]");
  if (link && (link.closest("#deathRows, #aliveRows") || link.classList.contains("pop-link"))) openOrganism(link.dataset.id);
  const gen = e.target.closest("tr[data-generation]");
  if (gen) {
    $("deathFilters").generation.value = gen.dataset.generation;
    state.page = 1;
    loadDeaths();
    $("deathFilters").scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
window.addEventListener("resize", () => {
  ["priceChart", "deathsChart", "moodChart"].forEach((id, i) => charts[i] && charts[i].resize($(id).clientWidth, $(id).clientHeight));
});
document.addEventListener("langchange", () => {
  if (state.report) {
    renderSummary(state.report);
    renderCauses(state.report);
    renderSurvival(state.report);
    renderEvolution(state.report);
    renderGenerations(state.report);
    renderAlive(state.report);
    loadDeaths();
  }
});

try {
  const saved = localStorage.getItem("pop.interval");
  if (saved) state.interval = saved;
} catch { /* storage unavailable */ }
const fromUrl = new URLSearchParams(window.location.search).get("interval");
if (fromUrl) state.interval = fromUrl;
load();
