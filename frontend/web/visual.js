// "Visual history" page: one image per logged forecast, drawn server-side
// like the dashboard chart - past candles, the AI forecast, and the real
// candles on top of it as they close. Images are redrawn as real candles
// arrive, so they're always fetched fresh.
//
// Two ways in, mirroring the folders on disk (evaluation/charts.py):
//   - by model -> interval, gallery grouped by day;
//   - by day, gallery grouped by model + interval.
// Each selection shows its own stats (GET /api/evaluation/stats with the
// same filters as the gallery), plus the general stats on top. The
// selection lives in the URL hash so any view can be linked to.

const PAGE_SIZE = 24;
const DAY_PAGE_SIZE = 500;
const INTERVALS = ["15m", "1h", "4h", "1d", "1w"];
const state = { view: "model", model: "", interval: "", day: "", horizon: "24", page: 1, pages: 1, stamp: Date.now() };
let models = [];
let days = [];

const $ = (id) => document.getElementById(id);
const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;
const locale = () => (getLang() === "ro" ? "ro-RO" : "en-GB");

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(iso) {
  return new Date(iso).toLocaleString(locale(), { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function fmtDay(day) {  // "YYYY-MM-DD" (UTC day) -> localized date, no tz shift
  const [y, m, d] = day.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString(locale(), { timeZone: "UTC", weekday: "short", year: "numeric", month: "2-digit", day: "2-digit" });
}

function localDayKey(iso) {
  const d = new Date(iso);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// The legacy model is the unchanged reference the tuned one is measured against.
const modelLabel = (m) => (m.endsWith("-legacy") ? `${m} (${t("visual.control")})` : m);
const fmtSkill = (v) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`);
const skillCls = (v) => (v == null ? "" : v > 0 ? "eval-ok" : "eval-err");

const fmtPct = (v, signed = false) => (v == null ? "—" : `${signed && v > 0 ? "+" : ""}${v.toFixed(2)}%`);

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  return body;
}

function query(extra = {}) {
  const params = new URLSearchParams();
  if (state.horizon) params.set("horizon_steps", state.horizon);
  for (const [k, v] of Object.entries(extra)) if (v !== "" && v != null) params.set(k, v);
  return params;
}

// ---- URL hash <-> state ------------------------------------------------------

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("view") === "day" || h.get("view") === "model") state.view = h.get("view");
  if (h.has("model")) state.model = h.get("model");
  if (h.has("interval")) state.interval = INTERVALS.includes(h.get("interval")) ? h.get("interval") : "";
  if (h.has("day")) state.day = h.get("day");
  if (h.has("h")) state.horizon = h.get("h") === "all" ? "" : h.get("h");
}

function writeHash() {
  const h = new URLSearchParams({ view: state.view, h: state.horizon || "all" });
  if (state.view === "model") {
    h.set("model", state.model);
    if (state.interval) h.set("interval", state.interval);
  } else if (state.day) {
    h.set("day", state.day);
  }
  history.replaceState(null, "", `#${h}`);
}

// ---- stats rendering ----------------------------------------------------------

function statTiles(s) {
  if (!s) return `<p class="eval-hint">${escapeHtml(t("eval.summary.empty"))}</p>`;
  const done = s.completed > 0;
  const dirCls = s.direction_accuracy_pct == null ? "" : s.direction_accuracy_pct >= 50 ? "eval-ok" : "eval-err";
  return `
    <div><span>${escapeHtml(t("eval.summary.predictions"))}</span><strong>${s.predictions}</strong>
      <small>${s.completed} ${escapeHtml(t("eval.summary.completed"))} · ${s.pending} ${escapeHtml(t("eval.summary.pending"))}${s.expired ? ` · ${s.expired} ${escapeHtml(t("eval.status.expired"))}` : ""}</small></div>
    <div><span>${escapeHtml(t("eval.summary.meanError"))}</span><strong>${fmtPct(s.mean_abs_pct_error)}</strong>${done ? "" : `<small>${escapeHtml(t("visual.stats.noneYet"))}</small>`}</div>
    <div><span>${escapeHtml(t("visual.stats.usd"))}</span><strong>${s.mean_abs_error_usd == null ? "—" : `$${s.mean_abs_error_usd.toFixed(2)}`}</strong></div>
    <div><span>${escapeHtml(t("eval.summary.direction"))}</span><strong class="${dirCls}">${s.direction_accuracy_pct == null ? "—" : `${s.direction_accuracy_pct.toFixed(1)}%`}</strong></div>
    <div><span>${escapeHtml(t("visual.stats.bias"))}</span><strong>${fmtPct(s.mean_pct_error, true)}</strong><small>${escapeHtml(t("visual.stats.biasHint"))}</small></div>
    <div><span>${escapeHtml(t("visual.stats.skill"))}</span><strong class="${skillCls(s.skill_score)}">${fmtSkill(s.skill_score)}</strong><small>${escapeHtml(t("visual.stats.skillHint"))}</small></div>
    <div><span>${escapeHtml(t("visual.stats.coverage"))}</span><strong>${s.band_coverage_pct == null ? "—" : `${s.band_coverage_pct.toFixed(0)}%`}</strong><small>${escapeHtml(t("visual.stats.coverageHint"))}</small></div>`;
}

function breakdownTable(firstCol, rows) {
  if (!rows.length) return "";
  const head = [firstCol, t("visual.col.predictions"), t("visual.col.completed"), t("visual.col.error"), t("visual.col.direction"), t("visual.col.skill"), t("visual.col.coverage")];
  return `
    <table>
      <thead><tr>${head.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((r) => `
        <tr data-jump="${escapeHtml(JSON.stringify(r.jump))}">
          <td>${escapeHtml(r.label)}</td><td>${r.s.predictions}</td><td>${r.s.completed}</td>
          <td>${fmtPct(r.s.mean_abs_pct_error)}</td>
          <td>${r.s.direction_accuracy_pct == null ? "—" : `${r.s.direction_accuracy_pct.toFixed(1)}%`}</td>
          <td class="${skillCls(r.s.skill_score)}">${fmtSkill(r.s.skill_score)}</td>
          <td>${r.s.band_coverage_pct == null ? "—" : `${r.s.band_coverage_pct.toFixed(0)}%`}</td>
        </tr>`).join("")}</tbody>
    </table>
    <p class="eval-hint">${escapeHtml(t("visual.jumpHint"))}</p>`;
}

// ---- navigation tabs ------------------------------------------------------------

function tabs(el, items, active, onPick) {
  el.innerHTML = items.map((it) => `
    <button type="button" data-value="${escapeHtml(it.value)}" class="${it.value === active ? "active" : ""}">
      ${escapeHtml(it.label)}${it.count != null ? `<small>${it.count}</small>` : ""}
    </button>`).join("");
  el.onclick = (e) => {
    const b = e.target.closest("button");
    if (b) onPick(b.dataset.value);
  };
}

function renderViewTabs() {
  for (const b of $("viewTabs").querySelectorAll("button")) b.classList.toggle("active", b.dataset.view === state.view);
  $("modelNav").hidden = state.view !== "model";
  $("dayNav").hidden = state.view !== "day";
}

async function renderNav() {
  const general = await getJson(`/api/evaluation/stats?${query()}`);
  $("overallStats").innerHTML = statTiles(general.overall);

  if (state.view === "model") {
    const count = Object.fromEntries(general.by_model.map((r) => [r.model, r.predictions]));
    tabs($("modelTabs"), models.map((m) => ({ value: m, label: modelLabel(m), count: count[m] || 0 })), state.model, (v) => go({ model: v, interval: "" }));
    const perInterval = Object.fromEntries(general.by_model_interval.filter((r) => r.model === state.model).map((r) => [r.interval, r.predictions]));
    const total = Object.values(perInterval).reduce((a, b) => a + b, 0);
    tabs(
      $("intervalTabs"),
      [{ value: "", label: t("visual.allIntervals"), count: total }, ...INTERVALS.map((i) => ({ value: i, label: i, count: perInterval[i] || 0 }))],
      state.interval,
      (v) => go({ interval: v }),
    );
  } else {
    tabs($("dayTabs"), days.map((d) => ({ value: d.day, label: fmtDay(d.day), count: d.predictions })), state.day, (v) => go({ day: v }));
  }
}

// ---- selection stats + gallery --------------------------------------------------

function card(r) {
  let status;
  if (r.status === "completed") {
    const cls = r.direction_correct ? "eval-ok" : "eval-err";
    status = `<span class="${cls}">${fmtPct(r.pct_error, true)} · ${r.direction_correct ? "✓" : "✗"}</span>`;
  } else if (r.status === "expired") {
    status = `<span class="eval-status eval-status-expired">${escapeHtml(t("eval.status.expired"))}</span>`;
  } else {
    status = `<span class="eval-status eval-status-pending">${escapeHtml(tf("visual.progress", { n: r.actual_path.length, total: r.horizon_steps }))}</span>`;
  }
  return `
    <figure class="visual-card" data-src="${escapeHtml(r.snapshot_url)}">
      <img loading="lazy" src="${escapeHtml(r.snapshot_url)}?t=${state.stamp}" alt="${escapeHtml(`${r.model_name} ${r.interval} ${r.created_at}`)}">
      <figcaption>
        <span class="visual-when">${fmtTime(r.created_at)}</span>
        <span class="visual-meta">${escapeHtml(modelLabel(r.model_name))} · ${escapeHtml(r.interval)} · ${escapeHtml(tf("visual.filter.steps", { n: r.horizon_steps }))}</span>
        ${status}
      </figcaption>
    </figure>`;
}

function renderGroups(groups) {
  $("galleryEmpty").hidden = groups.length > 0;
  $("gallery").innerHTML = groups.map((g) => `
    <section class="vis-group">
      <h3 class="vis-group-head">${escapeHtml(g.title)}<small>${g.rows.length}</small></h3>
      <div class="visual-gallery">${g.rows.map(card).join("")}</div>
    </section>`).join("");
}

function groupBy(rows, keyFn, titleFn) {
  const groups = [];
  for (const r of rows) {
    const key = keyFn(r);
    let g = groups[groups.length - 1];
    if (!g || g.key !== key) groups.push((g = { key, title: titleFn(r, key), rows: [] }));
    g.rows.push(r);
  }
  return groups;
}

async function loadModelView() {
  const filters = { model: state.model, interval: state.interval };
  const sel = await getJson(`/api/evaluation/stats?${query(filters)}`);
  $("selectionTitle").textContent = `${modelLabel(state.model)} · ${state.interval || t("visual.allIntervals")}`;
  $("selectionStats").innerHTML = statTiles(sel.overall);
  $("breakdown").innerHTML = state.interval
    ? breakdownTable(t("visual.col.day"), [...sel.by_created_day].reverse().map((r) => ({ label: fmtDay(r.created_day), s: r, jump: { view: "day", day: r.created_day } })))
    : breakdownTable(t("visual.col.interval"), INTERVALS.flatMap((i) => {
      const r = sel.by_interval.find((x) => x.interval === i);
      return r ? [{ label: i, s: r, jump: { interval: i } }] : [];
    }));

  await renderStepStats(filters);

  const data = await getJson(`/api/evaluation/predictions?${query({ ...filters, sort: "created_at", order: "desc", page: state.page, page_size: PAGE_SIZE })}`);
  state.pages = data.pages;
  renderGroups(groupBy(data.rows, (r) => localDayKey(r.created_at), (r) => new Date(r.created_at).toLocaleDateString(locale(), { weekday: "long", year: "numeric", month: "2-digit", day: "2-digit" })));
  $("pager").hidden = false;
  $("pageInfo").textContent = tf("eval.page", { page: data.page, pages: data.pages, total: data.total });
  $("prevPage").disabled = data.page <= 1;
  $("nextPage").disabled = data.page >= data.pages;
}

async function renderStepStats(filters) {
  // Per step only makes sense for one model + interval at one horizon.
  if (!filters.interval || !state.horizon) {
    $("stepStats").innerHTML = "";
    return;
  }
  const { steps } = await getJson(`/api/evaluation/stats/steps?${query(filters)}`);
  if (!steps.length) {
    $("stepStats").innerHTML = "";
    return;
  }
  const head = [t("visual.col.step"), t("visual.col.predictions"), t("visual.col.error"), t("visual.col.baseline"), t("visual.col.skill"), t("visual.col.coverage")];
  $("stepStats").innerHTML = `
    <div class="vis-selection-title" style="margin-top:14px">${escapeHtml(t("visual.steps.title"))}</div>
    <table>
      <thead><tr>${head.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${steps.map((s) => `
        <tr><td>${s.step}</td><td>${s.scored}</td><td>${fmtPct(s.mae_model_pct)}</td><td>${fmtPct(s.mae_baseline_pct)}</td>
          <td class="${skillCls(s.skill_score)}">${fmtSkill(s.skill_score)}</td>
          <td>${s.band_coverage_pct == null ? "—" : `${s.band_coverage_pct.toFixed(0)}%`}</td></tr>`).join("")}</tbody>
    </table>
    <p class="eval-hint">${escapeHtml(t("visual.steps.hint"))}</p>`;
}

async function loadDayView() {
  $("pager").hidden = true;
  $("stepStats").innerHTML = "";
  if (!state.day) {
    $("selectionTitle").textContent = "";
    $("selectionStats").innerHTML = statTiles(null);
    $("breakdown").innerHTML = "";
    renderGroups([]);
    return;
  }
  const filters = { date_from: state.day, date_to: state.day };
  const sel = await getJson(`/api/evaluation/stats?${query(filters)}`);
  $("selectionTitle").textContent = fmtDay(state.day);
  $("selectionStats").innerHTML = statTiles(sel.overall);
  const cells = [...sel.by_model_interval].sort((a, b) =>
    a.model.localeCompare(b.model) || INTERVALS.indexOf(a.interval) - INTERVALS.indexOf(b.interval));
  $("breakdown").innerHTML = breakdownTable(t("visual.col.modelInterval"), cells.map((r) => ({
    label: `${modelLabel(r.model)} · ${r.interval}`, s: r, jump: { view: "model", model: r.model, interval: r.interval },
  })));

  const data = await getJson(`/api/evaluation/predictions?${query({ ...filters, sort: "created_at", order: "desc", page: 1, page_size: DAY_PAGE_SIZE })}`);
  const rows = [...data.rows].sort((a, b) =>
    a.model_name.localeCompare(b.model_name) || INTERVALS.indexOf(a.interval) - INTERVALS.indexOf(b.interval) || b.created_at.localeCompare(a.created_at));
  renderGroups(groupBy(rows, (r) => `${r.model_name}|${r.interval}`, (r) => `${modelLabel(r.model_name)} · ${r.interval}`));
}

async function load() {
  state.stamp = Date.now();
  try {
    const [m, d] = await Promise.all([
      getJson("/api/evaluation/models"),
      getJson(`/api/evaluation/prediction-days?${query()}`),
    ]);
    models = m.models || [];
    days = d.days || [];
    if (!models.includes(state.model)) state.model = models.includes("sklearn-tuned") ? "sklearn-tuned" : models[0] || "";
    if (!days.some((x) => x.day === state.day)) state.day = days[0] ? days[0].day : "";
    writeHash();
    renderViewTabs();
    await renderNav();
    await (state.view === "model" ? loadModelView() : loadDayView());
    $("errorBanner").hidden = true;
  } catch (exc) {
    $("errorBanner").textContent = tf("eval.error.load", { msg: exc.message });
    $("errorBanner").hidden = false;
  }
}

function go(patch) {
  Object.assign(state, patch, { page: 1 });
  load();
  window.scrollTo({ top: $("viewTabs").getBoundingClientRect().top + window.scrollY - 90, behavior: "smooth" });
}

function renderStatic() {
  $("visualSubtitle").textContent = tf("visual.subtitle", { tz: userTz });
  $("horizonFilter").options[0].textContent = tf("visual.filter.steps", { n: 24 });
  $("horizonFilter").value = state.horizon;
}

$("viewTabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b && b.dataset.view !== state.view) go({ view: b.dataset.view });
});
$("horizonFilter").addEventListener("change", (e) => go({ horizon: e.target.value }));
$("breakdown").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-jump]");
  if (row) go(JSON.parse(row.dataset.jump));
});
$("prevPage").addEventListener("click", () => { if (state.page > 1) { state.page--; load(); } });
$("nextPage").addEventListener("click", () => { if (state.page < state.pages) { state.page++; load(); } });

$("gallery").addEventListener("click", (e) => {
  const c = e.target.closest(".visual-card");
  if (!c) return;
  $("chartImg").src = `${c.dataset.src}?t=${Date.now()}`;
  $("chartDialog").showModal();
});
$("chartDialog").addEventListener("click", (e) => { if (e.target === $("chartDialog")) $("chartDialog").close(); });

document.addEventListener("langchange", () => { renderStatic(); load(); });

readHash();
renderStatic();
load();
// Real candles land every minute server-side; new forecasts every few.
setInterval(load, 60000);
