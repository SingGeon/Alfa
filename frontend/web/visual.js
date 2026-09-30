// "Visual history" page: one image per logged forecast (newest first),
// drawn server-side like the dashboard chart - past candles, the AI
// forecast, and the real candles on top of it as they close. Images are
// redrawn as real candles arrive, so they're always fetched fresh.

const PAGE_SIZE = 24;
const visState = { page: 1, pages: 1, stamp: Date.now() };

const $ = (id) => document.getElementById(id);
const filtersForm = $("filters");
const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(iso) {
  return new Date(iso).toLocaleString(getLang() === "ro" ? "ro-RO" : "en-GB", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  return body;
}

function statusLine(r) {
  if (r.status === "completed") {
    const cls = r.direction_correct ? "eval-ok" : "eval-err";
    return `<span class="${cls}">${r.pct_error > 0 ? "+" : ""}${r.pct_error.toFixed(2)}% · ${r.direction_correct ? "✓" : "✗"}</span>`;
  }
  if (r.status === "expired") return `<span class="eval-status eval-status-expired">${escapeHtml(t("eval.status.expired"))}</span>`;
  return `<span class="eval-status eval-status-pending">${escapeHtml(tf("visual.progress", { n: r.actual_path.length, total: r.horizon_steps }))}</span>`;
}

function render(rows) {
  $("galleryEmpty").hidden = rows.length > 0;
  $("gallery").innerHTML = rows.map((r) => `
    <figure class="visual-card" data-src="${escapeHtml(r.snapshot_url)}">
      <img loading="lazy" src="${escapeHtml(r.snapshot_url)}?t=${visState.stamp}" alt="${escapeHtml(`${r.model_name} ${r.interval} ${r.created_at}`)}">
      <figcaption>
        <span class="visual-when">${fmtTime(r.created_at)}</span>
        <span class="visual-meta">${escapeHtml(r.model_name)} · ${escapeHtml(r.interval)} · ${escapeHtml(tf("visual.filter.steps", { n: r.horizon_steps }))}</span>
        ${statusLine(r)}
      </figcaption>
    </figure>`).join("");
}

async function load() {
  const params = new URLSearchParams();
  for (const [k, v] of new FormData(filtersForm)) if (v) params.set(k, v);
  params.set("sort", "created_at");
  params.set("order", "desc");
  params.set("page", visState.page);
  params.set("page_size", PAGE_SIZE);
  visState.stamp = Date.now();
  try {
    const data = await getJson(`/api/evaluation/predictions?${params}`);
    $("errorBanner").hidden = true;
    visState.pages = data.pages;
    render(data.rows);
    $("pageInfo").textContent = tf("eval.page", { page: data.page, pages: data.pages, total: data.total });
    $("prevPage").disabled = data.page <= 1;
    $("nextPage").disabled = data.page >= data.pages;
  } catch (exc) {
    $("errorBanner").textContent = tf("eval.error.load", { msg: exc.message });
    $("errorBanner").hidden = false;
  }
}

async function loadModels() {
  try {
    const { models } = await getJson("/api/evaluation/models");
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      $("modelFilter").appendChild(opt);
    }
  } catch { /* load() surfaces errors */ }
}

function renderStatic() {
  $("visualSubtitle").textContent = tf("visual.subtitle", { tz: userTz });
  $("horizonFilter").options[0].textContent = tf("visual.filter.steps", { n: 24 });
}

filtersForm.addEventListener("change", () => { visState.page = 1; load(); });
$("prevPage").addEventListener("click", () => { if (visState.page > 1) { visState.page--; load(); } });
$("nextPage").addEventListener("click", () => { if (visState.page < visState.pages) { visState.page++; load(); } });

$("gallery").addEventListener("click", (e) => {
  const card = e.target.closest(".visual-card");
  if (!card) return;
  $("chartImg").src = `${card.dataset.src}?t=${Date.now()}`;
  $("chartDialog").showModal();
});
$("chartDialog").addEventListener("click", (e) => { if (e.target === $("chartDialog")) $("chartDialog").close(); });

document.addEventListener("langchange", () => { renderStatic(); load(); });

renderStatic();
loadModels();
load();
// Real candles land every minute server-side; new forecasts every few.
setInterval(load, 60000);
