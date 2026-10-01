// "Prediction evaluation" page: per-model summary + the full, filterable,
// sortable, paginated table of logged predictions (/api/evaluation/...).
// The API speaks UTC only; every timestamp is converted to the viewer's
// own timezone here, at display time.

const PAGE_SIZE = 50;

const evalState = { sort: "created_at", order: "desc", page: 1, pages: 1 };

const $ = (id) => document.getElementById(id);
const filtersForm = $("filters");

const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

function fmtTime(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(getLang() === "ro" ? "ro-RO" : "en-GB", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function fmtMoney(v) {
  return v === null || v === undefined ? "—" : `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function fmtPct(v, signed = true) {
  if (v === null || v === undefined) return "—";
  return `${signed && v > 0 ? "+" : ""}${Number(v).toFixed(2)}%`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function showError(msg) {
  $("errorBanner").textContent = msg;
  $("errorBanner").hidden = false;
}

function filterParams() {
  const params = new URLSearchParams();
  for (const [k, v] of new FormData(filtersForm)) if (v) params.set(k, v);
  params.set("sort", evalState.sort);
  params.set("order", evalState.order);
  return params;
}

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  return body;
}

function renderSummary(summary) {
  const grid = $("summaryGrid");
  if (!summary.length) {
    grid.innerHTML = `<p class="eval-hint">${escapeHtml(t("eval.summary.empty"))}</p>`;
    return;
  }
  grid.innerHTML = summary.map((m) => `
    <div class="eval-summary-card">
      <div class="eval-summary-model">${escapeHtml(m.model_name.endsWith("-legacy") ? `${m.model_name} (control)` : m.model_name)}</div>
      <div class="eval-summary-stats">
        <div><span>${escapeHtml(t("eval.summary.predictions"))}</span><strong>${m.predictions}</strong>
          <small>${m.completed} ${escapeHtml(t("eval.summary.completed"))} · ${m.pending} ${escapeHtml(t("eval.summary.pending"))}</small></div>
        <div><span>${escapeHtml(t("eval.summary.meanError"))}</span><strong class="eval-err">${fmtPct(m.mean_abs_pct_error, false)}</strong></div>
        <div><span>${escapeHtml(t("eval.summary.direction"))}</span><strong class="eval-ok">${m.direction_accuracy_pct === null ? "—" : `${m.direction_accuracy_pct.toFixed(1)}%`}</strong></div>
      </div>
    </div>`).join("");
}

function renderRows(rows) {
  const tbody = $("rows");
  if (!rows.length) {
    tbody.innerHTML = `<tr class="empty-row"><td colspan="9">${escapeHtml(t("eval.empty"))}</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r) => {
    const dir = r.direction_correct === null || r.direction_correct === undefined
      ? "—"
      : `<span class="${r.direction_correct ? "eval-ok" : "eval-err"}">${r.direction_correct ? "✓" : "✗"} ${escapeHtml(t(r.direction_correct ? "eval.yes" : "eval.no"))}</span>`;
    const errCls = r.pct_error === null ? "" : Math.abs(r.pct_error) < 1 ? "eval-ok" : "eval-err";
    return `
      <tr class="eval-clickable" data-snapshot="${escapeHtml(r.snapshot_url)}">
        <td>${fmtTime(r.created_at)}</td>
        <td>${escapeHtml(r.interval)}</td>
        <td>${escapeHtml(r.model_name.endsWith("-legacy") ? `${r.model_name} (control)` : r.model_name)}</td>
        <td class="num">${fmtMoney(r.price_at_prediction)}</td>
        <td class="num price">${fmtMoney(r.predicted_final_price)}</td>
        <td class="num">${fmtMoney(r.actual_final_price)}</td>
        <td class="num ${errCls}">${fmtPct(r.pct_error)}</td>
        <td>${dir}</td>
        <td><span class="eval-status eval-status-${r.status}">${escapeHtml(t(`eval.status.${r.status}`))}</span></td>
      </tr>`;
  }).join("");
}

function renderSortIndicators() {
  document.querySelectorAll(".eval-table th[data-sort]").forEach((th) => {
    th.classList.toggle("sorted", th.dataset.sort === evalState.sort);
    th.dataset.order = th.dataset.sort === evalState.sort ? evalState.order : "";
  });
}

async function load() {
  const params = filterParams();
  $("csvLink").href = `/api/evaluation/predictions.csv?${params}`;
  params.set("page", evalState.page);
  params.set("page_size", PAGE_SIZE);
  try {
    const data = await getJson(`/api/evaluation/predictions?${params}`);
    $("errorBanner").hidden = true;
    evalState.pages = data.pages;
    renderSummary(data.summary);
    renderRows(data.rows);
    $("pageInfo").textContent = tf("eval.page", { page: data.page, pages: data.pages, total: data.total });
    $("prevPage").disabled = data.page <= 1;
    $("nextPage").disabled = data.page >= data.pages;
  } catch (exc) {
    showError(tf("eval.error.load", { msg: exc.message }));
  }
  renderSortIndicators();
}

async function loadModels() {
  try {
    const { models } = await getJson("/api/evaluation/models");
    const select = $("modelFilter");
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    }
  } catch { /* the table itself will surface the error */ }
}

function renderSubtitle() {
  $("evalSubtitle").textContent = tf("eval.subtitle", { tz: userTz });
}

document.querySelector(".eval-table thead").addEventListener("click", (e) => {
  const th = e.target.closest("th[data-sort]");
  if (!th) return;
  if (evalState.sort === th.dataset.sort) {
    evalState.order = evalState.order === "asc" ? "desc" : "asc";
  } else {
    evalState.sort = th.dataset.sort;
    evalState.order = th.dataset.sort === "created_at" ? "desc" : "asc";
  }
  evalState.page = 1;
  load();
});

filtersForm.addEventListener("change", () => { evalState.page = 1; load(); });
filtersForm.addEventListener("reset", () => setTimeout(() => { evalState.page = 1; load(); }, 0));
$("prevPage").addEventListener("click", () => { if (evalState.page > 1) { evalState.page--; load(); } });
$("nextPage").addEventListener("click", () => { if (evalState.page < evalState.pages) { evalState.page++; load(); } });

$("rows").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-snapshot]");
  if (!tr) return;
  $("chartImg").src = `${tr.dataset.snapshot}?t=${Date.now()}`;
  $("chartDialog").showModal();
});
$("chartDialog").addEventListener("click", (e) => { if (e.target === $("chartDialog")) $("chartDialog").close(); });

document.addEventListener("langchange", () => { renderSubtitle(); load(); });

renderSubtitle();
loadModels();
load();
// New predictions and completions arrive every minute server-side.
setInterval(load, 60000);
