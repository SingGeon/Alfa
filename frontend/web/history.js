// "Day history" page: the end-of-day overview images built by
// evaluation/daily_report.py, one per UTC day, with a date selector.

const $ = (id) => document.getElementById(id);

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `HTTP ${resp.status}`);
  return body;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const fmtPct = (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(2)}%`);

function stat(label, value, cls = "") {
  return `<div><span>${escapeHtml(label)}</span><strong class="${cls}">${escapeHtml(value)}</strong></div>`;
}

async function showDay(day) {
  const base = `/api/evaluation/days/${day}/overview`;
  $("overviewImg").src = `${base}.png`;
  $("overviewImg").alt = `${t("history.title")} ${day}`;
  $("overviewLink").href = `${base}.png`;
  $("overviewLink").hidden = false;
  try {
    const d = await getJson(`/api/evaluation/days/${day}`);
    const links = [`<a class="nav-link" href="${base}.png" target="_blank" rel="noopener">${escapeHtml(t("history.png"))}</a>`];
    if (d.gif) links.push(`<a class="nav-link" href="${base}.gif" target="_blank" rel="noopener">${escapeHtml(t("history.gif"))}</a>`);
    if (d.pdf) links.push(`<a class="nav-link" href="${base}.pdf" target="_blank" rel="noopener">${escapeHtml(t("history.pdf"))}</a>`);
    $("dayLinks").innerHTML = links.join("");
    $("dayStats").innerHTML = `
      <div class="eval-summary-card"><div class="eval-summary-stats">
        ${stat(t("eval.summary.predictions"), String(d.predictions))}
        ${stat(t("eval.summary.meanError"), fmtPct(d.mean_abs_pct_error), "eval-err")}
        ${stat(t("eval.summary.direction"), d.direction_accuracy_pct === null ? "—" : `${d.direction_accuracy_pct.toFixed(1)}%`, "eval-ok")}
        ${d.best_model ? stat(t("history.best"), `${d.best_model.model_name} (${fmtPct(d.best_model.mean_abs_pct_error)})`, "eval-ok") : ""}
        ${d.worst_model && d.worst_model.model_name !== d.best_model.model_name ? stat(t("history.worst"), `${d.worst_model.model_name} (${fmtPct(d.worst_model.mean_abs_pct_error)})`, "eval-err") : ""}
      </div></div>`;
  } catch (exc) {
    $("errorBanner").textContent = tf("history.error.load", { msg: exc.message });
    $("errorBanner").hidden = false;
  }
}

async function init() {
  try {
    const { days } = await getJson("/api/evaluation/days");
    const select = $("daySelect");
    select.innerHTML = days.map((d) => `<option value="${d}">${d}</option>`).join("");
    if (!days.length) {
      $("historyEmpty").hidden = false;
      select.disabled = true;
      return;
    }
    const wanted = new URLSearchParams(location.search).get("date");
    select.value = days.includes(wanted) ? wanted : days[0];
    showDay(select.value);
    select.addEventListener("change", () => {
      history.replaceState(null, "", `?date=${select.value}`);
      showDay(select.value);
    });
  } catch (exc) {
    $("errorBanner").textContent = tf("history.error.load", { msg: exc.message });
    $("errorBanner").hidden = false;
  }
}

document.addEventListener("langchange", () => { if ($("daySelect").value) showDay($("daySelect").value); });
init();
