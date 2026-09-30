"use strict";

const params = new URLSearchParams(window.location.search);
const assetType = params.get("type");
const assetId = params.get("id");

const els = {
  assetBadge: document.getElementById("assetBadge"),
  assetName: document.getElementById("assetName"),
  assetSymbol: document.getElementById("assetSymbol"),
  price: document.getElementById("price"),
  change: document.getElementById("change"),
  errorBanner: document.getElementById("errorBanner"),
  chart: document.getElementById("chart"),
  chartWrap: document.querySelector(".chart-wrap"),
  chartEmpty: document.getElementById("chartEmpty"),
  chartTooltip: document.getElementById("chartTooltip"),
  confidence: document.getElementById("confidence"),
  sentimentAvg: document.getElementById("sentimentAvg"),
  predictionRows: document.getElementById("predictionRows"),
  narrative: document.getElementById("narrative"),
  newsList: document.getElementById("newsList"),
  pinToggle: document.getElementById("pinToggle"),
};

const COLORS = { up: "#2fbf76", down: "#f0546b", pred: "#6d7cf5", grid: "rgba(255, 255, 255, 0.06)", text: "#676f83", surface: "#12151c" };

function fmtMoney(v) {
  if (v === null || v === undefined) return "—";
  const abs = Math.abs(v);
  let decimals = 2;
  if (abs > 0 && abs < 0.01) decimals = 6;
  else if (abs < 1) decimals = 4;
  return `$${Number(v).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}
function directionClass(v) {
  return v >= 0 ? "up" : "down";
}
function arrow(v) {
  return v >= 0 ? "▲" : "▼";
}
function qualityClass(confidence) {
  if (confidence < 40) return "quality-poor";
  if (confidence < 70) return "quality-decent";
  return "quality-good";
}
function fmtDay(iso) {
  return new Date(iso).toLocaleDateString("en-US", { day: "2-digit", month: "short" });
}
function toUnixSeconds(iso) {
  return Math.floor(new Date(iso).getTime() / 1000);
}

function showError(message) {
  els.errorBanner.hidden = false;
  els.errorBanner.textContent = message;
}

async function loadDetail() {
  if (!assetType || !assetId) {
    showError(t("detail.error.missingParams"));
    return;
  }

  try {
    const url = new URL("/api/scout/detail", window.location.origin);
    url.searchParams.set("asset_type", assetType);
    url.searchParams.set("id", assetId);
    url.searchParams.set("lang", getLang());
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `${resp.status}`);
    render(data);
  } catch (exc) {
    showError(tf("detail.error.analysisFailed", { msg: exc.message }));
    els.chartEmpty.hidden = false;
    els.chartEmpty.textContent = t("detail.error.analysisFailedShort");
  }
}

function render(data) {
  els.assetBadge.textContent = data.asset_type === "crypto" ? "🪙" : "🏢";
  els.assetName.textContent = data.name;
  els.assetSymbol.textContent = `${data.symbol} · ${t(data.asset_type === "crypto" ? "common.assetType.crypto" : "common.assetType.stock")}`;
  els.price.textContent = fmtMoney(data.current_price);
  els.change.textContent = "";
  document.title = `${data.symbol} — Scout AI`;

  els.confidence.textContent = `${Math.round(data.confidence)}/100`;
  els.confidence.className = `stat-value ${qualityClass(data.confidence)}`;
  els.sentimentAvg.textContent = data.sentiment_avg.toFixed(2);

  els.narrative.textContent = data.narrative;

  els.pinToggle.dataset.symbol = data.symbol;
  els.pinToggle.dataset.name = data.name;
  refreshPinState();

  renderChart(data);
  renderPredictionTable(data);
  renderNews(data.news);
}

let isPinned = false;
function setPinButton(pinned) {
  isPinned = pinned;
  els.pinToggle.classList.toggle("pinned", pinned);
  els.pinToggle.textContent = t(pinned ? "detail.pin.pinned" : "detail.pin.pin");
}
async function refreshPinState() {
  try {
    const url = new URL("/api/scout/pins", window.location.origin);
    url.searchParams.set("asset_type", assetType);
    const resp = await fetch(url);
    if (!resp.ok) return;
    const data = await resp.json();
    setPinButton((data.pins || []).some((p) => p.id === assetId));
  } catch {
    // best-effort - button just stays unpinned-looking if this fails
  }
}
async function togglePin() {
  try {
    if (isPinned) {
      await fetch(`/api/scout/pins?asset_type=${encodeURIComponent(assetType)}&id=${encodeURIComponent(assetId)}`, { method: "DELETE" });
      setPinButton(false);
    } else {
      await fetch("/api/scout/pins", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_type: assetType, id: assetId,
          symbol: els.pinToggle.dataset.symbol, name: els.pinToggle.dataset.name,
        }),
      });
      setPinButton(true);
    }
  } catch {
    // leave button state as-is; refreshPinState() will resync on next visit
  }
}
els.pinToggle.addEventListener("click", togglePin);

// installLeftEdgeClamp (shared with app.js) lives in chart-utils.js,
// loaded before this script.

function renderChart(data) {
  if (!window.LightweightCharts) {
    els.chartEmpty.hidden = false;
    els.chartEmpty.textContent = t("detail.error.libError");
    return;
  }
  els.chartEmpty.hidden = true;

  const chart = LightweightCharts.createChart(els.chart, {
    layout: { background: { color: "transparent" }, textColor: COLORS.text, fontFamily: "Sora, system-ui, -apple-system, 'Segoe UI', sans-serif" },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    localization: {
      timeFormatter: (time) => new Date(time * 1000).toLocaleString("en-US", { day: "2-digit", month: "short", year: "numeric" }),
    },
    rightPriceScale: { borderColor: COLORS.grid },
    timeScale: {
      borderColor: COLORS.grid, timeVisible: false, rightOffset: 8,
      tickMarkFormatter: (time) => new Date(time * 1000).toLocaleDateString("en-US", { day: "2-digit", month: "short" }),
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    ...CHART_PAN_ZOOM_OPTIONS,
  });

  // createChart() sizes to the container at that exact instant - on this
  // page the chart is built as soon as data arrives, which can be before
  // the surrounding layout has settled to its final width. Without this,
  // the chart starts too narrow, the zoom set below gets computed against
  // that wrong width, and only self-corrects once the ResizeObserver's
  // first callback fires - visible as a broken-looking flash of the wrong
  // range (e.g. isolated bars stranded in one corner) on first paint.
  chart.applyOptions({ width: els.chartWrap.clientWidth, height: els.chartWrap.clientHeight });

  const candleSeries = chart.addCandlestickSeries({
    upColor: COLORS.up, downColor: COLORS.down, borderUpColor: COLORS.up, borderDownColor: COLORS.down,
    wickUpColor: COLORS.up, wickDownColor: COLORS.down,
  });
  const volumeSeries = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "volume" });
  chart.priceScale("volume").applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
  const predSeries = chart.addLineSeries({ color: COLORS.pred, lineWidth: 2 });
  // lastValueVisible/priceLineVisible off: with candle + pred + upper +
  // lower all tagging the axis by default, the labels pile up and overlap
  // (worst on low-priced assets where they all round to similar values) -
  // the band's shape is already visible from the shaded area itself.
  //
  // Shaded confidence band (same technique as the ETH dashboard's chart,
  // app.js): an Area series fading from the upper bound to *fully
  // transparent* rather than a flat fill (or the classic "erase with an
  // opaque background-colored area on top" trick) - either of those
  // extends, opaque, all the way to the bottom of the price scale, which
  // blanked out the grid for the entire future/prediction span (a stark
  // black rectangle, seen live). A gradient ending in alpha 0 can't paint
  // over anything no matter how far down it notionally extends. The lower
  // bound is a plain dashed line - marks the boundary, fills nothing.
  const upperBandSeries = chart.addAreaSeries({
    lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, lineColor: COLORS.pred,
    topColor: "rgba(57,135,229,0.28)", bottomColor: "rgba(57,135,229,0)",
    crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });
  const lowerBandSeries = chart.addLineSeries({
    color: COLORS.pred, lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed,
    crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false,
  });

  const sorted = [...data.candles].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  candleSeries.setData(sorted.map((c) => ({ time: toUnixSeconds(c.timestamp), open: c.open, high: c.high, low: c.low, close: c.close })));
  volumeSeries.setData(
    sorted.map((c) => ({
      time: toUnixSeconds(c.timestamp),
      value: c.volume || 0,
      color: c.close >= c.open ? "rgba(25,158,112,0.5)" : "rgba(230,103,103,0.5)",
    }))
  );

  if (data.predictions.length) {
    const last = sorted[sorted.length - 1];
    const anchor = { time: toUnixSeconds(last.timestamp), value: last.close };
    predSeries.setData([anchor, ...data.predictions.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.predicted_price }))]);
    upperBandSeries.setData([anchor, ...data.predictions.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.upper }))]);
    lowerBandSeries.setData([anchor, ...data.predictions.map((p) => ({ time: toUnixSeconds(p.timestamp), value: p.lower }))]);
  }

  // Zoom in on the recent history + the forecast, not the whole ~200-day
  // history - otherwise the 7-day prediction shrinks to an unreadable
  // sliver at the right edge.
  const totalBars = sorted.length + data.predictions.length;
  const defaultWindow = Math.min(45, sorted.length);
  chart.timeScale().setVisibleLogicalRange({ from: sorted.length - defaultWindow, to: totalBars + 2 });

  new ResizeObserver(() => chart.applyOptions({ width: els.chartWrap.clientWidth, height: els.chartWrap.clientHeight })).observe(els.chartWrap);

  installLeftEdgeClamp(chart);

  // Floating price readout that tracks the mouse, same fix as app.js -
  // without it the only price readout is the axis-edge tag, far from
  // wherever you're actually looking on a wide chart.
  chart.subscribeCrosshairMove((param) => {
    if (!param.point) {
      els.chartTooltip.hidden = true;
      return;
    }
    // Read the price straight off the price scale at the cursor's Y pixel
    // (not off a series data point) so the number shown always matches
    // where it's drawn relative to the axis - see the same fix in app.js
    // for the full reasoning. Also works past "now" (forecast region),
    // where candleSeries has no data point to look up at all.
    const price = candleSeries.coordinateToPrice(param.point.y);
    if (price === null) {
      els.chartTooltip.hidden = true;
      return;
    }
    els.chartTooltip.hidden = false;
    els.chartTooltip.textContent = fmtMoney(price);

    const wrapWidth = els.chartWrap.clientWidth;
    const tooltipWidth = els.chartTooltip.offsetWidth || 90;
    let left = param.point.x + 14;
    if (left + tooltipWidth > wrapWidth) left = param.point.x - tooltipWidth - 14;
    els.chartTooltip.style.left = `${Math.max(4, left)}px`;
    els.chartTooltip.style.top = `${Math.max(4, param.point.y - 14)}px`;
  });
}

function renderPredictionTable(data) {
  const basePrice = data.current_price;
  if (!data.predictions.length) {
    els.predictionRows.innerHTML = `<tr class="empty-row"><td colspan="3">—</td></tr>`;
    return;
  }
  els.predictionRows.innerHTML = data.predictions
    .map((p) => {
      const pct = ((p.predicted_price - basePrice) / basePrice) * 100;
      const cls = directionClass(pct);
      return `<tr>
        <td>${fmtDay(p.timestamp)}</td>
        <td class="price price-${cls}">${fmtMoney(p.predicted_price)}</td>
        <td class="price-${cls}">${arrow(pct)} ${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%</td>
      </tr>`;
    })
    .join("");
}

function sentimentBadge(label) {
  const icons = { positive: "▲", negative: "▼", neutral: "●" };
  const safe = ["positive", "negative", "neutral"].includes(label) ? label : "neutral";
  return `<span class="sentiment-badge ${safe}">${icons[safe]} ${t(`common.sentiment.${safe}`)}</span>`;
}

function renderNews(news) {
  if (!news || !news.length) {
    els.newsList.innerHTML = `<p class="empty-hint">${t("detail.news.empty")}</p>`;
    return;
  }
  els.newsList.innerHTML = news
    .map(
      (a) => `<div class="news-item">
        <a class="news-title" href="${a.url}" target="_blank" rel="noopener noreferrer">${a.title}</a>
        <div class="news-meta">
          ${sentimentBadge(a.sentiment.label)}
          <span>score ${(a.sentiment.compound || 0).toFixed(2)}</span>
          <span>·</span>
          <span>${a.source || ""}</span>
          <span>·</span>
          <span>${new Date(a.published_at).toLocaleString("en-US", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}</span>
        </div>
      </div>`
    )
    .join("");
}

// Live price ticker: /api/scout/detail (candles+news+model) only needs to
// run once per visit, but the price itself should keep moving - polls a
// separate, lightweight, cached endpoint that does none of that heavy work.
let liveTickInFlight = false;
async function tickLivePrice() {
  if (liveTickInFlight) return;
  liveTickInFlight = true;
  try {
    const url = new URL("/api/scout/price", window.location.origin);
    url.searchParams.set("asset_type", assetType);
    url.searchParams.set("id", assetId);
    const resp = await fetch(url);
    const quote = await resp.json();
    if (!resp.ok) return;
    els.price.textContent = fmtMoney(quote.price);
    if (quote.change_24h_pct !== null && quote.change_24h_pct !== undefined) {
      const cls = directionClass(quote.change_24h_pct);
      els.change.textContent = `${arrow(quote.change_24h_pct)} ${Math.abs(quote.change_24h_pct).toFixed(2)}% (24h)`;
      els.change.className = `ticker-change ${cls}`;
    }
  } catch {
    // silent - best-effort live nudge, the initial loadDetail() call
    // already surfaced a real error banner if the asset itself is bad
  } finally {
    liveTickInFlight = false;
  }
}

loadDetail();
setInterval(tickLivePrice, 5000);

// Static labels re-apply themselves (i18n.js); the narrative/news dates
// come from the server (lang-aware /api/scout/detail), so a language
// switch re-fetches rather than just re-rendering cached data.
document.addEventListener("langchange", loadDetail);
