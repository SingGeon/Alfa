"use strict";

const els = {
  scannedAt: document.getElementById("scannedAt"),
  errorBanner: document.getElementById("errorBanner"),
  assetTypeGroup: document.getElementById("assetTypeGroup"),
  pinnedToggle: document.getElementById("pinnedToggle"),
  scoutRows: document.getElementById("scoutRows"),
};

const state = { assetType: "", pinnedOnly: false, pinnedKeys: new Set(), pinnedAssets: [] };

function pinKey(assetType, id) {
  return `${assetType}:${id}`;
}

function fmtMoney(v) {
  if (v === null || v === undefined) return "—";
  const abs = Math.abs(v);
  // Smart precision: sub-cent assets (meme coins etc.) need more decimals
  // to show anything meaningful at all.
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

// 3-tier quality read for confidence (0-100): red under 40 (weak/
// uncertain), yellow 40-70 (decent), green over 70 (precise/strong).
function qualityClass(confidence) {
  if (confidence < 40) return "quality-poor";
  if (confidence < 70) return "quality-decent";
  return "quality-good";
}

function fmtRelative(iso) {
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "chiar acum";
  if (mins < 60) return `acum ${mins} min`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `acum ${hours} h`;
  return `acum ${Math.round(hours / 24)} zile`;
}

async function loadPins() {
  try {
    const resp = await fetch("/api/scout/pins");
    if (!resp.ok) return;
    const data = await resp.json();
    state.pinnedAssets = data.pins || [];
    state.pinnedKeys = new Set(state.pinnedAssets.map((p) => pinKey(p.asset_type, p.id)));
  } catch {
    // best-effort - pin stars just won't light up if this fails
  }
}

async function loadScout() {
  try {
    await loadPins();
    const url = new URL("/api/scout", window.location.origin);
    if (state.assetType) url.searchParams.set("asset_type", state.assetType);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
    const data = await resp.json();
    render(data.results || []);
  } catch (exc) {
    els.errorBanner.hidden = false;
    els.errorBanner.textContent = `Nu pot obține rezultatele scout: ${exc.message}`;
  }
}

// A pinned asset can be present in the row form (from /api/scout, full
// numbers) or - if it dropped out of the latest scan - as a bare
// {asset_type, id, symbol, name} bookmark. Both render, just with "—" for
// the fields only a fresh scan/analysis has.
function orphanRow(pin) {
  return {
    asset_type: pin.asset_type, id: pin.id, symbol: pin.symbol, name: pin.name,
    current_price: null, predicted_price: null, predicted_change_pct: null,
    confidence: null, sentiment_avg: null, articles_analyzed: null, score: null,
    _orphan: true,
  };
}

function render(results) {
  let rows = results;
  if (state.pinnedOnly) {
    const inScan = results.filter((r) => state.pinnedKeys.has(pinKey(r.asset_type, r.id)));
    const inScanKeys = new Set(inScan.map((r) => pinKey(r.asset_type, r.id)));
    const orphans = state.pinnedAssets
      .filter((p) => !inScanKeys.has(pinKey(p.asset_type, p.id)))
      .filter((p) => !state.assetType || p.asset_type === state.assetType)
      .map(orphanRow);
    rows = [...inScan, ...orphans];
  }

  if (!rows.length) {
    els.scoutRows.innerHTML = state.pinnedOnly
      ? `<tr class="empty-row"><td colspan="10">Niciun activ fixat încă — apasă ☆ pe un rând pentru a-l memora.</td></tr>`
      : `<tr class="empty-row"><td colspan="10">Nicio scanare încă — prima scanare pornește automat cu run_scheduler.py și poate dura câteva minute (analizează ~80 active).</td></tr>`;
    els.scannedAt.textContent = "";
    return;
  }
  els.errorBanner.hidden = true;
  const withScanTime = results.filter((r) => r.scanned_at);
  if (withScanTime.length) {
    const latest = withScanTime.reduce((a, b) => (new Date(a.scanned_at) > new Date(b.scanned_at) ? a : b));
    els.scannedAt.textContent = `ultima scanare ${fmtRelative(latest.scanned_at)}`;
  }

  els.scoutRows.innerHTML = rows
    .map((r, i) => {
      const typeLabel = r.asset_type === "crypto" ? "Cripto" : "Acțiune";
      const href = `/detail.html?type=${encodeURIComponent(r.asset_type)}&id=${encodeURIComponent(r.id)}`;
      const pinned = state.pinnedKeys.has(pinKey(r.asset_type, r.id));
      const pinCell = `<td class="pin-cell"><button class="pin-btn ${pinned ? "pinned" : ""}" data-pin-type="${r.asset_type}" data-pin-id="${r.id}" data-pin-symbol="${r.symbol}" data-pin-name="${r.name}" title="${pinned ? "Scoate de la fixate" : "Fixează"}">${pinned ? "★" : "☆"}</button></td>`;
      // Mega-cap picks clear a much higher bar (see scanner.py's
      // MEGA_CAP_MIN_CHANGE_PCT) before ever showing up at all, so the
      // badge is a "this one's a big deal" flag, not a warning.
      const megaCapBadge = r.tier === "mega_cap" ? `<span class="mega-cap-badge" title="Mișcare dramatică prezisă pentru un activ mare, nu un underdog">⭐ Mega-cap</span>` : "";

      if (r._orphan) {
        return `<tr>
          ${pinCell}
          <td class="rank-cell">—</td>
          <td>
            <div class="asset-cell">
              <span class="asset-type-badge ${r.asset_type}">${typeLabel}</span>
              <div>
                <div class="asset-symbol">${r.symbol} ${megaCapBadge}</div>
                <div class="asset-name">${r.name}</div>
              </div>
            </div>
          </td>
          <td colspan="6" class="empty-hint-cell">nu mai e în ultima scanare · <a class="nav-link" href="${href}">vezi detalii</a></td>
        </tr>`;
      }

      const changeCls = directionClass(r.predicted_change_pct);
      const sentimentCls = directionClass(r.sentiment_avg);
      const scoreCls = directionClass(r.score);
      return `<tr class="clickable-row" data-href="${href}">
        ${pinCell}
        <td class="rank-cell">${i + 1}</td>
        <td>
          <div class="asset-cell">
            <span class="asset-type-badge ${r.asset_type}">${typeLabel}</span>
            <div>
              <div class="asset-symbol">${r.symbol} ${megaCapBadge}</div>
              <div class="asset-name">${r.name}</div>
            </div>
          </div>
        </td>
        <td>${fmtMoney(r.current_price)}</td>
        <td>${fmtMoney(r.predicted_price)}</td>
        <td class="price-${changeCls}">${arrow(r.predicted_change_pct)} ${r.predicted_change_pct >= 0 ? "+" : ""}${r.predicted_change_pct.toFixed(2)}%</td>
        <td class="${qualityClass(r.confidence)}">${Math.round(r.confidence)}/100</td>
        <td class="price-${sentimentCls}">${r.sentiment_avg.toFixed(2)}</td>
        <td>${r.articles_analyzed}</td>
        <td class="score-cell price-${scoreCls}">${r.score.toFixed(2)}</td>
      </tr>`;
    })
    .join("");
}

async function togglePin(btn) {
  const assetType = btn.dataset.pinType;
  const id = btn.dataset.pinId;
  const symbol = btn.dataset.pinSymbol;
  const name = btn.dataset.pinName;
  const key = pinKey(assetType, id);
  const currentlyPinned = state.pinnedKeys.has(key);
  try {
    if (currentlyPinned) {
      await fetch(`/api/scout/pins?asset_type=${encodeURIComponent(assetType)}&id=${encodeURIComponent(id)}`, { method: "DELETE" });
      state.pinnedKeys.delete(key);
      state.pinnedAssets = state.pinnedAssets.filter((p) => pinKey(p.asset_type, p.id) !== key);
    } else {
      await fetch("/api/scout/pins", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ asset_type: assetType, id, symbol, name }),
      });
      state.pinnedKeys.add(key);
      state.pinnedAssets.push({ asset_type: assetType, id, symbol, name });
    }
  } catch {
    return; // leave UI as-is; next loadScout() will resync
  }
  loadScout();
}

els.assetTypeGroup.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-type]");
  if (!btn) return;
  state.assetType = btn.dataset.type;
  [...els.assetTypeGroup.children].forEach((b) => b.classList.toggle("active", b === btn));
  loadScout();
});

els.pinnedToggle.addEventListener("click", () => {
  state.pinnedOnly = !state.pinnedOnly;
  els.pinnedToggle.classList.toggle("active", state.pinnedOnly);
  els.pinnedToggle.textContent = state.pinnedOnly ? "★ Fixate" : "☆ Fixate";
  loadScout();
});

els.scoutRows.addEventListener("click", (e) => {
  const pinBtn = e.target.closest("button.pin-btn");
  if (pinBtn) {
    togglePin(pinBtn);
    return;
  }
  const row = e.target.closest("tr.clickable-row");
  if (row && row.dataset.href) window.location.href = row.dataset.href;
});

loadScout();
setInterval(loadScout, 60000);
