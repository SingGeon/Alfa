"""Regenerate the README's confidence-band charts (docs/images/band-*.png).

Runs the walk-forward backtest (ml/walk_forward.py) twice on the same real
Binance candles: once with the tuned model as it was before band
calibration (ml/price_predictor.py at commit cecf999, read with git show),
once with the current one, then plots per-step coverage and band width.
Needs network (Binance) and ~4 minutes:

    .venv/bin/python docs/scripts/band_charts.py
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.price_predictor import PricePredictor  # noqa: E402
from ml.walk_forward import fetch_binance, run  # noqa: E402

BEFORE_COMMIT = "cecf999"


def _old_predictor_module():
    src = subprocess.run(["git", "show", f"{BEFORE_COMMIT}:ml/price_predictor.py"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "old_price_predictor.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("old_price_predictor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def backtest() -> dict:
    old = _old_predictor_module()
    eth = fetch_binance("ETHUSDT", "1h", 3000)
    btc = fetch_binance("BTCUSDT", "1h", 3000)
    out = {}
    for name, factory in (("before", lambda: old.PricePredictor(model_variant="tuned")),
                          ("after", lambda: PricePredictor(model_variant="tuned"))):
        res = run(eth, btc, "1h", 24, 40, predictor_factory=factory)
        w = [r for r in res.windows if len(r.predicted) == len(r.actual) == 24]
        inside = np.array([[lo <= a <= hi for lo, hi, a in zip(r.lower, r.upper, r.actual)] for r in w], float)
        width = np.array([[np.log(hi / lo) * 100 for lo, hi in zip(r.lower, r.upper)] for r in w])
        out[name] = {"coverage": inside.mean(0).tolist(), "width_pct": width.mean(0).tolist(), "windows": len(w)}
    return out


d = backtest()
out = ROOT / "docs" / "images"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
steps = list(range(1, 25))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

def base(title, subtitle, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=110)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.text(0.07, 0.94, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.885, subtitle, fontsize=10.5, color=INK2)
    ax.set_xlabel("Forecast step (hours ahead)", color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlim(0.5, 26.5); ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    fig.subplots_adjust(top=0.82, left=0.09, right=0.97, bottom=0.12)
    return fig, ax

def lines(ax, key, scale, fmt, legend_loc="upper left"):
    for name, color, label in (("before", ORANGE, "Before calibration"), ("after", BLUE, "After calibration (tuned)")):
        ys = [v * scale for v in d[name][key]]
        ax.plot(steps, ys, color=color, linewidth=2, label=label, solid_capstyle="round")
        ax.plot(steps[-1], ys[-1], "o", color=color, markersize=7, markeredgecolor=SURF, markeredgewidth=2)
        ax.annotate(fmt(ys[-1]), (steps[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
    ax.legend(loc=legend_loc, frameon=False, labelcolor=INK)

n = d["after"]["windows"]
fig, ax = base("How often the real price stayed inside the 80% band",
               f"Walk-forward backtest, ETH 1h, {n} forecast origins, 24-step forecasts", "Share of real prices inside the band")
ax.axhline(80, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(6.3, 84, "target 80%", color=INK2, fontsize=10)
lines(ax, "coverage", 100, lambda v: f"{v:.0f}%", legend_loc="lower left")
ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
fig.savefig(f"{out}/band-coverage.png", facecolor=SURF); plt.close(fig)

fig, ax = base("Width of the confidence band by forecast step",
               "Mean width (upper / lower, in %) over the same backtest - the calibrated band grows with the horizon", "Band width")
lines(ax, "width_pct", 1, lambda v: f"{v:.1f}%")
ax.set_ylim(0, None); ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
fig.savefig(f"{out}/band-width.png", facecolor=SURF); plt.close(fig)
print("wrote", out / "band-coverage.png", out / "band-width.png")
