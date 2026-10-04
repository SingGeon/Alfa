"""Regenerate the README's strategy charts (docs/images/strategy-*.png).

The fund charts read the reports ml/strategy_fund.py saves to
data/strategy_fund/ (run `.venv/bin/python -m ml.strategy_fund` first, ~1
min). The direction and strategy-lab charts plot the results of the runs
listed next to each number below (October 2026, ETH since August 2017),
so they need nothing but this file:

    .venv/bin/python docs/scripts/strategy_charts.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "images"
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, UNSEEN = "#2a78d6", "#eb6834", "#f0efec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

# Direction right, in % and its standard error (1 sd), each on candles the
# model hadn't seen. Source in the comment.
DIRECTION = [
    ("Evolving population, 15m, 1 ahead", 50.2, 0.09),     # ml.pretrain_evolution, 319,398 votes
    ("Evolving population, 1h, 1 ahead", 50.4, 0.18),      # 79,844 votes
    ("Evolving population, 4h, 1 ahead", 50.2, 0.35),      # 19,959 votes
    ("Evolving population, 1d, 1 ahead", 50.4, 0.87),      # 3,309 votes
    ("Legacy model, 1h, 24 ahead", 50.0, 3.5),             # walk-forward, 200 independent windows
    ("Legacy model, 15m, 24 ahead", 52.5, 3.5),            # walk-forward, 200 independent windows
    ("Logistic model, 1 day ahead", 52.2, 0.9),            # yearly refit, 2,828 days
    ("  its most confident fifth", 52.7, 2.1),             # 566 days
    ("Chart patterns, 1h, 24 ahead", 48.6, 0.5),           # ml.pattern_recognition, 9,091 calls
    ("Chart patterns, 4h, 24 ahead", 48.7, 1.0),           # 2,430 calls
]
ALWAYS_UP = 51.8  # share of 24h windows ETH closed higher

# Strategy lab + evolving strategies, TEST (2023 on): money from 100 EUR, worst drop %.
LAB = {
    "Buy & hold":                 {"1h": (230, 69), "4h": (229, 68), "1d": (225, 68)},
    "Vol-target long":            {"1h": (183, 44), "4h": (170, 39), "1d": (218, 60)},
    "Range":                      {"1h": (106, 14), "4h": (59, 62), "1d": (93, 50)},
    "Range, vol-sized":           {"1h": (69, 36), "4h": (121, 25), "1d": (65, 60)},
    "Vol-target + range":         {"1h": (92, 53), "4h": (147, 47), "1d": (198, 49)},
    "Vol-target + Box Breakout":  {"1h": (155, 38), "4h": (321, 48), "1d": (179, 58)},
    "Evolving strategies (fund)": {"1h": (164, 49), "4h": (501, 39), "1d": (291, 48)},
}


def frame(figsize=(10, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    fig.patch.set_facecolor(SURF)
    style(ax)
    return fig, ax


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def titles(fig, title, subtitle, x=0.07):
    fig.text(x, 0.94, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(x, 0.885, subtitle, fontsize=10.5, color=INK2)


def save(fig, name):
    fig.savefig(OUT / name, facecolor=SURF)
    plt.close(fig)
    print("wrote", OUT / name)


def direction_chart():
    fig, ax = frame((10, 6.2))
    titles(fig, "Nobody calls ETH's direction better than a coin flip",
           "Direction right on candles it hadn't seen, with its 95% range. ETH since 2017.", x=0.33)
    names = [d[0] for d in DIRECTION][::-1]
    vals = np.array([d[1] for d in DIRECTION][::-1])
    err = 1.96 * np.array([d[2] for d in DIRECTION][::-1])
    y = np.arange(len(names))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(50, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axvline(ALWAYS_UP, color=MUTED, linewidth=1, linestyle=(0, (1, 2)))
    ax.text(49.85, -0.9, "coin flip 50%", color=INK2, fontsize=9.5, ha="right")
    ax.text(ALWAYS_UP + 0.15, -0.9, f"always \"up\" {ALWAYS_UP}%", color=MUTED, fontsize=9.5)
    ax.hlines(y, vals - err, vals + err, color=BLUE, linewidth=2, alpha=0.45)
    ax.plot(vals, y, "o", color=BLUE, markersize=8, markeredgecolor=SURF, markeredgewidth=2)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.1f}%", (v, yi), xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    ax.set_xlim(40, 60)
    ax.set_xticks(range(40, 61, 2))
    ax.set_ylim(-1.3, len(names) - 0.1)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    fig.subplots_adjust(top=0.84, left=0.33, right=0.97, bottom=0.08)
    save(fig, "strategy-direction.png")


def load(interval):
    path = ROOT / "data" / "strategy_fund" / f"ethereum_{interval}.json"
    if not path.exists():
        sys.exit(f"{path} is missing: run `.venv/bin/python -m ml.strategy_fund` first")
    return json.loads(path.read_text())


def money_chart(r):
    c = pd.DataFrame(r["curve"])
    c["time"] = pd.to_datetime(c["time"])
    test = pd.Timestamp(r["test_from"])
    fig, ax = frame()
    titles(fig, f"Strategy fund vs buy & hold, ETH {r['interval']}",
           "100 € (virtual) each from Jan 2020, fees included, every decision from the past only. Log scale.")
    ax.axvspan(test, c["time"].iloc[-1], color=UNSEEN, zorder=0)
    for col, color, label in (("buy_hold", MUTED, "Buy & hold"), ("fund", BLUE, "Strategy fund")):
        ax.plot(c["time"], c[col], color=color, linewidth=2, label=label, solid_capstyle="round")
        last = c[col].iloc[-1]
        ax.plot(c["time"].iloc[-1], last, "o", color=color, markersize=7, markeredgecolor=SURF, markeredgewidth=2)
        ax.annotate(f"{last:,.0f} €", (c["time"].iloc[-1], last), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f} €"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axhline(100, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    t = r["test"]
    ax.text(test + pd.Timedelta(days=25), 0.97, f"unseen years (2023 on): fund {t['fund']['money']:.0f} €, "
            f"buy & hold {t['buy_hold']['money']:.0f} €", transform=ax.get_xaxis_transform(), color=INK2,
            fontsize=9.5, va="top")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(c["time"].iloc[0], c["time"].iloc[-1] + pd.Timedelta(days=140))
    ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    fig.subplots_adjust(top=0.82, left=0.1, right=0.97, bottom=0.08)
    save(fig, f"strategy-fund-{r['interval']}.png")


def years_chart(reports):
    fig, axes = plt.subplots(1, len(reports), figsize=(11, 5.2), dpi=110, sharey=True)
    fig.patch.set_facecolor(SURF)
    titles(fig, "Year by year: it earns less in the bull years and loses less, or earns, in the bad ones",
           "Return of 100 € in each calendar year, fees included. Shaded: unseen years (2023 on).", x=0.06)
    for ax, r in zip(axes, reports):
        style(ax)
        ys = pd.DataFrame(r["years"])
        x = np.arange(len(ys))
        w = 0.38
        test_year = pd.Timestamp(r["test_from"]).year
        first = int(np.argmax(ys["year"] >= test_year))
        ax.axvspan(first - 0.5, len(ys) - 0.5, color=UNSEEN, zorder=0)
        for off, col, color, label in ((-w / 2 - 0.01, "fund_pct", BLUE, "Strategy fund"),
                                       (w / 2 + 0.01, "buy_hold_pct", MUTED, "Buy & hold")):
            ax.bar(x + off, ys[col], width=w, color=color, label=label, zorder=2)
        for xi, v in zip(x, ys["fund_pct"]):
            ax.annotate(f"{v:+.0f}%", (xi - w / 2, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                        ha="center", color=INK, fontsize=8.5)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(ys["year"], color=INK2)
        ax.set_title(f"ETH {r['interval']}", color=INK, fontsize=12, loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axes[0].legend(loc="upper right", frameon=False, labelcolor=INK)
    fig.subplots_adjust(top=0.78, left=0.07, right=0.98, bottom=0.08, wspace=0.08)
    save(fig, "strategy-years.png")


def genes_chart(reports):
    genes = [("core", "Calm long weight"), ("box_w", "Box Breakout weight"),
             ("range_w", "Range weight"), ("long_short", "Share allowed to short")]
    fig, axes = plt.subplots(len(reports), len(genes), figsize=(11, 4.8), dpi=110, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURF)
    titles(fig, "What selection picked, month by month",
           "Average gene of the 4 best strategies (3 populations). Nobody set these by hand.", x=0.06)
    for row, r in zip(np.atleast_2d(axes), reports):
        g = pd.DataFrame(r["genes_over_time"])
        g["time"] = pd.to_datetime(g["time"])
        for ax, (key, label) in zip(row, genes):
            style(ax)
            ax.fill_between(g["time"], g[key], color=BLUE, alpha=0.12, linewidth=0)
            ax.plot(g["time"], g[key], color=BLUE, linewidth=2)
            ax.annotate(f"{g[key].iloc[-1]:.0%}", (g["time"].iloc[-1], g[key].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            if r is reports[0]:
                ax.set_title(label, color=INK, fontsize=10.5, loc="left")
        row[0].set_ylabel(f"ETH {r['interval']}", color=INK, fontsize=11)
    fig.subplots_adjust(top=0.78, left=0.08, right=0.97, bottom=0.08, wspace=0.12, hspace=0.25)
    save(fig, "strategy-genes.png")


def lab_table():
    intervals = ["1h", "4h", "1d"]
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=110)
    fig.patch.set_facecolor(SURF)
    ax.axis("off")
    titles(fig, "Unseen years (2023 on): money from 100 € and worst drop",
           "Strategy lab and evolving strategies, ETH, Binance fee 0.1% per trade. Bold: beat buy & hold on that interval.", x=0.03)
    hold = LAB["Buy & hold"]
    cells, weights = [], []
    for name, res in LAB.items():
        cells.append([name] + [f"{res[i][0]:,} €  ·  -{res[i][1]}%" for i in intervals])
        weights.append([False] + [name != "Buy & hold" and res[i][0] > hold[i][0] for i in intervals])
    table = ax.table(cellText=cells, colLabels=["Strategy"] + [f"ETH {i}" for i in intervals], loc="center",
                     cellLoc="right", colLoc="right", colWidths=[0.34, 0.22, 0.22, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.75)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        cell.set_facecolor(SURF)
        cell.visible_edges = "B"
        if c == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if r == 0:
            cell.set_text_props(color=INK2, fontweight="bold")
        else:
            bold = weights[r - 1][c]
            cell.set_text_props(color=INK, fontweight="bold" if bold else "normal")
            if bold:
                cell.set_facecolor("#e8f1fb")
    fig.subplots_adjust(top=0.84, left=0.03, right=0.97, bottom=0.02)
    save(fig, "strategy-lab-table.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    direction_chart()
    lab_table()
    reports = [load("4h"), load("1d")]
    for rep in reports:
        money_chart(rep)
    years_chart(reports)
    genes_chart(reports)
