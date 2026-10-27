"""Prediction images, drawn like the dashboard chart: the real candles the
model saw, the AI forecast (dashed line + confidence band) after them, and
the real candles that actually followed drawn *on top of* the forecast, so
the two can be compared at a glance.

Two files per prediction:
- the visual snapshot, data/snapshots/{model}/{interval}/YYYY-MM-DD/{interval}_{model}_{created_at}.png
  (date = day the prediction was made). Drawn as soon as the prediction is
  logged and redrawn every time new real candles arrive, so it always shows
  the latest state - this is the visual history.
- the final chart, data/charts/{model}/{interval}/YYYY-MM-DD/{same name}.png,
  where the date is the day the prediction *resolved* (its last forecast
  candle closed, see schema.sql) - so by the time the 00:05 UTC daily report
  runs, every chart for the day that just ended already exists. Idempotent:
  an existing file is never redrawn.

Both trees are browsable two ways: by model -> interval -> day (the real
files), and by day across every model/interval under by-date/YYYY-MM-DD/
(relative symlinks to those same files; a copy where symlinks aren't
supported). evaluation/stats.py writes a stats.json at each level.

Uses matplotlib's object API on an Agg canvas (no pyplot, no GUI) - pyplot
keeps global state that isn't safe to touch from the scheduler's worker
threads and Flask's request threads at the same time; a standalone Figure is.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import config  # noqa: E402
from evaluation import storage  # noqa: E402
from evaluation.storage import interval_delta, parse_iso  # noqa: E402

logger = logging.getLogger(__name__)

BACKGROUND = "#19171d"
TEXT = "#f3f8fa"
GRID = "#545355"
PREDICTED = "#498aa8"
ACTUAL = "#5fb894"
ERROR = "#d8706a"
HIGHLIGHT = "#e8b339"   # the trade snapshots' entry/exit marker - distinct from every other color here

BASELINE = "#a3a3a3"

FIG_SIZE = (10, 6)
DPI = 100


def model_label(model_name: str) -> str:
    """The legacy model is the unchanged reference the tuned one is measured
    against - say so wherever it's shown."""
    return f"{model_name} (control)" if model_name.endswith("-legacy") else model_name


BY_DATE_DIR = "by-date"
_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def model_dir_name(model_name: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in model_name)


def _file_name(pred: dict) -> str:
    created = parse_iso(pred["created_at"]).strftime("%Y%m%dT%H%M%SZ")
    return f"{pred['interval']}_{model_dir_name(pred['model_name'])}_{created}.png"


def _organized_path(root: str, model_name: str, interval: str, day: str, name: str) -> Path:
    return Path(root) / model_dir_name(model_name) / interval / day / name


def chart_path(pred: dict) -> Path:
    day = pred["resolved_at"][:10]
    return _organized_path(config.EVAL_CHARTS_DIR, pred["model_name"], pred["interval"], day, _file_name(pred))


def snapshot_path(pred: dict) -> Path:
    day = pred["created_at"][:10]
    return _organized_path(config.EVAL_SNAPSHOTS_DIR, pred["model_name"], pred["interval"], day, _file_name(pred))


def by_date_dir(root: str, day: str) -> Path:
    return Path(root) / BY_DATE_DIR / day


def _link_by_date(path: Path, root: str) -> None:
    """Expose `path` (.../{model}/{interval}/{day}/{name}) under
    by-date/{day}/{name} too."""
    link = by_date_dir(root, path.parent.name) / path.name
    if link.is_symlink():
        return
    os.makedirs(link.parent, exist_ok=True)
    try:
        os.symlink(os.path.relpath(path, link.parent), link)
    except FileExistsError:
        shutil.copy2(path, link)  # a copy from an earlier fallback: keep it current
    except OSError:
        shutil.copy2(path, link)


def organize_existing(root: str) -> int:
    """One-off migration from the old flat layout ({root}/YYYY-MM-DD/{name})
    into {root}/{model}/{interval}/YYYY-MM-DD/{name} + by-date links. The
    model and interval are read back from the file name. Returns how many
    files were moved."""
    base = Path(root)
    if not base.is_dir():
        return 0
    moved = 0
    for day_dir in [d for d in base.iterdir() if d.is_dir() and _DAY_DIR_RE.match(d.name)]:
        for f in day_dir.glob("*.png"):
            parts = f.stem.split("_")
            if f.name.endswith(".tmp.png") or len(parts) != 3:
                continue
            interval, model, _ = parts
            target = _organized_path(root, model, interval, day_dir.name, f.name)
            os.makedirs(target.parent, exist_ok=True)
            os.replace(f, target)
            _link_by_date(target, root)
            moved += 1
        if not any(day_dir.iterdir()):
            day_dir.rmdir()
    return moved


def _draw_candles(ax, candles: list[dict], width: float, alpha: float, zorder: float) -> None:
    for c in candles:
        x = mdates.date2num(parse_iso(c["timestamp"]))
        close = c.get("close", c.get("price"))
        o, h, low = c.get("open"), c.get("high"), c.get("low")
        if o is None or h is None or low is None:
            # Only a close is known (CoinGecko fallback) - a dot, not a fake candle.
            ax.plot([x], [close], marker="o", markersize=4, color=ACTUAL, alpha=alpha, zorder=zorder)
            continue
        color = ACTUAL if close >= o else ERROR
        ax.vlines(x, low, h, color=color, linewidth=1, alpha=alpha, zorder=zorder)
        body = abs(close - o) or max((h - low) * 0.02, 0.01)
        ax.bar(x, body, bottom=min(o, close), width=width, color=color, alpha=alpha, linewidth=0, zorder=zorder + 0.1)


def draw_prediction(pred: dict) -> Figure:
    delta = interval_delta(pred["interval"])
    predicted = pred["predicted_path"]
    actual = pred["actual_path"]
    history = pred.get("history_path") or []
    anchor_time = parse_iso(predicted[0]["timestamp"]) - delta
    anchor_price = pred["price_at_prediction"]
    width = delta.total_seconds() / 86400 * 0.62  # candle body, in matplotlib date units

    fig = Figure(figsize=FIG_SIZE, dpi=DPI, facecolor=BACKGROUND)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(BACKGROUND)

    # Forecast window, lightly shaded, like the dashboard's future area.
    end_time = parse_iso(predicted[-1]["timestamp"])
    ax.axvspan(anchor_time, end_time + delta / 2, color=PREDICTED, alpha=0.06, linewidth=0, zorder=0)

    if history:
        _draw_candles(ax, history, width, alpha=0.5, zorder=2)

    pred_x = [anchor_time] + [parse_iso(p["timestamp"]) for p in predicted]
    if all(p.get("lower") is not None and p.get("upper") is not None for p in predicted):
        ax.fill_between(
            pred_x, [anchor_price] + [p["lower"] for p in predicted], [anchor_price] + [p["upper"] for p in predicted],
            color=PREDICTED, alpha=0.18, linewidth=0, zorder=3, label="Bandă de încredere",
        )
    ax.plot(
        pred_x, [anchor_price] + [p["price"] for p in predicted], linestyle="--", linewidth=2, color=PREDICTED,
        marker="o", markersize=3, zorder=4, label="Predicție AI",
    )
    # Baseline every forecast is scored against: the price simply stays put.
    ax.plot(
        [anchor_time, end_time], [anchor_price, anchor_price], linestyle=(0, (2, 3)), linewidth=1.3,
        color=BASELINE, zorder=3.5, label="Fără schimbare",
    )

    # The real candles go on top of the forecast.
    if actual:
        _draw_candles(ax, actual, width, alpha=1.0, zorder=6)
        ax.plot(
            [anchor_time] + [parse_iso(p["timestamp"]) for p in actual], [anchor_price] + [p["price"] for p in actual],
            linestyle="-", linewidth=1.4, color=ACTUAL, alpha=0.9, zorder=7, label="Real",
        )
    else:
        ax.plot([], [], linestyle="-", linewidth=1.4, color=ACTUAL, label="Real")

    ax.axvline(anchor_time, color=TEXT, linestyle=":", linewidth=1, alpha=0.7, zorder=5)
    ax.scatter([anchor_time], [anchor_price], color=TEXT, zorder=8, s=30)
    created = parse_iso(pred["created_at"])
    ax.annotate(
        "predicție", xy=(anchor_time, 1), xycoords=("data", "axes fraction"), xytext=(4, -14),
        textcoords="offset points", color=TEXT, fontsize=9, alpha=0.8,
    )

    # Title row, then legend (left) + status box (right) above the plot:
    # nothing overlays the candles or the end of the forecast.
    fig.subplots_adjust(top=0.78)
    fig.text(
        0.125, 0.965, f"{model_label(pred['model_name'])} · {pred['interval']} · {pred['horizon_steps']} pași",
        color=TEXT, fontsize=14, fontweight="bold", va="top",
    )
    fig.text(
        0.125, 0.005, f"Creată {created.strftime('%Y-%m-%d %H:%M')} UTC · preț la predicție ${anchor_price:,.2f}",
        color=TEXT, alpha=0.7, fontsize=9, va="bottom",
    )

    from evaluation.evaluator import score_path  # evaluator doesn't import charts: no cycle

    if pred.get("status") == "completed":
        direction = "corectă" if pred["direction_correct"] else "greșită"
        box_text, box_color = f"Eroare {pred['pct_error']:+.2f}% · direcție {direction}", ERROR
    elif pred.get("status") == "expired":
        box_text, box_color = "Expirată (fără preț real)", GRID
    else:
        box_text, box_color = f"În desfășurare · {len(actual)}/{len(predicted)} candele reale", TEXT
    score = score_path(pred, actual)
    if score["steps_scored"]:
        skill = score["skill_score"]
        skill_text = "—" if skill is None else f"{skill:+.2f}"
        cover = score["band_coverage"]
        box_text += f"\nAbilitate vs fără schimbare {skill_text}"
        if cover is not None:
            box_text += f" · în bandă {cover * 100:.0f}%"
        if skill is not None and pred.get("status") == "completed":
            box_color = ACTUAL if skill > 0 else ERROR
    # Above the plot, not inside it, so it never hides the end of the forecast.
    ax.text(
        1.0, 1.015, box_text, transform=ax.transAxes, ha="right", va="bottom", color=box_color, fontsize=10.5,
        fontweight="bold", zorder=10, linespacing=1.4,
        bbox={"facecolor": BACKGROUND, "edgecolor": box_color, "boxstyle": "round,pad=0.35", "alpha": 0.9},
    )

    first = parse_iso(history[0]["timestamp"]) if history else anchor_time
    ax.set_xlim(first - delta, end_time + delta)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M", tz=created.tzinfo))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    legend = ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.015), ncol=2, facecolor=BACKGROUND, edgecolor=GRID, fontsize=9,
    )
    legend.set_zorder(10)
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.autofmt_xdate()
    return fig


def _save(fig: Figure, path: Path) -> Path:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    fig.savefig(tmp, facecolor=BACKGROUND)
    os.replace(tmp, path)  # atomic: a half-written file never looks "already generated"
    return path


def render_snapshot(pred: dict) -> Path:
    """(Re)draw the visual snapshot with the prediction's current state."""
    path = _save(draw_prediction(pred), snapshot_path(pred))
    _link_by_date(path, config.EVAL_SNAPSHOTS_DIR)
    return path


def render_prediction_chart(pred: dict, force: bool = False) -> Path | None:
    """Draw the final chart for one completed prediction; returns its path
    (None if the prediction isn't completed yet)."""
    if pred.get("status") != "completed":
        return None
    path = chart_path(pred)
    if path.exists() and not force:
        return path
    _save(draw_prediction(pred), path)
    _link_by_date(path, config.EVAL_CHARTS_DIR)
    return path


def refresh_snapshots(prediction_ids: list[int]) -> int:
    drawn = 0
    for pid in prediction_ids:
        pred = storage.get_prediction(pid)
        if not pred:
            continue
        try:
            render_snapshot(pred)
            drawn += 1
        except Exception:
            logger.exception("Failed to render snapshot for prediction %s", pid)
    return drawn


def generate_missing_charts(prediction_ids: list[int] | None = None) -> int:
    """Render the final chart for each completed prediction that doesn't
    have one yet (all of them, or just `prediction_ids`), refreshing its
    snapshot to the completed state too. Returns how many were drawn."""
    if prediction_ids is None:
        preds = storage.all_completed()
    else:
        preds = [p for p in (storage.get_prediction(i) for i in prediction_ids) if p]
    drawn = 0
    for pred in preds:
        if chart_path(pred).exists():
            continue
        try:
            render_prediction_chart(pred)
            render_snapshot(pred)
            drawn += 1
        except Exception:
            logger.exception("Failed to render chart for prediction %s", pred["id"])
    return drawn


# --- Trade snapshots (ml.evolution's fund: a picture of each entry/exit, not of a forecast) --
#
# ml.evolution.EvolutionaryForecaster's fund decides *when* to trade, not what price to
# predict - so, unlike every chart above, these are never drawn on a timer and never show a
# forecast line/band. evaluation.trade_watch calls render_trade_snapshot() exactly twice per
# trade: once the instant it opens ("entry"), once the instant it closes ("exit", showing both
# markers together so the whole trade reads at a glance).

def trade_snapshots_dir() -> Path:
    # A function, not a module-level constant: config.EVAL_SNAPSHOTS_DIR is monkeypatched
    # per-test (tests/conftest.py's evaluation_data_dir), and a constant computed once at
    # import time would freeze the first value it ever saw instead of tracking that.
    return Path(config.EVAL_SNAPSHOTS_DIR) / "trades"


def trade_snapshot_path(model_name: str, interval: str, trade: dict, kind: str) -> Path:
    stamp = parse_iso(trade["entry_time"]).strftime("%Y%m%dT%H%M%SZ")
    name = f"{interval}_{model_dir_name(model_name)}_{stamp}_{kind}.png"
    return trade_snapshots_dir() / model_dir_name(model_name) / interval / name


def draw_trade_snapshot(trade: dict, candles: list[dict], interval: str, model_name: str, kind: str) -> Figure:
    """Candles around one of the fund's own trades, with its entry (and exit, once it has
    one) highlighted in gold - never a predicted price line."""
    delta = interval_delta(interval)
    width = delta.total_seconds() / 86400 * 0.62
    long = trade["side"] == "long"
    entry_time = parse_iso(trade["entry_time"])
    has_exit = bool(trade.get("exit_time")) and trade.get("exit_price") is not None

    fig = Figure(figsize=FIG_SIZE, dpi=DPI, facecolor=BACKGROUND)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(BACKGROUND)

    if candles:
        _draw_candles(ax, candles, width, alpha=1.0, zorder=2)

    ax.scatter([entry_time], [trade["entry_price"]], marker="^" if long else "v", s=220,
               color=HIGHLIGHT, edgecolor=TEXT, linewidth=1.2, zorder=8)
    ax.annotate(
        f"{'LONG' if long else 'SHORT'} ${trade['entry_price']:,.2f}", xy=(entry_time, trade["entry_price"]),
        xytext=(0, 16), textcoords="offset points", ha="center", color=HIGHLIGHT,
        fontsize=10, fontweight="bold", zorder=9,
    )

    if has_exit:
        exit_time = parse_iso(trade["exit_time"])
        ret = trade.get("return_pct")
        exit_color = ACTUAL if (ret or 0) > 0 else ERROR
        ax.scatter([exit_time], [trade["exit_price"]], marker="o", s=160, color=exit_color,
                   edgecolor=TEXT, linewidth=1.2, zorder=8)
        ax.plot([entry_time, exit_time], [trade["entry_price"], trade["exit_price"]], linestyle=":",
                linewidth=1.3, color=HIGHLIGHT, alpha=0.8, zorder=6)
        ret_text = "—" if ret is None else f"{ret:+.2f}%"
        ax.annotate(
            f"IEȘIRE {ret_text}", xy=(exit_time, trade["exit_price"]), xytext=(0, -22),
            textcoords="offset points", ha="center", color=exit_color, fontsize=10, fontweight="bold", zorder=9,
        )
        status_text, status_color = f"Poziție închisă · {ret_text}", exit_color
    else:
        status_text, status_color = "Poziție deschisă", HIGHLIGHT

    fig.subplots_adjust(top=0.78)
    fig.text(0.125, 0.965, f"{model_label(model_name)} · {interval} · tranzacție fond", color=TEXT,
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.125, 0.005, f"Intrare {entry_time.strftime('%Y-%m-%d %H:%M')} UTC · {'long' if long else 'short'}",
             color=TEXT, alpha=0.7, fontsize=9, va="bottom")
    ax.text(1.0, 1.015, status_text, transform=ax.transAxes, ha="right", va="bottom", color=status_color,
           fontsize=10.5, fontweight="bold", zorder=10,
           bbox={"facecolor": BACKGROUND, "edgecolor": status_color, "boxstyle": "round,pad=0.35", "alpha": 0.9})

    if candles:
        first, last = parse_iso(candles[0]["timestamp"]), parse_iso(candles[-1]["timestamp"])
        ax.set_xlim(first - delta, last + delta)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M", tz=entry_time.tzinfo))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    fig.autofmt_xdate()
    return fig


def render_trade_snapshot(trade: dict, candles: list[dict], interval: str, model_name: str, kind: str) -> Path:
    """kind: "entry" the instant a position opens, "exit" the instant it closes."""
    path = trade_snapshot_path(model_name, interval, trade, kind)
    return _save(draw_trade_snapshot(trade, candles, interval, model_name, kind), path)


# --- Monte Carlo robustness report (ml.monte_carlo - Ch. 10: does a backtest hold up under -----
# small price noise, or is it a fluke of one specific historical path? Settings are replayed,
# never re-optimized, across every blurred run.

def draw_monte_carlo_report(results: list[dict], title: str, return_key: str = "total_return_pct",
                             drawdown_key: str = "max_drawdown_pct", sharpe_key: str = "sortino") -> Figure:
    """Top: every run's equity curve overlaid (an "equity_curve" list on the result dict, by
    trade number - absent runs are just skipped), with the median curve picked out. Bottom:
    three histograms - final return, max drawdown, and a risk-adjusted ratio - each with a
    fitted normal curve for scale. A visibly double-humped drawdown histogram is the book's own
    headline warning (p.274): two different outcome "regimes" hiding in what looked like one
    result."""
    from scipy.stats import norm

    from ml.monte_carlo import _is_bimodal

    returns = np.array([r[return_key] for r in results if r.get(return_key) is not None], dtype=float)
    drawdowns = np.array([r[drawdown_key] for r in results if r.get(drawdown_key) is not None], dtype=float)
    sharpes = np.array([r[sharpe_key] for r in results if r.get(sharpe_key) is not None], dtype=float)
    curves = [np.asarray(r["equity_curve"], dtype=float) for r in results if r.get("equity_curve")]

    fig = Figure(figsize=(12, 8), dpi=DPI, facecolor=BACKGROUND)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 3, height_ratios=(1.3, 1), hspace=0.5, wspace=0.3)

    ax_fan = fig.add_subplot(gs[0, :])
    ax_fan.set_facecolor(BACKGROUND)
    if curves:
        max_len = max(len(c) for c in curves)
        for c in curves:
            ax_fan.plot(np.arange(len(c)), c, color=ACTUAL, alpha=0.08, linewidth=1, zorder=2)
        # Shorter curves (a run with fewer closed trades) hold their last value flat so every
        # curve reaches max_len before the median is taken - no run is cut off mid-chart.
        padded = np.array([np.pad(c, (0, max_len - len(c)), mode="edge") for c in curves])
        ax_fan.plot(np.arange(max_len), np.median(padded, axis=0), color=TEXT, linewidth=2,
                    label="mediană", zorder=4)
        ax_fan.axhline(1.0, color=GRID, linestyle=":", linewidth=1, zorder=1)
        legend = ax_fan.legend(loc="upper left", facecolor=BACKGROUND, edgecolor=GRID, fontsize=9)
        for text in legend.get_texts():
            text.set_color(TEXT)
    else:
        ax_fan.text(0.5, 0.5, "fără curbe de echitate în rezultate", transform=ax_fan.transAxes,
                    ha="center", va="center", color=TEXT, alpha=0.5, fontsize=10)
    ax_fan.set_title("Curba de echitate, fiecare rulare Monte Carlo", color=TEXT, fontsize=12, fontweight="bold")
    ax_fan.set_xlabel("tranzacție #", color=TEXT, fontsize=9)
    ax_fan.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
    ax_fan.tick_params(colors=TEXT, labelsize=9)
    for spine in ax_fan.spines.values():
        spine.set_color(GRID)

    panels = (
        (returns, "Return final (%)", ACTUAL, False),
        (drawdowns, "Drawdown maxim (%)", ERROR, True),
        (sharpes, sharpe_key.capitalize(), PREDICTED, False),
    )
    for i, (values, label, color, check_bimodal) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(BACKGROUND)
        if len(values):
            ax.hist(values, bins=min(30, max(8, len(values) // 8)), color=color, alpha=0.7,
                    edgecolor=BACKGROUND, density=True)
            ax.axvline(float(np.median(values)), color=TEXT, linestyle="--", linewidth=1.2)
            if np.std(values) > 0:
                mu, sigma = norm.fit(values)
                xs = np.linspace(values.min(), values.max(), 200)
                ax.plot(xs, norm.pdf(xs, mu, sigma), color=TEXT, linewidth=1.3, alpha=0.9)
        ax.set_title(label, color=TEXT, fontsize=10.5, fontweight="bold")
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.6)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if check_bimodal and _is_bimodal(values):
            ax.text(0.5, 0.95, "⚠ distribuție bimodală", transform=ax.transAxes, ha="center", va="top",
                     color=HIGHLIGHT, fontsize=9.5, fontweight="bold")

    fig.subplots_adjust(top=0.90, bottom=0.08)
    fig.text(0.04, 0.965, title, color=TEXT, fontsize=14, fontweight="bold", va="top")
    fig.text(0.04, 0.015, f"{len(results)} rulări Monte Carlo, setările nemodificate (doar zgomot pe preț)",
             color=TEXT, alpha=0.7, fontsize=9, va="bottom")
    return fig


def monte_carlo_report_path(name: str) -> Path:
    return Path(config.EVAL_SNAPSHOTS_DIR) / "monte_carlo" / f"{name}.png"


def render_monte_carlo_report(results: list[dict], title: str, name: str, **kwargs) -> Path:
    return _save(draw_monte_carlo_report(results, title, **kwargs), monte_carlo_report_path(name))
