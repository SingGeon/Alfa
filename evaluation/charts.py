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
