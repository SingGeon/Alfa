"""Prediction images, drawn like the dashboard chart: the real candles the
model saw, the AI forecast (dashed line + confidence band) after them, and
the real candles that actually followed drawn *on top of* the forecast, so
the two can be compared at a glance.

Two files per prediction:
- the visual snapshot, data/snapshots/YYYY-MM-DD/{interval}_{model}_{created_at}.png
  (date = day the prediction was made). Drawn as soon as the prediction is
  logged and redrawn every time new real candles arrive, so it always shows
  the latest state - this is the visual history.
- the final chart, data/charts/YYYY-MM-DD/{same name}.png, where the date is
  the day the prediction *resolved* (its last forecast candle closed, see
  schema.sql) - so by the time the 00:05 UTC daily report runs, every chart
  for the day that just ended already exists. Idempotent: an existing file
  is never redrawn.

Uses matplotlib's object API on an Agg canvas (no pyplot, no GUI) - pyplot
keeps global state that isn't safe to touch from the scheduler's worker
threads and Flask's request threads at the same time; a standalone Figure is.
"""
from __future__ import annotations

import logging
import os
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

FIG_SIZE = (10, 6)
DPI = 100


def _file_name(pred: dict) -> str:
    created = parse_iso(pred["created_at"]).strftime("%Y%m%dT%H%M%SZ")
    model = "".join(c if c.isalnum() or c in "-." else "-" for c in pred["model_name"])
    return f"{pred['interval']}_{model}_{created}.png"


def chart_path(pred: dict) -> Path:
    return Path(config.EVAL_CHARTS_DIR) / pred["resolved_at"][:10] / _file_name(pred)


def snapshot_path(pred: dict) -> Path:
    return Path(config.EVAL_SNAPSHOTS_DIR) / pred["created_at"][:10] / _file_name(pred)


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

    ax.set_title(
        f"{pred['model_name']} · {pred['interval']} · {pred['horizon_steps']} pași",
        color=TEXT, fontsize=14, fontweight="bold", loc="left",
    )
    fig.text(
        0.125, 0.005, f"Creată {created.strftime('%Y-%m-%d %H:%M')} UTC · preț la predicție ${anchor_price:,.2f}",
        color=TEXT, alpha=0.7, fontsize=9, va="bottom",
    )

    if pred.get("status") == "completed":
        direction = "corectă" if pred["direction_correct"] else "greșită"
        box_text, box_color = f"Eroare {pred['pct_error']:+.2f}%\nDirecție {direction}", ERROR
    elif pred.get("status") == "expired":
        box_text, box_color = "Expirată\n(fără preț real)", GRID
    else:
        box_text, box_color = f"În desfășurare\n{len(actual)}/{len(predicted)} candele reale", TEXT
    ax.text(
        0.99, 0.97, box_text, transform=ax.transAxes, ha="right", va="top", color=box_color, fontsize=12,
        fontweight="bold", zorder=10,
        bbox={"facecolor": BACKGROUND, "edgecolor": box_color, "boxstyle": "round,pad=0.4", "alpha": 0.9},
    )

    first = parse_iso(history[0]["timestamp"]) if history else anchor_time
    ax.set_xlim(first - delta, end_time + delta)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M", tz=created.tzinfo))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    legend = ax.legend(loc="upper left", facecolor=BACKGROUND, edgecolor=GRID, fontsize=9)
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
    return _save(draw_prediction(pred), snapshot_path(pred))


def render_prediction_chart(pred: dict, force: bool = False) -> Path | None:
    """Draw the final chart for one completed prediction; returns its path
    (None if the prediction isn't completed yet)."""
    if pred.get("status") != "completed":
        return None
    path = chart_path(pred)
    if path.exists() and not force:
        return path
    return _save(draw_prediction(pred), path)


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
