"""End-of-day overview: every prediction chart of a day in one image.

Runs at 00:05 UTC for the day that just ended (see evaluation/jobs.py) and,
at startup, for any past day that has charts but no overview yet (the app
wasn't running at 00:05). Output, per day:

    data/daily/YYYY-MM-DD_overview.png   grid of all charts + summary header
    data/daily/YYYY-MM-DD.gif            optional, one frame per prediction
    data/daily/YYYY-MM-DD.pdf            optional, summary page + one page per chart

Charts are ordered by interval (15m, 1h, 4h, 1d, 1w), then by creation time.
"""
from __future__ import annotations

import logging
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import config
from evaluation import charts, storage
from evaluation.storage import INTERVAL_ORDER

logger = logging.getLogger(__name__)

THUMB_SIZE = (500, 300)
GAP = 12
HEADER_HEIGHT = 230
MAX_COLUMNS = 12
GIF_FRAME_SIZE = (800, 480)
GIF_FRAME_MS = 1200

_BG = charts.BACKGROUND
_TEXT = charts.TEXT
_MUTED = charts.GRID
_GOOD = charts.ACTUAL
_BAD = charts.ERROR

_DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # DejaVu Sans ships with matplotlib, so it's always there and has the
    # Romanian diacritics PIL's built-in bitmap font lacks.
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def overview_path(day: date) -> Path:
    return Path(config.EVAL_DAILY_DIR) / f"{day.isoformat()}_overview.png"


def gif_path(day: date) -> Path:
    return Path(config.EVAL_DAILY_DIR) / f"{day.isoformat()}.gif"


def pdf_path(day: date) -> Path:
    return Path(config.EVAL_DAILY_DIR) / f"{day.isoformat()}.pdf"


def day_chart_files(day: date) -> list[Path]:
    """This day's chart PNGs, ordered by interval then creation time (both
    read back from the file name: {interval}_{model}_{YYYYmmddTHHMMSSZ}.png)."""
    folder = charts.by_date_dir(config.EVAL_CHARTS_DIR, day.isoformat())
    files = [f for f in folder.glob("*.png") if not f.name.endswith(".tmp.png")] if folder.is_dir() else []

    def key(f: Path):
        interval = f.stem.split("_", 1)[0]
        created = f.stem.rsplit("_", 1)[-1]
        rank = INTERVAL_ORDER.index(interval) if interval in INTERVAL_ORDER else len(INTERVAL_ORDER)
        return rank, created, f.name

    return sorted(files, key=key)


def day_stats(day: date) -> dict:
    preds = storage.completed_resolved_on(day)
    errors = [abs(p["pct_error"]) for p in preds if p["pct_error"] is not None]
    hits = [p["direction_correct"] for p in preds if p["direction_correct"] is not None]
    by_model: dict[str, list[float]] = {}
    for p in preds:
        if p["pct_error"] is not None:
            by_model.setdefault(p["model_name"], []).append(abs(p["pct_error"]))
    model_errors = sorted(((sum(v) / len(v), k) for k, v in by_model.items()))
    return {
        "date": day.isoformat(),
        "predictions": len(preds),
        "mean_abs_pct_error": round(sum(errors) / len(errors), 3) if errors else None,
        "direction_accuracy_pct": round(sum(hits) / len(hits) * 100, 1) if hits else None,
        "best_model": {"model_name": model_errors[0][1], "mean_abs_pct_error": round(model_errors[0][0], 3)} if model_errors else None,
        "worst_model": {"model_name": model_errors[-1][1], "mean_abs_pct_error": round(model_errors[-1][0], 3)} if model_errors else None,
    }


def _fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"


_HEADER_PAD = 40
_ITEM_GAP = 70


def _header_items(stats: dict, chart_count: int) -> list[tuple[str, str, str]]:
    items = [
        ("Predicții", str(stats["predictions"] or chart_count), _TEXT),
        ("Eroare medie", _fmt_pct(stats["mean_abs_pct_error"]), _BAD),
        ("Acuratețe direcție", "—" if stats["direction_accuracy_pct"] is None else f"{stats['direction_accuracy_pct']:.1f}%", _GOOD),
    ]
    best, worst = stats["best_model"], stats["worst_model"]
    if best:
        items.append(("Cel mai bun model", f"{best['model_name']} ({_fmt_pct(best['mean_abs_pct_error'])})", _GOOD))
    if worst and worst["model_name"] != best["model_name"]:
        items.append(("Cel mai slab model", f"{worst['model_name']} ({_fmt_pct(worst['mean_abs_pct_error'])})", _BAD))
    return items


def _item_widths(items) -> list[int]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_font, value_font = _font(22), _font(34, bold=True)
    return [
        int(max(draw.textlength(label, font=label_font), draw.textlength(value, font=value_font)))
        for label, value, _ in items
    ]


def _header_width(stats: dict, chart_count: int) -> int:
    """Width the summary row needs, so a day with only a few charts (narrow
    grid) still gets an image wide enough for its header text."""
    widths = _item_widths(_header_items(stats, chart_count))
    return 2 * _HEADER_PAD + sum(widths) + _ITEM_GAP * (len(widths) - 1)


def _draw_header(img: Image.Image, day: date, stats: dict, chart_count: int) -> None:
    draw = ImageDraw.Draw(img)
    draw.text((_HEADER_PAD, 30), f"Evaluare predicții · {day.isoformat()} (UTC)", fill=_TEXT, font=_font(48, bold=True))
    items = _header_items(stats, chart_count)
    label_font, value_font = _font(22), _font(34, bold=True)
    x = _HEADER_PAD
    for (label, value, color), width in zip(items, _item_widths(items)):
        draw.text((x, 115), label, fill=_TEXT, font=label_font)
        draw.text((x, 148), value, fill=color, font=value_font)
        x += width + _ITEM_GAP
    draw.line((_HEADER_PAD, HEADER_HEIGHT - 12, img.width - _HEADER_PAD, HEADER_HEIGHT - 12), fill=_MUTED, width=2)


def _save_atomic(save, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    save(tmp)
    os.replace(tmp, path)


def build_daily_report(day: date, force: bool = False) -> dict | None:
    """Build the overview (and optional GIF/PDF) for `day`. Returns the
    written paths, or None when that day has no charts. Skips the work if
    the overview already exists, unless `force`."""
    files = day_chart_files(day)
    if not files:
        return None
    out = {"overview": overview_path(day)}
    if out["overview"].exists() and not force:
        return out

    stats = day_stats(day)
    n = len(files)
    cols = min(n, MAX_COLUMNS, max(3, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    # Smaller thumbnails on busy days (visual history logs every change of
    # the 24-step forecast), so the overview stays a sane size.
    tw, th = THUMB_SIZE if n <= 60 else (400, 240) if n <= 200 else (320, 192)
    width = max(cols * tw + (cols + 1) * GAP, 1600, _header_width(stats, n))
    height = HEADER_HEIGHT + rows * th + (rows + 1) * GAP

    overview = Image.new("RGB", (width, height), _BG)
    _draw_header(overview, day, stats, n)
    for i, f in enumerate(files):
        with Image.open(f) as chart:
            thumb = chart.convert("RGB").resize((tw, th), Image.LANCZOS)
        r, c = divmod(i, cols)
        overview.paste(thumb, (GAP + c * (tw + GAP), HEADER_HEIGHT + GAP + r * (th + GAP)))
    _save_atomic(lambda p: overview.save(p, "PNG", optimize=True), out["overview"])

    if config.EVAL_DAILY_GIF:
        frames = []
        for f in files:
            with Image.open(f) as chart:
                frames.append(chart.convert("RGB").resize(GIF_FRAME_SIZE, Image.LANCZOS).quantize(colors=128))
        out["gif"] = gif_path(day)
        _save_atomic(
            lambda p: frames[0].save(p, "GIF", save_all=True, append_images=frames[1:], duration=GIF_FRAME_MS, loop=0),
            out["gif"],
        )

    if config.EVAL_DAILY_PDF:
        cover = Image.new("RGB", (max(1600, _header_width(stats, n)), HEADER_HEIGHT + 40), _BG)
        _draw_header(cover, day, stats, n)
        pages = []
        for f in files:
            with Image.open(f) as chart:
                pages.append(chart.convert("RGB"))
        out["pdf"] = pdf_path(day)
        _save_atomic(lambda p: cover.save(p, "PDF", save_all=True, append_images=pages, resolution=100), out["pdf"])

    logger.info("Daily evaluation report for %s: %d charts -> %s", day, n, out["overview"])
    return out


def _ensure_charts_for(day: date) -> None:
    charts.generate_missing_charts([p["id"] for p in storage.completed_resolved_on(day)])


def daily_report_job(now: datetime | None = None) -> dict | None:
    """00:05 UTC job: report on the day that just ended."""
    yesterday = (now or datetime.now(timezone.utc)).date() - timedelta(days=1)
    _ensure_charts_for(yesterday)
    return build_daily_report(yesterday)


def recover_missing_days(now: datetime | None = None) -> list[date]:
    """Build the report for every past day (before today, UTC) that has
    completed predictions or charts but no overview yet - e.g. the app was
    down at 00:05. Today is left alone: it isn't over."""
    today = (now or datetime.now(timezone.utc)).date()
    days = set(storage.completed_resolved_days())
    charts_dir = Path(config.EVAL_CHARTS_DIR) / charts.BY_DATE_DIR
    if charts_dir.is_dir():
        days.update(date.fromisoformat(d.name) for d in charts_dir.iterdir() if d.is_dir() and _DAY_DIR_RE.match(d.name))
    built = []
    for day in sorted(d for d in days if d < today):
        if overview_path(day).exists():
            continue
        try:
            _ensure_charts_for(day)
            if build_daily_report(day):
                built.append(day)
        except Exception:
            logger.exception("Failed to recover daily evaluation report for %s", day)
    return built


def list_report_days() -> list[str]:
    folder = Path(config.EVAL_DAILY_DIR)
    if not folder.is_dir():
        return []
    return sorted((f.name[:10] for f in folder.glob("*_overview.png")), reverse=True)
