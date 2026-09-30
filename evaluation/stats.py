"""Accuracy statistics written next to the chart PNGs, one stats.json per
folder level of the layout (see evaluation/charts.py), so every way of
browsing the charts has its numbers right there:

    data/charts/stats.json                         overall + per model/interval/day
    data/charts/stats.csv                          one row per (model, interval)
    data/charts/{model}/stats.json                 that model: per interval, per day
    data/charts/{model}/{interval}/stats.json      that model+interval: per day
    data/charts/by-date/YYYY-MM-DD/stats.json      that day: per model, per model+interval

"day" is the day a prediction resolves (the same day its chart is filed
under). Counts include pending/expired predictions; errors and direction
accuracy only completed ones. A file is only rewritten when its content
changed.
"""
from __future__ import annotations

import csv
import io
import json
import os
from collections import defaultdict
from pathlib import Path

import config
from evaluation import storage
from evaluation.charts import BY_DATE_DIR, model_dir_name


def _strip(row: dict, *keys: str) -> dict:
    return {k: v for k, v in row.items() if k not in keys}


def compute_stats(filters: dict | None = None) -> dict:
    def g(*keys):
        return storage.grouped_stats(keys, filters)

    by_model = g("model")
    by_interval = g("interval")
    by_day = g("day")
    by_model_interval = g("model", "interval")
    by_model_day = g("model", "day")
    by_model_interval_day = g("model", "interval", "day")
    overall = g()
    return {
        "overall": overall[0] if overall else None,
        "by_model": by_model,
        "by_interval": by_interval,
        "by_day": by_day,
        "by_model_interval": by_model_interval,
        "by_model_day": by_model_day,
        "by_model_interval_day": by_model_interval_day,
    }


def _write_if_changed(path: Path, text: str) -> bool:
    data = text.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return False
    os.makedirs(path.parent, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def _json(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def _csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return buf.getvalue()


def write_stats_files(root: str | None = None) -> int:
    """Write every stats file under the charts tree; returns how many changed."""
    base = Path(root or config.EVAL_CHARTS_DIR)
    s = compute_stats()
    files: dict[Path, str] = {
        base / "stats.json": _json(s),
        base / "stats.csv": _csv(s["by_model_interval"]),
    }

    per_model: dict[str, dict] = defaultdict(lambda: {"by_interval": [], "by_day": []})
    per_model_interval: dict[tuple[str, str], list] = defaultdict(list)
    per_day: dict[str, dict] = defaultdict(lambda: {"by_model": [], "by_model_interval": []})
    for row in s["by_model"]:
        per_model[row["model"]]["overall"] = _strip(row, "model")
    for row in s["by_model_interval"]:
        per_model[row["model"]]["by_interval"].append(_strip(row, "model"))
    for row in s["by_model_day"]:
        per_model[row["model"]]["by_day"].append(_strip(row, "model"))
        per_day[row["day"]]["by_model"].append(_strip(row, "day"))
    for row in s["by_model_interval_day"]:
        per_model_interval[(row["model"], row["interval"])].append(_strip(row, "model", "interval"))
        per_day[row["day"]]["by_model_interval"].append(_strip(row, "day"))
    for row in s["by_day"]:
        per_day[row["day"]]["overall"] = _strip(row, "day")

    for model, data in per_model.items():
        files[base / model_dir_name(model) / "stats.json"] = _json({"model": model, **data})
    for row in s["by_model_interval"]:
        model, interval = row["model"], row["interval"]
        files[base / model_dir_name(model) / interval / "stats.json"] = _json({
            "model": model, "interval": interval,
            "overall": _strip(row, "model", "interval"),
            "by_day": per_model_interval[(model, interval)],
        })
    for day, data in per_day.items():
        files[base / BY_DATE_DIR / day / "stats.json"] = _json({"day": day, **data})

    return sum(_write_if_changed(path, text) for path, text in files.items())
