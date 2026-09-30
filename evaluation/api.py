"""HTTP API for the evaluation module, mounted at /api/evaluation.

    GET /predictions            table rows (filters, sort, pagination) + per-model summary
    GET /predictions.csv        same filters/sort, every matching row, as CSV
    GET /predictions/<id>/chart.png      final chart (completed only)
    GET /predictions/<id>/snapshot.png   visual snapshot, current state (any status)
    GET /models                 model names, for the filter dropdown
    GET /days                   dates that have a daily overview (newest first)
    GET /days/<YYYY-MM-DD>      that day's stats + which files exist
    GET /days/<YYYY-MM-DD>/overview.png | .gif | .pdf

Filters: interval, model, status, horizon_steps, date_from, date_to
(YYYY-MM-DD, UTC, inclusive). Sort: sort=<column>&order=asc|desc. Pagination: page, page_size.
All timestamps in responses are UTC ISO-8601 ("...Z"); the browser converts
them to the viewer's own timezone.
"""
from __future__ import annotations

import csv
import io
from datetime import date

from flask import Blueprint, Response, abort, jsonify, request, send_file

from evaluation import charts, daily_report, storage

evaluation_bp = Blueprint("evaluation", __name__)

MAX_PAGE_SIZE = 500

CSV_COLUMNS = [
    "id", "created_at", "interval", "model_name", "price_at_prediction", "horizon_steps", "target_time",
    "predicted_final_price", "confidence", "signal", "actual_final_price", "abs_error", "pct_error",
    "direction_correct", "status", "completed_at",
]


class _BadRequest(ValueError):
    pass


@evaluation_bp.errorhandler(_BadRequest)
def _bad_request(exc):
    return jsonify({"error": str(exc)}), 400


def _parse_date(name: str) -> date | None:
    value = request.args.get(name)
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise _BadRequest(f"{name} must be YYYY-MM-DD") from None


def _filters() -> dict:
    interval = request.args.get("interval") or None
    if interval and interval not in storage.INTERVAL_ORDER:
        raise _BadRequest(f"interval must be one of {list(storage.INTERVAL_ORDER)}")
    status = request.args.get("status") or None
    if status and status not in ("pending", "completed", "expired"):
        raise _BadRequest("status must be pending, completed or expired")
    horizon = request.args.get("horizon_steps") or None
    if horizon is not None and not horizon.isdigit():
        raise _BadRequest("horizon_steps must be an integer")
    return {
        "interval": interval,
        "horizon_steps": int(horizon) if horizon else None,
        "model": request.args.get("model") or None,
        "status": status,
        "date_from": _parse_date("date_from"),
        "date_to": _parse_date("date_to"),
    }


def _sort() -> tuple[str, str]:
    sort = request.args.get("sort", "created_at")
    if sort not in storage.SORTABLE_COLUMNS:
        raise _BadRequest(f"sort must be one of {sorted(storage.SORTABLE_COLUMNS)}")
    return sort, request.args.get("order", "desc")


def _public(row: dict) -> dict:
    row = dict(row)
    row.pop("history_path", None)  # only needed to draw the image; bulky
    row["has_chart"] = row["status"] == "completed"
    row["snapshot_url"] = f"/api/evaluation/predictions/{row['id']}/snapshot.png"
    return row


@evaluation_bp.get("/predictions")
def predictions():
    filters = _filters()
    sort, order = _sort()
    try:
        page = max(int(request.args.get("page", 1)), 1)
        page_size = min(max(int(request.args.get("page_size", 50)), 1), MAX_PAGE_SIZE)
    except ValueError:
        raise _BadRequest("page and page_size must be integers") from None
    rows, total = storage.query_predictions(filters, sort, order, page, page_size)
    return jsonify({
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": max(1, -(-total // page_size)),
        "sort": sort,
        "order": order,
        "summary": storage.summary_by_model(filters),
        "rows": [_public(r) for r in rows],
    })


@evaluation_bp.get("/predictions.csv")
def predictions_csv():
    sort, order = _sort()
    rows, _ = storage.query_predictions(_filters(), sort, order, page=1, page_size=0)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return Response(
        "﻿" + buf.getvalue(),  # BOM so Excel reads the diacritics as UTF-8
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@evaluation_bp.get("/predictions/<int:prediction_id>/chart.png")
def prediction_chart(prediction_id: int):
    pred = storage.get_prediction(prediction_id)
    if not pred or pred["status"] != "completed":
        abort(404)
    path = charts.render_prediction_chart(pred)  # no-op if it already exists
    return send_file(path, mimetype="image/png", max_age=86400)


@evaluation_bp.get("/predictions/<int:prediction_id>/snapshot.png")
def prediction_snapshot(prediction_id: int):
    pred = storage.get_prediction(prediction_id)
    if not pred:
        abort(404)
    path = charts.snapshot_path(pred)
    if not path.exists():
        path = charts.render_snapshot(pred)
    # Redrawn as real candles arrive - never let the browser keep a stale one.
    return send_file(path, mimetype="image/png", max_age=0)


@evaluation_bp.get("/models")
def models():
    return jsonify({"models": storage.list_models(), "intervals": list(storage.INTERVAL_ORDER)})


@evaluation_bp.get("/days")
def days():
    return jsonify({"days": daily_report.list_report_days()})


def _day(day_str: str) -> date:
    try:
        return date.fromisoformat(day_str)
    except ValueError:
        abort(404)


@evaluation_bp.get("/days/<day_str>")
def day_detail(day_str: str):
    day = _day(day_str)
    if not daily_report.overview_path(day).exists():
        abort(404)
    return jsonify({
        **daily_report.day_stats(day),
        "charts": len(daily_report.day_chart_files(day)),
        "gif": daily_report.gif_path(day).exists(),
        "pdf": daily_report.pdf_path(day).exists(),
    })


@evaluation_bp.get("/days/<day_str>/overview.<ext>")
def day_file(day_str: str, ext: str):
    day = _day(day_str)
    paths = {
        "png": (daily_report.overview_path(day), "image/png"),
        "gif": (daily_report.gif_path(day), "image/gif"),
        "pdf": (daily_report.pdf_path(day), "application/pdf"),
    }
    if ext not in paths or not paths[ext][0].exists():
        abort(404)
    path, mimetype = paths[ext]
    return send_file(path, mimetype=mimetype, max_age=3600)
