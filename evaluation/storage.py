"""SQLite storage for the prediction evaluation log (schema: schema.sql).

Deliberately separate from the Mongo `predictions` collection the dashboard
already uses (database/repository.py): this is an append-only audit log of
what each model said and what really happened, one row per *distinct*
prediction, which is what the evaluation table, the per-prediction charts
and the daily overview are built from.

Every call opens its own short-lived connection - sqlite3 connections
aren't meant to be shared across threads, and this module is used from
Flask request threads and APScheduler worker threads at the same time.
WAL mode lets those readers and the single writer run concurrently.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import config
from ml.price_predictor import _INTERVAL_TIMEDELTA

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

SIGNAL_LABELS = {"buy": "Cumpără", "sell": "Vinde", "wait": "Așteaptă"}

INTERVAL_ORDER = ("15m", "1h", "4h", "1d", "1w")

SORTABLE_COLUMNS = {
    "created_at", "interval", "model_name", "price_at_prediction", "predicted_final_price",
    "actual_final_price", "pct_error", "abs_error", "direction_correct", "status", "confidence",
    "horizon_steps", "target_time",
}

_initialized_paths: set[str] = set()
_init_lock = threading.Lock()
# Serializes log_prediction's check-then-insert (see its docstring).
_write_lock = threading.Lock()


def to_iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime(ISO_FORMAT)


def parse_iso(value: str) -> datetime:
    """Parse our own ISO_FORMAT as well as any isoformat() string (the
    prediction timestamps coming out of api.services use "+00:00")."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def interval_delta(interval: str) -> timedelta:
    return _INTERVAL_TIMEDELTA[interval]


def _db_path() -> str:
    # Read at call time, not import time, so tests can point it elsewhere.
    return config.EVAL_DB_PATH


@contextmanager
def connect():
    path = _db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if path not in _initialized_paths:
            with _init_lock:
                if path not in _initialized_paths:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
                    _migrate(conn)
                    _initialized_paths.add(path)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Columns added after a database may already exist on disk."""
    columns = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    if "history_path" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN history_path TEXT NOT NULL DEFAULT '[]'")


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["predicted_path"] = json.loads(d["predicted_path"] or "[]")
    d["actual_path"] = json.loads(d["actual_path"] or "[]")
    d["history_path"] = json.loads(d.get("history_path") or "[]")
    if d.get("direction_correct") is not None:
        d["direction_correct"] = bool(d["direction_correct"])
    return d


def log_prediction(
    *,
    interval: str,
    model_name: str,
    price_at_prediction: float,
    predictions: list[dict],
    confidence: float | None,
    signal: str | None,
    history: list[dict] | None = None,
    created_at: datetime | None = None,
) -> int | None:
    """Record one prediction; returns its id, or None if it was skipped.

    `predictions` is api.services.run_prediction's own list of
    {"timestamp", "predicted_price", "lower", "upper"}. `signal` is
    "buy" | "sell" | "wait" (ml.trading_signal) and is stored with its
    Romanian label.

    `history` is the real candles the model saw last ({"timestamp", "open",
    "high", "low", "close"}), for the visual snapshot.

    De-duplication:
    - the UNIQUE (interval, model_name, created_at) key (INSERT OR IGNORE);
    - visual-history horizons (config.EVAL_SNAPSHOT_STEPS, 24): a new row
      only when the forecast differs from the last one logged for the same
      (interval, model_name, horizon_steps) - i.e. every time the model
      retrains into a different answer, never for a repeat of the same one;
    - every other horizon: at most one row per (interval, model_name,
      horizon_steps) per candle of that interval.
    The dashboard re-requests /api/predict every 30s while the model
    underneath only retrains every ~15min, so without this one real
    forecast would be logged hundreds of times a day.
    The check and the insert run under one lock so two concurrent requests
    can't both pass the check.
    """
    if not predictions:
        return None
    created_at = created_at or datetime.now(timezone.utc)
    created_iso = to_iso(created_at)
    delta = interval_delta(interval)
    path = [
        {
            "timestamp": to_iso(parse_iso(p["timestamp"])),
            "price": float(p["predicted_price"]),
            "lower": float(p["lower"]) if p.get("lower") is not None else None,
            "upper": float(p["upper"]) if p.get("upper") is not None else None,
        }
        for p in predictions
    ]
    target_time = parse_iso(path[-1]["timestamp"])

    with _write_lock, connect() as conn:
        if len(path) == config.EVAL_SNAPSHOT_STEPS:
            last = conn.execute(
                "SELECT predicted_path FROM predictions WHERE interval = ? AND model_name = ? AND horizon_steps = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (interval, model_name, len(path)),
            ).fetchone()
            if last and _same_forecast(json.loads(last[0]), path):
                return None
        else:
            recent = conn.execute(
                "SELECT 1 FROM predictions WHERE interval = ? AND model_name = ? AND horizon_steps = ?"
                " AND created_at > ? LIMIT 1",
                (interval, model_name, len(path), to_iso(created_at - delta)),
            ).fetchone()
            if recent:
                return None
        cur = conn.execute(
            """INSERT OR IGNORE INTO predictions (
                   created_at, interval, model_name, price_at_prediction, horizon_steps,
                   target_time, resolved_at, predicted_path, predicted_final_price,
                   confidence, signal, history_path
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                created_iso, interval, model_name, float(price_at_prediction), len(path),
                to_iso(target_time), to_iso(target_time + delta), json.dumps(path),
                path[-1]["price"], confidence, SIGNAL_LABELS.get(signal, signal),
                json.dumps(history or []),
            ),
        )
        return cur.lastrowid if cur.rowcount else None


def _same_forecast(a: list[dict], b: list[dict]) -> bool:
    """Same timestamps and the same prices to the cent."""
    return len(a) == len(b) and all(
        p["timestamp"] == q["timestamp"] and round(p["price"], 2) == round(q["price"], 2) for p, q in zip(a, b)
    )


def get_prediction(prediction_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_pending() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE status = 'pending' ORDER BY created_at").fetchall()
    return [_row_to_dict(r) for r in rows]


def update_actual_path(prediction_id: int, actual_path: list[dict]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE predictions SET actual_path = ? WHERE id = ? AND status = 'pending'",
            (json.dumps(actual_path), prediction_id),
        )


def complete_prediction(
    prediction_id: int, *, actual_path: list[dict], actual_final_price: float, abs_error: float,
    pct_error: float, direction_correct: bool, completed_at: datetime | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE predictions SET actual_path = ?, actual_final_price = ?, abs_error = ?,
                   pct_error = ?, direction_correct = ?, status = 'completed', completed_at = ?
               WHERE id = ? AND status = 'pending'""",
            (
                json.dumps(actual_path), actual_final_price, abs_error, pct_error, int(direction_correct),
                to_iso(completed_at or datetime.now(timezone.utc)), prediction_id,
            ),
        )


def expire_prediction(prediction_id: int, actual_path: list[dict]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE predictions SET actual_path = ?, status = 'expired', completed_at = ?"
            " WHERE id = ? AND status = 'pending'",
            (json.dumps(actual_path), to_iso(datetime.now(timezone.utc)), prediction_id),
        )


def _where(filters: dict) -> tuple[str, list]:
    clauses, params = [], []
    if filters.get("interval"):
        clauses.append("interval = ?")
        params.append(filters["interval"])
    if filters.get("model"):
        clauses.append("model_name = ?")
        params.append(filters["model"])
    if filters.get("horizon_steps"):
        clauses.append("horizon_steps = ?")
        params.append(int(filters["horizon_steps"]))
    if filters.get("status"):
        clauses.append("status = ?")
        params.append(filters["status"])
    if filters.get("date_from"):
        clauses.append("created_at >= ?")
        params.append(to_iso(datetime.combine(filters["date_from"], datetime.min.time(), timezone.utc)))
    if filters.get("date_to"):
        # Inclusive of the whole date_to day.
        clauses.append("created_at < ?")
        params.append(to_iso(datetime.combine(filters["date_to"] + timedelta(days=1), datetime.min.time(), timezone.utc)))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def query_predictions(
    filters: dict | None = None, sort: str = "created_at", order: str = "desc",
    page: int = 1, page_size: int = 50,
) -> tuple[list[dict], int]:
    """(rows for this page, total matching rows). `sort` must be one of
    SORTABLE_COLUMNS (anything else falls back to created_at); NULLs always
    sort last so pending rows don't crowd the top of an error sort."""
    filters = filters or {}
    sort = sort if sort in SORTABLE_COLUMNS else "created_at"
    direction = "ASC" if str(order).lower() == "asc" else "DESC"
    where, params = _where(filters)
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM predictions{where}", params).fetchone()[0]
        query = f"SELECT * FROM predictions{where} ORDER BY {sort} IS NULL, {sort} {direction}, id {direction}"
        if page_size:
            query += " LIMIT ? OFFSET ?"
            params = [*params, int(page_size), (max(int(page), 1) - 1) * int(page_size)]
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows], total


def summary_by_model(filters: dict | None = None) -> list[dict]:
    """Per model: prediction count, completed count, mean |error %| and
    direction accuracy % over completed predictions (None when nothing has
    completed yet)."""
    where, params = _where(filters or {})
    with connect() as conn:
        rows = conn.execute(
            f"""SELECT model_name,
                       COUNT(*) AS predictions,
                       SUM(status = 'completed') AS completed,
                       SUM(status = 'pending') AS pending,
                       AVG(CASE WHEN status = 'completed' THEN ABS(pct_error) END) AS mean_abs_pct_error,
                       AVG(CASE WHEN status = 'completed' THEN direction_correct END) AS direction_accuracy
                FROM predictions{where}
                GROUP BY model_name ORDER BY model_name""",
            params,
        ).fetchall()
    return [
        {
            "model_name": r["model_name"],
            "predictions": r["predictions"],
            "completed": r["completed"] or 0,
            "pending": r["pending"] or 0,
            "mean_abs_pct_error": round(r["mean_abs_pct_error"], 3) if r["mean_abs_pct_error"] is not None else None,
            "direction_accuracy_pct": round(r["direction_accuracy"] * 100, 1) if r["direction_accuracy"] is not None else None,
        }
        for r in rows
    ]


def list_models() -> list[str]:
    with connect() as conn:
        return [r[0] for r in conn.execute("SELECT DISTINCT model_name FROM predictions ORDER BY model_name")]


def completed_resolved_on(day: date) -> list[dict]:
    """Completed predictions whose outcome became known on `day` (UTC) -
    exactly the ones whose charts live in that day's chart folder."""
    start = datetime.combine(day, datetime.min.time(), timezone.utc)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE status = 'completed' AND resolved_at >= ? AND resolved_at < ?"
            " ORDER BY created_at",
            (to_iso(start), to_iso(start + timedelta(days=1))),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def completed_resolved_days() -> list[date]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT substr(resolved_at, 1, 10) FROM predictions WHERE status = 'completed' ORDER BY 1"
        ).fetchall()
    return [date.fromisoformat(r[0]) for r in rows]


def all_completed() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM predictions WHERE status = 'completed' ORDER BY created_at").fetchall()
    return [_row_to_dict(r) for r in rows]
