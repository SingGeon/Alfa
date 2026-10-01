"""Evaluation module: error math, filling in real prices, PNG charts, the
daily overview and the /api/evaluation endpoints. Fully offline -
requests_mock stands in for Binance/CoinGecko, and conftest's
evaluation_data_dir fixture points the SQLite DB and output folders at a
per-test temp dir."""
from datetime import date, datetime, timedelta, timezone
import json
import re

import pytest
from PIL import Image

import config
from api import create_app
from database import repository
from evaluation import charts, daily_report, evaluator, stats, storage
from evaluation.storage import to_iso

BINANCE_KLINES = re.compile(r"https://api\.binance\.com/api/v3/klines")
COINGECKO_RANGE = re.compile(r"https://api\.coingecko\.com/api/v3/coins/ethereum/market_chart/range")

T0 = datetime(2026, 9, 29, 10, 0, tzinfo=timezone.utc)


def _log(created_at=T0, steps=3, interval="1h", model="sklearn-tuned", start_price=3000.0, step=10.0, anchor=None):
    """`anchor` = the last candle the model saw (defaults to created_at);
    forecast timestamps follow on from it, as in api.services."""
    delta = storage.interval_delta(interval)
    anchor = anchor or created_at
    preds = [
        {
            "timestamp": (anchor + delta * i).isoformat(),
            "predicted_price": start_price + step * i,
            "lower": start_price + step * i - 20,
            "upper": start_price + step * i + 20,
        }
        for i in range(1, steps + 1)
    ]
    return storage.log_prediction(
        interval=interval, model_name=model, price_at_prediction=start_price, predictions=preds,
        confidence=70.0, signal="buy", created_at=created_at,
    )


def _kline(open_time: datetime, close: float, interval="1h"):
    delta = storage.interval_delta(interval)
    open_ms = int(open_time.timestamp() * 1000)
    close_ms = int((open_time + delta).timestamp() * 1000) - 1
    return [open_ms, "1", "1", "1", str(close), "1", close_ms, "1", 1, "1", "1", "0"]


def _complete_one(requests_mock, closes=(3005.0, 3015.0, 3040.0)):
    pid = _log()
    requests_mock.get(BINANCE_KLINES, json=[_kline(T0 + timedelta(hours=i + 1), c) for i, c in enumerate(closes)])
    result = evaluator.complete_pending_predictions(now=T0 + timedelta(hours=10))
    assert result["completed_ids"] == [pid]
    return storage.get_prediction(pid)


# --- error math -------------------------------------------------------------

def test_compute_errors_overshoot_same_direction():
    e = evaluator.compute_errors(price_at_prediction=100, predicted_final_price=110, actual_final_price=105)
    assert e["abs_error"] == pytest.approx(5)
    assert e["pct_error"] == pytest.approx(5 / 105 * 100, rel=1e-4)
    assert e["direction_correct"] is True


def test_compute_errors_wrong_direction_is_negative_undershoot():
    e = evaluator.compute_errors(price_at_prediction=100, predicted_final_price=95, actual_final_price=104)
    assert e["pct_error"] < 0
    assert e["direction_correct"] is False


def test_compute_errors_flat_prediction_only_correct_if_flat():
    assert evaluator.compute_errors(100, 100, 100)["direction_correct"] is True
    assert evaluator.compute_errors(100, 100, 101)["direction_correct"] is False


# --- storage ----------------------------------------------------------------

def test_log_prediction_stores_fields_and_signal_label():
    pid = _log()
    row = storage.get_prediction(pid)
    assert row["status"] == "pending"
    assert row["horizon_steps"] == 3
    assert row["predicted_final_price"] == 3030.0
    assert row["target_time"] == to_iso(T0 + timedelta(hours=3))
    assert row["resolved_at"] == to_iso(T0 + timedelta(hours=4))
    assert row["signal"] == "Cumpără"
    assert row["actual_path"] == []
    assert row["created_at"].endswith("Z")


def test_log_prediction_deduplicates_within_one_candle():
    assert _log() is not None
    assert _log() is None                                     # same key
    assert _log(created_at=T0 + timedelta(minutes=20)) is None  # same 1h candle, throttled
    assert _log(created_at=T0 + timedelta(minutes=20), steps=1) is not None  # different horizon
    assert _log(created_at=T0 + timedelta(minutes=20), model="sklearn-legacy") is not None
    assert _log(created_at=T0 + timedelta(hours=1, minutes=1)) is not None  # next candle


def test_query_filters_sort_and_pagination():
    for i in range(5):
        _log(created_at=T0 + timedelta(hours=i), start_price=3000 + i)
    _log(created_at=T0, model="sklearn-legacy")
    rows, total = storage.query_predictions({"model": "sklearn-tuned"}, "price_at_prediction", "asc", page=2, page_size=2)
    assert total == 5
    assert [r["price_at_prediction"] for r in rows] == [3002, 3003]
    rows, total = storage.query_predictions({"date_from": date(2026, 9, 30)})
    assert total == 0


# --- filling in real prices -------------------------------------------------

def test_partial_fill_then_completion(requests_mock):
    pid = _log()
    # Two hours after T0: only the first forecast candle (opens T0+1h) has closed.
    requests_mock.get(BINANCE_KLINES, json=[_kline(T0 + timedelta(hours=1), 3005.0)])
    result = evaluator.complete_pending_predictions(now=T0 + timedelta(hours=2, minutes=1))
    assert result["updated"] == 1 and result["completed"] == 0
    row = storage.get_prediction(pid)
    assert row["status"] == "pending"
    assert [(p["timestamp"], p["price"]) for p in row["actual_path"]] == [(to_iso(T0 + timedelta(hours=1)), 3005.0)]

    requests_mock.get(BINANCE_KLINES, json=[_kline(T0 + timedelta(hours=2), 3015.0), _kline(T0 + timedelta(hours=3), 3020.0)])
    result = evaluator.complete_pending_predictions(now=T0 + timedelta(hours=4, minutes=1))
    assert result["completed_ids"] == [pid]
    row = storage.get_prediction(pid)
    assert row["status"] == "completed"
    assert [p["price"] for p in row["actual_path"]] == [3005.0, 3015.0, 3020.0]
    assert row["actual_final_price"] == 3020.0
    assert row["abs_error"] == pytest.approx(10.0)
    assert row["pct_error"] == pytest.approx(10 / 3020 * 100, rel=1e-3)
    assert row["direction_correct"] is True


def test_still_forming_candle_is_ignored(requests_mock):
    pid = _log()
    now = T0 + timedelta(hours=1, minutes=30)  # the T0+1h candle is still open
    requests_mock.get(BINANCE_KLINES, json=[_kline(T0 + timedelta(hours=1), 3005.0)])
    evaluator.complete_pending_predictions(now=now)
    assert storage.get_prediction(pid)["actual_path"] == []
    assert not requests_mock.called  # nothing had closed, so no API call at all


def test_binance_failure_falls_back_to_coingecko(requests_mock, monkeypatch):
    monkeypatch.setattr("data_collector.http_utils.time.sleep", lambda s: None)
    pid = _log(steps=1)
    requests_mock.get(BINANCE_KLINES, status_code=503)
    close_time = T0 + timedelta(hours=2)
    requests_mock.get(COINGECKO_RANGE, json={"prices": [
        [int((close_time - timedelta(minutes=10)).timestamp() * 1000), 2990.0],
        [int((close_time - timedelta(minutes=2)).timestamp() * 1000), 2995.0],
        [int((close_time + timedelta(minutes=3)).timestamp() * 1000), 3100.0],  # after the close - ignored
    ]})
    evaluator.complete_pending_predictions(now=T0 + timedelta(hours=3))
    row = storage.get_prediction(pid)
    assert row["status"] == "completed"
    assert row["actual_final_price"] == 2995.0
    assert row["direction_correct"] is False  # predicted up (3010), went down


def test_gap_stays_pending_then_expires(requests_mock, monkeypatch):
    monkeypatch.setattr("data_collector.http_utils.time.sleep", lambda s: None)
    pid = _log(steps=1)
    requests_mock.get(BINANCE_KLINES, json=[])  # hole in the data
    requests_mock.get(COINGECKO_RANGE, json={"prices": []})
    evaluator.complete_pending_predictions(now=T0 + timedelta(hours=3))
    assert storage.get_prediction(pid)["status"] == "pending"
    evaluator.complete_pending_predictions(now=T0 + timedelta(hours=2 + config.EVAL_EXPIRE_HOURS, minutes=1))
    assert storage.get_prediction(pid)["status"] == "expired"


# --- charts -----------------------------------------------------------------

def test_chart_png_generated_with_expected_path_and_size(requests_mock):
    pred = _complete_one(requests_mock)
    path = charts.render_prediction_chart(pred)
    assert path == charts.chart_path(pred)
    assert path.parent.name == "2026-09-29"  # resolved day
    assert path.parent.parent.name == "1h" and path.parent.parent.parent.name == "sklearn-tuned"
    assert path.name == "1h_sklearn-tuned_20260929T100000Z.png"
    by_date = charts.by_date_dir(config.EVAL_CHARTS_DIR, "2026-09-29") / path.name
    assert by_date.resolve() == path.resolve()
    with Image.open(path) as img:
        assert img.format == "PNG"
        assert img.size == (1000, 600)
        assert img.getpixel((2, 2))[:3] == (0x19, 0x17, 0x1D)  # dark background


def test_chart_generation_is_idempotent(requests_mock):
    pred = _complete_one(requests_mock)
    path = charts.render_prediction_chart(pred)
    mtime = path.stat().st_mtime_ns
    assert charts.render_prediction_chart(pred) == path
    assert path.stat().st_mtime_ns == mtime
    assert charts.generate_missing_charts() == 0


def test_pending_prediction_has_no_chart():
    pid = _log()
    assert charts.render_prediction_chart(storage.get_prediction(pid)) is None


# --- daily report -----------------------------------------------------------

def _complete_several(requests_mock):
    ids = [
        _log(created_at=T0, interval="1h", model="sklearn-tuned"),
        _log(created_at=T0, interval="15m", model="sklearn-legacy", step=1.0),
        _log(created_at=T0 + timedelta(minutes=15), interval="15m", model="sklearn-tuned", step=1.0),
    ]
    klines = {
        "1h": [_kline(T0 + timedelta(hours=i + 1), 3005.0 + i) for i in range(3)],
        "15m": [_kline(T0 + timedelta(minutes=15 * (i + 1)), 3002.0 + i, "15m") for i in range(4)],
    }
    requests_mock.get(BINANCE_KLINES, json=lambda req, ctx: klines[req.qs["interval"][0]])
    evaluator.complete_pending_predictions(now=T0 + timedelta(hours=10))
    charts.generate_missing_charts()
    return ids


def test_daily_report_orders_by_interval_then_time_and_writes_outputs(requests_mock):
    _complete_several(requests_mock)
    day = date(2026, 9, 29)
    names = [f.name for f in daily_report.day_chart_files(day)]
    assert [n.split("_")[0] for n in names] == ["15m", "15m", "1h"]
    assert names[0].endswith("100000Z.png") and names[1].endswith("101500Z.png")

    out = daily_report.build_daily_report(day)
    assert out["overview"].name == "2026-09-29_overview.png"
    with Image.open(out["overview"]) as img:
        assert img.width >= 1600 and img.height > daily_report.HEADER_HEIGHT
    with Image.open(out["gif"]) as gif:
        assert gif.n_frames == 3
    assert out["pdf"].read_bytes().startswith(b"%PDF")

    stats = daily_report.day_stats(day)
    assert stats["predictions"] == 3
    assert stats["best_model"]["model_name"] != stats["worst_model"]["model_name"]


def test_recover_missing_days_builds_past_days_only(requests_mock):
    _complete_several(requests_mock)
    built = daily_report.recover_missing_days(now=datetime(2026, 9, 30, 8, tzinfo=timezone.utc))
    assert built == [date(2026, 9, 29)]
    assert daily_report.recover_missing_days(now=datetime(2026, 9, 30, 8, tzinfo=timezone.utc)) == []
    # Still the 29th: that day isn't over, nothing to build.
    daily_report.overview_path(date(2026, 9, 29)).unlink()
    assert daily_report.recover_missing_days(now=datetime(2026, 9, 29, 23, tzinfo=timezone.utc)) == []


# --- API --------------------------------------------------------------------

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_api_predictions_table_and_summary(client, requests_mock):
    _complete_several(requests_mock)
    _log(created_at=T0 + timedelta(days=1))  # still pending
    resp = client.get("/api/evaluation/predictions?sort=pct_error&order=asc&page_size=2")
    assert resp.status_code == 200
    body = resp.json
    assert body["total"] == 4 and body["pages"] == 2 and len(body["rows"]) == 2
    summary = {s["model_name"]: s for s in body["summary"]}
    assert summary["sklearn-tuned"]["predictions"] == 3
    assert summary["sklearn-tuned"]["completed"] == 2
    assert summary["sklearn-tuned"]["direction_accuracy_pct"] is not None

    resp = client.get("/api/evaluation/predictions?interval=15m&model=sklearn-legacy")
    assert resp.json["total"] == 1


def test_api_rejects_bad_params(client):
    assert client.get("/api/evaluation/predictions?sort=drop_table").status_code == 400
    assert client.get("/api/evaluation/predictions?date_from=yesterday").status_code == 400
    assert client.get("/api/evaluation/predictions?interval=5m").status_code == 400


def test_api_csv_export(client, requests_mock):
    _complete_several(requests_mock)
    resp = client.get("/api/evaluation/predictions.csv?interval=1h")
    assert resp.status_code == 200
    assert resp.mimetype == "text/csv"
    lines = resp.get_data(as_text=True).lstrip("﻿").strip().splitlines()
    assert lines[0].startswith("id,created_at,interval,model_name")
    assert len(lines) == 2
    assert "Cumpără" in lines[1]


def test_api_chart_and_daily_files(client, requests_mock):
    ids = _complete_several(requests_mock)
    assert client.get(f"/api/evaluation/predictions/{ids[0]}/chart.png").mimetype == "image/png"
    assert client.get("/api/evaluation/predictions/99999/chart.png").status_code == 404

    daily_report.build_daily_report(date(2026, 9, 29))
    assert client.get("/api/evaluation/days").json == {"days": ["2026-09-29"]}
    detail = client.get("/api/evaluation/days/2026-09-29").json
    assert detail["predictions"] == 3 and detail["gif"] and detail["pdf"]
    assert client.get("/api/evaluation/days/2026-09-29/overview.png").mimetype == "image/png"
    assert client.get("/api/evaluation/days/2026-09-29/overview.pdf").mimetype == "application/pdf"
    assert client.get("/api/evaluation/days/2026-09-28/overview.png").status_code == 404


# --- hook into the existing prediction code ---------------------------------

def test_run_prediction_logs_to_evaluation(client, synthetic_candles):
    repository.save_candles("ethereum", "1h", synthetic_candles(200))
    resp = client.get("/api/predict?interval=1h&steps=6&model_variant=legacy")
    assert resp.status_code == 200
    rows, total = storage.query_predictions()
    assert total == 1
    row = rows[0]
    assert row["model_name"] == "sklearn-legacy"
    assert row["horizon_steps"] == 6
    assert row["price_at_prediction"] == resp.json["last_known_price"]
    assert row["signal"] in ("Cumpără", "Vinde", "Așteaptă")
    # A dashboard refresh 30s later is the same forecast - not logged again.
    client.get("/api/predict?interval=1h&steps=6&model_variant=legacy")
    assert storage.query_predictions()[1] == 1


# --- visual history (24-step snapshots) --------------------------------------

def test_24_step_forecast_logged_on_every_change_only():
    first = _log(steps=24)
    assert first is not None
    # Same forecast again (dashboard refresh, cache not retrained) - skipped,
    # even hours later.
    assert _log(steps=24, created_at=T0 + timedelta(hours=2), anchor=T0) is None
    # Model retrained into a different answer - logged, even within the same candle.
    changed = _log(steps=24, created_at=T0 + timedelta(minutes=15), anchor=T0, step=11.0)
    assert changed is not None and changed != first
    assert _log(steps=24, created_at=T0 + timedelta(minutes=20), anchor=T0, step=11.0) is None


def test_actual_path_keeps_real_ohlc(requests_mock):
    pred = _complete_one(requests_mock)
    point = pred["actual_path"][0]
    assert {"timestamp", "price", "open", "high", "low"} <= set(point)
    assert point["price"] == 3005.0


def test_snapshot_drawn_for_pending_and_redrawn_as_real_candles_arrive(requests_mock):
    history = [
        {"timestamp": to_iso(T0 - timedelta(hours=i)), "open": 2990.0, "high": 3010.0, "low": 2980.0, "close": 3000.0}
        for i in range(5, -1, -1)
    ]
    pid = storage.log_prediction(
        interval="1h", model_name="sklearn-tuned", price_at_prediction=3000.0,
        predictions=[{"timestamp": (T0 + timedelta(hours=i)).isoformat(), "predicted_price": 3000 + i, "lower": 2990, "upper": 3010} for i in (1, 2)],
        confidence=60.0, signal="wait", history=history, created_at=T0,
    )
    pred = storage.get_prediction(pid)
    assert len(pred["history_path"]) == 6
    path = charts.render_snapshot(pred)
    assert path.parent.name == "2026-09-29"  # created day
    assert path.parent.parent.name == "1h" and path.parent.parent.parent.name == "sklearn-tuned"
    assert (charts.by_date_dir(config.EVAL_SNAPSHOTS_DIR, "2026-09-29") / path.name).resolve() == path.resolve()
    first_bytes = path.read_bytes()

    requests_mock.get(BINANCE_KLINES, json=[_kline(T0 + timedelta(hours=1), 3004.0)])
    result = evaluator.complete_pending_predictions(now=T0 + timedelta(hours=2, minutes=1))
    assert result["updated_ids"] == [pid]
    assert charts.refresh_snapshots(result["updated_ids"]) == 1
    assert path.read_bytes() != first_bytes  # the real candle is now on top of the forecast
    with Image.open(path) as img:
        assert img.size == (1000, 600)


def test_api_snapshot_for_pending_prediction(client):
    pid = _log()
    resp = client.get(f"/api/evaluation/predictions/{pid}/snapshot.png")
    assert resp.status_code == 200 and resp.mimetype == "image/png"
    row = client.get("/api/evaluation/predictions?horizon_steps=3").json["rows"][0]
    assert row["snapshot_url"].endswith(f"/{pid}/snapshot.png")
    assert "history_path" not in row
    assert client.get("/api/evaluation/predictions?horizon_steps=24").json["total"] == 0


def test_run_prediction_24_steps_stores_history_and_draws_snapshot(client, synthetic_candles):
    repository.save_candles("ethereum", "1h", synthetic_candles(200))
    assert client.get("/api/predict?interval=1h&steps=24&model_variant=tuned").status_code == 200
    row = storage.query_predictions({"horizon_steps": 24})[0][0]
    assert len(row["history_path"]) == config.EVAL_SNAPSHOT_HISTORY_CANDLES
    assert charts.snapshot_path(row).exists()


def test_old_database_gets_history_column(tmp_path, monkeypatch):
    import sqlite3

    db = tmp_path / "old.db"
    schema = storage.SCHEMA_PATH.read_text(encoding="utf-8").replace(
        "    history_path          TEXT    NOT NULL DEFAULT '[]',\n", ""
    )
    conn = sqlite3.connect(db)
    conn.executescript(schema)
    conn.close()
    monkeypatch.setattr(config, "EVAL_DB_PATH", str(db))
    assert _log() is not None
    assert storage.get_prediction(1)["history_path"] == []


# --- folder layout + stats --------------------------------------------------

def test_organize_existing_moves_flat_layout(tmp_path):
    root = tmp_path / "charts"
    old = root / "2026-09-29" / "4h_sklearn-legacy_20260929T100000Z.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"png")
    assert charts.organize_existing(str(root)) == 1
    new = root / "sklearn-legacy" / "4h" / "2026-09-29" / old.name
    assert new.read_bytes() == b"png"
    assert (root / charts.BY_DATE_DIR / "2026-09-29" / old.name).read_bytes() == b"png"
    assert not (root / "2026-09-29").exists()
    assert charts.organize_existing(str(root)) == 0  # idempotent


def test_stats_files_at_every_level(requests_mock):
    _complete_several(requests_mock)
    _log(created_at=T0 + timedelta(hours=1), interval="1h", model="sklearn-legacy")  # still pending
    assert stats.write_stats_files() > 0
    root = charts.Path(config.EVAL_CHARTS_DIR)

    overall = json.loads((root / "stats.json").read_text())["overall"]
    assert overall["predictions"] == 4 and overall["completed"] == 3 and overall["pending"] == 1

    tuned = json.loads((root / "sklearn-tuned" / "stats.json").read_text())
    assert tuned["overall"]["completed"] == 2
    assert sorted(r["interval"] for r in tuned["by_interval"]) == ["15m", "1h"]

    cell = json.loads((root / "sklearn-tuned" / "1h" / "stats.json").read_text())
    assert cell["overall"]["completed"] == 1 and cell["by_day"][0]["day"] == "2026-09-29"

    day = json.loads((root / charts.BY_DATE_DIR / "2026-09-29" / "stats.json").read_text())
    assert {r["model"] for r in day["by_model"]} == {"sklearn-tuned", "sklearn-legacy"}

    assert "sklearn-tuned,1h," in (root / "stats.csv").read_text()
    assert stats.write_stats_files() == 0  # unchanged -> nothing rewritten


def test_stats_endpoint(client, requests_mock):
    _complete_several(requests_mock)
    body = client.get("/api/evaluation/stats").json
    assert body["overall"]["completed"] == 3
    assert {(r["model"], r["interval"]) for r in body["by_model_interval"]} == {
        ("sklearn-legacy", "15m"), ("sklearn-tuned", "15m"), ("sklearn-tuned", "1h"),
    }


def test_stats_endpoint_filters_and_prediction_days(client, requests_mock):
    _complete_several(requests_mock)
    body = client.get("/api/evaluation/stats?model=sklearn-tuned&interval=15m").json
    assert body["overall"]["predictions"] == 1
    assert [r["created_day"] for r in body["by_created_day"]] == ["2026-09-29"]
    days = client.get("/api/evaluation/prediction-days").json["days"]
    assert days == [{"day": "2026-09-29", "predictions": 3}]


# --- step-by-step scoring vs "no change" ----------------------------------------

def test_scores_filled_step_by_step_and_final(requests_mock):
    pid = _log()  # forecast 3010, 3020, 3030 (band +-20), price at prediction 3000
    requests_mock.get(BINANCE_KLINES, json=[_kline(T0 + timedelta(hours=1), 3005.0)])
    evaluator.complete_pending_predictions(now=T0 + timedelta(hours=2, minutes=1))
    partial = storage.get_prediction(pid)
    assert partial["status"] == "pending" and partial["steps_scored"] == 1
    assert partial["skill_score"] == pytest.approx(0.0)  # both 5 away from 3005

    requests_mock.get(BINANCE_KLINES, json=[
        _kline(T0 + timedelta(hours=i + 1), c) for i, c in enumerate((3005.0, 3015.0, 3040.0))
    ])
    evaluator.complete_pending_predictions(now=T0 + timedelta(hours=10))
    done = storage.get_prediction(pid)
    model = (5 / 3005 + 5 / 3015 + 10 / 3040) / 3 * 100
    base = (5 / 3005 + 15 / 3015 + 40 / 3040) / 3 * 100
    assert done["status"] == "completed" and done["steps_scored"] == 3
    assert done["skill_score"] == pytest.approx(1 - model / base, abs=1e-4)
    assert done["band_coverage"] == pytest.approx(1.0)


def test_step_stats_and_pooled_skill(requests_mock):
    _complete_one(requests_mock)
    steps = stats.step_stats()
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert steps[2]["mae_baseline_pct"] == pytest.approx(40 / 3040 * 100, abs=1e-3)
    assert steps[2]["skill_score"] > 0
    overall = storage.grouped_stats()[0]
    assert overall["scored"] == 1 and overall["skill_score"] > 0 and overall["band_coverage_pct"] == 100.0


def test_diagnostics_stored_and_served(client):
    pid = storage.log_prediction(
        interval="1h", model_name="sklearn-tuned", price_at_prediction=3000.0,
        predictions=[{"timestamp": (T0 + timedelta(hours=1)).isoformat(), "predicted_price": 3001, "lower": 2990, "upper": 3010}],
        confidence=60.0, signal="wait", created_at=T0, diagnostics={"method": "direct", "horizons": [{"h": 1, "alpha": 0.1}]},
    )
    body = client.get(f"/api/evaluation/predictions/{pid}/diagnostics").json
    assert body["diagnostics"]["horizons"][0]["alpha"] == 0.1
    row = client.get("/api/evaluation/predictions").json["rows"][0]
    assert "diagnostics" not in row
