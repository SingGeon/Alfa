from datetime import datetime, timedelta, timezone

import pytest

from evaluation import box_breakout_robustness as robustness


@pytest.fixture(autouse=True)
def _isolated_state_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(robustness, "STATE_DIR", tmp_path)


def test_load_returns_none_when_nothing_computed():
    assert robustness.load("1h") is None


def test_report_round_trips_through_load():
    report = {"interval": "1h", "computed_at": datetime.now(timezone.utc).isoformat(), "n_runs": 10}
    path = robustness.report_path("1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(report))
    assert robustness.load("1h") == report


def test_is_stale_true_when_nothing_computed():
    assert robustness.is_stale("1h") is True


def test_is_stale_false_within_the_refresh_window():
    now = datetime.now(timezone.utc)
    report = {"interval": "1h", "computed_at": (now - timedelta(hours=1)).isoformat()}
    path = robustness.report_path("1h")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(report))
    assert robustness.is_stale("1h", now=now) is False
    assert robustness.is_stale("1h", now=now + timedelta(hours=24)) is True


def test_refresh_all_only_recomputes_stale_intervals(monkeypatch):
    now = datetime.now(timezone.utc)
    fresh = {"interval": "4h", "computed_at": now.isoformat()}
    path = robustness.report_path("4h")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(fresh))

    computed = []
    monkeypatch.setattr(robustness, "compute", lambda interval: computed.append(interval))
    robustness.refresh_all()
    assert computed == ["1h"]   # 4h's report is fresh, only 1h (never computed) needed a rerun


def test_compute_rejects_unsupported_interval():
    with pytest.raises(ValueError):
        robustness.compute("1w")
