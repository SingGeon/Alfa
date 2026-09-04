from datetime import datetime, timezone

from database import repository


def test_save_and_get_candles(synthetic_candles):
    candles = synthetic_candles(20)
    repository.save_candles("ethereum", "1h", candles)
    stored = repository.get_candles("ethereum", "1h", limit=100)
    assert len(stored) == 20
    # chronological order preserved
    assert stored[0]["timestamp"] <= stored[-1]["timestamp"]


def test_save_candles_upserts_without_duplicating(synthetic_candles):
    candles = synthetic_candles(10)
    repository.save_candles("ethereum", "1h", candles)
    repository.save_candles("ethereum", "1h", candles)  # re-run same batch
    stored = repository.get_candles("ethereum", "1h", limit=100)
    assert len(stored) == 10


def test_save_news_items_dedupes_by_url():
    items = [
        {"title": "A", "url": "http://x.test/1", "source": "t", "published_at": datetime.now(timezone.utc), "summary": ""},
        {"title": "A dup", "url": "http://x.test/1", "source": "t", "published_at": datetime.now(timezone.utc), "summary": ""},
    ]
    inserted = repository.save_news_items(items)
    assert inserted == 1
    assert len(repository.get_recent_news(10)) == 1


def test_price_snapshot_roundtrip():
    repository.save_price_snapshot("ethereum", {"price": 3000.0, "source": "test"})
    history = repository.get_price_history("ethereum", limit=10)
    assert len(history) == 1
    assert history[0]["price"] == 3000.0


def test_prediction_roundtrip():
    assert repository.get_latest_prediction("ethereum") is None
    repository.save_prediction("ethereum", {"confidence": 80.0})
    latest = repository.get_latest_prediction("ethereum")
    assert latest["confidence"] == 80.0
