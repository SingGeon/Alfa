"""data_collector.jobs - background job logic not already covered via the
API layer in test_api.py.
"""
from api.services import get_prediction_accuracy
from data_collector.jobs import backfill_prediction_history
from database import repository
from database.mongo_client import get_db


def test_backfill_prediction_history_fills_gaps(synthetic_candles):
    # 70 days of history, nothing recorded yet - a stand-in for "the app
    # wasn't running for a while, then someone asks for the backfill".
    repository.save_candles("ethereum", "1d", synthetic_candles(70, interval_hours=24))

    assert repository.get_predicted_target_timestamps("ethereum", "1d") == set()

    backfilled = backfill_prediction_history("ethereum", "1d", max_days=30)

    # The earliest candidates in the 30-day window don't have
    # MIN_CANDLES_FOR_TRAINING (40) candles *before* them yet (the walk-
    # forward cutoff only sees the 70-30=40 candles preceding the window
    # at best, shrinking further for earlier candidates), so not every one
    # of the 30 candidates can be backfilled - but most of the later ones
    # should be.
    assert backfilled >= 15

    targets_after = repository.get_predicted_target_timestamps("ethereum", "1d")
    assert len(targets_after) == backfilled

    result = get_prediction_accuracy("ethereum", "1d", days=30)
    assert result["count"] == backfilled
    assert all(p["actual_price"] is not None for p in result["points"])


def test_backfill_is_idempotent(synthetic_candles):
    # Running it twice shouldn't double up predictions for the same target.
    repository.save_candles("ethereum", "1d", synthetic_candles(70, interval_hours=24))
    first = backfill_prediction_history("ethereum", "1d", max_days=20)
    second = backfill_prediction_history("ethereum", "1d", max_days=20)
    assert first > 0
    assert second == 0


def test_backfill_skips_targets_that_already_have_a_prediction(synthetic_candles):
    repository.save_candles("ethereum", "1d", synthetic_candles(70, interval_hours=24))
    stored = repository.get_candles("ethereum", "1d")
    target = stored[-1]["timestamp"]
    repository.save_prediction(
        "ethereum",
        {
            "interval": "1d",
            "backend": "sklearn",
            "predictions": [{"timestamp": target.isoformat(), "predicted_price": stored[-1]["close"] + 1}],
        },
    )

    backfill_prediction_history("ethereum", "1d", max_days=20)

    # Exactly one prediction should exist for that target - the pre-seeded
    # one, not a second one written on top of it.
    docs = list(get_db().predictions.find({"coin_id": "ethereum", "interval": "1d"}))
    matching = [d for d in docs if d["predictions"][0]["timestamp"] == target.isoformat()]
    assert len(matching) == 1
    assert matching[0]["predictions"][0]["predicted_price"] == stored[-1]["close"] + 1
