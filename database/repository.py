"""Data-access helpers. Every function degrades gracefully to a no-op / empty
result when Mongo is unavailable, so the rest of the app can call these
without special-casing "no database" everywhere.
"""
import logging
from datetime import datetime, timezone
from typing import Iterable

from pymongo import UpdateOne
from pymongo.errors import PyMongoError

from database.mongo_client import get_db

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


def iso_utc(ts: datetime) -> str:
    """`ts.isoformat()`, guaranteed to carry an explicit UTC offset.

    The real MongoClient is configured tz_aware (see mongo_client.py), so
    datetimes read back here normally already have tzinfo - but callers
    that compare a candle timestamp's isoformat() against a prediction's
    stored target string (itself always built from a tz-aware pandas/
    datetime value - see ml.price_predictor's recursive forecast) need
    both sides in the *same* form, or a naive one silently never matches
    an aware one despite being the same instant (bare `.isoformat()`
    omits the offset entirely for a naive datetime, e.g. "...12:00:00" vs
    "...12:00:00+00:00"). Naive input here is assumed to already be UTC
    (never true local time) since that's the only thing that flows
    through this app.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


# --- Prices (ticker snapshots) ---------------------------------------------

def save_price_snapshot(coin_id: str, snapshot: dict) -> None:
    db = get_db()
    if db is None:
        return
    doc = {"coin_id": coin_id, "timestamp": _now(), **snapshot}
    try:
        db.prices.insert_one(doc)
    except PyMongoError as exc:
        logger.warning("Failed to save price snapshot: %s", exc)


def get_price_history(coin_id: str, limit: int = 500) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    cursor = db.prices.find({"coin_id": coin_id}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return list(cursor)[::-1]


# --- Candles (OHLC) ----------------------------------------------------------

def save_candles(coin_id: str, interval: str, candles: Iterable[dict]) -> None:
    """Upsert OHLC candles keyed by (coin_id, interval, timestamp) to avoid duplicates."""
    db = get_db()
    if db is None:
        return
    ops = [
        UpdateOne(
            {"coin_id": coin_id, "interval": interval, "timestamp": c["timestamp"]},
            {"$set": {**c, "coin_id": coin_id, "interval": interval}},
            upsert=True,
        )
        for c in candles
    ]
    if not ops:
        return
    try:
        db.candles.bulk_write(ops, ordered=False)
    except PyMongoError as exc:
        logger.warning("Failed to save candles: %s", exc)


def get_candles(coin_id: str, interval: str, limit: int = 500, before: datetime | None = None) -> list[dict]:
    """`before`, when given, excludes candles at/after that timestamp - lets
    a caller ask "what history was actually available as of that point in
    time" (see data_collector.jobs.backfill_prediction_history's walk-
    forward backtest, which needs exactly this to avoid training on data
    that wouldn't have existed yet on the day it's simulating).
    """
    db = get_db()
    if db is None:
        return []
    query = {"coin_id": coin_id, "interval": interval}
    if before is not None:
        query["timestamp"] = {"$lt": before}
    cursor = db.candles.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return list(cursor)[::-1]


# --- News + sentiment ---------------------------------------------------------

def save_news_items(items: Iterable[dict]) -> int:
    """Upsert by URL so re-running the collector never duplicates an article."""
    db = get_db()
    if db is None:
        return 0
    ops = [UpdateOne({"url": item["url"]}, {"$setOnInsert": item}, upsert=True) for item in items]
    if not ops:
        return 0
    try:
        result = db.news.bulk_write(ops, ordered=False)
        return result.upserted_count
    except PyMongoError as exc:
        logger.warning("Failed to save news items: %s", exc)
        return 0


def get_recent_news(limit: int = 50) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    cursor = db.news.find({}, {"_id": 0}).sort("published_at", -1).limit(limit)
    return list(cursor)


def get_news_since(since: datetime) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    cursor = db.news.find({"published_at": {"$gte": since}}, {"_id": 0})
    return list(cursor)


# --- Predictions (audit trail of what the model predicted, for later evaluation) --

def save_prediction(coin_id: str, prediction: dict) -> None:
    db = get_db()
    if db is None:
        return
    doc = {"coin_id": coin_id, "created_at": _now(), **prediction}
    try:
        db.predictions.insert_one(doc)
    except PyMongoError as exc:
        logger.warning("Failed to save prediction: %s", exc)


def get_latest_prediction(coin_id: str) -> dict | None:
    db = get_db()
    if db is None:
        return None
    return db.predictions.find_one({"coin_id": coin_id}, {"_id": 0}, sort=[("created_at", -1)])


def get_predicted_target_timestamps(coin_id: str, interval: str) -> set[str]:
    """Every target timestamp (ISO string) a genuine one-step-ahead
    forecast has been recorded for, any backend, any time it was made -
    used by data_collector.jobs.backfill_prediction_history to find which
    past days still have no such forecast before spending a training run
    on them.

    Only predictions.0 (get_prediction_accuracy_series only ever reads
    that index too - see its own docstring) - a single saved document can
    hold a much longer multi-step trajectory (the dashboard's horizon
    slider goes up to 72 steps), and distinct("predictions.timestamp")
    without the index would count *every* one of those future steps as
    "already predicted", not just the one-step-ahead target the document
    was actually fresh for. Confirmed this the hard way: an old 45-step
    scan from days ago made every date in its trajectory look already
    covered, silently blocking backfill for genuinely missing days that
    happened to fall inside that stale window.
    """
    db = get_db()
    if db is None:
        return set()
    return set(db.predictions.distinct("predictions.0.timestamp", {"coin_id": coin_id, "interval": interval}))


def save_signal_record(coin_id: str, record: dict) -> None:
    """One row = one point-in-time "what would the Trading Signal card have
    said here" call, from data_collector.jobs.backfill_signal_history -
    the walk-forward backtest behind the dashboard's "signal accuracy
    history" panel (see ml/trading_signal.py for the rule being evaluated).
    """
    db = get_db()
    if db is None:
        return
    doc = {"coin_id": coin_id, "recorded_at": _now(), **record}
    try:
        db.signal_track_record.insert_one(doc)
    except PyMongoError as exc:
        logger.warning("Failed to save signal record: %s", exc)


def get_signal_recorded_timestamps(coin_id: str, interval: str) -> set[str]:
    """Every entry timestamp (ISO string) already backfilled for this
    (coin, interval) - lets backfill_signal_history skip candidates it's
    already scored, the same idempotency trick
    get_predicted_target_timestamps uses for the price-prediction backfill.
    """
    db = get_db()
    if db is None:
        return set()
    try:
        raw = db.signal_track_record.distinct("timestamp", {"coin_id": coin_id, "interval": interval})
    except PyMongoError as exc:
        logger.warning("Failed to read recorded signal timestamps: %s", exc)
        return set()
    return {iso_utc(ts) for ts in raw}


def get_signal_accuracy_summary(coin_id: str, interval: str, since: datetime | None = None) -> dict:
    """Per-signal-type win rate: of the buy/sell calls this exact heuristic
    would have made historically, how many actually played out as
    predicted (price higher later for a "buy", lower for a "sell") - "wait"
    makes no directional claim, so it's counted but never scored right/wrong.
    """
    empty = {
        "buy": {"count": 0, "correct": 0},
        "sell": {"count": 0, "correct": 0},
        "wait": {"count": 0},
    }
    db = get_db()
    if db is None:
        return empty
    query: dict = {"coin_id": coin_id, "interval": interval}
    if since is not None:
        query["timestamp"] = {"$gte": since}
    try:
        records = list(db.signal_track_record.find(query, {"_id": 0}))
    except PyMongoError as exc:
        logger.warning("Failed to read signal accuracy summary: %s", exc)
        return empty
    summary = {
        "buy": {"count": 0, "correct": 0},
        "sell": {"count": 0, "correct": 0},
        "wait": {"count": 0},
    }
    for r in records:
        signal = r.get("signal")
        if signal not in summary:
            continue
        summary[signal]["count"] += 1
        if signal in ("buy", "sell") and r.get("correct"):
            summary[signal]["correct"] += 1
    return summary


def get_signal_records(coin_id: str, interval: str, limit: int = 50) -> list[dict]:
    """Most recent scored signal calls, newest first - the small detail
    table under the win-rate summary."""
    db = get_db()
    if db is None:
        return []
    cursor = (
        db.signal_track_record.find({"coin_id": coin_id, "interval": interval}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_prediction_accuracy_series(
    coin_id: str, interval: str, since: datetime, backend: str | None = None, model_variant: str | None = None,
) -> list[dict]:
    """One row per distinct next-period target timestamp *dated* since
    `since`, keeping only the earliest recorded forecast for each target.

    The dashboard polls /api/predict and /api/outlook every 30s while the
    underlying model only retrains every ~15min (see api/services.py's
    prediction cache), so a single real forecast for a given target candle
    ends up saved dozens of times - grouping in the database (rather than
    pulling everything into Python) keeps this proportional to the number
    of distinct forecasts made, not the number of times a client polled.

    Filters on the target's own date (predictions.0.timestamp), not on
    when the forecast was recorded (created_at) - those coincide closely
    for a live prediction (made ~a day before its target), but not for one
    from data_collector.jobs.backfill_prediction_history, which can record
    a forecast for a months-old target right now. A "last 30 days" chart
    should mean the last 30 days of *targets*, not of Mongo insert times,
    or backfilling would make an old gap bleed straight through the "1
    Lună" filter into view.

    `backend` (None = any) narrows to one model - without it, sklearn and
    LSTM forecasts for the same target/interval get grouped together
    (whichever was recorded first "wins"), which would make a per-backend
    accuracy chart meaningless.

    `model_variant` (None = any, see ml.price_predictor.MODEL_VARIANTS)
    narrows to one GBR hyperparameter profile - "tuned" or "legacy", for
    the dashboard's two side-by-side accuracy lines. "legacy" also counts
    records saved before this field existed: this app ran on sklearn's
    plain defaults (what "legacy" now means) for essentially its entire
    history before the tuned hyperparameters were added, so untagged
    history genuinely *is* legacy, not an unlabeled mix - attributing it
    to "tuned" instead (an earlier version of this logic did exactly that)
    was wrong, confirmed directly: it made the "new model" line show
    months of history from a date before "tuned" ever existed. "tuned" has
    no such backfill - it only starts existing once a prediction is
    actually tagged that way, so an exact match is the honest read for it.
    """
    db = get_db()
    if db is None:
        return []
    match: dict = {
        "coin_id": coin_id,
        "interval": interval,
        "predictions.0.timestamp": {"$gte": iso_utc(since)},
    }
    if backend is not None:
        match["backend"] = backend
    if model_variant == "legacy":
        match["$or"] = [{"model_variant": "legacy"}, {"model_variant": {"$exists": False}}]
    elif model_variant is not None:
        match["model_variant"] = model_variant
    pipeline = [
        {"$match": match},
        {"$sort": {"created_at": 1}},
        {"$group": {
            "_id": {"$arrayElemAt": ["$predictions.timestamp", 0]},
            "predicted_price": {"$first": {"$arrayElemAt": ["$predictions.predicted_price", 0]}},
            "predicted_at": {"$first": "$created_at"},
        }},
        {"$sort": {"_id": 1}},
    ]
    try:
        results = list(db.predictions.aggregate(pipeline))
    except PyMongoError as exc:
        logger.warning("Failed to aggregate prediction accuracy: %s", exc)
        return []
    return [
        {"target": r["_id"], "predicted_price": r["predicted_price"], "predicted_at": r["predicted_at"]}
        for r in results
    ]


# --- Scout results (underdog-asset scan, see scout/) ---------------------------

def save_scout_results(results: Iterable[dict]) -> None:
    """Replace the whole scout_results collection with the latest scan.

    A full replace (not upsert-merge) is deliberate: an asset that no
    longer clears the universe filters (e.g. dropped off a Yahoo screen)
    should disappear from the results, not linger with a stale score.
    """
    db = get_db()
    if db is None:
        return
    docs = [{**r, "scanned_at": _now()} for r in results]
    try:
        db.scout_results.delete_many({})
        if docs:
            db.scout_results.insert_many(docs)
    except PyMongoError as exc:
        logger.warning("Failed to save scout results: %s", exc)


def get_scout_results(asset_type: str | None = None, limit: int = 100) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    query = {"asset_type": asset_type} if asset_type else {}
    cursor = db.scout_results.find(query, {"_id": 0}).sort("score", -1).limit(limit)
    return list(cursor)


# --- Pinned assets (user bookmarks, see the Scout page's "Pinned" filter) ------
#
# Kept as its own tiny collection rather than a flag on scout_results: a
# pin needs to survive save_scout_results()'s full delete+reinsert (an
# asset can legitimately drop out of a scan - filters changed, it stopped
# rising - without the user wanting to lose track of it).

def pin_asset(asset_type: str, asset_id: str, symbol: str, name: str) -> None:
    db = get_db()
    if db is None:
        return
    try:
        db.pinned_assets.update_one(
            {"asset_type": asset_type, "id": asset_id},
            {"$set": {"asset_type": asset_type, "id": asset_id, "symbol": symbol, "name": name},
             "$setOnInsert": {"pinned_at": _now()}},
            upsert=True,
        )
    except PyMongoError as exc:
        logger.warning("Failed to pin asset: %s", exc)


def unpin_asset(asset_type: str, asset_id: str) -> None:
    db = get_db()
    if db is None:
        return
    try:
        db.pinned_assets.delete_one({"asset_type": asset_type, "id": asset_id})
    except PyMongoError as exc:
        logger.warning("Failed to unpin asset: %s", exc)


def get_pinned_assets(asset_type: str | None = None) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    query = {"asset_type": asset_type} if asset_type else {}
    cursor = db.pinned_assets.find(query, {"_id": 0}).sort("pinned_at", -1)
    return list(cursor)


def get_pinned_asset(asset_type: str, asset_id: str) -> dict | None:
    db = get_db()
    if db is None:
        return None
    return db.pinned_assets.find_one({"asset_type": asset_type, "id": asset_id}, {"_id": 0})
