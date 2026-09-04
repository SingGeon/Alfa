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


def get_candles(coin_id: str, interval: str, limit: int = 500) -> list[dict]:
    db = get_db()
    if db is None:
        return []
    cursor = (
        db.candles.find({"coin_id": coin_id, "interval": interval}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
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


# --- Pinned assets (user bookmarks, see scout.html "Fixate") -------------------
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
