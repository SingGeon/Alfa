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
