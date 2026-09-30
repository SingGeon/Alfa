"""Thin wrapper around the MongoDB connection.

The rest of the app must never crash just because Mongo is unreachable
(e.g. during local dev without a running Mongo instance) - get_db()
returns None in that case, and callers should treat persistence as
best-effort (log + skip) rather than failing the request.
"""
import logging

from pymongo import MongoClient
from pymongo.errors import PyMongoError

import config

logger = logging.getLogger(__name__)

_client: MongoClient | None = None
_db = None


def get_db():
    """Return the Mongo database handle, connecting lazily. None if unavailable."""
    global _client, _db
    if _db is not None:
        return _db
    try:
        # tz_aware so datetimes read back from Mongo carry explicit UTC
        # tzinfo - without it, .isoformat() drops the offset entirely
        # (e.g. "2026-08-26T15:15:00" instead of "...+00:00"), and
        # JavaScript's `new Date()` silently misreads such a string as
        # LOCAL time instead of UTC, shifting every timestamp shown in
        # the dashboard by the viewer's timezone offset.
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000, tz_aware=True)
        _client.admin.command("ping")
        _db = _client[config.MONGO_DB]
        _ensure_indexes(_db)
        logger.info("Connected to MongoDB at %s", config.MONGO_URI)
        return _db
    except PyMongoError as exc:
        logger.warning("MongoDB unavailable (%s) - continuing without persistence", exc)
        _client = None
        _db = None
        return None


def _ensure_indexes(db) -> None:
    db.prices.create_index([("coin_id", 1), ("timestamp", -1)])
    db.candles.create_index([("coin_id", 1), ("interval", 1), ("timestamp", -1)])
    db.news.create_index([("url", 1)], unique=True)
    db.news.create_index([("published_at", -1)])
    db.predictions.create_index([("coin_id", 1), ("created_at", -1)])
    # Backs get_prediction_accuracy_series's $match (coin_id + interval +
    # created_at range) - without `interval` in the index, that query can
    # only narrow by coin_id+created_at and then filter interval per
    # document, which gets slower as this collection grows (every /api/predict,
    # /api/outlook and /api/summary call inserts a new prediction document
    # on every dashboard refresh, so this is one of the fastest-growing
    # collections in the app - already tens of thousands of rows after a
    # couple of days of normal use).
    db.predictions.create_index([("coin_id", 1), ("interval", 1), ("created_at", -1)])
    db.scout_results.create_index([("asset_type", 1), ("score", -1)])
