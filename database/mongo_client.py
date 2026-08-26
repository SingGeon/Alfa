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
        _client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=3000)
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
