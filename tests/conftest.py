"""Shared fixtures. No test here touches the real network or a real Mongo -
mongomock stands in for the database and requests_mock intercepts any
HTTP calls that slip through, so `pytest` runs fully offline.
"""
from datetime import datetime, timedelta, timezone

import mongomock
import numpy as np
import pytest

import database.mongo_client as mongo_client


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    client = mongomock.MongoClient()
    db = client["eth_predictor_test"]
    mongo_client._ensure_indexes(db)
    monkeypatch.setattr(mongo_client, "_client", client)
    monkeypatch.setattr(mongo_client, "_db", db)
    yield db


@pytest.fixture
def synthetic_candles():
    def _make(n=200, interval_hours=1, seed=0):
        rng = np.random.default_rng(seed)
        start = datetime.now(timezone.utc) - timedelta(hours=n * interval_hours)
        prices = 3000 + np.cumsum(rng.normal(0, 10, n))
        candles = []
        for i in range(n):
            p = float(prices[i])
            candles.append(
                {
                    "timestamp": start + timedelta(hours=i * interval_hours),
                    "open": p,
                    "high": p + 5,
                    "low": p - 5,
                    "close": p,
                    "volume": 1000 + i,
                }
            )
        return candles

    return _make
