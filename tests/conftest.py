"""Shared fixtures. No test here touches the real network or a real Mongo -
mongomock stands in for the database and requests_mock intercepts any
HTTP calls that slip through, so `pytest` runs fully offline.
"""
from datetime import datetime, timedelta, timezone

import mongomock
import numpy as np
import pytest

import api.routes as routes
import api.services as services
import data_collector.news_collector as news_collector
import database.mongo_client as mongo_client


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    client = mongomock.MongoClient()
    db = client["eth_predictor_test"]
    mongo_client._ensure_indexes(db)
    monkeypatch.setattr(mongo_client, "_client", client)
    monkeypatch.setattr(mongo_client, "_db", db)
    yield db


@pytest.fixture(autouse=True)
def reset_prediction_cache():
    """api.services caches trained models across requests (see its module
    docstring) - without clearing it here, a model trained in one test
    (against that test's mongomock data) would leak into a later test that
    expects fresh/different data or an InsufficientDataError.
    """
    services._prediction_cache.clear()
    yield


@pytest.fixture(autouse=True)
def reset_news_cache():
    """news_collector caches raw RSS entries across calls (see its module
    docstring) - without clearing it here, a test that monkeypatches
    feedparser.parse would silently get a previous test's cached result
    instead of exercising its own mock.
    """
    news_collector._raw_entries_cache = None
    yield


@pytest.fixture(autouse=True)
def reset_defillama_cache():
    """data_collector.defillama_client caches both the TVL history and the
    L2 chain list at module level - same leak risk across tests as the
    other module-level caches here if left uncleared.
    """
    import data_collector.defillama_client as defillama_client

    defillama_client._cache.clear()
    yield


@pytest.fixture(autouse=True)
def reset_scout_caches():
    """api.routes caches /api/scout/detail and /api/scout/price responses
    per (asset_type, id) (see routes.py) - same leak risk as the other
    module-level caches above if left uncleared between tests.
    """
    routes._scout_detail_cache.clear()
    routes._scout_price_cache.clear()
    yield


@pytest.fixture(autouse=True)
def stub_external_feature_sources(monkeypatch):
    """btc_return_1/tvl_momentum (ml/features.py) are fetched in
    api/services.py via live Binance/DeFiLlama calls with no requests_mock
    route registered for most tests - without this they'd quietly hit the
    real network, breaking the "fully offline" guarantee above. Stubbing to
    None/{} is faithful, not a shortcut: it's exactly what the real code
    already degrades to on a fetch failure (see _btc_close_series /
    get_ethereum_tvl_by_date's own docstrings).
    """
    monkeypatch.setattr(services, "_btc_close_series", lambda interval, coin_id: None)
    monkeypatch.setattr(services, "get_ethereum_tvl_by_date", lambda: {})
    yield


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


@pytest.fixture(autouse=True)
def evaluation_data_dir(tmp_path, monkeypatch):
    """Point the evaluation module's SQLite DB and chart/report folders at a
    per-test temp dir - api.services.run_prediction logs every prediction
    there, so without this the test suite would write into the real data/."""
    import config

    data_dir = tmp_path / "eval_data"
    monkeypatch.setattr(config, "EVAL_DATA_DIR", str(data_dir))
    monkeypatch.setattr(config, "EVAL_DB_PATH", str(data_dir / "predictions.db"))
    monkeypatch.setattr(config, "EVAL_CHARTS_DIR", str(data_dir / "charts"))
    monkeypatch.setattr(config, "EVAL_DAILY_DIR", str(data_dir / "daily"))
    monkeypatch.setattr(config, "EVAL_SNAPSHOTS_DIR", str(data_dir / "snapshots"))
    yield data_dir


@pytest.fixture(autouse=True)
def _isolated_evolution_state(tmp_path, monkeypatch):
    """The "tuned" population persists to data/evolution/ (ml/evolution.py);
    tests must neither read the real saved populations nor write into them."""
    from ml import evolution

    monkeypatch.setattr(evolution, "STATE_DIR", tmp_path / "evolution")
    evolution._registry.clear()
    evolution._saved_mtime.clear()
    yield
    evolution._registry.clear()
    evolution._saved_mtime.clear()


@pytest.fixture(autouse=True)
def _no_live_market_context(monkeypatch):
    """ml/market_context.live_sources fetches Fear & Greed and Binance
    futures over the network - offline tests get no live sources."""
    from ml import market_context

    monkeypatch.setattr(market_context, "live_sources", lambda symbol, interval: {})
