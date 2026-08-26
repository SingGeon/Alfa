"""The actual work each scheduled job performs. Kept separate from the
APScheduler wiring (run_scheduler.py) so jobs can also be triggered
on-demand (e.g. from a CLI or a Flask admin route) without needing a
running scheduler.
"""
from __future__ import annotations

import logging

import config
from data_collector import market_data, news_collector
from database import repository
from nlp import sentiment

logger = logging.getLogger(__name__)


def collect_market_data_job(coin_id: str = config.COIN_ID) -> None:
    """Fetch current price + latest candles for every supported interval, persist to Mongo."""
    try:
        snapshot = market_data.get_current_price(coin_id)
        repository.save_price_snapshot(coin_id, snapshot)
        logger.info("Saved price snapshot for %s: %s", coin_id, snapshot.get("price"))
    except Exception:
        logger.exception("Failed to collect current price for %s", coin_id)

    for interval in market_data.SUPPORTED_INTERVALS:
        try:
            candles = market_data.get_ohlc_candles(interval=interval, coin_id=coin_id)
            repository.save_candles(coin_id, interval, candles)
            logger.info("Saved %d candles (%s) for %s", len(candles), interval, coin_id)
        except Exception:
            logger.exception("Failed to collect %s candles for %s", interval, coin_id)


def collect_news_job() -> None:
    """Fetch recent articles, score sentiment, persist new ones (dedup by URL)."""
    try:
        articles = news_collector.collect_news()
        scored = [sentiment.score_article(a) for a in articles]
        inserted = repository.save_news_items(scored)
        logger.info("Collected %d articles, %d new", len(scored), inserted)
    except Exception:
        logger.exception("Failed to collect news")


def run_all_once() -> None:
    """Convenience for manual runs / first-time seeding."""
    collect_market_data_job()
    collect_news_job()
