"""Standalone scheduler process: periodically refreshes market data and news.

Run alongside the Flask API + dashboard (`python run_api.py`), e.g. in
separate terminals/containers:
    python run_scheduler.py
"""
import logging
import threading

from apscheduler.schedulers.blocking import BlockingScheduler

import config
from data_collector.jobs import collect_market_data_job, collect_news_job, run_all_once
from scout.scanner import run_scout_scan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Seeding initial data...")
    run_all_once()

    # A full scout scan takes a few minutes (real model per asset, ~80
    # assets) - run the first one in the background so it doesn't delay
    # market/news data (which the dashboard needs immediately) from coming
    # up. Later scans are handled by the normal interval job below.
    threading.Thread(target=run_scout_scan, daemon=True, name="scout-initial-scan").start()

    scheduler = BlockingScheduler(timezone="UTC")
    # No explicit next_run_time here: omitting it (rather than passing None,
    # which leaves the job unscheduled forever - the bug that silently
    # stopped candles/news from ever refreshing after the initial seed
    # above) lets APScheduler compute the normal first run at now+interval.
    scheduler.add_job(
        collect_market_data_job,
        "interval",
        minutes=config.COLLECT_INTERVAL_MINUTES,
        id="collect_market_data",
    )
    scheduler.add_job(
        collect_news_job,
        "interval",
        minutes=config.NEWS_INTERVAL_MINUTES,
        id="collect_news",
    )
    scheduler.add_job(
        run_scout_scan,
        "interval",
        minutes=config.SCOUT_INTERVAL_MINUTES,
        id="scout_scan",
    )
    logger.info(
        "Scheduler started: market data every %dmin, news every %dmin, scout scan every %dmin",
        config.COLLECT_INTERVAL_MINUTES,
        config.NEWS_INTERVAL_MINUTES,
        config.SCOUT_INTERVAL_MINUTES,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
