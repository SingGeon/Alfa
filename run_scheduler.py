"""Standalone scheduler process: periodically refreshes market data and news.

Run alongside the Flask API (`python run_api.py`) and the Streamlit
dashboard, e.g. in separate terminals/containers:
    python run_scheduler.py
"""
import logging

from apscheduler.schedulers.blocking import BlockingScheduler

import config
from data_collector.jobs import collect_market_data_job, collect_news_job, run_all_once

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Seeding initial data...")
    run_all_once()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        collect_market_data_job,
        "interval",
        minutes=config.COLLECT_INTERVAL_MINUTES,
        id="collect_market_data",
        next_run_time=None,
    )
    scheduler.add_job(
        collect_news_job,
        "interval",
        minutes=config.NEWS_INTERVAL_MINUTES,
        id="collect_news",
        next_run_time=None,
    )
    logger.info(
        "Scheduler started: market data every %dmin, news every %dmin",
        config.COLLECT_INTERVAL_MINUTES,
        config.NEWS_INTERVAL_MINUTES,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
