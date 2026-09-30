"""Flask API entrypoint: python run_api.py"""
import logging
import multiprocessing
import os
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler

import config
from api import create_app
from api.services import warm_prediction_cache
from data_collector.jobs import (
    backfill_signal_history,
    collect_market_data_job,
    collect_news_job,
    record_prediction_snapshots_job,
    run_all_once,
)
from evaluation import jobs as evaluation_jobs
from scout.scanner import run_scout_scan

app = create_app()
logger = logging.getLogger(__name__)

# Refreshed well before _PREDICTION_CACHE_TTL_SECONDS (900s) can expire, so
# a real request never has to pay the ~6s-per-interval training cost itself
# - see warm_prediction_cache()'s docstring for the full reasoning.
_WARM_UP_INTERVAL_SECONDS = 300

_market_data_scheduler: BackgroundScheduler | None = None


def _run_scout_scan_subprocess() -> None:
    """Run the Scout AI scan (real model trained per asset, ~135 assets,
    a few minutes) in its own OS process instead of a thread here.

    A thread would share this process's GIL with the ETH dashboard's own
    sklearn/LSTM training - confirmed directly: with the scan running
    in-process, the first page load after startup took nearly 2 minutes to
    get a prediction back, because the scan's CPU-bound work was starving
    every other thread of GIL time for its full duration. A separate
    process sidesteps the GIL entirely; `spawn` rather than the default
    (on Linux) `fork` because by the time this runs, prediction cache
    warm-up may have already loaded TensorFlow in this process, and
    forking a process that has native/threaded libraries like that already
    initialized is a known source of hangs and crashes in the child.
    """
    ctx = multiprocessing.get_context("spawn")
    ctx.Process(target=run_scout_scan, daemon=True, name="scout-scan-proc").start()


def _warm_up_loop() -> None:
    while True:
        try:
            warm_prediction_cache()
        except Exception:
            logger.exception("Prediction cache warm-up loop iteration failed")
        time.sleep(_WARM_UP_INTERVAL_SECONDS)


def _start_market_data_scheduler() -> None:
    """Keep candles/news/Scout AI results fresh for as long as this process runs.

    Used to live only in a separate `run_scheduler.py` process that had to
    be started alongside this one by hand - easy to forget, and the
    failure mode was silent (the dashboard just kept serving increasingly
    stale candles with no error, which is exactly what happened: a day's
    worth of drift before anyone noticed). Running it in-process here means
    `python run_api.py` alone is always enough to stay current;
    run_scheduler.py still works standalone for anyone who'd rather run
    collection as its own process (e.g. multiple API workers sharing one
    collector instead of each duplicating the work).
    """
    global _market_data_scheduler
    logger.info("Seeding market data/news on startup...")
    run_all_once()
    _run_scout_scan_subprocess()

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(collect_market_data_job, "interval", minutes=config.COLLECT_INTERVAL_MINUTES, id="collect_market_data")
    scheduler.add_job(collect_news_job, "interval", minutes=config.NEWS_INTERVAL_MINUTES, id="collect_news")
    # Same cadence as market data collection, so the accuracy chart's 15m/1h
    # history accumulates roughly one point per fresh candle rather than by
    # coincidence of who has the dashboard open (see the job's docstring).
    scheduler.add_job(record_prediction_snapshots_job, "interval", minutes=config.COLLECT_INTERVAL_MINUTES, id="record_prediction_snapshots")
    # A handful of candidates per cycle (idempotent, see the job's docstring)
    # keeps this growing continuously without ever blocking a cycle for long.
    scheduler.add_job(
        backfill_signal_history, "interval", minutes=config.COLLECT_INTERVAL_MINUTES, id="backfill_signal_history",
        kwargs={"interval": "1h", "steps": 24, "max_candidates": 5},
    )
    scheduler.add_job(_run_scout_scan_subprocess, "interval", minutes=config.SCOUT_INTERVAL_MINUTES, id="scout_scan")
    # Prediction evaluation: real prices every minute, daily overview at 00:05 UTC.
    evaluation_jobs.register_jobs(scheduler)
    scheduler.start()
    evaluation_jobs.startup_recovery()
    _market_data_scheduler = scheduler  # keep a reference so it isn't GC'd
    logger.info(
        "Background scheduler started: market data every %dmin, news every %dmin, scout scan every %dmin",
        config.COLLECT_INTERVAL_MINUTES, config.NEWS_INTERVAL_MINUTES, config.SCOUT_INTERVAL_MINUTES,
    )


if __name__ == "__main__":
    # Guard against Werkzeug's debug-mode reloader running this module twice
    # (once in the file-watching parent, once in the child that actually
    # serves requests) - without it, debug mode would seed/schedule jobs in
    # both processes. Skipped entirely when debug is off, since then
    # there's only ever the one process anyway.
    if not config.FLASK_DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_start_market_data_scheduler, daemon=True, name="market-data-scheduler-init").start()
        threading.Thread(target=_warm_up_loop, daemon=True, name="prediction-cache-warmup").start()
    # threaded=True so one slow/rate-limited upstream call (CoinGecko, Binance)
    # can't freeze every other request on this single-process dev server.
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True)
