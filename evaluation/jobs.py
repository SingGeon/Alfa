"""APScheduler wiring for the evaluation module.

    register_jobs(scheduler)   adds the recurring jobs to an existing
                               BackgroundScheduler (must use timezone="UTC")
    startup_recovery()         catch-up work for time the app was down

run_api.py and run_scheduler.py both call these, so evaluation keeps
running whichever process does the collecting.
"""
from __future__ import annotations

import logging

import config
from evaluation import charts, daily_report, evaluator, stats, trade_watch

logger = logging.getLogger(__name__)


def evaluate_and_chart_job() -> None:
    """Every minute: fill real prices, complete what resolved, chart it."""
    try:
        result = evaluator.complete_pending_predictions()
        # New real candles -> redraw those snapshots on top of their forecast.
        charts.refresh_snapshots(result["updated_ids"])
        drawn = charts.generate_missing_charts(result["completed_ids"]) if result["completed_ids"] else 0
        stats.write_stats_files()
        if result["completed"] or result["expired"]:
            logger.info(
                "Evaluation: %d updated, %d completed, %d expired, %d charts drawn",
                result["updated"], result["completed"], result["expired"], drawn,
            )
    except Exception:
        logger.exception("Prediction evaluation job failed")


def snapshot_predictions_job() -> None:
    """Make the visual-history forecasts (config.EVAL_SNAPSHOT_STEPS steps,
    each of config.EVAL_SNAPSHOT_INTERVALS) on a timer, so the history fills
    in even when nobody has the dashboard open. These hit the warm model
    cache (api.services), so they're cheap; storage only logs one when the
    forecast has actually changed since the last one.

    "tuned" (ml.evolution) is excluded: that model's job is deciding *when*
    to trade, not forecasting a price, so a timer-driven predicted-price
    snapshot would misrepresent it - its own trade entries/exits get their
    own snapshot instead, drawn the instant they happen (see
    evolution_trade_snapshots_job / evaluation.trade_watch)."""
    from api.services import MODEL_VARIANTS, InsufficientDataError, run_prediction

    for interval in config.EVAL_SNAPSHOT_INTERVALS:
        for model_variant in MODEL_VARIANTS:
            if model_variant == "tuned":
                continue
            try:
                run_prediction(interval=interval, steps=config.EVAL_SNAPSHOT_STEPS, model_variant=model_variant)
            except InsufficientDataError:
                logger.info("Skipping %s snapshot prediction (%s) - not enough candles yet", interval, model_variant)
            except Exception:
                logger.exception("Snapshot prediction failed for %s/%s", interval, model_variant)


def evolution_trade_snapshots_job() -> None:
    """Every minute: draw a snapshot for any evolution-fund trade that opened or closed since
    the last check (ml.evolution.EvolutionaryForecaster.fund_trades_report, via
    evaluation.trade_watch) - not on a prediction timer, never a forecast line."""
    try:
        drawn = trade_watch.check_all()
        if drawn:
            logger.info("Evolution trade snapshots: %d drawn", drawn)
    except Exception:
        logger.exception("Evolution trade snapshot job failed")


def daily_report_job() -> None:
    try:
        daily_report.daily_report_job()
    except Exception:
        logger.exception("Daily evaluation report job failed")


def startup_recovery() -> None:
    """Complete whatever resolved while the app was down, draw any charts
    that are missing, then build the overviews of days that never got one."""
    try:
        # Files drawn before the model/interval/day layout existed.
        for root in (config.EVAL_CHARTS_DIR, config.EVAL_SNAPSHOTS_DIR):
            moved = charts.organize_existing(root)
            if moved:
                logger.info("Moved %d evaluation PNGs in %s into the model/interval/day layout", moved, root)
        rescored = evaluator.rescore_unscored()
        if rescored:
            logger.info("Scored %d older predictions step by step", rescored)
        result = evaluator.complete_pending_predictions()
        charts.refresh_snapshots(result["updated_ids"])
        drawn = charts.generate_missing_charts()
        stats.write_stats_files()
        built = daily_report.recover_missing_days()
        if drawn or built:
            logger.info("Evaluation recovery: %d charts drawn, daily reports built for %s", drawn, built)
    except Exception:
        logger.exception("Evaluation startup recovery failed")
    snapshot_predictions_job()
    evolution_trade_snapshots_job()


def register_jobs(scheduler) -> None:
    scheduler.add_job(
        snapshot_predictions_job, "interval", minutes=config.EVAL_SNAPSHOT_EVERY_MINUTES, id="snapshot_predictions",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        evaluate_and_chart_job, "interval", minutes=1, id="evaluate_predictions",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        evolution_trade_snapshots_job, "interval", minutes=1, id="evolution_trade_snapshots",
        max_instances=1, coalesce=True,
    )
    scheduler.add_job(
        daily_report_job, "cron", hour=0, minute=5, timezone="UTC", id="daily_evaluation_report",
        max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
