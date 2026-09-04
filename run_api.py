"""Flask API entrypoint: python run_api.py"""
import logging
import threading
import time

import config
from api import create_app
from api.services import warm_prediction_cache

app = create_app()
logger = logging.getLogger(__name__)

# Refreshed well before _PREDICTION_CACHE_TTL_SECONDS (900s) can expire, so
# a real request never has to pay the ~6s-per-interval training cost itself
# - see warm_prediction_cache()'s docstring for the full reasoning. Runs as
# a background thread in this same process (the prediction cache is a
# plain in-memory dict, not shared with run_scheduler.py's separate
# process, so warming it up has to happen here).
_WARM_UP_INTERVAL_SECONDS = 300


def _warm_up_loop() -> None:
    while True:
        try:
            warm_prediction_cache()
        except Exception:
            logger.exception("Prediction cache warm-up loop iteration failed")
        time.sleep(_WARM_UP_INTERVAL_SECONDS)


if __name__ == "__main__":
    threading.Thread(target=_warm_up_loop, daemon=True).start()
    # threaded=True so one slow/rate-limited upstream call (CoinGecko, Binance)
    # can't freeze every other request on this single-process dev server.
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG, threaded=True)
