"""Centralized configuration, loaded from environment variables / .env.

Every setting has a default so the app runs out of the box in dev mode.
Add new coins by extending COIN_ID / VS_CURRENCY per-request rather than
hardcoding "ethereum" deeper in the codebase (see data_collector/market_data.py).
"""
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# --- MongoDB ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "eth_predictor")

# --- Market data ---
COINGECKO_API_BASE = os.getenv("COINGECKO_API_BASE", "https://api.coingecko.com/api/v3")
BINANCE_API_BASE = os.getenv("BINANCE_API_BASE", "https://api.binance.com/api/v3")
COIN_ID = os.getenv("COIN_ID", "ethereum")          # CoinGecko coin id
BINANCE_SYMBOL = os.getenv("BINANCE_SYMBOL", "ETHUSDT")
VS_CURRENCY = os.getenv("VS_CURRENCY", "usd")

# --- On-chain data (optional) ---
# https://etherscan.io/apis - free tier. Used as a price fallback + gas price signal.
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")

# --- News ---
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWS_RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.theblock.co/rss.xml",
    "https://cryptoslate.com/feed/",
    "https://cryptonews.com/news/feed/",
    "https://u.today/rss",
    "https://www.newsbtc.com/feed/",
    "https://bitcoinist.com/feed/",
    "https://beincrypto.com/feed/",
]
NEWS_KEYWORDS = ["ethereum", "ether", "eth"]

# --- Optional narrative summary ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Scheduler ---
COLLECT_INTERVAL_MINUTES = int(os.getenv("COLLECT_INTERVAL_MINUTES", "15"))
NEWS_INTERVAL_MINUTES = int(os.getenv("NEWS_INTERVAL_MINUTES", "5"))
# A full scout scan trains a real model per asset across ~80 assets, so it
# runs far less often than the single-coin market/news jobs above.
SCOUT_INTERVAL_MINUTES = int(os.getenv("SCOUT_INTERVAL_MINUTES", "60"))

# --- Prediction ---
PREDICTION_HORIZON_HOURS = int(os.getenv("PREDICTION_HORIZON_HOURS", "24"))
PREDICTION_BACKEND = os.getenv("PREDICTION_BACKEND", "sklearn")

# --- Flask ---
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = _bool("FLASK_DEBUG", False)

# Base URL the frontend dashboard uses to reach the Flask API.
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{FLASK_PORT}")

# --- Prediction evaluation (evaluation/) ---
# SQLite log of every prediction + its real outcome, per-prediction PNG
# charts and the end-of-day overview images. Relative paths resolve against
# the project root so it doesn't matter which directory the app starts from.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_DATA_DIR = os.path.join(_PROJECT_ROOT, os.getenv("EVAL_DATA_DIR", "data"))
EVAL_DB_PATH = os.path.join(EVAL_DATA_DIR, os.getenv("EVAL_DB_FILE", "predictions.db"))
EVAL_CHARTS_DIR = os.path.join(EVAL_DATA_DIR, "charts")
EVAL_DAILY_DIR = os.path.join(EVAL_DATA_DIR, "daily")
# A prediction whose real price still can't be found this many hours after
# its target candle closed (API outage + a real hole in the data) is marked
# "expired" instead of staying pending forever.
EVAL_EXPIRE_HOURS = int(os.getenv("EVAL_EXPIRE_HOURS", "72"))
# The optional GIF/PDF alongside the daily overview PNG.
EVAL_DAILY_GIF = _bool("EVAL_DAILY_GIF", True)
EVAL_DAILY_PDF = _bool("EVAL_DAILY_PDF", True)
# Visual history: predictions with this many steps get a new row + image
# every time the forecast actually changes (model retrained), not just once
# per candle; EVAL_SNAPSHOT_INTERVALS are forecast at that horizon on a
# timer so the history fills in even with no dashboard open.
EVAL_SNAPSHOT_STEPS = int(os.getenv("EVAL_SNAPSHOT_STEPS", "24"))
EVAL_SNAPSHOT_INTERVALS = [s.strip() for s in os.getenv("EVAL_SNAPSHOT_INTERVALS", "1h").split(",") if s.strip()]
EVAL_SNAPSHOT_EVERY_MINUTES = int(os.getenv("EVAL_SNAPSHOT_EVERY_MINUTES", "5"))
# How many real candles before the prediction the visual snapshot shows.
EVAL_SNAPSHOT_HISTORY_CANDLES = int(os.getenv("EVAL_SNAPSHOT_HISTORY_CANDLES", "48"))
EVAL_SNAPSHOTS_DIR = os.path.join(EVAL_DATA_DIR, "snapshots")
