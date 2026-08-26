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
]
NEWS_KEYWORDS = ["ethereum", "eth "]

# --- Optional narrative summary ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Scheduler ---
COLLECT_INTERVAL_MINUTES = int(os.getenv("COLLECT_INTERVAL_MINUTES", "15"))
NEWS_INTERVAL_MINUTES = int(os.getenv("NEWS_INTERVAL_MINUTES", "30"))

# --- Prediction ---
PREDICTION_HORIZON_HOURS = int(os.getenv("PREDICTION_HORIZON_HOURS", "24"))
PREDICTION_BACKEND = os.getenv("PREDICTION_BACKEND", "sklearn")

# --- Flask ---
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = _bool("FLASK_DEBUG", False)

# Base URL the Streamlit dashboard uses to reach the Flask API.
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{FLASK_PORT}")
