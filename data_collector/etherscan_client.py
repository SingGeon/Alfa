"""Etherscan integration (optional, needs ETHERSCAN_API_KEY - free tier at
https://etherscan.io/apis).

Etherscan is an on-chain explorer, not a market-data provider: it has no
historical OHLC candles, so it can't replace CoinGecko/Binance for the
price-history/prediction pipeline. What it *does* give us for free:
  - a live ETH/USD price point (useful as one more fallback), and
  - the current on-chain gas price, which CoinGecko/Binance don't expose
    and which is a nice extra signal for the dashboard.
"""
from __future__ import annotations

import logging

import config
from data_collector.http_utils import ExternalAPIError, get_json

logger = logging.getLogger(__name__)

BASE_URL = "https://api.etherscan.io/api"


def is_configured() -> bool:
    return bool(config.ETHERSCAN_API_KEY)


def get_eth_price() -> dict | None:
    """Current ETH/USD (and ETH/BTC) price straight from Etherscan. None if unavailable."""
    if not is_configured():
        return None
    try:
        data = get_json(
            BASE_URL,
            params={"module": "stats", "action": "ethprice", "apikey": config.ETHERSCAN_API_KEY},
        )
    except ExternalAPIError as exc:
        logger.warning("Etherscan ethprice failed: %s", exc)
        return None
    result = data.get("result") or {}
    if not result:
        return None
    return {
        "price": float(result["ethusd"]),
        "eth_btc": float(result.get("ethbtc", 0)) or None,
        "source": "etherscan",
    }


def get_gas_price_gwei() -> dict | None:
    """Current on-chain gas price (Safe/Propose/Fast, in gwei). None if unavailable."""
    if not is_configured():
        return None
    try:
        data = get_json(
            BASE_URL,
            params={"module": "gastracker", "action": "gasoracle", "apikey": config.ETHERSCAN_API_KEY},
        )
    except ExternalAPIError as exc:
        logger.warning("Etherscan gasoracle failed: %s", exc)
        return None
    result = data.get("result") or {}
    if not result:
        return None
    return {
        "safe_gwei": float(result.get("SafeGasPrice", 0)),
        "propose_gwei": float(result.get("ProposeGasPrice", 0)),
        "fast_gwei": float(result.get("FastGasPrice", 0)),
    }
