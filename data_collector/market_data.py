"""Live Ethereum market data.

Primary source is CoinGecko (keyless, generous rate limits) for the current
ticker (price/volume/24h change). For OHLC candles we prefer Binance's
/klines endpoint because its `interval` parameter (1h, 1d, 1w, ...) maps
1:1 onto what a candlestick chart needs, and fall back to CoinGecko's OHLC
endpoint if Binance is unreachable (e.g. blocked in some regions).

To support another coin later, pass a different `coin_id` (CoinGecko id,
e.g. "bitcoin") and `symbol` (Binance symbol, e.g. "BTCUSDT") into the
functions below - nothing here is Ethereum-specific except the config
defaults.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
from data_collector.http_utils import ExternalAPIError, get_json

logger = logging.getLogger(__name__)

# Binance kline intervals are already exactly what we want to expose.
SUPPORTED_INTERVALS = {"15m", "1h", "4h", "1d", "1w"}

# CoinGecko `days` parameter used as a fallback per requested interval.
_COINGECKO_DAYS_BY_INTERVAL = {"15m": 1, "1h": 1, "4h": 14, "1d": 30, "1w": 90}


class MarketDataError(ExternalAPIError):
    """Raised when all configured market data sources fail."""


def _request_with_retry(url: str, params: dict | None = None, max_retries: int = 3, timeout: int = 10):
    try:
        return get_json(url, params=params, max_retries=max_retries, timeout=timeout)
    except ExternalAPIError as exc:
        raise MarketDataError(str(exc)) from exc


def get_current_price(coin_id: str = config.COIN_ID, vs_currency: str = config.VS_CURRENCY) -> dict:
    """Current price, 24h volume, 24h change and market cap.

    Tries Binance first, falls back to CoinGecko. Binance is preferred
    because it's where our candle history (get_ohlc_candles) also comes
    from - using the same exchange for both keeps the ticker and the
    chart's last candle consistent with each other, and Binance's ticker
    is a real, tick-by-tick exchange price rather than CoinGecko's
    aggregated read (which only refreshes on CoinGecko's own cadence,
    independent of how often we poll it).
    """
    if vs_currency.lower() == "usd":
        try:
            url = f"{config.BINANCE_API_BASE}/ticker/24hr"
            data = _request_with_retry(url, {"symbol": config.BINANCE_SYMBOL}, max_retries=1, timeout=6)
            return {
                "price": float(data["lastPrice"]),
                "market_cap": None,
                "volume_24h": float(data["quoteVolume"]),
                "change_24h_pct": float(data["priceChangePercent"]),
                "source": "binance",
            }
        except MarketDataError as exc:
            logger.warning("Binance current price failed (%s), falling back to CoinGecko", exc)

    try:
        url = f"{config.COINGECKO_API_BASE}/simple/price"
        params = {
            "ids": coin_id,
            "vs_currencies": vs_currency,
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }
        data = _request_with_retry(url, params, max_retries=1, timeout=6)[coin_id]
        return {
            "price": data.get(vs_currency),
            "market_cap": data.get(f"{vs_currency}_market_cap"),
            "volume_24h": data.get(f"{vs_currency}_24h_vol"),
            "change_24h_pct": data.get(f"{vs_currency}_24h_change"),
            "source": "coingecko",
        }
    except (MarketDataError, KeyError) as exc:
        logger.warning("CoinGecko current price failed (%s), falling back to Etherscan", exc)

    # Last resort: Etherscan (ETH/USD only, no volume/market cap - needs ETHERSCAN_API_KEY).
    if coin_id == "ethereum" and vs_currency.lower() == "usd":
        from data_collector.etherscan_client import get_eth_price

        etherscan_price = get_eth_price()
        if etherscan_price:
            return {
                "price": etherscan_price["price"],
                "market_cap": None,
                "volume_24h": None,
                "change_24h_pct": None,
                "source": "etherscan",
            }

    raise MarketDataError(f"All configured price sources failed for {coin_id}/{vs_currency}")


def get_ohlc_candles(
    interval: str = "1h",
    limit: int = 1000,
    coin_id: str = config.COIN_ID,
    symbol: str = config.BINANCE_SYMBOL,
    vs_currency: str = config.VS_CURRENCY,
) -> list[dict]:
    """OHLC candlestick history for `interval` in {"15m", "1h", "4h", "1d", "1w"}."""
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"interval must be one of {SUPPORTED_INTERVALS}, got {interval!r}")

    try:
        url = f"{config.BINANCE_API_BASE}/klines"
        raw = _request_with_retry(url, {"symbol": symbol, "interval": interval, "limit": limit})
        return [
            {
                "timestamp": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                # Buyer-initiated (taker-buy) share of volume - Binance breaks
                # every kline down into buy-side vs sell-side taker volume,
                # a real proxy for "who's more aggressive right now" (cererea
                # vs. oferta) rather than volume alone. See ml/features.py's
                # taker_buy_ratio.
                "taker_buy_volume": float(k[9]),
            }
            for k in raw
        ]
    except MarketDataError as exc:
        logger.warning("Binance klines failed (%s), falling back to CoinGecko OHLC", exc)

    days = _COINGECKO_DAYS_BY_INTERVAL[interval]
    url = f"{config.COINGECKO_API_BASE}/coins/{coin_id}/ohlc"
    raw = _request_with_retry(url, {"vs_currency": vs_currency, "days": days})
    candles = [
        {
            "timestamp": datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": None,  # CoinGecko's OHLC endpoint doesn't include volume
        }
        for row in raw
    ]
    return candles[-limit:]
