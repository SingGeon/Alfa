"""Which assets the scout scans - underdogs, not blue chips.

The point of a "will rise a lot" scout is mid/small-cap names with real
room to move, not BTC/ETH/AAPL/MSFT (already priced efficiently, rarely
move double digits). So both universes deliberately exclude the mega-caps
and pull from sources built for exactly this:

Crypto: mid/small-cap coins (CoinGecko rank 30-250, i.e. real projects
with real trading history, just not yet mega-cap) + coins CoinGecko's own
"trending" (real search-interest spike) list surfaces outside the top 30.

Stocks: Yahoo Finance's own live screeners for this - small_cap_gainers
(momentum already showing), aggressive_small_caps and
undervalued_growth_stocks (statistical/fundamental angle) - real,
continuously-refreshed lists, not a static curated set.

Both universes are further filtered down to what's actually tradeable on
Revolut (see revolut_allowlist.py) - a hot pick the user can't buy through
their own broker isn't useful, however strong the signal.
"""
from __future__ import annotations

import logging

import config
from data_collector.http_utils import ExternalAPIError, get_json
from scout.revolut_allowlist import REVOLUT_CRYPTO_SYMBOLS, REVOLUT_STOCK_EXCHANGES

logger = logging.getLogger(__name__)

# Well-known stablecoins to exclude from the crypto universe (id -> price
# pegged near $1 makes "will it moonshot" a meaningless question). Backed up
# by a price-band check below for anything not on this list.
_STABLECOIN_IDS = {
    "tether", "usd-coin", "dai", "usds", "usd1-wlfi", "ethena-usde",
    "global-dollar", "true-usd", "frax", "paxos-standard", "gemini-dollar",
    "binance-usd", "first-digital-usd", "usdd",
    # Not $1-pegged, but still low-volatility-by-design (gold/yield-backed
    # real-world-asset tokens) - not what an "underdog" scout is for either.
    "tether-gold", "pax-gold", "hashnote-usyc", "ondo-us-dollar-yield",
}

# Exclude these from "underdog" consideration - already mega-cap, not what
# an underdog scout is for.
_MEGA_CAP_RANK_CUTOFF = 30

_STOCK_SCREENS = ("small_cap_gainers", "aggressive_small_caps", "undervalued_growth_stocks")


def _is_stablecoin(coin: dict) -> bool:
    if coin.get("id") in _STABLECOIN_IDS:
        return True
    price = coin.get("current_price")
    return price is not None and 0.985 <= price <= 1.015


def get_crypto_universe(limit: int = 40) -> list[dict]:
    """Mid/small-cap coins + trending-but-not-mega-cap coins, stablecoins excluded.

    Returns [{"coin_id", "symbol", "name", "binance_symbol"}, ...]. The
    Binance pair is a best-effort guess (SYMBOL+USDT); market_data.py's
    existing Binance->CoinGecko fallback covers coins where that pair
    doesn't actually exist.
    """
    seen_ids: set[str] = set()
    universe: list[dict] = []

    def _add(coin_id: str, symbol: str, name: str) -> None:
        if coin_id in seen_ids or not symbol:
            return
        # The user trades through Revolut - a pick they can't actually buy
        # there isn't a real opportunity for them, however strong the signal.
        if symbol.upper() not in REVOLUT_CRYPTO_SYMBOLS:
            return
        seen_ids.add(coin_id)
        universe.append(
            {"coin_id": coin_id, "symbol": symbol.upper(), "name": name, "binance_symbol": f"{symbol.upper()}USDT"}
        )

    # 1) Broad mid/small-cap pool: fetch enough of the market-cap-ranked
    # list to slice off everything above the mega-cap cutoff.
    try:
        markets = get_json(
            f"{config.COINGECKO_API_BASE}/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1, "sparkline": "false"},
            max_retries=1, timeout=10,
        )
    except ExternalAPIError as exc:
        logger.warning("Failed to fetch crypto market-cap list: %s", exc)
        markets = []
    for coin in markets[_MEGA_CAP_RANK_CUTOFF:]:
        if _is_stablecoin(coin):
            continue
        _add(coin.get("id", ""), coin.get("symbol", ""), coin.get("name", ""))
        if len(universe) >= limit:
            break

    # 2) Real-time "trending" (search-interest spike) coins, same mega-cap exclusion.
    if len(universe) < limit:
        try:
            trending = get_json(f"{config.COINGECKO_API_BASE}/search/trending", max_retries=1, timeout=10)
        except ExternalAPIError as exc:
            logger.warning("Failed to fetch trending coins: %s", exc)
            trending = {}
        for entry in trending.get("coins", []):
            item = entry.get("item", {})
            rank = item.get("market_cap_rank")
            if rank is not None and rank <= _MEGA_CAP_RANK_CUTOFF:
                continue
            _add(item.get("id", ""), item.get("symbol", ""), item.get("name", ""))
            if len(universe) >= limit:
                break

    return universe


def get_crypto_mega_cap_universe(limit: int = _MEGA_CAP_RANK_CUTOFF) -> list[dict]:
    """The top-{limit}-by-market-cap coins get_crypto_universe excludes on
    purpose (see _MEGA_CAP_RANK_CUTOFF) - scanned separately so a mega-cap
    can still surface in Scout AI when the model calls for a genuinely
    dramatic move, without loosening the "underdog" bar for everything
    else. See scanner.py's MEGA_CAP_MIN_CHANGE_PCT for the stricter
    inclusion threshold this tier is held to.
    """
    universe: list[dict] = []
    try:
        markets = get_json(
            f"{config.COINGECKO_API_BASE}/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": limit, "page": 1, "sparkline": "false"},
            max_retries=1, timeout=10,
        )
    except ExternalAPIError as exc:
        logger.warning("Failed to fetch top-tier crypto market-cap list: %s", exc)
        return []
    for coin in markets:
        if _is_stablecoin(coin):
            continue
        symbol = (coin.get("symbol") or "").upper()
        coin_id = coin.get("id", "")
        if not symbol or not coin_id or symbol not in REVOLUT_CRYPTO_SYMBOLS:
            continue
        universe.append(
            {"coin_id": coin_id, "symbol": symbol, "name": coin.get("name", ""), "binance_symbol": f"{symbol}USDT"}
        )
    return universe


# Words that show up in the *name* of derivative instruments (warrants,
# rights, units, preferred shares) rather than common stock. yfinance
# tags all of these as quoteType "EQUITY" too, so it can't be filtered
# that way - the name is the only reliable, honest signal available.
# These trade on thin volume with erratic, near-random price action
# (nothing to do with the underlying company's prospects), which is
# exactly what produced 0/100-confidence, all-noise predictions.
_NON_COMMON_STOCK_MARKERS = ("warrant", "right", " unit", "preferred", " pfd")


def _is_common_stock(name: str) -> bool:
    lowered = name.lower()
    return not any(marker in lowered for marker in _NON_COMMON_STOCK_MARKERS)


def get_stock_universe(limit: int = 40) -> list[dict]:
    """Real, live Yahoo Finance screeners for small/growth names - see module docstring."""
    import yfinance as yf

    from scout import retry_yfinance

    seen: set[str] = set()
    universe: list[dict] = []
    for screen_id in _STOCK_SCREENS:
        result = retry_yfinance(
            lambda sid=screen_id: yf.screen(sid, count=25),
            is_empty=lambda r: not r or not r.get("quotes"),
            label=f"screen({screen_id})",
        )
        if not result:
            logger.warning("Yahoo screen %r came back empty after retries", screen_id)
            continue
        for quote in result.get("quotes", []):
            symbol = quote.get("symbol")
            name = quote.get("shortName") or symbol
            if not symbol or symbol in seen or not _is_common_stock(name):
                continue
            # Same reasoning as the crypto allowlist above: only exchanges
            # Revolut Trading actually covers, see revolut_allowlist.py.
            if quote.get("exchange") not in REVOLUT_STOCK_EXCHANGES:
                continue
            seen.add(symbol)
            universe.append({"ticker": symbol, "name": name})
            if len(universe) >= limit:
                return universe
    return universe


# "Large/mega cap" floor for get_stock_mega_cap_universe below - well above
# anything get_stock_universe's small-cap screens would ever surface, so
# the two universes stay meaningfully distinct tiers rather than overlapping.
_MEGA_CAP_MARKET_CAP_FLOOR = 50_000_000_000  # $50B


def get_stock_mega_cap_universe(limit: int = 25) -> list[dict]:
    """Real, live large/mega-cap movers (Yahoo's day_gainers screen,
    filtered to an actual mega-cap market cap floor) - the tier
    get_stock_universe's small-cap screens structurally never surface.
    See scanner.py's MEGA_CAP_MIN_CHANGE_PCT for the stricter bar this
    tier has to clear before it actually shows up in Scout AI results.
    """
    import yfinance as yf

    from scout import retry_yfinance

    seen: set[str] = set()
    universe: list[dict] = []
    result = retry_yfinance(
        lambda: yf.screen("day_gainers", count=50),
        is_empty=lambda r: not r or not r.get("quotes"),
        label="screen(day_gainers)",
    )
    if not result:
        logger.warning("Yahoo day_gainers screen came back empty after retries")
        return []
    for quote in result.get("quotes", []):
        symbol = quote.get("symbol")
        name = quote.get("shortName") or symbol
        if not symbol or symbol in seen or not _is_common_stock(name):
            continue
        if (quote.get("marketCap") or 0) < _MEGA_CAP_MARKET_CAP_FLOOR:
            continue
        if quote.get("exchange") not in REVOLUT_STOCK_EXCHANGES:
            continue
        seen.add(symbol)
        universe.append({"ticker": symbol, "name": name})
        if len(universe) >= limit:
            break
    return universe
