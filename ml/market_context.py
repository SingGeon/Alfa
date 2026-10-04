"""Market context for the evolving population (ml/evolution.py): sentiment,
DeFi TVL and Binance futures, aligned to candles without look-ahead.

Columns added by add_context() (each only from information that existed
when the candle opened):
    fear_greed       Crypto Fear & Greed Index (alternative.me), previous
                     day's value, scaled to [-1, 1] ((v - 50) / 50).
                     Daily history since Feb 2018.
    fear_greed_chg   its change over the previous 7 days.
    news_sentiment   the app's own news sentiment (VADER, nlp/sentiment.py),
                     previous day's average in [-1, 1]; NaN on days without
                     articles (the app only has news since it started
                     collecting).
    tvl_chg_1d/7d    Ethereum DeFi TVL change (DeFiLlama), previous day's.
    funding          Binance perpetual funding rate, last one published at
                     or before the candle's open (every 8h, since ~2019).
    premium          Binance perpetual premium index (perp vs spot) on the
                     same candle, closed when the candle closes.

Every fetch is best-effort: a source that fails just leaves its column
missing, and the organisms that use it can't be born (ml/evolution.py).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_json(url, params=None):
    """GET with retries/backoff (data_collector.http_utils): a long paging
    over the whole history shouldn't die on one slow response."""
    from data_collector.http_utils import get_json

    return get_json(url, params=params, max_retries=6, timeout=30)

CONTEXT_COLUMNS = ("fear_greed", "fear_greed_chg", "news_sentiment", "tvl_chg_1d", "tvl_chg_7d", "funding", "premium")
_FAPI = "https://fapi.binance.com/fapi/v1"
_cache: dict[tuple, tuple[float, object]] = {}
_cache_lock = threading.Lock()


def _cached(key: tuple, ttl: float, fn):
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _cache_lock:
        _cache[key] = (time.monotonic(), value)
    return value


# --- sources ------------------------------------------------------------------

def fetch_fear_greed() -> pd.Series:
    """Daily index value (0-100), indexed by UTC date (Timestamp at 00:00)."""
    data = _get_json("https://api.alternative.me/fng/", params={"limit": 0})["data"]
    s = pd.Series({pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"): float(d["value"]) for d in data})
    return s.sort_index()


def fetch_funding(symbol: str, start: pd.Timestamp | None = None) -> pd.Series:
    """Funding rates by funding time. From `start` (paged), or the last 1000."""
    out = []
    params = {"symbol": symbol, "limit": 1000}
    if start is None:
        out = _get_json(f"{_FAPI}/fundingRate", params=params)
    else:
        params["startTime"] = int(start.timestamp() * 1000)
        while True:
            page = _get_json(f"{_FAPI}/fundingRate", params=params)
            if not isinstance(page, list) or not page:
                break
            out += page
            if len(page) < 1000:
                break
            params["startTime"] = page[-1]["fundingTime"] + 1
    if not isinstance(out, list):
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(r["fundingTime"], unit="ms", tz="UTC"): float(r["fundingRate"]) for r in out})
    return s.sort_index()


def fetch_premium(symbol: str, interval: str, total: int = 1000) -> pd.Series:
    """Premium index klines' close by open time, the newest `total`."""
    out, end = [], None
    while len(out) < total:
        params = {"symbol": symbol, "interval": interval, "limit": 1500}
        if end:
            params["endTime"] = end
        page = _get_json(f"{_FAPI}/premiumIndexKlines", params=params)
        if not isinstance(page, list) or not page:
            break
        out = page + out
        end = page[0][0] - 1
        if len(page) < 1500:
            break
    s = pd.Series({pd.Timestamp(k[0], unit="ms", tz="UTC"): float(k[4]) for k in out})
    return s[~s.index.duplicated()].sort_index().iloc[-total:]


def live_sources(symbol: str, interval: str) -> dict:
    """The sources for the live app, cached (they change slowly)."""
    def safe(name, fn):
        try:
            return fn()
        except Exception:
            logger.warning("Market context: %s unavailable", name, exc_info=True)
            return None

    return {
        "fear_greed": _cached(("fng",), 3600, lambda: safe("fear & greed", fetch_fear_greed)),
        "funding": _cached(("funding", symbol), 600, lambda: safe("funding", lambda: fetch_funding(symbol))),
        "premium": _cached(("premium", symbol, interval), 120, lambda: safe("premium", lambda: fetch_premium(symbol, interval))),
    }


# --- alignment ------------------------------------------------------------------

def _previous_day(index: pd.DatetimeIndex, daily: pd.Series) -> np.ndarray:
    """For each candle, the daily value of the day *before* its own UTC day
    (a day's value is only complete once the day is over)."""
    daily = daily.copy()
    daily.index = pd.DatetimeIndex(daily.index).tz_convert("UTC").normalize()
    daily = daily[~daily.index.duplicated(keep="last")].sort_index()
    days = index.tz_convert("UTC").normalize() - pd.Timedelta(days=1)
    filled = daily.reindex(daily.index.union(days.unique())).ffill()
    return filled.reindex(days).to_numpy(float)


def add_context(df: pd.DataFrame, fear_greed: pd.Series | None = None, funding: pd.Series | None = None,
                premium: pd.Series | None = None, sentiment_by_date: dict | None = None,
                tvl_by_date: dict | None = None) -> pd.DataFrame:
    """`df` (indexed by candle open time, UTC) with the CONTEXT_COLUMNS that
    could be built from the given sources."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if fear_greed is not None and len(fear_greed):
        fg = (fear_greed - 50) / 50
        out["fear_greed"] = _previous_day(idx, fg)
        out["fear_greed_chg"] = out["fear_greed"] - _previous_day(idx - pd.Timedelta(days=7), fg)
    if sentiment_by_date:
        news = pd.Series({pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d): v
                          for d, v in sentiment_by_date.items()}, dtype=float)
        # Days without articles stay NaN instead of carrying old news forward.
        days = idx.tz_convert("UTC").normalize() - pd.Timedelta(days=1)
        news.index = pd.DatetimeIndex(news.index).normalize()
        out["news_sentiment"] = news.reindex(days).to_numpy(float)
    if tvl_by_date:
        tvl = pd.Series({pd.Timestamp(d): float(v) for d, v in tvl_by_date.items()}).sort_index()
        tvl.index = pd.DatetimeIndex(tvl.index).tz_localize("UTC") if tvl.index.tz is None else tvl.index
        out["tvl_chg_1d"] = _previous_day(idx, tvl.pct_change(fill_method=None))
        out["tvl_chg_7d"] = _previous_day(idx, tvl.pct_change(7, fill_method=None))
    if funding is not None and len(funding):
        f = funding.sort_index()
        # Published at or before the candle's open, and not stale (> 1 day).
        pos = f.index.searchsorted(idx, side="right") - 1
        vals = np.where(pos >= 0, f.to_numpy()[np.clip(pos, 0, None)], np.nan)
        age = idx - f.index[np.clip(pos, 0, None)]
        out["funding"] = np.where((pos >= 0) & (age <= timedelta(days=1)), vals, np.nan)
    if premium is not None and len(premium):
        out["premium"] = premium.reindex(idx).to_numpy(float)
    return out
