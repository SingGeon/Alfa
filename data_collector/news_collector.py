"""Crypto news collection: CryptoPanic + NewsAPI (both optional, need API
keys) with an RSS fallback (CoinDesk/Cointelegraph) that needs no key at
all, so the app has *something* to analyze out of the box.

Every source is normalized to the same shape:
    {"title", "url", "source", "published_at": datetime, "summary"}
so nlp/sentiment.py and the rest of the pipeline don't care where an
article came from.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

import feedparser

import config
from data_collector.http_utils import ExternalAPIError, get_json

logger = logging.getLogger(__name__)

# Fetching + parsing all configured RSS feeds is the dominant cost of
# collect_news() (~2s across 10 feeds) and is keyword-independent - every
# caller re-fetches the exact same feed content and just filters it
# differently. That's fine for the ETH dashboard (one caller, minutes
# apart) but was multiplying badly for the scout scan (one fetch per
# asset x ~80 assets) and made every detail-page click pay the full 2s
# too. Caching the unfiltered entries for a few minutes fixes both -
# RSS feeds don't publish anywhere near that fast anyway.
_RAW_ENTRIES_CACHE_TTL_SECONDS = 300
_raw_entries_cache: list[dict] | None = None
_raw_entries_cached_at: float = 0.0


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    """Word-boundary match so short tokens like "eth" don't hit "method"/"wealth"."""
    pattern = r"\b(?:" + "|".join(re.escape(kw) for kw in keywords) + r")\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _from_cryptopanic() -> list[dict]:
    if not config.CRYPTOPANIC_API_KEY:
        return []
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {"auth_token": config.CRYPTOPANIC_API_KEY, "currencies": "ETH", "public": "true"}
    try:
        data = get_json(url, params=params)
    except ExternalAPIError as exc:
        logger.warning("CryptoPanic fetch failed: %s", exc)
        return []
    items = []
    for post in data.get("results", []):
        published = post.get("published_at")
        items.append(
            {
                "title": post.get("title", ""),
                "url": post.get("url", ""),
                "source": (post.get("source") or {}).get("title", "cryptopanic"),
                "published_at": _parse_datetime(published),
                "summary": "",
            }
        )
    return items


def _from_newsapi() -> list[dict]:
    if not config.NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "ethereum",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 50,
        "apiKey": config.NEWSAPI_KEY,
    }
    try:
        data = get_json(url, params=params)
    except ExternalAPIError as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return []
    items = []
    for art in data.get("articles", []):
        items.append(
            {
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": (art.get("source") or {}).get("name", "newsapi"),
                "published_at": _parse_datetime(art.get("publishedAt")),
                "summary": art.get("description") or "",
            }
        )
    return items


def _fetch_raw_rss_entries() -> list[dict]:
    """Every entry from every configured feed, unfiltered - cached for
    _RAW_ENTRIES_CACHE_TTL_SECONDS (see module docstring for why).
    """
    global _raw_entries_cache, _raw_entries_cached_at
    now = time.monotonic()
    if _raw_entries_cache is not None and now - _raw_entries_cached_at < _RAW_ENTRIES_CACHE_TTL_SECONDS:
        return _raw_entries_cache

    items = []
    for feed_url in config.NEWS_RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:  # feedparser swallows most errors, but be defensive
            logger.warning("RSS fetch failed for %s: %s", feed_url, exc)
            continue
        feed_title = parsed.feed.get("title", feed_url)
        for entry in parsed.entries:
            published = None
            if entry.get("published_parsed"):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            items.append(
                {
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "source": feed_title,
                    "published_at": published or datetime.now(timezone.utc),
                    "summary": entry.get("summary", ""),
                }
            )
    _raw_entries_cache = items
    _raw_entries_cached_at = now
    return items


def _from_rss(keywords: list[str] | None = None) -> list[dict]:
    """No API key needed - always available, used as the baseline source."""
    keywords = keywords or config.NEWS_KEYWORDS
    return [
        item
        for item in _fetch_raw_rss_entries()
        if _matches_keywords(f"{item['title']} {item['summary']}", keywords)
    ]


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def collect_news(keywords: list[str] | None = None) -> list[dict]:
    """Aggregate + dedupe (by URL) articles from every enabled source.

    `keywords` defaults to config.NEWS_KEYWORDS (ETH) - pass a different
    list (e.g. ["bitcoin", "btc"]) to filter the same RSS feeds for a
    different asset, as scout/pipeline.py does for the multi-asset scan.
    CryptoPanic/NewsAPI stay ETH-only (their queries are hardcoded) since
    they're optional, keyed integrations outside this app's default path.
    """
    all_items = [*_from_cryptopanic(), *_from_newsapi(), *_from_rss(keywords)]
    seen = set()
    deduped = []
    for item in all_items:
        if not item["url"] or item["url"] in seen:
            continue
        seen.add(item["url"])
        deduped.append(item)
    return deduped
