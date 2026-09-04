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
from datetime import datetime, timezone

import feedparser

import config
from data_collector.http_utils import ExternalAPIError, get_json

logger = logging.getLogger(__name__)


def _matches_keywords(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


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


def _from_rss() -> list[dict]:
    """No API key needed - always available, used as the baseline source."""
    items = []
    for feed_url in config.NEWS_RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as exc:  # feedparser swallows most errors, but be defensive
            logger.warning("RSS fetch failed for %s: %s", feed_url, exc)
            continue
        for entry in parsed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            if not _matches_keywords(f"{title} {summary}", config.NEWS_KEYWORDS):
                continue
            published = None
            if entry.get("published_parsed"):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            items.append(
                {
                    "title": title,
                    "url": entry.get("link", ""),
                    "source": parsed.feed.get("title", feed_url),
                    "published_at": published or datetime.now(timezone.utc),
                    "summary": summary,
                }
            )
    return items


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def collect_news() -> list[dict]:
    """Aggregate + dedupe (by URL) articles from every enabled source."""
    all_items = [*_from_cryptopanic(), *_from_newsapi(), *_from_rss()]
    seen = set()
    deduped = []
    for item in all_items:
        if not item["url"] or item["url"] in seen:
            continue
        seen.add(item["url"])
        deduped.append(item)
    return deduped
