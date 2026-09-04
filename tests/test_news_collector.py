from types import SimpleNamespace

import config
from data_collector import news_collector


def _fake_feed(entries):
    return SimpleNamespace(feed={"title": "Fake Feed"}, entries=entries)


def test_from_rss_filters_by_keyword(monkeypatch):
    entries = [
        {"title": "Ethereum hits new high", "summary": "ETH news", "link": "http://a.test/1", "published_parsed": None},
        {"title": "Random cat video goes viral", "summary": "", "link": "http://a.test/2", "published_parsed": None},
    ]
    monkeypatch.setattr(news_collector.feedparser, "parse", lambda url: _fake_feed(entries))
    items = news_collector._from_rss()
    urls = [i["url"] for i in items]
    assert "http://a.test/1" in urls
    assert "http://a.test/2" not in urls


def test_from_rss_handles_parse_errors(monkeypatch):
    def boom(url):
        raise ValueError("bad feed")

    monkeypatch.setattr(news_collector.feedparser, "parse", boom)
    assert news_collector._from_rss() == []


def test_from_cryptopanic_skipped_without_key(monkeypatch):
    monkeypatch.setattr(config, "CRYPTOPANIC_API_KEY", "")
    assert news_collector._from_cryptopanic() == []


def test_collect_news_dedupes_across_sources(monkeypatch):
    monkeypatch.setattr(news_collector, "_from_cryptopanic", lambda: [
        {"title": "A", "url": "http://x.test/1", "source": "cp", "published_at": None, "summary": ""}
    ])
    monkeypatch.setattr(news_collector, "_from_newsapi", lambda: [
        {"title": "A dup", "url": "http://x.test/1", "source": "na", "published_at": None, "summary": ""}
    ])
    monkeypatch.setattr(news_collector, "_from_rss", lambda keywords=None: [
        {"title": "B", "url": "http://x.test/2", "source": "rss", "published_at": None, "summary": ""}
    ])
    result = news_collector.collect_news()
    assert len(result) == 2
    assert {r["url"] for r in result} == {"http://x.test/1", "http://x.test/2"}
