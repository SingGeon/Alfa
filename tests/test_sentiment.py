from datetime import datetime, timezone

from nlp.sentiment import aggregate_daily, score_article, score_text


def test_score_text_positive():
    result = score_text("Ethereum surges to new all-time high, investors thrilled")
    assert result["compound"] > 0
    assert result["label"] == "positive"


def test_score_text_negative():
    result = score_text("Ethereum crashes amid regulatory crackdown fears")
    assert result["compound"] < 0
    assert result["label"] == "negative"


def test_score_text_empty():
    result = score_text("")
    assert result["compound"] == 0.0
    assert result["label"] == "neutral"


def test_score_article_preserves_fields():
    article = {"title": "ETH rallies", "url": "http://x.test", "summary": "great news"}
    scored = score_article(article)
    assert scored["url"] == "http://x.test"
    assert "sentiment" in scored
    assert "compound" in scored["sentiment"]


def test_aggregate_daily_averages_by_date():
    now = datetime.now(timezone.utc)
    articles = [
        {"published_at": now, "sentiment": {"compound": 0.5}},
        {"published_at": now, "sentiment": {"compound": -0.1}},
    ]
    daily = aggregate_daily(articles)
    assert len(daily) == 1
    assert abs(list(daily.values())[0] - 0.2) < 1e-9
