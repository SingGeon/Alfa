import pytest

import config
from api import create_app
from database import repository
from nlp.sentiment import score_article


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def seeded(synthetic_candles):
    repository.save_candles("ethereum", "1h", synthetic_candles(200))
    repository.save_news_items(
        [
            score_article(
                {
                    "title": "Ethereum rallies on ETF optimism",
                    "url": "http://x.test/1",
                    "source": "test",
                    "published_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                    "summary": "",
                }
            )
        ]
    )


def test_health(client):
    assert client.get("/api/health").json == {"status": "ok"}


def test_price_history(client, seeded):
    resp = client.get("/api/price/history?interval=1h&limit=5")
    assert resp.status_code == 200
    assert len(resp.json["candles"]) == 5


def test_price_history_rejects_bad_interval(client):
    resp = client.get("/api/price/history?interval=5m")
    assert resp.status_code == 400


def test_predict_insufficient_data(client):
    resp = client.get("/api/predict?interval=1h&steps=3")
    assert resp.status_code == 409


def test_predict_success(client, seeded):
    resp = client.get("/api/predict?interval=1h&steps=3")
    assert resp.status_code == 200
    body = resp.json
    assert len(body["predictions"]) == 3
    assert 0 <= body["confidence"] <= 100


def test_news_and_sentiment(client, seeded):
    news_resp = client.get("/api/news")
    assert news_resp.status_code == 200
    assert len(news_resp.json["news"]) == 1

    sentiment_resp = client.get("/api/sentiment")
    assert sentiment_resp.status_code == 200
    assert sentiment_resp.json["articles_analyzed"] == 1


def test_summary(client, seeded):
    resp = client.get("/api/summary?interval=1h")
    assert resp.status_code == 200
    assert "narrative" in resp.json
    assert isinstance(resp.json["narrative"], str) and len(resp.json["narrative"]) > 0


def test_gas_not_configured(client, monkeypatch):
    monkeypatch.setattr(config, "ETHERSCAN_API_KEY", "")
    resp = client.get("/api/gas")
    assert resp.status_code == 501
