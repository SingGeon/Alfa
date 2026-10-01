import pytest

import config
from api import create_app
from api.services import get_prediction_accuracy
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
    # The evolving "tuned" population doesn't read news sentiment: no
    # importance to report, rather than a made-up one.
    assert body["sentiment_contribution_pct"] is None
    assert body["trained_ago_seconds"] >= 0


def test_predict_legacy_reports_sentiment_contribution(client, seeded):
    resp = client.get("/api/predict?interval=1h&steps=3&model_variant=legacy")
    assert resp.status_code == 200
    # sklearn's own learned feature importance for "sentiment" - real
    # signal from the trained model, not a fixed/made-up number - should
    # be a genuine share of 1.0 across all features.
    assert 0 <= resp.json["sentiment_contribution_pct"] <= 100


def test_predict_sentiment_contribution_none_without_sentiment(client, seeded):
    resp = client.get("/api/predict?interval=1h&steps=3&use_sentiment=false")
    assert resp.status_code == 200
    assert resp.json["sentiment_contribution_pct"] is None


def test_predict_rejects_bad_backend(client):
    resp = client.get("/api/predict?interval=1h&steps=3&backend=not-a-backend")
    assert resp.status_code == 400


def test_news_and_sentiment(client, seeded):
    news_resp = client.get("/api/news")
    assert news_resp.status_code == 200
    assert len(news_resp.json["news"]) == 1

    sentiment_resp = client.get("/api/sentiment")
    assert sentiment_resp.status_code == 200
    assert sentiment_resp.json["articles_analyzed"] == 1


def test_summary(client, seeded, synthetic_candles):
    # The daily summary always uses the "1d" interval now (see
    # run_combined_summary's docstring) regardless of what's passed here -
    # `seeded` alone (1h candles) isn't enough, so seed "1d" too.
    repository.save_candles("ethereum", "1d", synthetic_candles(200, interval_hours=24))
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    assert "narrative" in resp.json
    assert isinstance(resp.json["narrative"], str) and len(resp.json["narrative"]) > 0


def test_summary_agrees_with_daily_outlook_direction(client, seeded, synthetic_candles):
    # Regression test: the daily narrative used to be built from whatever
    # interval the main chart happened to be on (24 steps of it), which
    # could - and did, live - point the opposite direction from the "24h"
    # outlook tile's own 1d-interval forecast, despite both being shown
    # side by side under a "next ~24h" framing. Both now come from the
    # exact same interval="1d", steps=1 prediction, so they can't disagree.
    repository.save_candles("ethereum", "1d", synthetic_candles(200, interval_hours=24))
    # /api/outlook also computes a weekly leg - needs its own data too, or
    # the whole call 409s on the weekly leg alone.
    repository.save_candles("ethereum", "1w", synthetic_candles(200, interval_hours=24 * 7))

    outlook_resp = client.get("/api/outlook")
    assert outlook_resp.status_code == 200
    daily_change_pct = outlook_resp.json["daily"]["change_pct"]

    summary_resp = client.get("/api/summary")
    assert summary_resp.status_code == 200
    narrative = summary_resp.json["narrative"]

    if daily_change_pct == 0:
        assert "stay about where it is" in narrative
        assert "rise" not in narrative and "fall" not in narrative
    else:
        expected_word = "rise" if daily_change_pct > 0 else "fall"
        unexpected_word = "fall" if daily_change_pct > 0 else "rise"
        assert expected_word in narrative
        assert unexpected_word not in narrative


def test_predict_accuracy(client, synthetic_candles):
    repository.save_candles("ethereum", "1d", synthetic_candles(40, interval_hours=24))
    # Read back through the same repository call the service under test
    # uses, rather than the pre-insert Python objects: Mongo (real or
    # mongomock) truncates datetimes to millisecond precision, so the
    # exact microsecond-precision timestamps synthetic_candles() generates
    # wouldn't round-trip-match otherwise.
    stored = repository.get_candles("ethereum", "1d")
    target = stored[-1]["timestamp"]
    repository.save_prediction(
        "ethereum",
        {
            "interval": "1d",
            "predictions": [{"timestamp": repository.iso_utc(target), "predicted_price": stored[-1]["close"] + 10}],
        },
    )

    resp = client.get("/api/predict/accuracy?interval=1d&days=30")
    assert resp.status_code == 200
    body = resp.json
    assert len(body["points"]) == 1
    assert body["points"][0]["actual_price"] == round(stored[-1]["close"], 2)
    assert body["mape"] is not None


def test_predict_accuracy_rejects_bad_interval(client):
    resp = client.get("/api/predict/accuracy?interval=5m")
    assert resp.status_code == 400


def test_predict_accuracy_rejects_bad_backend(client):
    resp = client.get("/api/predict/accuracy?interval=1d&backend=not-a-backend")
    assert resp.status_code == 400


def test_predict_accuracy_filters_by_backend(synthetic_candles):
    # get_prediction_accuracy(backend=...) must only ever return that one
    # backend's own recorded forecasts, not every backend's blended
    # together under the same target timestamp - exercised directly
    # against the service layer (not the /api/predict/accuracy route,
    # which only ever validates "sklearn" since that's the only backend
    # PREDICTION_BACKENDS exposes today) so the underlying filtering
    # mechanism stays covered even with a single active backend.
    repository.save_candles("ethereum", "1d", synthetic_candles(40, interval_hours=24))
    stored = repository.get_candles("ethereum", "1d")
    target = stored[-1]["timestamp"]
    repository.save_prediction(
        "ethereum",
        {
            "interval": "1d",
            "backend": "sklearn",
            "predictions": [{"timestamp": repository.iso_utc(target), "predicted_price": stored[-1]["close"] + 10}],
        },
    )
    repository.save_prediction(
        "ethereum",
        {
            "interval": "1d",
            "backend": "other-model",
            "predictions": [{"timestamp": repository.iso_utc(target), "predicted_price": stored[-1]["close"] + 999}],
        },
    )

    result = get_prediction_accuracy("ethereum", "1d", days=30, backend="sklearn")
    assert len(result["points"]) == 1
    assert result["points"][0]["predicted_price"] == round(stored[-1]["close"] + 10, 2)

    result = get_prediction_accuracy("ethereum", "1d", days=30, backend="other-model")
    assert len(result["points"]) == 1
    assert result["points"][0]["predicted_price"] == round(stored[-1]["close"] + 999, 2)


def test_predict_accuracy_limit_trims_to_most_recent(client, synthetic_candles):
    repository.save_candles("ethereum", "1h", synthetic_candles(10, interval_hours=1))
    stored = repository.get_candles("ethereum", "1h")
    for candle in stored:
        repository.save_prediction(
            "ethereum",
            {
                "interval": "1h",
                "predictions": [{"timestamp": repository.iso_utc(candle["timestamp"]), "predicted_price": candle["close"] + 1}],
            },
        )

    resp = client.get("/api/predict/accuracy?interval=1h&limit=3")
    assert resp.status_code == 200
    body = resp.json
    assert body["limit"] == 3
    assert body["count"] == 3
    assert len(body["points"]) == 3
    # The trimmed points must be the 3 most recent (chronologically last), not the first 3.
    assert body["points"][-1]["timestamp"] == repository.iso_utc(stored[-1]["timestamp"])


def test_l2_tvls(client, requests_mock):
    requests_mock.get(
        "https://api.llama.fi/v2/chains",
        json=[
            {"name": "Arbitrum", "tvl": 400_000_000},
            {"name": "Base", "tvl": 10_000_000},
            {"name": "Unrelated Chain", "tvl": 999_000_000},
        ],
    )
    resp = client.get("/api/l2")
    assert resp.status_code == 200
    chains = resp.json["chains"]
    names = [c["name"] for c in chains]
    assert "Unrelated Chain" not in names
    assert names[0] == "Arbitrum"
    assert chains[0]["tier"] == "established"
    assert next(c for c in chains if c["name"] == "Base")["tier"] == "emerging"


def test_gas_not_configured(client, monkeypatch):
    monkeypatch.setattr(config, "ETHERSCAN_API_KEY", "")
    resp = client.get("/api/gas")
    assert resp.status_code == 501


def test_box_breakout_endpoint(client, seeded):
    resp = client.get("/api/box-breakout?interval=1h&lookback=10")
    assert resp.status_code == 200
    body = resp.json
    assert body["mode"] == "adaptive"
    assert {"boxes", "trades", "stats", "box_stats", "segments", "current", "active_box", "open_trade"} <= set(body)
    assert body["stats"]["bars"] > 0
    assert len(body["segments"]) >= 1


def test_box_breakout_endpoint_errors(client, seeded):
    assert client.get("/api/box-breakout?interval=3m").status_code == 400
    assert client.get("/api/box-breakout?interval=1d").status_code == 409  # no 1d candles seeded


def test_population_endpoints(client, synthetic_candles):
    repository.save_candles("ethereum", "1h", synthetic_candles(700))
    assert client.get("/api/evolution/report?interval=1h").status_code == 404  # not born yet
    assert client.get("/api/predict?interval=1h&steps=3&model_variant=tuned").status_code == 200
    report = client.get("/api/evolution/report?interval=1h")
    assert report.status_code == 200
    body = report.json
    assert {"summary", "causes", "timeline", "generations", "alive", "track_record"} <= set(body)
    assert body["summary"]["deaths"] == sum(c["deaths"] for c in body["causes"])
    deaths = client.get("/api/evolution/deaths?interval=1h&page_size=5").json
    assert deaths["total"] == body["summary"]["deaths"]
    if deaths["rows"]:
        oid = deaths["rows"][0]["id"]
        org = client.get(f"/api/evolution/organism/{oid}?interval=1h").json
        assert org["organism"]["id"] == oid and org["organism"]["dead"] is True
    assert client.get("/api/evolution/deaths?interval=1h&cause=nope").status_code == 400
    assert client.get("/api/evolution/organism/999999?interval=1h").status_code == 404
