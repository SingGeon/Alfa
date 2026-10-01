"""Per-asset analysis for the scout: real price history + real news +
the same PricePredictor/sentiment stack the ETH dashboard uses.

Two consumers share the same fetch + train/predict core:
  - scan_crypto_asset/scan_stock_asset: compact score for the ranked table
    (scout/scanner.py), only the final predicted point.
  - detail_for_crypto_asset/detail_for_stock_asset: full candle history +
    the whole prediction series + news + a narrative, for the per-asset
    detail page (api/routes.py's /api/scout/detail).

Deliberately decoupled from api/services.py: that module reads candles
back out of our Mongo `candles` collection, which only ever stores ETH
(config.COIN_ID). Scanning 30+ other assets on every run would mean
storing full OHLC history for all of them too - unnecessary, since both
consumers here only need a fresh read per asset, not a persisted history.
So this fetches candles directly from the source (Binance/CoinGecko for
crypto, yfinance for stocks) on demand and only the scan path persists a
*result* (see scout/scanner.py -> database/repository.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import config
from data_collector import market_data, news_collector
from data_collector.http_utils import ExternalAPIError, get_json
from ml.features import build_feature_frame
from ml.price_predictor import PricePredictor
from nlp import sentiment
from scout.narrative import generate_asset_narrative

_BTC_COIN_ID = "bitcoin"

logger = logging.getLogger(__name__)

MIN_CANDLES = 60
PREDICTION_STEPS = 7  # ~1 week ahead at daily granularity
PREDICTION_INTERVAL = "1d"


def _is_placeholder_candle(c: dict) -> bool:
    """A real trading day never has open==high==low==close AND zero
    volume - that combination is yfinance's tell for a synthetic
    "no trading happened" row (seen trailing the real history, i.e. a
    forward-filled placeholder for a day the market hasn't actually
    traded yet). Left in, it becomes the anchor for the whole forecast:
    a fake flat "last price" with no real high/low to build features from.
    """
    return c["open"] == c["high"] == c["low"] == c["close"] and not c.get("volume")


def _drop_incomplete_candles(candles: list[dict]) -> list[dict]:
    """Drop NaN-OHLC rows (e.g. yfinance's "today" row with a NaN close
    while the market's still open - using that as the anchor poisons every
    downstream feature) and trailing zero-volume placeholder rows.
    """
    cleaned = [c for c in candles if all(c[k] == c[k] for k in ("open", "high", "low", "close"))]
    while cleaned and _is_placeholder_candle(cleaned[-1]):
        cleaned.pop()
    return cleaned


def _momentum_score(change_pct: float, confidence: float, sentiment_avg: float) -> float:
    """Simple, transparent heuristic - not a guarantee. Weighs predicted
    upside by how confident the model is in it, plus a sentiment nudge.
    """
    return change_pct * (confidence / 100) + sentiment_avg * 10


# Fallback trailing daily volatility (as a fraction, e.g. 0.01 = 1%/day) for
# the handful of rows where volatility_24 comes back NaN/zero (not enough
# rolling history yet) - a middling, not-too-calm-not-too-wild default
# rather than treating missing data as either extreme.
_DEFAULT_DAILY_VOL = 0.01

# See _scout_confidence's docstring for why this replaces
# ml.combined_predictor.compute_confidence here specifically.
_CONFIDENCE_SCALE = 50.0


def _scout_confidence(df, predictions: list[dict], sentiment_avg: float) -> float:
    """Confidence scaled by each asset's OWN trailing volatility, not
    ml.combined_predictor.compute_confidence's fixed absolute-width cutoff
    (~20% interval width -> 0 confidence, tuned around ETH's own typically
    calm volatility on the dashboard that function actually serves).

    Reusing that fixed cutoff here was a real bug, confirmed directly
    against Scout's live universe: two genuine picks predicting +17% and
    +26% 7-day moves scored 9/100 and 5/100 confidence purely because
    their interval was wide in absolute price terms - not because the
    model was actually any less sure about them than usual, but because a
    volatile underdog's honest interval is *always* wide in absolute
    terms. Meanwhile assets predicting next to no movement at all (-1% to
    +1%) scored 70-80/100 just for being calm. That's backwards for a
    "will it rise a lot" scout: it structurally suppressed exactly the
    dramatic-but-genuine calls Scout AI exists to surface, both directly
    (MIN_CONFIDENCE cutoff in scanner.py) and indirectly (score is
    change_pct * confidence/100, so even picks that cleared the cutoff
    got their score gutted).

    Fix: compare the interval's width not to a fixed number, but to what
    a well-behaved random-walk forecast over this same horizon would
    produce given *this asset's own* recent volatility (volatility_24 -
    already a feature column, no extra computation) - i.e. "is this
    interval wide even accounting for how much this specific asset
    normally moves", not just "is this interval wide in dollar terms".
    A calm mega-cap with a proportionally wide interval still reads as
    low-confidence; a volatile memecoin with an interval no wider than
    its own history would predict no longer does.
    """
    last = predictions[-1]
    interval_width_pct = (last["upper"] - last["lower"]) / max(last["predicted_price"], 1e-9)

    daily_vol = df["volatility_24"].iloc[-1]
    if daily_vol != daily_vol or daily_vol <= 0:  # NaN or non-positive
        daily_vol = _DEFAULT_DAILY_VOL
    # Random-walk assumption: an N-day-ahead price's spread grows with
    # sqrt(N) times the daily volatility - the same "expected width" a
    # calibrated model should produce purely from this asset's own normal
    # day-to-day noise, before any real predictive signal narrows it.
    expected_width_pct = daily_vol * (PREDICTION_STEPS ** 0.5)
    relative_width = interval_width_pct / max(expected_width_pct, 1e-9)
    base_confidence = min(100.0, max(0.0, 100.0 - relative_width * _CONFIDENCE_SCALE))

    first_price = predictions[0]["predicted_price"]
    predicted_direction = 1 if last["predicted_price"] >= first_price else -1
    sentiment_direction = 1 if sentiment_avg > 0.05 else (-1 if sentiment_avg < -0.05 else 0)
    if sentiment_direction == 0:
        agreement_adjustment = 0.0
    elif sentiment_direction == predicted_direction:
        agreement_adjustment = 5.0
    else:
        agreement_adjustment = -10.0
    return round(min(100.0, max(0.0, base_confidence + agreement_adjustment)), 1)


# -- fetching (per asset type) -------------------------------------------

def _btc_close_series(asset_coin_id: str):
    """BTC's own close series, for ml/features.py's btc_return_1 - most alts
    move with BTC far more than on their own fundamentals, so "what's the
    broader crypto market doing" is a real, cheap signal to give the model.
    None for BTC itself (never actually reached - scout's universe excludes
    mega-caps) or if the fetch fails.
    """
    import pandas as pd

    if asset_coin_id == _BTC_COIN_ID:
        return None
    try:
        btc_candles = market_data.get_ohlc_candles(
            interval=PREDICTION_INTERVAL, limit=200, coin_id=_BTC_COIN_ID, symbol="BTCUSDT"
        )
    except ExternalAPIError:
        logger.warning("BTC candle fetch failed - btc_return_1 will default neutral")
        return None
    btc_candles = _drop_incomplete_candles(btc_candles)
    if not btc_candles:
        return None
    sorted_candles = sorted(btc_candles, key=lambda c: c["timestamp"])
    index = pd.to_datetime([c["timestamp"] for c in sorted_candles], utc=True)
    return pd.Series([c["close"] for c in sorted_candles], index=index)


def _fetch_crypto(asset: dict) -> tuple[list[dict], list[dict], object]:
    """`asset` = {"coin_id", "symbol", "name", "binance_symbol"} from scout.universe."""
    candles = market_data.get_ohlc_candles(
        interval=PREDICTION_INTERVAL, limit=200,
        coin_id=asset["coin_id"], symbol=asset["binance_symbol"],
    )
    candles = _drop_incomplete_candles(candles)
    keywords = [asset["name"].lower(), asset["symbol"].lower()]
    try:
        articles = news_collector.collect_news(keywords=keywords)
    except Exception:
        logger.exception("News collection failed for %s", asset["coin_id"])
        articles = []
    btc_close = _btc_close_series(asset["coin_id"])
    return candles, articles, btc_close


def _fetch_stock(ticker: str) -> tuple[list[dict], list[dict]]:
    import yfinance as yf

    from scout import retry_yfinance

    # A fresh yf.Ticker() built inside the retried call itself, not once
    # before it - Ticker binds yfinance's process-wide session singleton
    # at construction time (TickerBase.__init__: self._data = YfData(...)),
    # so reusing the same Ticker object across retries would keep reusing
    # the exact same (possibly stuck) session retry_yfinance is trying to
    # get past. See retry_yfinance's docstring for why that session can
    # get stuck at all.
    t = None

    def _fetch_history():
        nonlocal t
        t = yf.Ticker(ticker)
        return t.history(period="1y", interval="1d")

    hist = retry_yfinance(_fetch_history, is_empty=lambda h: h is None or h.empty, label=f"history({ticker})")
    if hist is None or hist.empty:
        return [], []

    candles = _drop_incomplete_candles(
        [
            {
                "timestamp": ts.to_pydatetime().astimezone(timezone.utc),
                "open": float(row.Open),
                "high": float(row.High),
                "low": float(row.Low),
                "close": float(row.Close),
                "volume": float(row.Volume),
            }
            for ts, row in hist.iterrows()
        ]
    )

    articles = []
    try:
        for item in t.news or []:
            c = item.get("content", {})
            pub_raw = c.get("pubDate")
            try:
                published = datetime.fromisoformat(pub_raw.replace("Z", "+00:00")) if pub_raw else None
            except ValueError:
                published = None
            articles.append(
                {
                    "title": c.get("title", ""),
                    "url": (c.get("canonicalUrl") or {}).get("url", ""),
                    "source": (c.get("provider") or {}).get("displayName", "Yahoo Finance"),
                    "published_at": published or datetime.now(timezone.utc),
                    "summary": c.get("summary", ""),
                }
            )
    except Exception:
        logger.warning("News fetch failed for %s", ticker)
    return candles, articles


# -- shared train/predict core --------------------------------------------

def _train_and_predict(candles: list[dict], articles: list[dict], btc_close=None, ensemble_size: int = 3):
    """Returns (df, predictions, sentiment_avg, scored_articles) or None if
    there isn't enough real data to train on.

    ensemble_size=1 for the on-demand detail view (someone is actively
    waiting on this one), full ensemble for the background scan (no one's
    watching a spinner, so it can afford the smoother/steadier score - see
    ml/price_predictor.py's PricePredictor.__init__ for why this trades
    smoothing for speed rather than accuracy for speed).
    """
    if len(candles) < MIN_CANDLES:
        return None

    scored_articles = [sentiment.score_article(a) for a in articles]
    sentiment_map = sentiment.aggregate_daily(scored_articles)
    sentiment_avg = sum(sentiment_map.values()) / len(sentiment_map) if sentiment_map else 0.0

    df = build_feature_frame(candles, sentiment_map, btc_close=btc_close)
    train_df = df.dropna()
    if len(train_df) < 30:
        return None

    # Scout uses the original ("legacy") model on purpose: the evolving
    # "tuned" population is for the main dashboard's coins.
    predictor = PricePredictor(backend=config.PREDICTION_BACKEND, ensemble_size=ensemble_size, model_variant="legacy")
    try:
        predictor.fit(train_df)
        predictions = predictor.predict(df, steps=PREDICTION_STEPS, interval=PREDICTION_INTERVAL)
    except Exception:
        logger.exception("Training/prediction failed")
        return None
    if not predictions:
        return None

    return df, predictions, sentiment_avg, scored_articles


def _scan_result(asset_type: str, asset_id: str, symbol: str, name: str, candles, articles, btc_close=None) -> dict | None:
    trained = _train_and_predict(candles, articles, btc_close=btc_close)
    if trained is None:
        return None
    df, predictions, sentiment_avg, scored_articles = trained

    last_price = float(df["close"].iloc[-1])
    final = predictions[-1]
    change_pct = (final["predicted_price"] - last_price) / last_price * 100 if last_price else 0.0
    confidence = _scout_confidence(df, predictions, sentiment_avg)
    score = _momentum_score(change_pct, confidence, sentiment_avg)

    return {
        "asset_type": asset_type,
        "id": asset_id,
        "symbol": symbol,
        "name": name,
        "current_price": last_price,
        "predicted_price": final["predicted_price"],  # full precision - some trade at sub-cent values
        "predicted_change_pct": round(change_pct, 2),
        "confidence": confidence,
        "sentiment_avg": round(sentiment_avg, 3),
        "articles_analyzed": len(scored_articles),
        "score": round(score, 2),
        "horizon_days": PREDICTION_STEPS,
    }


def _detail_result(
    asset_type: str, asset_id: str, symbol: str, name: str, candles, articles, btc_close=None, lang: str = "en",
) -> dict | None:
    trained = _train_and_predict(candles, articles, btc_close=btc_close, ensemble_size=1)
    if trained is None:
        return None
    df, predictions, sentiment_avg, scored_articles = trained

    last_price = float(df["close"].iloc[-1])
    confidence = _scout_confidence(df, predictions, sentiment_avg)
    narrative = generate_asset_narrative(name, symbol, last_price, predictions, sentiment_avg, scored_articles, lang=lang)

    return {
        "asset_type": asset_type,
        "id": asset_id,
        "symbol": symbol,
        "name": name,
        "current_price": last_price,
        "confidence": confidence,
        "sentiment_avg": round(sentiment_avg, 3),
        "narrative": narrative,
        "horizon_days": PREDICTION_STEPS,
        "candles": [
            {"timestamp": ts.isoformat(), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
            for ts, r in df.iterrows()
        ],
        "predictions": [
            {
                "timestamp": p["timestamp"].isoformat(),
                "predicted_price": round(p["predicted_price"], 8),
                "lower": round(p["lower"], 8),
                "upper": round(p["upper"], 8),
            }
            for p in predictions
        ],
        "news": [
            {
                "title": a["title"],
                "url": a["url"],
                "source": a["source"],
                "published_at": a["published_at"].isoformat(),
                "sentiment": a["sentiment"],
            }
            for a in scored_articles
        ],
    }


# -- public API -------------------------------------------------------------

def analyze_crypto_asset(asset: dict) -> dict | None:
    try:
        candles, articles, btc_close = _fetch_crypto(asset)
    except Exception:
        logger.warning("Candle fetch failed for %s", asset["coin_id"])
        return None
    return _scan_result("crypto", asset["coin_id"], asset["symbol"], asset["name"], candles, articles, btc_close)


def analyze_stock_asset(asset: dict) -> dict | None:
    try:
        candles, articles = _fetch_stock(asset["ticker"])
    except Exception:
        logger.warning("History fetch failed for %s", asset["ticker"])
        return None
    return _scan_result("stock", asset["ticker"], asset["ticker"], asset.get("name", asset["ticker"]), candles, articles)


def detail_for_crypto(coin_id: str, symbol: str, name: str, binance_symbol: str, lang: str = "en") -> dict | None:
    candles, articles, btc_close = _fetch_crypto(
        {"coin_id": coin_id, "symbol": symbol, "name": name, "binance_symbol": binance_symbol}
    )
    return _detail_result("crypto", coin_id, symbol, name, candles, articles, btc_close, lang=lang)


def detail_for_stock(ticker: str, name: str, lang: str = "en") -> dict | None:
    candles, articles = _fetch_stock(ticker)
    return _detail_result("stock", ticker, ticker, name, candles, articles, lang=lang)


def get_live_price_crypto(binance_symbol: str, coin_id: str) -> dict | None:
    """Fast standalone quote (no candles, no news, no model) for the detail
    page's live-price poll - Binance first, CoinGecko fallback.
    """
    try:
        data = get_json(
            f"{config.BINANCE_API_BASE}/ticker/24hr", {"symbol": binance_symbol}, max_retries=1, timeout=6
        )
        return {"price": float(data["lastPrice"]), "change_24h_pct": float(data["priceChangePercent"]), "source": "binance"}
    except ExternalAPIError:
        pass
    try:
        data = get_json(
            f"{config.COINGECKO_API_BASE}/simple/price",
            {"ids": coin_id, "vs_currencies": "usd", "include_24hr_change": "true"},
            max_retries=1, timeout=6,
        )[coin_id]
        return {"price": data.get("usd"), "change_24h_pct": data.get("usd_24h_change"), "source": "coingecko"}
    except (ExternalAPIError, KeyError):
        return None


def get_live_price_stock(ticker: str) -> dict | None:
    """Fast standalone quote for a stock - fast_info is a lightweight
    endpoint (no full history download), unlike .history().
    """
    import yfinance as yf

    from scout import retry_yfinance

    def _fetch():
        info = yf.Ticker(ticker).fast_info
        price = info.get("lastPrice")
        if price is None:
            return None
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else None
        return {"price": float(price), "change_24h_pct": change_pct, "source": "yahoo"}

    result = retry_yfinance(_fetch, is_empty=lambda r: r is None, label=f"fast_info({ticker})")
    if result is None:
        logger.warning("Live price fetch failed for %s", ticker)
    return result
