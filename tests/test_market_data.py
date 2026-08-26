import config
from data_collector import market_data


def test_get_current_price_uses_coingecko_when_available(requests_mock):
    requests_mock.get(
        f"{config.COINGECKO_API_BASE}/simple/price",
        json={"ethereum": {"usd": 3200.5, "usd_market_cap": 1e9, "usd_24h_vol": 5e8, "usd_24h_change": 1.2}},
    )
    result = market_data.get_current_price("ethereum", "usd")
    assert result["price"] == 3200.5
    assert result["source"] == "coingecko"


def test_get_current_price_falls_back_to_binance(requests_mock):
    requests_mock.get(f"{config.COINGECKO_API_BASE}/simple/price", status_code=500)
    requests_mock.get(
        f"{config.BINANCE_API_BASE}/ticker/24hr",
        json={"lastPrice": "3100.0", "quoteVolume": "123456", "priceChangePercent": "-0.5"},
    )
    result = market_data.get_current_price("ethereum", "usd")
    assert result["price"] == 3100.0
    assert result["source"] == "binance"


def test_get_current_price_raises_when_all_sources_fail(requests_mock):
    requests_mock.get(f"{config.COINGECKO_API_BASE}/simple/price", status_code=500)
    requests_mock.get(f"{config.BINANCE_API_BASE}/ticker/24hr", status_code=500)
    try:
        market_data.get_current_price("ethereum", "usd")
        assert False, "expected MarketDataError"
    except market_data.MarketDataError:
        pass


def test_get_ohlc_candles_prefers_binance_klines(requests_mock):
    kline = [1710000000000, "3000", "3100", "2900", "3050", "12.5"]
    requests_mock.get(f"{config.BINANCE_API_BASE}/klines", json=[kline])
    candles = market_data.get_ohlc_candles("1h", limit=1)
    assert len(candles) == 1
    assert candles[0]["open"] == 3000.0
    assert candles[0]["close"] == 3050.0


def test_get_ohlc_candles_falls_back_to_coingecko(requests_mock):
    requests_mock.get(f"{config.BINANCE_API_BASE}/klines", status_code=500)
    row = [1710000000000, 3000, 3100, 2900, 3050]
    requests_mock.get(
        f"{config.COINGECKO_API_BASE}/coins/{config.COIN_ID}/ohlc",
        json=[row],
    )
    candles = market_data.get_ohlc_candles("1h", limit=10)
    assert len(candles) == 1
    assert candles[0]["close"] == 3050


def test_invalid_interval_raises():
    try:
        market_data.get_ohlc_candles("5m")
        assert False, "expected ValueError"
    except ValueError:
        pass
