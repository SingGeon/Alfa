import numpy as np
import pandas as pd

from ml import strategy_evolution as se
from ml import strategy_lab as lab


def _daily(n=1600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D", tz="UTC")
    # Volatility that clusters (calm and wild stretches), like the real thing.
    vol = np.where((np.arange(n) // 120) % 2 == 0, 0.01, 0.05)
    close = 3000 * np.exp(np.cumsum(rng.normal(0.0005, 1, n) * vol))
    return pd.DataFrame({"open": close, "high": close * 1.02, "low": close * 0.98, "close": close,
                         "volume": rng.uniform(900, 1100, n), "taker_buy_volume": 500.0}, index=idx)


def test_every_trade_pays_the_fee():
    n = 10
    mask = np.ones(n, dtype=bool)
    flat = np.zeros(n)
    r = lab.simulate(np.ones(n), flat, 365, mask)        # buy once, price never moves
    assert r.money == round(lab.START * (1 - lab.FEE), 2) and r.trades == 1
    flip = np.array([1.0, -1.0] * 5)                     # long, short, long...: 2x the money changes hands each time
    assert lab.simulate(flip, flat, 365, mask).fees > 9 * 2 * lab.FEE * 90


def test_wealth_matches_simulate():
    df = _daily(400)
    r_next = np.log(df["close"]).diff().shift(-1).fillna(0).values
    pos = np.sign(np.sin(np.arange(400) / 9))
    mask = np.ones(400, dtype=bool)
    assert round(float(lab.wealth(pos, r_next, mask)[-1]), 2) == lab.simulate(pos, r_next, 365, mask).money


def test_volatility_forecast_uses_only_the_past():
    df = _daily()
    full = lab.forecast_volatility(df, "1d")
    cut = df.index[1200]
    short = lab.forecast_volatility(df[df.index <= cut], "1d")
    both = full[short.index].dropna()
    assert len(both) > 300
    assert np.allclose(both.values, short[both.index].values)


def test_volatility_forecast_follows_the_regime():
    df = _daily()
    sigma = lab.forecast_volatility(df, "1d")
    realized = np.log(df["close"]).diff()[::-1].rolling(24).std()[::-1].shift(-1)
    ok = sigma.notna() & realized.notna()
    assert np.corrcoef(np.log(sigma[ok]), np.log(realized[ok]))[0, 1] > 0.3


def test_strategy_fund_never_looks_ahead():
    # The fund on the first 1300 days must hold exactly what the fund on all
    # 1600 held on those days: later candles can't change past decisions.
    df = _daily()
    full_ctx = lab.context(df, "1d")
    short_ctx = lab.context(df.iloc[:1300], "1d")
    full, _, _ = se.evolve(full_ctx, seed=1)
    short, log, _ = se.evolve(short_ctx, seed=1)
    assert log, "the fund should have started trading"
    assert np.allclose(full[:1300], short)


def test_selection_keeps_what_made_money():
    # In a market that only goes up, the surviving strategies hold more ETH
    # than the random ones they started from.
    df = _daily()
    df["close"] = 3000 * np.exp(np.cumsum(0.004 + np.random.default_rng(3).normal(0, 0.02, len(df))))
    df["high"], df["low"], df["open"] = df["close"] * 1.01, df["close"] * 0.99, df["close"]
    ctx = lab.context(df, "1d")
    fund, log, final = se.evolve(ctx, seed=2)
    assert len(final) == se.TOP_K
    assert fund[-100:].mean() > 0.3
