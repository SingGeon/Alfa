from datetime import date

import numpy as np
import pandas as pd

from ml.market_context import add_context
from ml.evolution import EvolutionaryForecaster, mood_signal

IDX = pd.date_range("2026-03-01", periods=72, freq="h", tz="UTC")
DF = pd.DataFrame({"close": np.linspace(100, 110, len(IDX))}, index=IDX)


def test_daily_values_come_from_the_previous_day():
    fg = pd.Series([20.0, 80.0, 50.0], index=pd.date_range("2026-03-01", periods=3, freq="D", tz="UTC"))
    out = add_context(DF, fear_greed=fg)
    assert np.isnan(out.loc["2026-03-01 05:00", "fear_greed"])           # no day before
    assert out.loc["2026-03-02 23:00", "fear_greed"] == (20 - 50) / 50   # Mar 1's value all of Mar 2
    assert out.loc["2026-03-03 00:00", "fear_greed"] == (80 - 50) / 50


def test_funding_only_once_published():
    f = pd.Series([0.001, 0.002], index=pd.to_datetime(["2026-03-01 08:00", "2026-03-01 16:00"], utc=True))
    out = add_context(DF, funding=f)
    assert np.isnan(out.loc["2026-03-01 07:00", "funding"])
    assert out.loc["2026-03-01 08:00", "funding"] == 0.001
    assert out.loc["2026-03-01 15:00", "funding"] == 0.001
    assert out.loc["2026-03-01 16:00", "funding"] == 0.002
    assert np.isnan(out.loc["2026-03-03 20:00", "funding"])               # > 1 day stale


def test_news_only_on_days_with_articles():
    out = add_context(DF, sentiment_by_date={date(2026, 3, 1): 0.4})
    assert out.loc["2026-03-02 10:00", "news_sentiment"] == 0.4
    assert np.isnan(out.loc["2026-03-03 10:00", "news_sentiment"])


def test_mood_signal_prefers_news_then_fear_greed():
    df = pd.DataFrame({"news_sentiment": [0.3, np.nan, np.nan], "fear_greed": [-0.5, -0.5, np.nan]})
    value, source = mood_signal(df)
    assert list(value) == [0.3, -0.5, 0.0]
    assert list(source) == ["news", "fear_greed", ""]


def test_context_gene_not_used_before_its_data_exists():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=900, freq="h", tz="UTC")
    close = 3000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    df = pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996, "close": close,
                       "volume": rng.uniform(900, 1100, len(idx))}, index=idx)
    df["funding"] = np.nan
    df.iloc[600:, df.columns.get_loc("funding")] = rng.normal(0, 1e-4, 300)
    df["premium"] = np.nan
    ev = EvolutionaryForecaster().fit(df)
    assert "futures" in ev.group_names
    assert "futures" not in ev._usable_groups(500)
    for o in ev.alive:
        if "futures" in o.groups:
            assert o.born_at > 600
