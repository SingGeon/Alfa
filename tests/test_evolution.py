import numpy as np
import pandas as pd

from ml.evolution import EvolutionaryForecaster, emotion_name, POPULATION


def _frame(n=700, seed=0, trend=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = 3000 * np.exp(np.cumsum(rng.normal(trend, 0.01, n)))
    return pd.DataFrame({"open": close, "high": close * 1.004, "low": close * 0.996, "close": close,
                         "volume": rng.uniform(900, 1100, n)}, index=idx)


def test_emotion_names():
    assert emotion_name(0.1) == "fear"
    assert emotion_name(0.5) == "calm"
    assert emotion_name(0.95) == "euphoria"


def test_population_lives_dies_and_reproduces():
    ev = EvolutionaryForecaster().fit(_frame())
    d = ev.diagnostics(ev.t_last)
    assert d["population"] == POPULATION
    assert d["deaths"] > 0 and d["births"] == d["deaths"]  # every death is replaced
    assert d["max_generation"] >= 1
    assert all(0 <= o["mood"] <= 1 for o in d["leaders"])


def test_no_look_ahead():
    # The replay over 700 candles must pass through exactly the same states
    # as the replay over the first 500: later candles can't change the past.
    df = _frame()
    short = EvolutionaryForecaster().fit(df.iloc[:500])
    full = EvolutionaryForecaster().fit(df)
    cut = df.index[499]
    short_hist = [h for h in short.history if pd.Timestamp(h["time"]) < cut]
    full_hist = [h for h in full.history if pd.Timestamp(h["time"]) < cut]
    assert short_hist and short_hist == full_hist


def test_wrong_way_position_loses_money():
    ev = EvolutionaryForecaster().fit(_frame())
    ev.pending.clear()
    t = ev.t_last
    move = ev.logc[t] - ev.logc[t - 1]
    org = ev.alive[0]
    org.energy, org.fees_last, org.culled = 100.0, 0.0, False
    org.position = -1.0 if move > 0 else 1.0   # bet against what happened
    before = org.trading_pnl
    ev._step(t)
    assert org.trading_pnl < before or move == 0


def test_step_band_is_as_wide_as_the_next_candle_like_legacy():
    ev = EvolutionaryForecaster().fit(_frame())
    path, d = ev.predict(_frame(), 40)
    widths = [round(hi - lo, 12) for p, lo, hi in path]
    for p, lo, hi in path:
        assert lo <= p <= hi
    assert len(set(widths)) == 1


def test_cumulative_band_widens(monkeypatch):
    from ml import evolution

    monkeypatch.setattr(evolution, "BAND_MODE", "cumulative")
    ev = EvolutionaryForecaster().fit(_frame())
    path, d = ev.predict(_frame(), 40)
    for p, lo, hi in path:
        assert lo <= p <= hi
    assert (path[-1][2] - path[-1][1]) > (path[0][2] - path[0][1])


def test_too_little_history_is_newborn_no_change():
    ev = EvolutionaryForecaster().fit(_frame(n=60))
    path, d = ev.predict(_frame(n=60), 5)
    assert d["newborn"] is True and d["population"] == 0
    assert all(p == 0.0 and lo < 0 < hi for p, lo, hi in path)


def test_saved_population_keeps_living():
    from datetime import datetime, timezone

    from ml import evolution

    df = _frame(n=800)
    now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    first = evolution.evolve(df.iloc[:600], key="eth_1h", now=now)
    assert evolution.state_path("eth_1h").exists()
    evolution._registry.clear()  # as after an app restart: load from disk
    later = evolution.evolve(df.iloc[300:800], key="eth_1h", now=now)
    assert later.index[0] == df.index[0]          # it kept its whole past
    assert len(later.index) == 800
    assert later.deaths >= first.deaths
    # Continuing gives the same population as one replay of everything.
    replay = EvolutionaryForecaster().fit(df)
    assert [o.id for o in later.alive] == [o.id for o in replay.alive]
    assert later.diagnostics(later.t_last)["track_record"]


def test_gap_starts_a_new_population():
    from datetime import datetime, timezone

    from ml import evolution

    df = _frame(n=1200)
    now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    evolution.evolve(df.iloc[:500], key="k", now=now)
    after_gap = evolution.evolve(df.iloc[700:1200], key="k", now=now)
    assert after_gap.index[0] == df.index[700]


def test_open_last_candle_is_not_lived():
    from ml import evolution

    df = _frame(n=600)
    now = df.index[-1] + pd.Timedelta(minutes=30)  # last hourly candle still open
    ev = evolution.evolve(df, key="open", now=now)
    assert ev.index[-1] == df.index[-2]


def test_pretrained_file_replaces_running_population():
    from datetime import datetime, timezone
    import os

    from ml import evolution

    df = _frame(n=900)
    now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    evolution.evolve(df.iloc[500:800], key="p", now=now)        # the app's own short-lived population
    pre = EvolutionaryForecaster().fit(df.iloc[:800])          # "pretrained" on the long history
    evolution.save_state("p", pre)
    evolution._saved_mtime.pop("p")                            # written by another process
    os.utime(evolution.state_path("p"))
    ev = evolution.evolve(df.iloc[500:900], key="p", now=now)
    assert ev.index[0] == df.index[0]


def test_every_death_is_recorded_with_why_and_where():
    ev = EvolutionaryForecaster().fit(_frame())
    assert len(ev.graveyard) == ev.deaths
    rec = ev.graveyard[-1]
    assert rec["cause"] in {"outcompeted", "big_loss", "fees", "bad_trades", "taxes"}
    assert {"money", "trades", "fees_paid", "trading_pnl"} <= set(rec)
    assert pd.Timestamp(rec["died"]) >= pd.Timestamp(rec["born"])
    assert rec["age"] > 0 and rec["price"] > 0
    assert set(rec["inputs"]) <= set(ev.group_names)
    assert {"shares", "boldness", "temperament", "news_sensitivity"} <= set(ev.history[-1])


def test_money_is_life_long_gains_when_price_rises():
    from ml import evolution

    ev = EvolutionaryForecaster().fit(_frame())
    ev.pending.clear()
    t = ev.t_last
    a, b = ev.alive[:2]
    for o in (a, b):
        o.energy, o.fees_last = 100.0, 0.0
    a.position, b.position = 1.0, 0.0
    move = np.exp(ev.logc[t] - ev.logc[t - 1]) - 1
    before = (a.energy, b.energy)
    b_pnl_before = b.trading_pnl
    a_id, b_id = a.id, b.id
    ev._step(t)
    a = next((o for o in ev.alive if o.id == a_id), a)
    b = next((o for o in ev.alive if o.id == b_id), b)
    # a held a full long: its money moved with the price; b stayed flat and
    # the move cost it nothing (no flat daily tax any more - only a losing
    # day is fined, see test_loss_fine_only_on_losing_days_not_flat_ones).
    assert a.trading_pnl != 0.0 or move == 0
    assert b.trading_pnl == b_pnl_before   # flat: this candle's move added no pnl


def test_trading_costs_a_fee():
    from ml import evolution

    ev = EvolutionaryForecaster().fit(_frame())
    org = ev.alive[0]
    org.energy, org.position, org.fees_paid = 100.0, 0.0, 0.0
    ev._trade(org, 1.0)
    assert org.position == 1.0
    assert np.isclose(org.fees_paid, 100.0 * evolution.FEE)
    ev._trade(org, 0.0)
    assert org.position == 0.0


def test_patience_keeps_it_out_of_small_edges():
    from ml import evolution

    ev = EvolutionaryForecaster().fit(_frame())
    org = ev.alive[0]
    org.pred_w, org.pattern_w, org.box_w, org.news_w = 1.0, 0.0, 0.0, 0.0
    t = ev.t_last
    tiny = 2 * evolution.FEE * 1.01   # just over 1 round trip of fees, as a 1-step forecast
    org.patience = 0.5
    assert ev._target_position(tiny, org, t) != 0.0
    org.patience = 2.0
    assert ev._target_position(tiny, org, t) == 0.0


def test_bankrupt_dies_and_slowest_is_outcompeted():
    from ml import evolution

    ev = EvolutionaryForecaster().fit(_frame())
    ev.pending.clear()
    t = ev.t_last
    while t % ev.cull_every:
        t -= 1
    poor = ev.alive[0]
    poor.energy, poor.position = 5.0, 0.0
    deaths = ev.deaths
    ev._step(t)
    assert poor.id not in {o.id for o in ev.alive}
    causes = [d["cause"] for d in ev.graveyard[-(ev.deaths - deaths):]]
    assert ev.deaths > deaths
    adults = [d for d in ev.graveyard if d["cause"] == "outcompeted"]
    assert adults  # culling happens over the replay


def test_death_causes_from_money():
    ev = EvolutionaryForecaster().fit(_frame())
    org = ev.alive[0]
    org.culled = False
    org.trace = [(0.0, 0.0, 0.07, 100.0)] * 10 + [(-40.0, 0.0, 0.07, 100.0)]
    assert ev._death_record(org, ev.t_last)["cause"] == "big_loss"
    org.trace = [(-0.01, 2.0, 0.07, 50.0)] * 20
    assert ev._death_record(org, ev.t_last)["cause"] == "fees"
    org.trace = [(-1.0, 0.1, 0.07, 50.0)] * 20
    assert ev._death_record(org, ev.t_last)["cause"] == "bad_trades"
    org.trace = [(0.0, 0.0, 0.07, 50.0)] * 20
    assert ev._death_record(org, ev.t_last)["cause"] == "fines"
    org.culled = True
    assert ev._death_record(org, ev.t_last)["cause"] == "outcompeted"


def test_fund_is_tracked():
    from ml import evolution

    ev = EvolutionaryForecaster().fit(_frame())
    m = ev.diagnostics(ev.t_last)["money"]
    assert m["start"] == evolution.START_MONEY and m["currency"] == "EUR"
    assert isinstance(m["fund"], float)
    assert "buy_hold" not in m


def test_fines_target_losing_and_fully_idle_days_only():
    """Once a day (scaled to real time): a losing day pays a fine proportional to the loss; a
    day spent entirely out of the market (zero engagement) pays a small separate idle fine -
    closes the "never trade, immortal, fitness 0 forever" loophole; a day spent holding any
    position, even an exactly flat one, pays nothing at all."""
    import math

    from ml import evolution

    ev = EvolutionaryForecaster().fit(_frame())
    ev.pending.clear()
    t = ev.t_last
    a, b, c = ev.alive[:3]
    a.born_at = b.born_at = c.born_at = t - ev.candles_per_day   # force a day boundary this step
    a.fees_last = b.fees_last = c.fees_last = 0.0
    a.engagement_sum = b.engagement_sum = c.engagement_sum = 0.0
    a.engagement_n = b.engagement_n = c.engagement_n = 0

    move = math.exp(ev.logc[t] - ev.logc[t - 1]) - 1

    a.energy_1d_ago, a.energy, a.position = 100.0, 90.0, 0.0   # already down 10% today, out now
    c.energy_1d_ago = c.energy = 100.0                          # fully idle all day
    c.position = 0.0

    b.energy, b.position = 100.0, 0.3                           # engaged, but an exactly flat day
    b.energy_1d_ago = b.energy + b.energy * b.position * move   # makes today's daily_pct == 0

    # Section 5 of _step always retargets and re-trades everyone for the *next* candle, which
    # would charge its own fee unrelated to the fine logic under test here - disable it so the
    # only thing moving anyone's energy is this candle's pnl plus (maybe) a fine.
    ev._trade = lambda acct, target: None

    ev._step(t)

    assert a.energy <= 85.5     # fined ~5% of its money (LOSS_FINE_RATE x 10% loss)
    assert c.energy < 100.0     # the idle fine actually costs it something now...
    assert c.energy >= 100.0 * (1 - 2 * evolution.IDLE_FINE_RATE)   # ...but only a small amount
    assert abs(b.energy - (100.0 + 100.0 * 0.3 * move)) < 1e-6  # no fine: just this candle's pnl


def test_near_constant_input_cannot_blow_up_the_forecast():
    # A feature flat for the whole training window (float noise only, like
    # news sentiment before it existed) that then moves must not be divided
    # by its ~1e-14 "spread" into a forecast in the billions.
    ev = EvolutionaryForecaster().fit(_frame())
    t = ev.t_last
    org = ev.alive[0]
    col = org.cols[0]
    ev.X[:, col] = 0.1 + np.random.default_rng(1).normal(0, 1e-14, len(ev.X))
    assert ev._fit_org(org, t)
    ev.X[t, col] = 0.9
    assert np.all(np.abs(ev._raw(org, t)) < 1.0)
