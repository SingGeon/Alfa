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


def test_wrong_models_lose_energy():
    ev = EvolutionaryForecaster().fit(_frame())
    org = ev.alive[0]
    before = org.energy
    org.recent = [(0, -2.0)]  # a resolved prediction much worse than "no change"
    ev.pending.clear()
    ev._step(ev.t_last)
    assert org.energy < before or org not in ev.alive


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


def test_longer_horizons_and_combos_earn_more_energy():
    ev = EvolutionaryForecaster().fit(_frame())
    ev.pending.clear()
    a, b, c = ev.alive[:3]
    for o in (a, b, c):
        o.energy = 100.0
    a.recent = [(0, 0.5)]                                   # right 1 candle ahead
    b.recent = [(5, 0.5)]                                   # right 24 candles ahead
    c.recent = [(0, 0.5), (3, 0.5), (5, 0.5)]               # right on 3 horizons at once
    combos_before = (a.combos, c.combos)
    ev._step(ev.t_last)
    # Same skill, but the 3-horizon combo earns the bonus on top.
    assert c.energy > b.energy
    assert (a.combos, c.combos) == (combos_before[0], combos_before[1] + 1)


def test_bigger_misses_cost_more_than_proportionally():
    ev = EvolutionaryForecaster().fit(_frame())
    ev.pending.clear()
    t = ev.t_last
    y = ev.logc[t] - ev.logc[t - 1]
    a, b = ev.alive[:2]
    for o in (a, b):
        o.energy, o.recent = 100.0, []
    scale = ev.scale[0]
    # Predictions for horizon 1 made at t-1: a misses by 1 "scale", b by 2.
    ev.pending[t - 1] = {a.id: np.full(6, y + np.sign(y or 1) * (abs(y) + 1 * scale)),
                         b.id: np.full(6, y + np.sign(y or 1) * (abs(y) + 2 * scale))}
    ev._step(t)
    loss_a, loss_b = 100.0 - a.energy, 100.0 - b.energy
    assert loss_b > 2 * loss_a


def test_every_death_is_recorded_with_why_and_where():
    ev = EvolutionaryForecaster().fit(_frame())
    assert len(ev.graveyard) == ev.deaths
    rec = ev.graveyard[-1]
    assert rec["cause"] in {"fatal_miss", "repeated_misses", "exhaustion"}
    assert pd.Timestamp(rec["died"]) >= pd.Timestamp(rec["born"])
    assert rec["age"] > 0 and rec["price"] > 0
    assert set(rec["inputs"]) <= set(ev.group_names)
    assert {"shares", "boldness", "temperament", "news_sensitivity"} <= set(ev.history[-1])


def test_fatal_miss_cause():
    ev = EvolutionaryForecaster().fit(_frame())
    org = ev.alive[0]
    org.trace = [(0.0, 0.0)] * 10 + [(0.0, 40.0)]
    assert ev._death_record(org, ev.t_last)["cause"] == "fatal_miss"
    org.trace = [(0.0, 2.0)] * 20
    assert ev._death_record(org, ev.t_last)["cause"] == "repeated_misses"
    org.trace = [(0.1, 0.0)] * 20
    assert ev._death_record(org, ev.t_last)["cause"] == "exhaustion"
