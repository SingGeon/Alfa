"""Evolutionary forecaster with emotions (the "tuned" model variant).

A population of small organisms lives on the price history. **The only goal is money, full
stop - never prediction accuracy.** Each organism combines four weighted "tools" into one
position every candle:

  - price: a 1-step ridge regression (log return over the next candle), sized by half-Kelly
    and scaled by the organism's mood/boldness (see Emotions below) - a cheaper, less noisy
    signal than forecasting far ahead, and just one tool among others, not the model's purpose.
  - pattern: ml.pattern_recognition's geometric/candlestick signal, recomputed on a cadence
    scaled to real time (every ~24h), shared by the whole population (not gene-dependent).
  - box: ml.box_breakout's adaptive Box Breakout position, same walk-forward mechanism the
    app's own Box Breakout feature uses, shared by the whole population.
  - news: the market's sentiment (mood_signal: news where there is any, else Fear & Greed),
    traded directly, not just used to sway mood.

Each organism's `pred_w`/`pattern_w`/`box_w`/`news_w` genes decide how much it leans on each
tool; evolution (not the author) decides, per interval, which tools are actually worth
listening to.

  - Money is life. Every organism gets START_MONEY (1000 EUR, virtual) and trades it on its
    combined position: long, short or out of the market, up to its whole money (no leverage),
    paying FEE per side like Binance. There is no flat daily tax - instead, once a day
    (scaled to real time per interval - see INTERVAL_HOURS), an organism that lost money that
    day pays a fine proportional to the loss (none if it made money or stayed flat); dying
    happens only below BANKRUPT (~0 EUR), not at some arbitrary fraction of its start money.
    Every CULL_EVERY candles (also real-time-scaled) the adult with the worst fitness is
    replaced by a child of the rich ("outcompeted"), even if it isn't bankrupt - fitness is
    growth(t) plus a bonus that grows as its average daily return approaches, then passes, a
    10%/day aspirational target (only while genuinely engaged - a flat position earns no
    bonus for "0% risked"), minus a drawdown penalty past a per-interval threshold.
  - Birth. Dead organisms are replaced by children of the strongest living ones (tournament
    selection on money): genes mixed from two parents, then mutated.
  - Genes. Which price-feature inputs it looks at (momentum, trend, long context, volatility,
    volume, oscillators, BTC, and when available: sentiment, news, DeFi TVL, Binance futures -
    see ml/market_context.py), how much history it learns from, how strongly it is
    regularized, its boldness, temperament, news sensitivity, patience, and its four tool
    weights (pred_w/pattern_w/box_w/news_w).
  - Time-scaling. CULL_EVERY/REFIT_EVERY/ADULT_AGE and the window gene's range are candle
    counts derived from a fixed real-time duration (INTERVAL_HOURS), not copied unscaled
    across intervals - "24 candles" is 1 day at 1h but 24 weeks at 1w, and judging/refitting
    an organism that fast relative to its own interval starves it of real signal. This was the
    actual cause of early losses at 15m/1h; see notebooks/genetic_strategy_lab.ipynb.
  - Emotions. Each organism has a mood in [0, 1]: its recent rate of profitable candles,
    remembered for about 1 / temperament candles. The news moves it too: the day's sentiment
    shifts its mood by up to +/-0.5 x its news sensitivity gene (which can be negative, a
    contrarian). Its *price* prediction is scaled by mood x boldness before entering the
    weighted position - a frightened organism barely commits to its own forecast; a confident
    one does. Moods are named fear / caution / calm / confidence / euphoria.

The forecast shown on the price chart/band is still the energy-weighted ensemble of the
TOP_K strongest adults' price tool, flat-extrapolated across HORIZONS for compatibility with
the existing band calibration and track record - but that forecast is diagnostic only. The
fund's actual trading, and every organism's fitness, is driven purely by the combined 4-tool
position; `fund_trades` (ml/evolution.py's own entry/exit log, same shape as
ml/box_breakout.py's) is what should be charted, not the predicted price.

Everything is a deterministic replay (fixed seed) of the history given to fit(), one candle at
a time, using only data available at that candle. A population can also keep living:
evolve(df, key, interval) continues the saved population for `key` with the candles that
closed since, instead of starting over, and saves it to data/evolution/. Run
`python -m ml.pretrain_evolution` once to let it live through years of old candles first.
"""
from __future__ import annotations

import copy
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HORIZONS = (1, 2, 4, 8, 12, 24)
H_MAX = max(HORIZONS)
HORIZON_ARRAY = np.array(HORIZONS, dtype=float)
POPULATION = 24
TOP_K = 8
# Money (virtual, EUR). Returns are ETH/USDT's, so the EUR/USD rate is ignored.
START_MONEY = 1000.0
START_ENERGY = START_MONEY   # energy is money
FEE = 0.001                 # per side, of the traded amount (Binance spot taker)
KELLY_FRACTION = 0.5        # half-Kelly: the edge is only an estimate, full Kelly over-bets
BANKRUPT = 0.01              # dies only below ~0 EUR, not at some fraction of START_MONEY
MIN_REBALANCE = 0.05         # ignore position changes smaller than this (fees) - validated in
                              # notebooks/test_rebalance_sweep.py: larger values hurt 15m
VOTE_WEIGHT_CAP = 3 * START_MONEY
# Profit-only fitness (no prediction-accuracy term anywhere): growth(t) (log money growth per
# candle) plus a bonus that grows as the average daily return approaches, then exceeds,
# TARGET_DAILY_PCT - but only while genuinely engaged (ENGAGEMENT_MIN: a flat position earns
# no bonus for "0% risked"), minus a drawdown penalty past the interval's dd threshold.
# Losing days (not flat ones) pay LOSS_FINE_RATE x the loss, once a day - this replaces the
# old flat DAILY_TAX, which taxed the whole population even on days nobody actually lost
# trading money (see notebooks/genetic_strategy_lab.ipynb for why that sank the old fund).
# A day spent entirely out of the market (zero engagement, not just zero pnl) pays a small
# separate IDLE_FINE_RATE - without it, "never trade, ever" is a risk-free, zero-cost,
# fitness-0-forever strategy that no amount of real edge in an active trader can beat at
# culling time (fitness almost always < 0 given ~50% direction accuracy and fees), so the
# population drifts toward total paralysis instead of toward money. Small enough that an
# organism trading occasionally (patience gene) barely notices it; an organism that never
# once engages compounds it for as long as it lives.
TARGET_DAILY_PCT = 10.0
NEAR_BONUS = 0.05
EXCESS_BONUS = 0.10
LOSS_FINE_RATE = 0.5
IDLE_FINE_RATE = 0.0005
ENGAGEMENT_MIN = 0.1
DD_PENALTY = 1.0
# dd threshold swept per interval in notebooks/sweep_dd_threshold.py +
# notebooks/sweep_fast_intervals.py; 0.0 won at the fast intervals, 0.1 at the slow ones.
DD_THRESHOLD_BY_INTERVAL = {"15m": 0.0, "1h": 0.0, "4h": 0.0, "1d": 0.1, "1w": 0.1}
# Real-time anchors (hours) for every interval-scaled "candle count" constant below, taken
# from 4h's values (24/12/[150,1500] candles @ 4h = 96h/48h/[600,6000]h) - already proven
# correct unscaled at 4h/1d/1w; only 15m/1h needed scaling UP to match (see
# notebooks/test_time_scaled.py). INTERVAL_HOURS lets every one of them be expressed as "this
# many real hours" and converted to native candles for whichever interval is live.
INTERVAL_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0, "1w": 168.0}
CULL_EVERY_HOURS = 96.0
ADULT_AGE_HOURS = 48.0
CULL_MIN_AGE_HOURS = 4 * CULL_EVERY_HOURS   # ratio preserved from the original 96/24 = 4x
WINDOW_HOURS = (600.0, 6000.0)               # the organism `window` gene's random-init range
WINDOW_MUTATE_HOURS = (400.0, 12000.0)       # its wider range after mutation/crossover
# ml/box_breakout.py's TRAIN_BARS=500/REOPT_EVERY=100/PARAM_GRID lookback/padding are native
# candle counts, already proven fine unscaled everywhere except 15m (see
# notebooks/test_box_scaled.py: scaling them DOWN for 1d made it worse, so only 15m gets a
# rescaled grid - x4, anchored to 1h's real-time meaning, same anchor used for 15m's organism
# constants above).
BOX_SCALE_INTERVALS = {"15m": 4.0}
# Replaying a live, growing dataset's box_breakout.run_adaptive() from scratch on every
# extend() call would re-walk years of history just to append a few candles' worth of signal -
# too slow for a live API. extend() replays only box_context_candles() worth (one train+reopt
# cycle) of recent history instead of the whole thing; fit() (used once, offline, by
# ml/pretrain_evolution.py) always replays in full, which is what
# notebooks/genetic_strategy_lab.ipynb's validated results are based on.
MIN_TRAIN_ROWS = 30         # minimum rows for a ridge fit to be trusted
REPLAY_START_FLOOR = 50     # candles of warm-up before the replay begins, regardless of
                              # interval - a fixed 200 (as this used to be) ate 42% of 1w's
                              # ~477-candle history; see notebooks/test_1d_1w_fixes.py.
DEATH_LOOKBACK = 48
BIG_LOSS = 0.25
MAX_Z = 8.0                 # an input this many sd from its training mean counts as only this far
BAND_QUANTILES = (0.25, 0.75)
_BAND_Z = 0.6745              # normal z of the 75% quantile, for the fallback below
BAND_MODE = "step"
MIN_RESIDUALS = 40          # ensemble errors needed before the band is calibrated on them
RESIDUAL_MEMORY = 600
HORIZON_EXPONENT = 0.57     # band growth past the last horizon (see ml/price_predictor.py)
SEED = 7
RECENT_TRACK = 500          # resolved forecasts in the "recent" track record
COMBO_MIN = 3
_FRAME_COLS = ("open", "high", "low", "close", "volume", "taker_buy_volume", "btc_return_1",
               "fear_greed", "fear_greed_chg", "news_sentiment", "tvl_chg_1d", "tvl_chg_7d", "funding", "premium")
# Always computable from the candles; the other groups exist only when
# their source data does (ml/market_context.py), and only from when it does.
CORE_GROUPS = ("momentum", "trend", "long", "volatility", "volume", "oscillators", "btc")
NEWS_MOOD_WEIGHT = 0.5      # max mood shift from the news, x the news sensitivity gene
STATE_VERSION = 8
STATE_DIR = Path(__file__).resolve().parent.parent / "data" / "evolution"
DEFAULT_INTERVAL = "1h"

EMOTIONS = ((0.30, "fear"), (0.45, "caution"), (0.60, "calm"), (0.75, "confidence"), (1.01, "euphoria"))


def emotion_name(mood: float) -> str:
    for limit, name in EMOTIONS:
        if mood < limit:
            return name
    return EMOTIONS[-1][1]


# --- Inputs ------------------------------------------------------------------

def long_window(n_rows: int) -> int:
    """96 candles of context, shorter when the history itself is short."""
    return int(min(96, max(24, n_rows // 4)))


def build_inputs(df: pd.DataFrame, long_w: int = 96) -> tuple[np.ndarray, dict[str, list[int]], list[str]]:
    """Scale-free inputs from the candles (and BTC's return when present),
    grouped into the genes an organism can switch on. Row t only uses
    candles <= t. `long_w` is the longest look-back (see long_window)."""
    c = df["close"].astype(float)
    lc = np.log(c)
    lr = lc.diff()
    high = df["high"].astype(float) if "high" in df else c
    low = df["low"].astype(float) if "low" in df else c
    vol = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)
    cols: dict[str, tuple[str, pd.Series]] = {}

    def add(group, name, series):
        cols[name] = (group, series)

    for k in (1, 3, 6, 12):
        add("momentum", f"ret_{k}", lc - lc.shift(k))
    add("trend", "close_vs_sma_24", c / c.rolling(24).mean() - 1)
    add("trend", "close_vs_sma_long", c / c.rolling(long_w).mean() - 1)
    add("trend", "ema_12_vs_26", c.ewm(span=12).mean() / c.ewm(span=26).mean() - 1)
    add("long", "ret_24", lc - lc.shift(24))
    add("long", "ret_long", lc - lc.shift(long_w))
    hi96, lo96 = high.rolling(long_w).max(), low.rolling(long_w).min()
    add("long", "range_pos_long", (c - lo96) / (hi96 - lo96).replace(0, np.nan) - 0.5)
    vol24 = lr.rolling(24).std()
    add("volatility", "vol_24", vol24)
    add("volatility", "vol_ratio_24_long", vol24 / lr.rolling(long_w).std() - 1)
    add("volatility", "range_1", (high - low) / c)
    add("volume", "volume_vs_24", np.log((vol / vol.rolling(24).mean()).clip(lower=1e-6)))
    if "taker_buy_volume" in df:
        add("volume", "taker_buy_ratio", df["taker_buy_volume"].astype(float) / vol.replace(0, np.nan) - 0.5)
    d = c.diff()
    gain, loss = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    add("oscillators", "rsi_14", (100 - 100 / (1 + gain / loss.replace(0, 1e-9)) - 50) / 50)
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    add("oscillators", "bb_position", (c - sma20) / (2 * sd20).replace(0, np.nan))
    if "btc_return_1" in df:
        btc = df["btc_return_1"].astype(float).fillna(0.0)
        add("btc", "btc_return_1", btc)
        add("btc", "btc_ret_24", btc.rolling(24).sum())

    if "fear_greed" in df:
        add("sentiment", "fear_greed", df["fear_greed"].astype(float))
        add("sentiment", "fear_greed_chg", df["fear_greed_chg"].astype(float))
    if "news_sentiment" in df:
        add("news", "news_sentiment", df["news_sentiment"].astype(float))
    if "tvl_chg_1d" in df:
        add("defi", "tvl_chg_1d", df["tvl_chg_1d"].astype(float))
        add("defi", "tvl_chg_7d", df["tvl_chg_7d"].astype(float))
    if "funding" in df:
        f = df["funding"].astype(float)
        add("futures", "funding", f)
        add("futures", "funding_ma_24", f.rolling(24).mean())
    if "premium" in df:
        pr = df["premium"].astype(float)
        add("futures", "premium", pr)
        add("futures", "premium_ma_24", pr.rolling(24).mean())
        add("futures", "premium_chg_24", pr - pr.shift(24))

    names = list(cols)
    X = np.column_stack([cols[n][1].to_numpy(float) for n in names])
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        groups.setdefault(cols[n][0], []).append(i)
    return X, groups, names


def mood_signal(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The market's sentiment per candle, in [-1, 1]: the app's news
    sentiment where there was news, else the Fear & Greed Index, else 0.
    Returns (values, source) with source "news" / "fear_greed" / ""."""
    n = len(df)
    news = df["news_sentiment"].to_numpy(float) if "news_sentiment" in df else np.full(n, np.nan)
    fg = df["fear_greed"].to_numpy(float) if "fear_greed" in df else np.full(n, np.nan)
    value = np.where(np.isfinite(news), news, np.where(np.isfinite(fg), fg, 0.0))
    source = np.where(np.isfinite(news), "news", np.where(np.isfinite(fg), "fear_greed", ""))
    return np.clip(value, -1, 1), source


def _pattern_score(res: dict) -> float:
    """ml.pattern_recognition.analyze()'s result collapsed to one directional score in
    [-1, 1]: the leading geometric pattern (full weight, half if unconfirmed) plus the last
    candlestick pattern (half weight)."""
    score = 0.0
    geo = res.get("geometric") or []
    if geo:
        g = geo[0]
        d = 1.0 if g.get("direction") == "bullish" else -1.0 if g.get("direction") == "bearish" else 0.0
        score += d * (1.0 if g.get("confirmed", True) else 0.5)
    candle = res.get("candlestick") or []
    if candle:
        last = candle[-1]
        d = 1.0 if last.get("direction") == "bullish" else -1.0 if last.get("direction") == "bearish" else 0.0
        score += 0.5 * d
    return float(np.clip(score, -1, 1))


def pattern_signal(df: pd.DataFrame, interval: str, refresh_hours: float = 24.0) -> np.ndarray:
    """Directional pattern score per candle, shared by the whole population (not gene-
    dependent - genes only decide how much a given organism listens to it). Recomputed every
    `refresh_hours` of real time (scaled to native candles per interval), held constant
    between refreshes - rerunning geometric pattern detection on every single candle would be
    far more expensive for no benefit at this signal's natural update rate."""
    from ml import pattern_recognition

    cadence = max(1, round(refresh_hours / INTERVAL_HOURS[interval]))
    n = len(df)
    signal = np.zeros(n)
    if n == 0:
        return signal
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    last_val = 0.0
    for t in range(0, n, cadence):
        window = ohlcv.iloc[max(0, t - 149): t + 1]
        candles = [{"timestamp": ts.to_pydatetime(), **{k: float(v) for k, v in row.items()}} for ts, row in window.iterrows()]
        try:
            last_val = _pattern_score(pattern_recognition.analyze(candles))
        except Exception:
            pass
        signal[t: min(n, t + cadence)] = last_val
    return signal


def _box_kwargs(interval: str) -> dict:
    scale = BOX_SCALE_INTERVALS.get(interval)
    if not scale:
        return {}
    from ml import box_breakout
    return dict(
        train_bars=round(box_breakout.TRAIN_BARS * scale),
        reopt_every=round(box_breakout.REOPT_EVERY * scale),
        param_grid=[{**p, "lookback": max(2, round(p["lookback"] * scale)),
                    "padding": max(1, round(p["padding"] * scale))} for p in box_breakout.PARAM_GRID],
    )


def box_context_candles(interval: str) -> int:
    """The minimum window extend() needs to replay for box_signal() to produce a valid value
    for its newest rows: one training window plus one reopt cycle (not
    BOX_FULL_REPLAY_MAX_CANDLES's old flat cap, which replayed up to 15 whole reopt cycles -
    each a 48-candidate grid search re-simulating the training window - just to get the
    signal's last 1-2 values on every single incremental extend(), the actual cause of the
    live app's lag after this signal was added; see ml/evolution.py's module docstring)."""
    from ml import box_breakout

    kw = _box_kwargs(interval)
    train_bars = kw.get("train_bars", box_breakout.TRAIN_BARS)
    reopt_every = kw.get("reopt_every", box_breakout.REOPT_EVERY)
    return train_bars + reopt_every + 50


def box_signal(df: pd.DataFrame, interval: str) -> np.ndarray:
    """Position per candle from ml.box_breakout's walk-forward adaptive strategy (the same
    mechanism and, for every interval but 15m, the same settings as the app's own Box
    Breakout feature) - shared by the whole population. Best-effort: too little history (or
    any other failure) just leaves it flat (zeros), the same philosophy as
    ml/market_context.py's optional sources."""
    from ml import strategy_lab as lab

    n = len(df)
    if n < 210:
        return np.zeros(n)
    try:
        return lab.box_positions(df, interval, **_box_kwargs(interval))
    except Exception:
        logger.warning("Evolution: box signal unavailable for %s (%d candles)", interval, n, exc_info=True)
        return np.zeros(n)


# --- Organisms ---------------------------------------------------------------

@dataclass
class Organism:
    id: int
    born_at: int
    generation: int
    groups: tuple[str, ...]
    window: int            # rows of history it learns from
    log_alpha: float       # ridge regularization (log10)
    boldness: float        # how far it dares to move its *price* forecast
    temperament: float     # how fast its mood reacts to wins/losses
    parents: tuple[int, ...] = ()
    news_sensitivity: float = 0.0  # how much the market's sentiment sways its mood (< 0: contrarian)
    patience: float = 1.0          # the price tool only fires if the expected move > patience x round-trip fee
    pred_w: float = 1.0     # how much weight each of its 4 tools gets in its combined position
    pattern_w: float = 0.0
    box_w: float = 0.0
    news_w: float = 0.0
    energy: float = START_ENERGY   # = its money
    mood: float = 0.5
    wins: int = 0
    losses: int = 0
    combos: int = 0        # candles where it was right on every resolved horizon (price tool only)
    trace: list = field(default_factory=list)  # last DEATH_LOOKBACK candles: (pnl, fees, fine, money before)
    worst: tuple | None = None  # its worst resolved price-tool prediction on the last candle: (h, predicted, actual, skill)
    position: float = 0.0       # fraction of its money: > 0 long, < 0 short, 0 out
    peak_money: float = START_MONEY
    trades: int = 0
    fees_paid: float = 0.0
    trading_pnl: float = 0.0
    fees_last: float = 0.0      # fees paid when it last traded, charged to the next candle's result
    culled: bool = False        # replaced for its fitness being the worst
    energy_1d_ago: float = START_ENERGY
    last_daily_pct: float = 0.0
    engagement_sum: float = 0.0
    engagement_n: int = 0
    last_daily_engagement: float = 0.0

    def growth(self, t: int) -> float:
        """Log growth of its money per candle since birth."""
        return math.log(max(self.energy, 1e-9) / START_MONEY) / max(self.age(t), 1)
    cols: np.ndarray | None = None
    W: np.ndarray | None = None       # (n_features + 1,) - a single column: the 1-step price tool
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    recent: list[float] = field(default_factory=list)

    def age(self, t: int) -> int:
        return t - self.born_at

    def genes(self) -> dict:
        return {"inputs": list(self.groups), "window": self.window, "ridge_alpha": round(10 ** self.log_alpha, 3),
                "boldness": round(self.boldness, 2), "temperament": round(self.temperament, 2),
                "news_sensitivity": round(self.news_sensitivity, 2), "patience": round(self.patience, 2),
                "pred_w": round(self.pred_w, 2), "pattern_w": round(self.pattern_w, 2),
                "box_w": round(self.box_w, 2), "news_w": round(self.news_w, 2)}


class EvolutionaryForecaster:
    def __init__(self, population: int = POPULATION, seed: int = SEED, interval: str = DEFAULT_INTERVAL):
        self.population_size = population
        self.seed = seed
        self.interval = interval
        self.fitted = False

    # -- per-interval, real-time-scaled constants ------------------------------

    def _scale_constants(self) -> None:
        # Only 15m/1h are scaled UP from 4h's real-time meaning (CULL_EVERY_HOURS etc. are
        # already 4h's native values in hours - 24 candles x 4h/candle = 96h). 4h/1d/1w all use
        # those same native candle counts unscaled: clamping the divisor at 4h means a slower
        # interval's hours-per-candle never shrinks the result below 4h's own native values.
        # Scaling 1d/1w DOWN (dividing by their own much larger hours-per-candle) was tried and
        # made things worse, not better - see notebooks/test_1d_1w_fixes.py.
        h = min(INTERVAL_HOURS.get(self.interval, 4.0), 4.0)
        self.cull_every = max(1, round(CULL_EVERY_HOURS / h))
        self.refit_every = self.cull_every
        self.adult_age = max(1, round(ADULT_AGE_HOURS / h))
        self.cull_min_age = max(1, round(CULL_MIN_AGE_HOURS / h))
        self.win_lo = max(50, round(WINDOW_HOURS[0] / h))
        self.win_hi = max(self.win_lo + 50, round(WINDOW_HOURS[1] / h))
        self.win_mutate_lo = max(30, round(WINDOW_MUTATE_HOURS[0] / h))
        self.win_mutate_hi = max(self.win_mutate_lo + 50, round(WINDOW_MUTATE_HOURS[1] / h))
        true_h = INTERVAL_HOURS.get(self.interval, 4.0)   # "a day" must use the REAL hours/candle, not the clamp above
        self.candles_per_day = max(1, round(24.0 / true_h))
        self.dd_threshold = DD_THRESHOLD_BY_INTERVAL.get(self.interval, 0.1)

    # -- life cycle ------------------------------------------------------------

    def _usable_groups(self, t: int) -> list[str]:
        """Input groups with enough resolved rows by candle t to learn from
        (a source that starts in 2020 can't be a gene in 2018)."""
        end = t - 1
        need = max(self.min_train // 2, 1)
        return [g for g in self.group_names if end >= 0 and self.group_rows[g][end] >= need]

    def _random_organism(self, t: int) -> Organism:
        r = self.rng
        usable = self._usable_groups(t) or [g for g in CORE_GROUPS if g in self.groups]
        k = int(r.integers(1, min(4, len(usable)) + 1))
        groups = tuple(sorted(str(g) for g in r.choice(usable, size=k, replace=False)))
        return self._new(t, groups, int(r.integers(self.win_lo, self.win_hi)), float(r.uniform(-1, 3)),
                         float(r.uniform(0.3, 1.5)), float(r.uniform(0.05, 0.4)), (), 0, float(r.uniform(-1, 1)),
                         float(r.uniform(0, 3)), float(r.uniform(0, 1.5)), float(r.uniform(0, 1.5)),
                         float(r.uniform(0, 1.5)), float(r.uniform(0, 1.5)))

    def _new(self, t, groups, window, log_alpha, boldness, temperament, parents, generation,
             news_sensitivity=0.0, patience=1.0, pred_w=1.0, pattern_w=0.0, box_w=0.0, news_w=0.0) -> Organism:
        org = Organism(self.next_id, t, generation, groups, window, log_alpha, boldness, temperament, parents,
                       news_sensitivity, patience, pred_w, pattern_w, box_w, news_w)
        org.cols = np.array(sorted(i for g in groups for i in self.groups[g]))
        self.next_id += 1
        self.births += 1
        return org

    def _child(self, t: int) -> Organism:
        r = self.rng

        def pick():  # tournament of 3 on money: the richest have the most children
            contenders = r.choice(len(self.alive), size=min(3, len(self.alive)), replace=False)
            return max((self.alive[i] for i in contenders), key=lambda o: o.energy)

        a, b = pick(), pick()
        usable = set(self._usable_groups(t))
        pool = sorted((set(a.groups) | set(b.groups)) & usable) or sorted(usable)
        groups = {g for g in pool if r.random() < 0.6} or {pool[int(r.integers(len(pool)))]}
        if r.random() < 0.3 and usable:  # mutation: switch one input group on/off
            options = sorted(usable)
            g = options[int(r.integers(len(options)))]
            groups = groups ^ {g} or {g}
        mix = lambda x, y, sd, lo, hi: float(np.clip((x + y) / 2 + r.normal(0, sd), lo, hi))
        window = int(np.clip(np.exp((math.log(a.window) + math.log(b.window)) / 2 + r.normal(0, 0.25)),
                             self.win_mutate_lo, self.win_mutate_hi))
        return self._new(t, tuple(sorted(str(g) for g in groups)), window,
                         mix(a.log_alpha, b.log_alpha, 0.3, -1, 3), mix(a.boldness, b.boldness, 0.15, 0.1, 2.0),
                         mix(a.temperament, b.temperament, 0.05, 0.02, 0.6), (a.id, b.id),
                         max(a.generation, b.generation) + 1,
                         mix(a.news_sensitivity, b.news_sensitivity, 0.15, -1, 1),
                         mix(a.patience, b.patience, 0.3, 0, 5),
                         mix(a.pred_w, b.pred_w, 0.2, 0, 2), mix(a.pattern_w, b.pattern_w, 0.2, 0, 2),
                         mix(a.box_w, b.box_w, 0.2, 0, 2), mix(a.news_w, b.news_w, 0.2, 0, 2))

    def _fit_org(self, org: Organism, t: int) -> bool:
        # Rows whose 1-step target (the next candle's log return) had resolved by t.
        end = t - 1
        start = max(self.first_row, end - org.window)
        rows = np.arange(start, end + 1)
        rows = rows[self.y1_ok[rows] & np.isfinite(self.X[np.ix_(rows, org.cols)]).all(axis=1)]
        if len(rows) < self.min_train // 2:
            return False
        X = self.X[np.ix_(rows, org.cols)]
        mu, sd = X.mean(axis=0), X.std(axis=0)
        # Near-constant over the window (e.g. news sentiment before it
        # existed): float noise, not spread - dividing by it blew a later
        # non-constant value up to forecasts in the billions.
        sd[sd < 1e-8 * np.maximum(np.abs(mu), 1.0)] = 1.0
        Z = np.column_stack([_standardize(X, mu, sd), np.ones(len(rows))])
        y = self.y1[rows]
        reg = (10 ** org.log_alpha) * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0  # don't shrink the intercept
        org.W = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        org.mu, org.sd = mu, sd
        return True

    def _feeling(self, org: Organism, t: int) -> float:
        """Its mood right now: earned mood, swayed by the market's sentiment."""
        return float(np.clip(org.mood + NEWS_MOOD_WEIGHT * org.news_sensitivity * self.S[t], 0.0, 1.0))

    def _raw(self, org: Organism, t: int) -> float:
        """The price tool's raw 1-step forecast (log return over the next candle), before
        mood/boldness scaling."""
        x = self.X[t, org.cols]
        if org.W is None or not np.isfinite(x).all():
            return 0.0
        return float(np.append(_standardize(x, org.mu, org.sd), 1.0) @ org.W)

    # -- training = replaying history ----------------------------------------

    def _load_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """(Re)build every per-row array from the candle frame; returns the
        rows whose inputs are complete."""
        n_old = len(self.frame) if hasattr(self, "frame") else 0
        self.frame = frame
        self.index = frame.index
        self.X, self.groups, self.feature_names = build_inputs(frame, self.long_w)
        self.group_names = sorted(self.groups)
        logc = np.log(frame["close"].astype(float).to_numpy())
        n = len(frame)
        self.logc = logc
        self.y1 = np.full(n, np.nan)
        self.y1[:-1] = logc[1:] - logc[:-1]
        self.y1_ok = np.isfinite(self.y1)
        # Multi-horizon targets: used only to score the diagnostic track record/band shown on
        # the chart (ml/price_predictor.py's predict()), never to fit or to decide fitness.
        self.Y = np.full((n, len(HORIZONS)), np.nan)
        for j, h in enumerate(HORIZONS):
            self.Y[: n - h, j] = logc[h:] - logc[: n - h]
        self.S, self.S_source = mood_signal(frame)

        if n_old and hasattr(self, "pattern_sig") and n > n_old:
            # extend(): replay only a bounded recent window for the new rows' box/pattern
            # signal (just enough for one box_breakout train+reopt cycle - see
            # box_context_candles), not the whole (possibly years-long) history on every call -
            # fit() (used once, offline, by ml/pretrain_evolution.py) always replays in full.
            ctx = min(n, box_context_candles(self.interval))
            tail = frame.iloc[-ctx:]
            n_new = n - n_old
            self.pattern_sig = np.concatenate([self.pattern_sig, pattern_signal(tail, self.interval)[-n_new:]])
            self.box_sig = np.concatenate([self.box_sig, box_signal(tail, self.interval)[-n_new:]])
        else:
            self.pattern_sig = pattern_signal(frame, self.interval)
            self.box_sig = box_signal(frame, self.interval)

        self.y_ok = np.isfinite(self.Y).all(axis=1)
        # Per group: how many rows with complete inputs and a resolved 1-step target exist up
        # to each row (cumulative), for _usable_groups.
        self.group_rows = {}
        for g, idx in self.groups.items():
            ok = np.isfinite(self.X[:, idx]).all(axis=1) & self.y1_ok
            self.group_rows[g] = np.cumsum(ok)
        core = [i for g in CORE_GROUPS if g in self.groups for i in self.groups[g]]
        return np.isfinite(self.X[:, core]).all(axis=1)

    def fit(self, df: pd.DataFrame, interval: str | None = None) -> "EvolutionaryForecaster":
        if interval is not None:
            self.interval = interval
        self.version = STATE_VERSION
        self._scale_constants()
        self.rng = np.random.default_rng(self.seed)
        self.long_w = long_window(len(df))
        frame = df[[c for c in _FRAME_COLS if c in df.columns]].copy()
        feats_ok = self._load_frame(frame)
        logc, n = self.logc, len(frame)
        self.track = {"err": [[] for _ in HORIZONS], "base": [[] for _ in HORIZONS], "hit": [[] for _ in HORIZONS],
                      "life_err": np.zeros(len(HORIZONS)), "life_base": np.zeros(len(HORIZONS)),
                      "life_hits": np.zeros(len(HORIZONS)), "life_calls": np.zeros(len(HORIZONS)),
                      "life_n": np.zeros(len(HORIZONS)), "first_time": None}
        finite_rows = np.flatnonzero(feats_ok)
        if len(finite_rows) == 0:
            raise ValueError("No complete input rows to train on")
        self.first_row = int(finite_rows[0])
        self.min_train = int(min(MIN_TRAIN_ROWS, max(10, (n - self.first_row) // 3)))
        start = self.first_row + max(self.min_train, REPLAY_START_FLOOR)
        self.newborn = start >= n - 1
        if self.newborn:
            # Too little history for any organism to have lived: no opinion
            # ("no change"), band from the real volatility so far.
            lr = np.diff(logc)
            sd = float(np.nanstd(lr)) if len(lr) > 1 else 0.01
            self.t_last = n - 1
            self.scale = np.array([sd * math.sqrt(h) * 0.8 for h in HORIZONS])
            self.alive, self.residuals, self.history, self.death_log = [], [[] for _ in HORIZONS], [], []
            self.life_stats, self.graveyard = {}, []
            self.births = self.deaths = 0
            self.fitted = True
            logger.warning("Evolution: only %d rows, population not born yet (needs > %d)", n, start + 1)
            return self

        self.next_id, self.births, self.deaths = 0, 0, 0
        self.death_log: list[dict] = []
        self.graveyard: list[dict] = []                 # every death, see _death_record
        self.life_stats: dict[str, list[float]] = {}   # gene -> [sum of lifespans, deaths]
        self.alive = [self._random_organism(start) for _ in range(self.population_size)]
        for org in self.alive:
            self._fit_org(org, start)
        # The population's fund: the vote trades START_MONEY the same way the organisms do.
        self.fund = Organism(-1, start, 0, (), 0, 0.0, 1.0, 0.0)
        self.fund_start = self.index[start].isoformat()
        self.fund_t0 = start
        self.fund_trades: list[dict] = []
        self.births = 0  # the founders aren't counted as births
        # Typical |real move| per horizon, the yardstick for the diagnostic track record.
        valid_y = self.Y[feats_ok & self.y_ok & (np.arange(n) < start - H_MAX)]
        self.scale = np.abs(valid_y).mean(axis=0) if len(valid_y) else np.full(len(HORIZONS), 1e-3)
        self.pending: dict[int, dict[int, float]] = {}   # t -> {organism id: raw*boldness*feeling}
        self.ens_pending: dict[int, np.ndarray] = {}
        self.residuals = [[] for _ in HORIZONS]
        self.history: list[dict] = []
        for t in range(start, n):
            self._step(t)
        self.t_last = n - 1
        self.fitted = True
        return self

    def extend(self, df: pd.DataFrame) -> int:
        """Keep living through the rows of `df` newer than the last one
        lived. They must follow on directly (the last lived candle has to be
        in `df`), otherwise the inputs across the gap would be wrong:
        raises ValueError. Returns how many candles were lived."""
        if self.newborn:
            raise ValueError("newborn population: nothing to continue")
        last = self.index[-1]
        if last not in df.index:
            raise ValueError(f"gap: last lived candle {last} is not in the new data")
        new = df.loc[df.index > last, [c for c in self.frame.columns if c in df.columns]]
        if new.empty:
            return 0
        n_old = len(self.index)
        self._load_frame(pd.concat([self.frame, new.reindex(columns=self.frame.columns)]))
        for t in range(n_old, len(self.index)):
            self._step(t)
        self.t_last = len(self.index) - 1
        return len(self.index) - n_old

    def _target_position(self, raw_pred: float, org: Organism, t: int) -> float:
        """The organism's combined position: its four tools (price, pattern, box, news),
        weighted by its own pred_w/pattern_w/box_w/news_w genes - never a prediction-accuracy
        judgment, only what it decides to bet.

        Price tool sizing: half-Kelly (the mood/boldness-scaled 1-step forecast / the
        population's recent per-candle variance) - at most all its money (no leverage), and
        only fires if the expected move clears `patience` round trips of fees."""
        sigma = 1.2533 * max(self.scale[0], 1e-9)   # std of a candle's move (from its mean |move|)
        size = float(np.clip(KELLY_FRACTION * raw_pred / sigma ** 2, -1.0, 1.0))
        price_pos = size if abs(raw_pred) > org.patience * 2 * FEE else 0.0
        w_sum = org.pred_w + org.pattern_w + org.box_w + org.news_w
        combined = (org.pred_w * price_pos + org.pattern_w * self.pattern_sig[t]
                    + org.box_w * self.box_sig[t] + org.news_w * self.S[t]) / max(w_sum, 1e-9)
        return float(np.clip(combined, -1.0, 1.0))

    def _trade(self, acct: Organism, target: float) -> None:
        change = abs(target - acct.position)
        # Small resizes of an open position aren't worth their fees.
        if change == 0.0 or (change < MIN_REBALANCE and target != 0.0 and acct.position != 0.0):
            return
        fee = change * acct.energy * FEE
        acct.energy -= fee
        acct.fees_paid += fee
        acct.fees_last += fee
        acct.trades += 1
        acct.position = target

    def _fund_step(self, move: float, t: int) -> None:
        f = self.fund
        f.energy += f.energy * f.position * move
        f.fees_last = 0.0
        f.peak_money = max(f.peak_money, f.energy)

    def _record_fund_trade(self, prev_position: float, new_position: float, t: int) -> None:
        """Builds the fund's own entry/exit log (same shape as ml/box_breakout.py's trades),
        for the chart: markers for when the fund opened/closed a position, not a predicted
        price line (see memory feedback-chart-entry-exit-not-prediction)."""
        from_sign, to_sign = np.sign(prev_position), np.sign(new_position)
        if from_sign != 0 and to_sign != from_sign and self.fund_trades and self.fund_trades[-1]["exit_bar"] is None:
            open_trade = self.fund_trades[-1]
            open_trade["exit_bar"] = t
            open_trade["exit_time"] = self.index[t].isoformat()
            open_trade["exit_price"] = float(math.exp(self.logc[t]))
            entry_price = open_trade["entry_price"]
            sign = 1 if open_trade["side"] == "long" else -1
            open_trade["return_pct"] = round((open_trade["exit_price"] / entry_price - 1) * sign * 100, 3)
        if to_sign != 0 and to_sign != from_sign:
            self.fund_trades.append({
                "side": "long" if to_sign > 0 else "short", "entry_bar": t, "entry_time": self.index[t].isoformat(),
                "entry_price": float(math.exp(self.logc[t])), "entry_amount": round(abs(new_position) * self.fund.energy, 2),
                "exit_bar": None, "exit_time": None,
                "exit_price": None, "return_pct": None,
            })
        del self.fund_trades[:-200]

    def _death_record(self, o: Organism, t: int) -> dict:
        """Everything about a death: who, when, where (the market at that
        moment) and why."""
        trading_loss = -sum(p for p, _, _, _ in o.trace)
        fees = sum(f for _, f, _, _ in o.trace)
        fines = sum(r for _, _, r, _ in o.trace)
        worst_candle = max((-p / b for p, _, _, b in o.trace if b > 0), default=0.0)
        if getattr(o, "culled", False):
            cause = "outcompeted"
        elif worst_candle >= BIG_LOSS:
            cause = "big_loss"
        elif fees >= max(trading_loss, fines):
            cause = "fees"
        elif trading_loss >= fines:
            cause = "bad_trades"
        else:
            cause = "fines"
        lr = np.diff(self.logc[max(0, t - 24): t + 1])
        lr_long = np.diff(self.logc[max(0, t - 24 * 30): t + 1])
        vol, vol_long = (float(np.std(lr)) if len(lr) > 1 else 0.0), (float(np.std(lr_long)) if len(lr_long) > 1 else 0.0)
        worst = o.worst
        return {
            "id": o.id, "generation": o.generation, "parents": list(o.parents),
            "born": self.index[o.born_at].isoformat(), "died": self.index[t].isoformat(), "age": o.age(t),
            "cause": cause,
            "money": round(o.energy, 2), "peak_money": round(o.peak_money, 2),
            "growth_pct_per_100": round((math.exp(o.growth(t) * 100) - 1) * 100, 2),
            "trades": o.trades, "fees_paid": round(o.fees_paid, 2), "trading_pnl": round(o.trading_pnl, 2),
            "fines_paid": round(fines, 2),
            "last_48": {"trading_loss": round(trading_loss, 2), "fees": round(fees, 2), "fines": round(fines, 2),
                        "worst_candle_pct": round(worst_candle * 100, 1)},
            "killing_prediction": None if worst is None else {
                "h": worst[0], "predicted_pct": round((math.exp(min(max(worst[1], -10.0), 10.0)) - 1) * 100, 3),
                "actual_pct": round((math.exp(worst[2]) - 1) * 100, 3)},
            "price": round(float(math.exp(self.logc[t])), 2),
            "move_24_pct": round(float((math.exp(self.logc[t] - self.logc[max(0, t - 24)]) - 1) * 100), 2),
            "volatility_vs_month": round(vol / vol_long, 2) if vol_long else None,
            "market_sentiment": round(float(self.S[t]), 2), "sentiment_source": str(self.S_source[t]) or None,
            "emotion": emotion_name(self._feeling(o, t)), "earned_mood": round(o.mood, 3),
            "wins": o.wins, "losses": o.losses, "combos": o.combos,
            **o.genes(),
        }

    def _gene_snapshot(self) -> dict:
        alive = self.alive
        n = max(len(alive), 1)
        return {
            "shares": {g: round(sum(g in o.groups for o in alive) / n, 3) for g in self.group_names},
            "boldness": round(float(np.mean([o.boldness for o in alive])), 3) if alive else None,
            "temperament": round(float(np.mean([o.temperament for o in alive])), 3) if alive else None,
            "news_sensitivity": round(float(np.mean([o.news_sensitivity for o in alive])), 3) if alive else None,
            "window": round(float(np.mean([o.window for o in alive])), 1) if alive else None,
            "avg_generation": round(float(np.mean([o.generation for o in alive])), 1) if alive else None,
            "avg_energy": round(float(np.mean([o.energy for o in alive])), 1) if alive else None,
            "patience": round(float(np.mean([o.patience for o in alive])), 3) if alive else None,
            "pred_w": round(float(np.mean([o.pred_w for o in alive])), 3) if alive else None,
            "pattern_w": round(float(np.mean([o.pattern_w for o in alive])), 3) if alive else None,
            "box_w": round(float(np.mean([o.box_w for o in alive])), 3) if alive else None,
            "news_w": round(float(np.mean([o.news_w for o in alive])), 3) if alive else None,
        }

    def _ensemble_members(self, t: int) -> list[Organism]:
        adults = [o for o in self.alive if o.age(t) >= self.adult_age and o.W is not None] or \
                 [o for o in self.alive if o.W is not None]
        return sorted(adults, key=lambda o: o.energy, reverse=True)[:TOP_K]

    @staticmethod
    def _combine(members: list[Organism], preds: dict[int, float]) -> float:
        """Energy-weighted average of the price tool's raw*boldness*feeling forecasts - a
        diagnostic ensemble view (what the chart's band is calibrated on), not what the fund
        trades on (see fund_target in _step)."""
        if not members:
            return 0.0
        w = np.array([min(max(o.energy, 1e-6), VOTE_WEIGHT_CAP) for o in members])
        p = np.array([preds[o.id] for o in members])
        return float((w * p).sum() / w.sum())

    def _snapshot(self, t: int) -> dict:
        members = self._ensemble_members(t)
        w = np.array([max(o.energy, 1e-6) for o in members]) if members else np.array([1.0])
        mood = float((w * np.array([self._feeling(o, t) for o in members])).sum() / w.sum()) if members else 0.5
        return {"time": self.index[t].isoformat(), "alive": len(self.alive), "deaths": self.deaths,
                "births": self.births, "mood": round(mood, 3),
                "max_generation": max((o.generation for o in self.alive), default=0),
                "price": round(float(math.exp(self.logc[t])), 2), **self._gene_snapshot(),
                "fund": round(self.fund.energy, 2) if hasattr(self, "fund") else None,
                "richest": round(max((o.energy for o in self.alive), default=0.0), 2)}

    def _step(self, t: int) -> None:
        # 1. Outcomes that resolve at candle t: score every organism's price tool against them
        # (diagnostic track record only - never used for fitness/selection).
        for j, h in enumerate(HORIZONS):
            past = t - h
            preds = self.pending.get(past)
            if preds is None:
                continue
            y = self.logc[t] - self.logc[past]
            self.scale[j] = 0.99 * self.scale[j] + 0.01 * abs(y)
            for org in self.alive:
                raw = preds.get(org.id)
                if raw is None:
                    continue
                p_h = raw * h   # flat extrapolation: the 1-step forecast held constant per candle
                skill = float(np.clip((abs(y) - abs(p_h - y)) / max(self.scale[j], 1e-9), -10, 2))
                org.recent.append((j, skill))
                if org.worst is None or skill < org.worst[3]:
                    org.worst = (h, float(p_h), float(y), skill)
            ens = self.ens_pending.get(past)
            if ens is not None:
                res = self.residuals[j]
                res.append(y - ens[j])
                if len(res) > RESIDUAL_MEMORY:
                    del res[0]
                tr = self.track
                err, base = abs(ens[j] - y), abs(y)
                called = abs(ens[j]) > 1e-9 and y != 0
                for key, v in (("err", err), ("base", base)):
                    tr[key][j].append(v)
                    del tr[key][j][:-RECENT_TRACK]
                if called:
                    tr["hit"][j].append(float(np.sign(ens[j]) == np.sign(y)))
                    del tr["hit"][j][:-RECENT_TRACK]
                    tr["life_hits"][j] += float(np.sign(ens[j]) == np.sign(y))
                    tr["life_calls"][j] += 1
                tr["life_err"][j] += err
                tr["life_base"][j] += base
                tr["life_n"][j] += 1
                if tr["first_time"] is None:
                    tr["first_time"] = self.index[t].isoformat()
        for past in [k for k in self.pending if k <= t - H_MAX]:
            del self.pending[past]
            self.ens_pending.pop(past, None)

        # 2. Money: the position held since the last candle pays (or costs). Once a day
        # (scaled to real time), a losing day pays a fine proportional to the loss, and a day
        # spent entirely out of the market pays a small separate idle fine (see
        # IDLE_FINE_RATE); a day spent holding any position - flat or profitable - pays
        # nothing. Mood follows profitable vs losing candles.
        move = math.exp(self.logc[t] - self.logc[t - 1]) - 1 if t > 0 else 0.0
        for org in self.alive:
            before = org.energy
            pnl = org.energy * org.position * move
            org.energy += pnl
            org.trading_pnl += pnl
            if org.position != 0.0:
                won = pnl - org.fees_last > 0
                org.wins += won
                org.losses += not won
                org.mood += org.temperament * ((1.0 if won else 0.0) - org.mood)
            org.engagement_sum += abs(org.position)
            org.engagement_n += 1
            fine = 0.0
            if (t - org.born_at) % self.candles_per_day == 0:
                daily_pct = (org.energy / max(org.energy_1d_ago, 1e-9) - 1) * 100
                if daily_pct < 0:
                    fine = LOSS_FINE_RATE * abs(daily_pct) / 100 * org.energy
                elif org.engagement_sum == 0.0:
                    fine = IDLE_FINE_RATE * org.energy
                org.energy -= fine
                org.energy_1d_ago = org.energy
                org.last_daily_pct = daily_pct
                org.last_daily_engagement = org.engagement_sum / max(org.engagement_n, 1)
                org.engagement_sum, org.engagement_n = 0.0, 0
            org.trace.append((pnl, org.fees_last, fine, before))
            del org.trace[:-DEATH_LOOKBACK]
            org.fees_last = 0.0
            org.peak_money = max(org.peak_money, org.energy)
            if org.recent:
                if len(org.recent) >= COMBO_MIN and all(sk > 0 for _, sk in org.recent):
                    org.combos += 1
                org.recent = []
            else:
                org.worst = None
        if t > self.fund_t0:
            self._fund_step(move, t)

        # 3. Death and birth. Fitness is purely profit: growth(t) plus a bonus that grows as
        # the average daily return approaches, then passes, TARGET_DAILY_PCT (only while
        # genuinely engaged), minus a drawdown penalty past the interval's threshold.
        def fitness(o: Organism) -> float:
            growth = math.log(max(o.energy / START_MONEY, 1e-9)) / max(o.age(t), 1)
            engaged = o.last_daily_engagement > ENGAGEMENT_MIN
            daily_pct = o.last_daily_pct if engaged else 0.0
            near = NEAR_BONUS * float(np.clip(daily_pct / TARGET_DAILY_PCT, 0, 1)) if engaged else 0.0
            excess = EXCESS_BONUS * max(0.0, daily_pct - TARGET_DAILY_PCT) / 100 if engaged else 0.0
            current_dd = 1.0 - o.energy / max(o.peak_money, 1e-9)
            dd_penalty = DD_PENALTY * max(0.0, current_dd - self.dd_threshold)
            return growth + near + excess - dd_penalty

        dead = [o for o in self.alive if o.energy < BANKRUPT]
        if t % self.cull_every == 0:
            adults = [o for o in self.alive if o.age(t) >= self.cull_min_age and o.energy >= BANKRUPT]
            if len(adults) > 1:
                slowest = min(adults, key=fitness)
                slowest.culled = True
                dead.append(slowest)
        if dead:
            dead_ids = {o.id for o in dead}
            self.alive = [o for o in self.alive if o.id not in dead_ids]
            for o in dead:
                self.deaths += 1
                self.death_log.append({"id": o.id, "t": t, "age": o.age(t), "generation": o.generation})
                self.graveyard.append(self._death_record(o, t))
                sens = "news_follower" if o.news_sensitivity > 0.25 else "news_contrarian" if o.news_sensitivity < -0.25 else "news_indifferent"
                for gene in (*o.groups, sens):
                    acc = self.life_stats.setdefault(gene, [0.0, 0])
                    acc[0] += o.age(t)
                    acc[1] += 1
            del self.death_log[:-50]
            if not self.alive:  # extinction: start again from scratch
                self.alive = [self._random_organism(t) for _ in range(self.population_size)]
            attempts = 0
            while len(self.alive) < self.population_size:
                attempts += 1
                child = self._child(t) if attempts <= 20 else self._random_organism(t)
                if self._fit_org(child, t):
                    self.alive.append(child)
                elif attempts > 60:
                    break

        # 4. Learning: each organism refits on its own (staggered) schedule.
        for org in self.alive:
            if (t + org.id) % self.refit_every == 0:
                self._fit_org(org, t)

        # 5. Everyone predicts its price tool and takes its own profit-seeking combined
        # position; the fund trades the energy-weighted average of the TOP_K adults'
        # positions (not a combined prediction - see class docstring).
        preds, weighted = {}, self._ensemble_members(t)
        positions_now: dict[int, float] = {}
        for org in self.alive:
            raw = self._raw(org, t) * org.boldness * self._feeling(org, t)
            preds[org.id] = raw
            positions_now[org.id] = self._target_position(raw, org, t)
            self._trade(org, positions_now[org.id])
        self.pending[t] = preds
        ens_scalar = self._combine(weighted, preds)
        self.ens_pending[t] = ens_scalar * HORIZON_ARRAY
        if weighted:
            w = np.array([max(o.energy, 1e-6) for o in weighted])
            fund_target = float((w * np.array([positions_now[o.id] for o in weighted])).sum() / w.sum())
        else:
            fund_target = 0.0
        prev_fund_position = self.fund.position
        self._trade(self.fund, fund_target)
        if self.fund.position != prev_fund_position:
            self._record_fund_trade(prev_fund_position, self.fund.position, t)
        if t % 24 == 0 or t == len(self.logc) - 1:
            self.history.append(self._snapshot(t))

    # -- forecasting ----------------------------------------------------------

    def predict(self, df: pd.DataFrame, steps: int) -> tuple[list[tuple[float, float, float]], dict]:
        """([(log_return, log_low, log_high) cumulative from the last close,
        one per step 1..steps], diagnostics). Predicts from the last row
        fit() replayed; if `df` has newer rows, from its last row.

        Diagnostic only (the ensemble's *price tool*, flat-extrapolated) - the fund's actual
        trading is the combined 4-tool position (see fund_trades / _target_position), not this
        path."""
        t = self.t_last
        X_last = None
        s_now = self.S[t]
        if len(df) and df.index[-1] != self.index[t]:
            X_new, _, _ = build_inputs(df, self.long_w)
            if X_new.shape[1] == self.X.shape[1]:
                X_last = X_new[-1]
            s_now = mood_signal(df)[0][-1]
        feeling = lambda o: float(np.clip(o.mood + NEWS_MOOD_WEIGHT * o.news_sensitivity * s_now, 0.0, 1.0))
        members = self._ensemble_members(t)
        preds = {}
        for org in self.alive:
            if X_last is not None:
                x = X_last[org.cols]
                raw = float(np.append(_standardize(x, org.mu, org.sd), 1.0) @ org.W) if org.W is not None and np.isfinite(x).all() else 0.0
            else:
                raw = self._raw(org, t)
            preds[org.id] = raw * org.boldness * feeling(org)
        ens_scalar = self._combine(members, preds)
        point = ens_scalar * HORIZON_ARRAY

        low_off, high_off = [], []
        for j in range(len(HORIZONS)):
            res = self.residuals[j]
            if len(res) >= MIN_RESIDUALS:
                lo, hi = np.quantile(res, BAND_QUANTILES)
            else:  # not enough own mistakes yet: the typical real move, as a normal band
                lo, hi = -_BAND_Z * self.scale[j] * 1.25, _BAND_Z * self.scale[j] * 1.25
            low_off.append(min(lo, 0.0))
            high_off.append(max(hi, 0.0))

        # Between trained horizons: linear in k (from 0 at k=0). Past the
        # last one: same drift per step, band widening as k ** exponent
        # (cumulative mode; see BAND_MODE).
        knots = [0, *HORIZONS]
        path = []
        for k in range(1, steps + 1):
            if k <= H_MAX:
                p = np.interp(k, knots, [0.0, *point])
                lo = np.interp(k, knots, [0.0, *low_off])
                hi = np.interp(k, knots, [0.0, *high_off])
            else:
                growth = (k / H_MAX) ** HORIZON_EXPONENT
                p, lo, hi = point[-1] * k / H_MAX, low_off[-1] * growth, high_off[-1] * growth
            if BAND_MODE == "step":
                lo, hi = low_off[0], high_off[0]
            path.append((float(p), float(p + lo), float(p + hi)))
        return path, self.diagnostics(t, members)

    def _money_summary(self, t: int) -> dict | None:
        if not hasattr(self, "fund"):
            return None
        alive = self.alive
        rich = max(alive, key=lambda o: o.energy) if alive else None
        f = self.fund
        days = max((self.index[t] - self.index[self.fund_t0]).total_seconds() / 86400, 1e-9)
        avg_daily_pct = (math.exp(math.log(max(f.energy / START_MONEY, 1e-9)) / days) - 1) * 100
        return {
            "start": START_MONEY, "currency": "EUR", "since": self.fund_start,
            "fund": round(f.energy, 2), "fund_return_pct": round((f.energy / START_MONEY - 1) * 100, 2),
            "avg_daily_return_pct": round(avg_daily_pct, 4), "target_daily_pct": TARGET_DAILY_PCT,
            "fund_peak": round(f.peak_money, 2), "fund_trades": f.trades, "fund_fees": round(f.fees_paid, 2),
            "fund_position": round(f.position, 2),
            "total_alive_money": round(sum(o.energy for o in alive), 2),
            "richest": None if rich is None else {"id": rich.id, "money": round(rich.energy, 2), "age": rich.age(t)},
            "long": sum(o.position > 0 for o in alive), "short": sum(o.position < 0 for o in alive),
            "out": sum(o.position == 0 for o in alive),
        }

    def fund_trades_report(self) -> dict:
        """The fund's own entry/exit history, same shape as ml/box_breakout.py's trades - for
        charting intent/exit markers instead of a predicted price line."""
        trades = getattr(self, "fund_trades", [])
        open_trade = trades[-1] if trades and trades[-1]["exit_bar"] is None else None
        return {"trades": [t for t in trades], "open_trade": open_trade,
                "last_close": round(float(math.exp(self.logc[self.t_last])), 2) if self.fitted and not self.newborn else None,
                "last_time": self.index[self.t_last].isoformat() if self.fitted and not self.newborn and len(self.index) else None}

    def diagnostics(self, t: int, members: list[Organism] | None = None) -> dict:
        members = members if members is not None else self._ensemble_members(t)
        snap = self._snapshot(t)
        ids = {o.id for o in members}

        def org_json(o: Organism) -> dict:
            feel = self._feeling(o, t)
            return {"id": o.id, "age": o.age(t), "generation": o.generation, "energy": round(o.energy, 1),
                    "money": round(o.energy, 2), "position": round(o.position, 2), "trades": o.trades,
                    "mood": round(feel, 3), "earned_mood": round(o.mood, 3), "news_effect": round(feel - o.mood, 3),
                    "emotion": emotion_name(feel), "wins": o.wins, "losses": o.losses,
                    "combos": getattr(o, "combos", 0),
                    "parents": list(o.parents), "voting": o.id in ids, **o.genes()}

        emotions: dict[str, int] = {}
        for o in self.alive:
            e = emotion_name(self._feeling(o, t))
            emotions[e] = emotions.get(e, 0) + 1
        stats = getattr(self, "life_stats", {})
        survival = {g: {"avg_lifespan": round(v[0] / v[1], 1), "deaths": int(v[1])} for g, v in stats.items() if v[1]}
        alive_sens = [o.news_sensitivity for o in self.alive]
        usage = {g: round(sum(g in o.groups for o in members) / max(len(members), 1), 2) for g in self.group_names}
        deaths_recent = [d for d in self.death_log if d["t"] > t - 100]
        track = []
        tr = getattr(self, "track", None)
        for j, h in enumerate(HORIZONS):
            if not tr or not tr["life_n"][j]:
                continue
            rb, re_ = sum(tr["base"][j]), sum(tr["err"][j])
            track.append({
                "h": h,
                "n": int(tr["life_n"][j]),
                "skill": round(float(1 - tr["life_err"][j] / tr["life_base"][j]), 4) if tr["life_base"][j] else None,
                "direction_pct": round(float(tr["life_hits"][j] / tr["life_calls"][j] * 100), 1) if tr["life_calls"][j] else None,
                "recent_n": len(tr["err"][j]),
                "recent_skill": round(1 - re_ / rb, 4) if rb else None,
                "recent_direction_pct": round(float(np.mean(tr["hit"][j]) * 100), 1) if tr["hit"][j] else None,
            })
        return {
            "method": "evolution",
            "interval": self.interval,
            "newborn": self.newborn,
            "lived_from": self.index[0].isoformat() if len(self.index) else None,
            "candles_lived": len(self.index),
            "track_record": track,
            "track_from": tr["first_time"] if tr else None,
            "population": len(self.alive),
            "births": self.births,
            "deaths": self.deaths,
            "deaths_last_100": len(deaths_recent),
            "avg_lifespan_of_dead": round(float(np.mean([d["age"] for d in self.death_log])), 1) if self.death_log else None,
            "max_generation": snap["max_generation"],
            "mood": snap["mood"],
            "emotion": emotion_name(snap["mood"]),
            "emotions": emotions,
            "input_usage": usage,
            "available_inputs": list(self.group_names),
            "gene_survival": survival,
            "market_sentiment": {"value": round(float(self.S[t]), 3), "source": str(self.S_source[t]) or None},
            "avg_news_sensitivity": round(float(np.mean(alive_sens)), 3) if alive_sens else None,
            "money": self._money_summary(t),
            "band_calibrated": all(len(r) >= MIN_RESIDUALS for r in self.residuals),
            "leaders": [org_json(o) for o in members],
            "history": [{"time": h["time"], "mood": h["mood"]} for h in self.history[-60:]],
        }


# --- A population that keeps living (persisted per coin/interval) -----------

_registry: dict[str, EvolutionaryForecaster] = {}
_registry_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}
_saved_mtime: dict[str, float] = {}   # what this process last wrote, per key


def _closed_rows(df: pd.DataFrame, now: datetime | None) -> pd.DataFrame:
    """Drop a last candle whose period hasn't ended (its close would still
    change): only closed candles become part of the population's life."""
    if len(df) < 3:
        return df
    step = pd.Series(df.index).diff().median()
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    return df.iloc[:-1] if df.index[-1] + step > now else df


def _standardize(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return np.clip((x - mu) / sd, -MAX_Z, MAX_Z)


def state_path(key: str) -> Path:
    return STATE_DIR / f"{key}.joblib"


def save_state(key: str, ev: EvolutionaryForecaster) -> None:
    import joblib

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = state_path(key).with_suffix(".tmp")
    joblib.dump(ev, tmp)
    tmp.replace(state_path(key))
    _saved_mtime[key] = state_path(key).stat().st_mtime


def load_state(key: str) -> EvolutionaryForecaster | None:
    import joblib

    path = state_path(key)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        logger.exception("Evolution: could not load %s, starting a new population", path)
        return None


def evolve(df: pd.DataFrame, key: str | None = None, now: datetime | None = None,
           interval: str = DEFAULT_INTERVAL) -> EvolutionaryForecaster:
    """A fitted population for `df`. Without a key: a fresh replay of `df`.
    With one: the saved population for that key keeps living through the
    candles that closed since (a fresh replay only if there is none yet, or
    the new data doesn't follow on from it). The returned object is never
    mutated afterwards - the next call continues a copy of it."""
    closed = _closed_rows(df, now)
    if key is None:
        return EvolutionaryForecaster(interval=interval).fit(closed)
    with _registry_lock:
        lock = _key_locks.setdefault(key, threading.Lock())
    with lock:
        path = state_path(key)
        # A file this process didn't write (e.g. ml.pretrain_evolution ran
        # meanwhile) wins over the in-memory population.
        on_disk_is_newer = path.exists() and path.stat().st_mtime != _saved_mtime.get(key)
        state = load_state(key) if on_disk_is_newer or key not in _registry else _registry[key]
        if on_disk_is_newer and state is not None:
            _saved_mtime[key] = path.stat().st_mtime
        ev, lived = None, 0
        if state is not None and getattr(state, "version", 1) != STATE_VERSION:
            logger.warning("Evolution %s: saved population is from an older version, starting a new one", key)
            state = None
        if state is not None:
            ev = copy.deepcopy(state)
            try:
                lived = ev.extend(closed)
            except ValueError as exc:
                logger.warning("Evolution %s: %s - starting a new population on the available candles", key, exc)
                ev = None
        if ev is None:
            ev = EvolutionaryForecaster(interval=interval).fit(closed)
            lived = len(closed)
        _registry[key] = ev
        if lived and not ev.newborn:
            try:
                save_state(key, ev)
            except Exception:
                logger.exception("Evolution: could not save %s", key)
        return ev
