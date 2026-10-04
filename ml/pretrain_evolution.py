"""Let the "tuned" population live through years of old candles.

Downloads closed candles from Binance (the coin and BTC, for the BTC inputs)
and the market context (ml/market_context.py: Fear & Greed, Binance funding
and premium, DeFi TVL, the app's own news sentiment), replays the whole
history through the evolving population (ml/evolution.py) and saves it to
data/evolution/<coin>_<interval>.joblib. From then on the app continues that
same population with every new candle instead of starting from the last
1000 candles.

To show whether the new data helps, the same candles are lived twice: once
with only the price-based genes, once with the context genes and the
news-swayed moods too, and both track records are printed side by side.

    .venv/bin/python -m ml.pretrain_evolution                    # ETH, 15m 1h 4h 1d 1w
    .venv/bin/python -m ml.pretrain_evolution --interval 1h --candles 50000

Prints the population's real track record over that history: how often its
vote beat "the price stays the same" and called the direction, on candles it
hadn't seen yet when it voted - with and without the new data - and how
long models with each gene lived on average.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from ml import evolution, market_context
from ml.features import build_feature_frame
from ml.walk_forward import fetch_binance

# Default history per interval: everything Binance has for ETHUSDT (Aug 2017).
DEFAULT_CANDLES = {"15m": 330_000, "1h": 83_000, "4h": 21_000, "1d": 3_500, "1w": 500}


def _context_sources(symbol: str, interval: str, start: pd.Timestamp, total: int) -> dict:
    from api.services import _sentiment_by_date
    from data_collector.defillama_client import get_ethereum_tvl_by_date

    sources = {"sentiment_by_date": _sentiment_by_date(lookback_days=5000), "tvl_by_date": get_ethereum_tvl_by_date()}
    for name, fn in (("fear_greed", market_context.fetch_fear_greed),
                     ("funding", lambda: market_context.fetch_funding(symbol, start)),
                     ("premium", lambda: market_context.fetch_premium(symbol, interval, total))):
        try:
            sources[name] = fn()
        except Exception as exc:  # a missing source just means fewer genes
            print(f"  ! {name} unavailable: {exc}")
    for name, src in sources.items():
        n = len(src) if src is not None else 0
        first = (min(src) if isinstance(src, dict) else src.index[0]) if n else None
        print(f"  {name}: {n} values" + (f" since {pd.Timestamp(first):%Y-%m-%d}" if first is not None else ""))
    return sources


def _print_track(label: str, d: dict) -> None:
    rows = {r["h"]: r for r in d["track_record"]}
    cells = [f"h={h}: skill {rows[h]['skill']:+.4f} dir {rows[h]['direction_pct']}%" for h in (1, 4, 24) if h in rows]
    print(f"  {label:<22} " + " | ".join(cells))


def pretrain(coin_id: str, symbol: str, interval: str, candles: int, compare: bool = True) -> evolution.EvolutionaryForecaster:
    t0 = time.time()
    rows = fetch_binance(symbol, interval, candles)
    btc = fetch_binance("BTCUSDT", interval, candles)
    btc_close = pd.DataFrame(btc).set_index("timestamp")["close"]
    btc_close.index = pd.to_datetime(btc_close.index, utc=True)
    df = build_feature_frame(rows, {}, btc_close=btc_close, tvl_by_date=None).dropna()
    print(f"{interval}: {len(df)} candles {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d %H:%M} "
          f"(downloaded in {time.time() - t0:.0f}s)", flush=True)
    full = market_context.add_context(df, **_context_sources(symbol, interval, df.index[0], len(df)))

    if compare:
        t1 = time.time()
        base = evolution.EvolutionaryForecaster().fit(df)
        print(f"  price-only population lived in {time.time() - t1:.0f}s", flush=True)
    t1 = time.time()
    ev = evolution.EvolutionaryForecaster().fit(full)
    key = f"{coin_id}_{interval}"
    evolution.save_state(key, ev)
    d = ev.diagnostics(ev.t_last)
    print(f"  with context lived in {time.time() - t1:.0f}s: {d['births']} born, {d['deaths']} died, "
          f"generation {d['max_generation']}, mood {d['emotion']} -> {evolution.state_path(key)}")
    if compare:
        _print_track("price only:", base.diagnostics(base.t_last))
    _print_track("with context + news:", d)
    surv = d["gene_survival"]
    core = [v["avg_lifespan"] for g, v in surv.items() if g in evolution.CORE_GROUPS]
    print(f"  avg lifespan of the dead: price genes {sum(core) / len(core):.0f} candles" if core else "", end="")
    for g in ("sentiment", "news", "defi", "futures", "news_follower", "news_contrarian", "news_indifferent"):
        if g in surv:
            print(f" | {g} {surv[g]['avg_lifespan']:.0f} ({surv[g]['deaths']})", end="")
    print(f"\n  voters use: {d['input_usage']}", flush=True)
    m = d["money"]
    if m:
        from collections import Counter
        causes = Counter(x["cause"] for x in ev.graveyard)
        print(f"  money (from {m['since'][:10]}, {m['start']:.0f} EUR each): fund {m['fund']:.2f} EUR "
              f"({m['fund_return_pct']:+.1f}%, {m['fund_trades']} trades, {m['fund_fees']:.2f} EUR fees) | "
              f"buy & hold {m['buy_hold']:.2f} EUR ({m['buy_hold_return_pct']:+.1f}%) | "
              f"richest alive #{m['richest']['id']} {m['richest']['money']:.2f} EUR", flush=True)
        print(f"  deaths by cause: {dict(causes)}", flush=True)
    return ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coin-id", default="ethereum")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", action="append", help="repeatable; default: all of " + ", ".join(DEFAULT_CANDLES))
    parser.add_argument("--candles", type=int, help="history length (default per interval: see DEFAULT_CANDLES)")
    parser.add_argument("--no-compare", action="store_true", help="skip the price-only comparison run")
    args = parser.parse_args()
    for interval in args.interval or list(DEFAULT_CANDLES):
        pretrain(args.coin_id, args.symbol, interval, args.candles or DEFAULT_CANDLES[interval], not args.no_compare)
    print("Restart the app (or wait for its next refresh) to continue these populations live.")


if __name__ == "__main__":
    main()
