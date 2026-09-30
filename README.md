# Alfa — ETH Price Predictor + Scout AI

A Python web app with two faces:

- **ETH Price Predictor** — a live Ethereum dashboard: candlestick chart,
  price prediction with a confidence interval, technical indicators, news
  sentiment, and a Buy/Sell signal panel.
- **Scout AI** — scans the crypto and stock market for **underdogs**
  (mid/small-cap with real room to grow, not BTC/ETH/AAPL/MSFT), scored by
  model + news sentiment, with one exception: if a large asset (mega-cap)
  has a genuinely dramatic signal, it shows up too, flagged separately.

Combines:
- live market data (Binance primary, CoinGecko/Etherscan fallback),
- an ensemble of time-series forecasting models (scikit-learn, pluggable with
  Prophet/LSTM), with real market features: RSI/MACD/Bollinger, taker
  buy/sell ratio, correlation with BTC, DeFi TVL momentum,
- sentiment analysis (VADER) across 10 crypto RSS sources,
- a calibrated confidence score and a daily narrative summary.

## Contents

- [Structure](#structure)
- [Installation](#installation)
- [Running](#running)
- [ETH Dashboard](#eth-dashboard)
- [Scout AI](#scout-ai)
- [API](#api)
- [Prediction evaluation](#prediction-evaluation)
- [Prediction models](#prediction-models)
- [Performance](#performance)
- [Tests](#tests)
- [Extending to other cryptocurrencies](#extending-to-other-cryptocurrencies)

## Structure

```
config.py                     # centralized configuration (.env)
run_api.py                    # Flask API entrypoint + prediction cache warm-up thread
run_scheduler.py               # periodic job entrypoint (APScheduler)

database/
  mongo_client.py             # Mongo connection (tz_aware), graceful fallback if Mongo is down
  repository.py                # CRUD: prices, candles, news, predictions, scout_results, pinned_assets

data_collector/
  http_utils.py                # GET with retry/backoff (rate limits, 5xx, timeouts)
  market_data.py               # current price + candlestick OHLC (Binance -> CoinGecko -> Etherscan)
  etherscan_client.py          # ETH/USD price + on-chain gas price (optional, needs an API key)
  defillama_client.py          # historical Ethereum DeFi TVL (public, no key, 6h cache)
  news_collector.py            # 10 crypto RSS sources, 5min cache on the raw fetch
  jobs.py                      # the actual work each scheduled job performs

nlp/
  sentiment.py                  # VADER score per article + daily aggregation

ml/
  features.py                   # feature engineering: technical + sentiment + taker_buy_ratio +
                                 # btc_return_1 + tvl_momentum
  price_predictor.py            # PricePredictor: sklearn ensemble (default) / prophet / lstm
  combined_predictor.py         # confidence score + narrative summary

scout/
  universe.py                   # what gets scanned: underdogs + a separate mega-cap tier
  revolut_allowlist.py          # only assets tradeable on Revolut (hand-editable list)
  pipeline.py                   # per-asset analysis: candles + news + model -> prediction + narrative
  scanner.py                    # orchestrates a full scan, saves ranked results
  narrative.py                  # per-asset "why and when" text

evaluation/                     # prediction evaluation (see "Prediction evaluation" below)
  schema.sql                    # SQLite schema of the `predictions` table
  storage.py                    # SQLite access: log, query/filter/sort/paginate, per-model summary
  recorder.py                   # the single hook called from api/services.run_prediction
  evaluator.py                  # fills real prices (Binance -> CoinGecko), errors, completion
  charts.py                     # per-prediction images: live snapshot + final chart (matplotlib, Agg)
  daily_report.py               # end-of-day overview PNG (+ GIF, PDF), missed-day recovery
  jobs.py                       # APScheduler wiring (every minute + 00:05 UTC)
  api.py                        # /api/evaluation/* blueprint (table, CSV, charts, days)

api/
  routes.py                     # every endpoint (see the table below)
  services.py                   # orchestrates candles + sentiment + BTC + TVL + model -> prediction

frontend/
  web/
    index.html / landing.css / landing.js       # landing page (hero, how it works, features)
    dashboard.html / style.css / app.js         # ETH dashboard
    scout.html / scout.css / scout.js           # Scout AI list (filters, pin, mega-cap badge)
    detail.html / detail.css / detail.js        # per-asset detail (chart, prediction, narrative, live price)
    chart-utils.js                              # shared chart helper (app.js + detail.js)
    i18n.js                                     # shared EN/RO i18n runtime (language toggle)
    evaluation.html / evaluation.css / .js      # prediction evaluation table
    history.html / history.js                   # day history (daily overviews)
    visual.html / visual.js                     # visual history (one image per forecast)

tests/                          # pytest, fully offline (mongomock + requests-mock)
```

The code is organized so that adding another cryptocurrency to the main
dashboard just means changing the `coin_id` / `symbol` parameters — nothing
in the pipeline is hardcoded to Ethereum outside the defaults in
`config.py`. Scout AI already scans dozens of different assets out of the box.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in whichever API keys you have (all optional)
```

The app works out of the box **with no API key at all**: prices come from
Binance/CoinGecko (public), TVL from DeFiLlama (public), and news from 10 RSS
feeds. The optional keys (`CRYPTOPANIC_API_KEY`, `NEWSAPI_KEY`,
`ETHERSCAN_API_KEY`) improve coverage but aren't required — a missing
`ETHERSCAN_API_KEY` just disables the gas price badge.

MongoDB is optional for quickly testing the API: if it isn't available,
`database/mongo_client.py` keeps running without persistence (logs a
warning) instead of crashing the app. For real use you need a Mongo
instance:

```bash
docker run -d --name eth-mongo -p 27017:27017 mongo:7
```

## Running

A single process. Flask serves the API, all the static frontend
(`frontend/web/`), and periodic data collection (candles + news + Scout AI
scanning), all on the same port:

```bash
cd /home/singeon/Documents/Alfa

# Flask API + dashboard + Scout AI + data collector, at http://localhost:5000
.venv/bin/python run_api.py
```

On startup it immediately seeds current data, then refreshes it
automatically on an interval (candles/price every `COLLECT_INTERVAL_MINUTES`,
news every `NEWS_INTERVAL_MINUTES`, Scout AI scans every
`SCOUT_INTERVAL_MINUTES`) for as long as the process runs — there's no
second process to start separately, and no risk of forgetting to start it
and being left with stale (unrefreshed) candles with no visible error.

If you're running multiple API instances (e.g. several workers) and want
ONE central collector instead of each instance collecting independently,
you can start collection as its own process with
`.venv/bin/python run_scheduler.py` — see that file's docstring for details.

The ETH prediction needs at least ~40 stored historical candles for the
requested interval (default 1h) — let the app run for a bit after startup,
or run a single manual collection:

```bash
python -c "from data_collector.jobs import run_all_once; run_all_once()"
```

Scout AI automatically kicks off a first scan (on a background thread, so it
doesn't block the scheduler from starting) — it takes a few minutes, since it
trains a real model per asset across ~135 assets.

## ETH Dashboard

`http://localhost:5000/dashboard.html` (the landing page at `/` links to it) — a candlestick chart (Lightweight Charts) with:
- an overlaid prediction + confidence band, SMA/EMA/Bollinger/Fibonacci indicators,
- a price tooltip that tracks the mouse (read directly off the price scale
  at the cursor's Y position, so it always matches the axis exactly — also
  works over the prediction region, where there's no real candle),
- a live price ticker (polled every 3s) + 24h change,
- a "Pinned" panel — assets bookmarked from Scout AI show up right next to the ETH price,
- a Buy/Sell signal panel from RSI + MACD + trend + sentiment,
- separate predictions for the 24h and 7-day horizons.

## Scout AI

`http://localhost:5000/scout.html` — scans:
- **Underdogs**: crypto ranked 30-250 on CoinGecko + "trending" coins, stocks
  from the Yahoo Finance screeners `small_cap_gainers` / `aggressive_small_caps` /
  `undervalued_growth_stocks`. They show up if the prediction is positive and
  the model's confidence is ≥ 25/100.
- **Mega-cap (escape hatch)**: the top 30 crypto by market cap + large stocks
  (≥ $50B) from the `day_gainers` screener. These assets do NOT appear
  automatically — they need a prediction of **at least +10% over 7 days**
  with confidence ≥ 40/100 to pass the filter, flagged with a **⭐ Mega-cap**
  badge. The goal: if a big name genuinely has a dramatic signal, it still
  shows up, without diluting the rest with noise from a mega-cap's ordinary moves.
- Both universes are filtered to **only assets tradeable on Revolut**
  (`scout/revolut_allowlist.py` — a hand-editable list).
- **Pin**: any asset can be bookmarked (☆ → ★), saved in MongoDB — it stays
  accessible even if it drops out of the current scan, and shows up both in
  Scout AI's "Pinned" tab and next to the ETH price on the main dashboard.
- Clicking a row → a per-asset detail page, with a chart, a 7-day
  prediction, a "why and when" narrative, analyzed news, and a live price.

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/price/current` | Current ETH price, 24h volume, 24h change |
| `GET /api/price/history?interval=15m\|1h\|4h\|1d\|1w&limit=` | OHLC history |
| `GET /api/predict?interval=&steps=&use_sentiment=` | Price prediction + confidence interval + score |
| `GET /api/outlook?use_sentiment=` | Quick 24h + 7-day prediction |
| `GET /api/news?limit=` | Latest articles + sentiment |
| `GET /api/sentiment?days=` | Aggregated daily sentiment score |
| `GET /api/summary?interval=&lang=en\|ro` | Prediction + narrative summary + supporting news |
| `GET /api/gas` | On-chain gas price (needs `ETHERSCAN_API_KEY`) |
| `GET /api/patterns?interval=&lang=en\|ro` | Rule-based candlestick + geometric chart pattern detection |
| `GET /api/scout?asset_type=crypto\|stock&limit=` | Latest Scout AI scan results, ranked |
| `GET /api/scout/detail?asset_type=&id=&lang=en\|ro` | Full per-asset detail (chart, prediction, narrative, news) |
| `GET /api/scout/price?asset_type=&id=` | Lightweight live price for the detail page |
| `GET /api/scout/pins?asset_type=` | List of pinned assets |
| `POST /api/scout/pins` | Pin an asset (`{asset_type, id, symbol, name}`) |
| `DELETE /api/scout/pins?asset_type=&id=` | Unpin an asset |

`lang` (where supported) picks the language of any human-readable
narrative/note text in the response - `"en"` (default) or `"ro"`; it's pure
output-language selection over already-computed numbers and never triggers
a re-prediction.

## Prediction evaluation

Every prediction any model makes (`api.services.run_prediction`: dashboard
chart, outlook tiles, summary, the recurring snapshot job) is logged to a
SQLite database, scored against the real Binance price once its target candle
has closed, and drawn as a PNG. The models themselves are untouched; one call
at the end of `run_prediction` (`evaluation/recorder.py`) records their output
and never raises.

**How it runs.** Nothing extra to start: `python run_api.py` (or
`run_scheduler.py`) registers the jobs on the existing APScheduler:

| Job | When | What |
|---|---|---|
| `snapshot_predictions` | every 5 min | 24-step forecast for 1h, both model variants (warm cache, cheap); logged only if it differs from the last one |
| `evaluate_predictions` | every minute | append every real candle that has closed and redraw that prediction's snapshot; when the last one has closed, compute `abs_error`, `pct_error`, `direction_correct`, set `completed`, draw the final PNG |
| `daily_evaluation_report` | 00:05 UTC | `data/daily/YYYY-MM-DD_overview.png` (+ `.gif`, `.pdf`) for the day that just ended |
| startup recovery | on start | finish whatever resolved while the app was down, draw missing charts, build the overviews for any past days that don't have one |

**Storage.** `data/predictions.db`, table `predictions`
([evaluation/schema.sql](evaluation/schema.sql)). The unique key is
`(interval, model_name, created_at)`. On top of that, at most one row is kept
per `(interval, model_name, horizon_steps)` per candle of that interval. The
dashboard polls `/api/predict` every 30s while the model only retrains every
~15min, so without this one real forecast would be logged hundreds of times.
`model_name` is `<backend>-<variant>` (`sklearn-tuned`, `sklearn-legacy`). All
times are UTC (`YYYY-MM-DDTHH:MM:SSZ`). The browser converts them to the
viewer's timezone. The date filters are UTC dates.

**Scoring.** A forecast point at timestamp `T` means the close of the candle
that opens at `T` (the same convention the models train on), so the real
price for `T` is known at `T + interval`. `target_time` is the last forecast
candle and `resolved_at = target_time + interval`.
- `pct_error = (predicted - actual) / actual * 100`, signed: positive means
  the model was too high.
- `direction_correct` is true when the model called the same side of
  `price_at_prediction` (up / down / flat) as reality.
- If Binance fails after retries, or has a hole for a closed candle, CoinGecko
  price history fills the gap (last price at or before the candle close).
- A prediction whose final price is still missing `EVAL_EXPIRE_HOURS` (72h)
  after `resolved_at` is marked `expired` and left out of the statistics.

**Visual history.** Every logged prediction gets an image drawn like the
dashboard chart. It shows the last 48 real candles the model saw, then the AI
forecast (dashed line + confidence band), with the real candles drawn *on top
of* the forecast as they close. The image is saved to
`data/snapshots/YYYY-MM-DD/…png` (the day it was made) the moment the
prediction is logged, and redrawn every minute that new real candles arrive.
24-step forecasts (`EVAL_SNAPSHOT_STEPS`) get a new row + image every time the
forecast actually changes (the model retrained into a different answer). A
repeat of the same forecast is skipped. Other horizons keep one per candle.
`/visual.html` shows them as a gallery, newest first.

**Charts.** `data/charts/YYYY-MM-DD/{interval}_{model}_{created_at}.png`.
The date is the day the prediction resolved, so the 00:05 report always finds
every chart of the day that just ended. Charts are idempotent: an existing
file is never redrawn.

**Dashboard.** `/evaluation.html` has the per-model summary and the table,
with filters (interval, model, status, dates), sorting by clicking a column,
pagination, CSV export, and a click on a completed row to open its chart.
`/history.html` shows the daily overviews with a date selector.
`/visual.html` is the visual history gallery.

**API** (`/api/evaluation`):

| Endpoint | Description |
|---|---|
| `GET /predictions?interval=&model=&status=&horizon_steps=&date_from=&date_to=&sort=&order=&page=&page_size=` | Table rows + per-model summary (count, mean \|error %\|, direction accuracy %) |
| `GET /predictions.csv?…same filters…` | Every matching row as CSV |
| `GET /predictions/<id>/chart.png` | Final chart of one completed prediction |
| `GET /predictions/<id>/snapshot.png` | Visual snapshot, current state (any status) |
| `GET /models` | Model names and intervals, for the filters |
| `GET /days` | Days that have an overview, newest first |
| `GET /days/<YYYY-MM-DD>` | That day's stats + which files exist |
| `GET /days/<YYYY-MM-DD>/overview.png\|gif\|pdf` | The daily overview files |

**Settings** (`.env`): `EVAL_DATA_DIR` (default `data`), `EVAL_DB_FILE`
(`predictions.db`), `EVAL_EXPIRE_HOURS` (72), `EVAL_DAILY_GIF` / `EVAL_DAILY_PDF`
(both on), `EVAL_SNAPSHOT_STEPS` (24), `EVAL_SNAPSHOT_INTERVALS` (`1h`, comma
separated), `EVAL_SNAPSHOT_EVERY_MINUTES` (5), `EVAL_SNAPSHOT_HISTORY_CANDLES` (48).

**By hand:**

```bash
.venv/bin/python -c "from evaluation.evaluator import complete_pending_predictions as f; print(f())"
.venv/bin/python -c "from evaluation.charts import generate_missing_charts as f; print(f())"
.venv/bin/python -c "from datetime import date; from evaluation.daily_report import build_daily_report as f; print(f(date(2026, 9, 29), force=True))"
```

## Prediction models

`PREDICTION_BACKEND` in `.env` picks the backend:
- `sklearn` (default) — an ensemble of `GradientBoostingRegressor` (3
  different seeds, averaged) on log returns, with a real confidence interval
  via quantile regression.
- `prophet` — needs `pip install prophet`.
- `lstm` — an extension point (needs `pip install tensorflow`).

Features used for training (`ml/features.py`):
- technical: SMA/EMA, RSI(14), volatility, MACD, Bollinger position, relative
  volume, 6 price lags,
- **taker buy ratio** — what percentage of each candle's volume was
  aggressive buying (straight from Binance's own breakdown) — a real proxy
  for demand vs. supply,
- **btc_return_1** — BTC's own return on the same time grid — most altcoins
  (ETH included) move more in lockstep with BTC than on their own
  fundamentals,
- **tvl_momentum** — the daily change in Ethereum DeFi TVL (DeFiLlama) — a
  real "network utility" signal, not just price and volume,
- news sentiment, aggregated daily.

Any feature that can't be computed for a given asset (e.g. taker-buy doesn't
exist for stocks, TVL only applies to ETH) falls back to a neutral value
instead of dropping the row from training.

## Performance

Training one model (a full ensemble) takes ~6s — not acceptable on a
request. Mitigations in place:
- **Prediction cache** (`api/services.py`) with a 900s TTL, aligned with the
  actual data-collection cadence (15 min) — there's no point expiring it
  faster than the underlying data changes anyway.
- **Warm-up thread** (`run_api.py`) — proactively trains, in the background,
  at startup and every 5 min, the 3 combinations the dashboard actually uses
  (1h/1d/1w), so a real request always finds a warm cache.
- The lock around training covers the whole read-train-write cycle, not just
  the dict access — two concurrent requests for the same key (e.g. the chart
  + the daily summary on refresh) no longer train it twice.
- A dedicated cache (3s) for the Scout AI detail page — a second click on
  the same asset is instant instead of retraining.

Measured result: a full dashboard load dropped from 8-15s (cold) to ~1.8s.

## Tests

```bash
pytest tests/ -q
```

All tests run offline: the database is `mongomock`, HTTP calls to
Binance/CoinGecko/DeFiLlama are intercepted with `requests-mock` or stubbed
explicitly (see `tests/conftest.py`).

## Extending to other cryptocurrencies

Every function in `data_collector`, `ml`, and `api/services.py` takes
`coin_id` (CoinGecko id) and, where relevant, `symbol` (Binance symbol) as
optional parameters defaulting from `config.py`. For the main dashboard, it's
enough to call the endpoints with `?coin_id=bitcoin` (after also extending
`BINANCE_SYMBOL`/`COIN_ID` per coin). Scout AI already scans dozens of assets
at once, independent of whichever coin the dashboard is configured for.
