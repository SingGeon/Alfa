# ETH Price Predictor

Aplicație web în Python care prezice prețul Ethereum (ETH) combinând:
- date de piață live (CoinGecko / Binance, cu fallback pe Etherscan),
- un model de time-series forecasting (scikit-learn, pluggable cu Prophet/LSTM),
- analiză de sentiment (VADER) pe știri crypto (CryptoPanic / NewsAPI / RSS),
- un scor de încredere și un rezumat narativ zilnic (opțional prin Claude API).

## Structură

```
config.py                  # configurare centralizată (.env)
run_api.py                 # entrypoint Flask API
run_scheduler.py           # entrypoint job-uri periodice (APScheduler)

database/
  mongo_client.py          # conexiune Mongo, fallback grațios dacă Mongo e oprit
  repository.py            # CRUD: prices, candles, news, predictions

data_collector/
  http_utils.py            # GET cu retry/backoff (rate limits, 5xx, timeouts)
  market_data.py           # preț curent + candlestick OHLC (CoinGecko -> Binance -> Etherscan)
  etherscan_client.py       # preț ETH/USD + gas price on-chain (opțional, are API key)
  news_collector.py        # CryptoPanic + NewsAPI (opționale) + RSS (CoinDesk/Cointelegraph)
  jobs.py                  # job-urile efective apelate de scheduler

nlp/
  sentiment.py              # scor VADER per articol + agregare zilnică

ml/
  features.py                # feature engineering (indicatori tehnici + sentiment)
  price_predictor.py         # PricePredictor: backend sklearn (implicit) / prophet / lstm
  combined_predictor.py      # scor de încredere + rezumat narativ (Claude, opțional)

api/
  routes.py                  # /api/price, /api/predict, /api/news, /api/sentiment, /api/summary, /api/gas
  services.py                 # orchestrează candles + sentiment + model -> predicție combinată

frontend/
  dashboard.py                # dashboard Streamlit (grafic + predicție + știri + sentiment)

tests/                         # pytest, complet offline (mongomock + requests-mock)
```

Codul e organizat astfel încât adăugarea unei alte criptomonede înseamnă doar
schimbarea parametrilor `coin_id` / `symbol` la apelurile din `data_collector`
și `ml` — nimic din pipeline nu e hardcodat pe Ethereum în afara valorilor
implicite din `config.py`.

## Instalare

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completează cheile API pe care le ai (toate sunt opționale)
```

Aplicația funcționează din prima **fără nicio cheie API**: prețul vine de la
CoinGecko/Binance (publice), iar știrile din feed-urile RSS CoinDesk/Cointelegraph.
Cheile opționale (`CRYPTOPANIC_API_KEY`, `NEWSAPI_KEY`, `ETHERSCAN_API_KEY`,
`ANTHROPIC_API_KEY`) îmbunătățesc acoperirea, dar nu sunt obligatorii.

MongoDB e opțional și el pentru a testa rapid API-ul: dacă nu e disponibil,
`database/mongo_client.py` continuă fără persistență (logează un warning) în
loc să crape aplicația. Pentru rulare reală ai nevoie de o instanță Mongo:

```bash
docker run -d --name eth-mongo -p 27017:27017 mongo:7
```

## Rulare

Trei procese separate (terminale/containere diferite):

```bash
# 1) Colectorul de date (seedează imediat, apoi rulează periodic)
python run_scheduler.py

# 2) API-ul Flask
python run_api.py

# 3) Dashboard-ul
streamlit run frontend/dashboard.py
```

Predicția are nevoie de minimum ~40 de candles istorice salvate pentru
intervalul cerut (implicit 1h) — lasă scheduler-ul să ruleze puțin, sau
rulează o singură colectare manuală:

```bash
python -c "from data_collector.jobs import run_all_once; run_all_once()"
```

## API

| Endpoint | Descriere |
|---|---|
| `GET /api/price/current` | Preț curent, volum 24h, variație 24h |
| `GET /api/price/history?interval=1h\|1d\|1w&limit=` | Istoric OHLC |
| `GET /api/predict?interval=&steps=&use_sentiment=` | Predicție preț + interval de încredere + scor de încredere |
| `GET /api/news?limit=` | Ultimele articole + sentiment |
| `GET /api/sentiment?days=` | Scor agregat de sentiment pe zi |
| `GET /api/summary?interval=` | Predicție + rezumat narativ + știri suport |
| `GET /api/gas` | Gas price on-chain (necesită `ETHERSCAN_API_KEY`) |

## Modele de predicție

`PREDICTION_BACKEND` din `.env` alege backend-ul:
- `sklearn` (implicit) — `GradientBoostingRegressor` pe randamente logaritmice,
  cu interval de încredere real via regresie quantile (nu necesită dependențe
  suplimentare).
- `prophet` — necesită `pip install prophet`.
- `lstm` — punct de extensie (necesită `pip install tensorflow`); vezi
  `ml/price_predictor.py::_fit_lstm` pentru unde se conectează.

Sentimentul din știri intră în model ca feature suplimentar (`sentiment`,
agregat zilnic din articolele colectate), iar `ml/combined_predictor.py`
calculează separat un scor de încredere combinând lățimea intervalului de
predicție cu acordul dintre direcția prezisă și sentiment.

## Teste

```bash
pytest tests/ -q
```

Toate testele rulează offline: baza de date e `mongomock`, iar apelurile
HTTP către CoinGecko/Binance sunt interceptate cu `requests-mock`.

## Extindere la alte criptomonede

Toate funcțiile din `data_collector`, `ml` și `api/services.py` primesc
`coin_id` (id CoinGecko) și, unde e cazul, `symbol` (simbol Binance) ca
parametri opționali cu valoare implicită din `config.py`. Pentru a adăuga
Bitcoin, de exemplu, e suficient să apelezi endpoint-urile cu
`?coin_id=bitcoin` (după ce extinzi și `BINANCE_SYMBOL`/`COIN_ID` per coin,
sau adaugi un mic registry coin_id -> symbol dacă vrei mai multe monede simultan).
