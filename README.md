# Alfa — ETH Price Predictor + Scout AI

Aplicație web în Python, cu două fețe:

- **ETH Price Predictor** — dashboard live pentru Ethereum: grafic candlestick,
  predicție de preț cu interval de încredere, indicatori tehnici, sentiment din
  știri și un panou de semnal Cumpără/Vinde.
- **Scout AI** — scanează piața de cripto și acțiuni în căutare de **underdogs**
  (mid/small-cap cu loc real de creștere, nu BTC/ETH/AAPL/MSFT), scorate după
  model + sentiment din știri, cu o excepție: dacă un activ mare (mega-cap)
  are un semnal cu adevărat dramatic, apare și el, marcat separat.

Combină:
- date de piață live (Binance primar, CoinGecko/Etherscan fallback),
- un ansamblu de modele de time-series forecasting (scikit-learn, pluggable cu
  Prophet/LSTM), cu feature-uri reale de piață: RSI/MACD/Bollinger, raport
  cumpărare/vânzare (taker buy ratio), corelație cu BTC, momentum TVL DeFi,
- analiză de sentiment (VADER) pe 10 surse RSS crypto,
- un scor de încredere calibrat și un rezumat narativ zilnic.

## Cuprins

- [Structură](#structură)
- [Instalare](#instalare)
- [Rulare](#rulare)
- [Dashboard ETH](#dashboard-eth)
- [Scout AI](#scout-ai)
- [API](#api)
- [Modele de predicție](#modele-de-predicție)
- [Performanță](#performanță)
- [Teste](#teste)
- [Extindere la alte criptomonede](#extindere-la-alte-criptomonede)

## Structură

```
config.py                     # configurare centralizată (.env)
run_api.py                    # entrypoint Flask API + thread de warm-up al cache-ului de predicții
run_scheduler.py               # entrypoint job-uri periodice (APScheduler)

database/
  mongo_client.py             # conexiune Mongo (tz_aware), fallback grațios dacă Mongo e oprit
  repository.py                # CRUD: prices, candles, news, predictions, scout_results, pinned_assets

data_collector/
  http_utils.py                # GET cu retry/backoff (rate limits, 5xx, timeouts)
  market_data.py               # preț curent + candlestick OHLC (Binance -> CoinGecko -> Etherscan)
  etherscan_client.py          # preț ETH/USD + gas price on-chain (opțional, are API key)
  defillama_client.py          # TVL istoric DeFi pe Ethereum (public, fără cheie, cache 6h)
  news_collector.py            # 10 surse RSS crypto, cache 5 min pe fetch-ul brut
  jobs.py                      # job-urile efective apelate de scheduler

nlp/
  sentiment.py                  # scor VADER per articol + agregare zilnică

ml/
  features.py                   # feature engineering: tehnice + sentiment + taker_buy_ratio +
                                 # btc_return_1 + tvl_momentum
  price_predictor.py            # PricePredictor: ansamblu sklearn (implicit) / prophet / lstm
  combined_predictor.py         # scor de încredere + rezumat narativ

scout/
  universe.py                   # ce active se scanează: underdogs + tier separat de mega-caps
  revolut_allowlist.py          # doar active tranzacționabile pe Revolut (listă editabilă)
  pipeline.py                   # analiză per-activ: candles + știri + model -> predicție + narativ
  scanner.py                    # orchestrează un scan complet, salvează rezultatele ranked
  narrative.py                  # text „de ce și când" per activ

api/
  routes.py                     # toate endpoint-urile (vezi tabelul de mai jos)
  services.py                   # orchestrează candles + sentiment + BTC + TVL + model -> predicție

frontend/
  web/
    index.html / style.css / app.js       # dashboard ETH
    scout.html / scout.css / scout.js     # listă Scout AI (filtre, pin, badge mega-cap)
    detail.html / detail.css / detail.js  # detaliu per-activ (grafic, predicție, narativ, live price)

tests/                          # pytest, complet offline (mongomock + requests-mock)
```

Codul e organizat astfel încât adăugarea unei alte criptomonede pe dashboard-ul
principal înseamnă doar schimbarea parametrilor `coin_id` / `symbol` — nimic
din pipeline nu e hardcodat pe Ethereum în afara valorilor implicite din
`config.py`. Scout AI scanează deja zeci de active diferite din start.

## Instalare

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completează cheile API pe care le ai (toate sunt opționale)
```

Aplicația funcționează din prima **fără nicio cheie API**: prețurile vin de la
Binance/CoinGecko (publice), TVL de la DeFiLlama (public), iar știrile din 10
feed-uri RSS. Cheile opționale (`CRYPTOPANIC_API_KEY`, `NEWSAPI_KEY`,
`ETHERSCAN_API_KEY`) îmbunătățesc acoperirea, dar nu sunt obligatorii —
`ETHERSCAN_API_KEY` lipsă doar dezactivează badge-ul de gas price.

MongoDB e opțional pentru a testa rapid API-ul: dacă nu e disponibil,
`database/mongo_client.py` continuă fără persistență (logează un warning) în
loc să crape aplicația. Pentru rulare reală ai nevoie de o instanță Mongo:

```bash
docker run -d --name eth-mongo -p 27017:27017 mongo:7
```

## Rulare

Două procese separate (terminale/containere diferite). Flask servește atât
API-ul cât și tot frontend-ul static (`frontend/web/`) pe același port:

```bash
cd /home/singeon/Documents/Alfa

# 1) Colectorul de date (seedează imediat, apoi rulează periodic + scanarea Scout AI)
.venv/bin/python run_scheduler.py

# 2) API-ul Flask + dashboard-ul + Scout AI, la http://localhost:5000
.venv/bin/python run_api.py
```

Predicția ETH are nevoie de minimum ~40 de candles istorice salvate pentru
intervalul cerut (implicit 1h) — lasă scheduler-ul să ruleze puțin, sau
rulează o singură colectare manuală:

```bash
python -c "from data_collector.jobs import run_all_once; run_all_once()"
```

Scout AI pornește automat un prim scan (pe un thread în fundal, ca să nu
blocheze pornirea scheduler-ului) — durează câteva minute, pentru că
antrenează un model real per activ pe ~135 de active.

## Dashboard ETH

`http://localhost:5000/` — grafic candlestick (Lightweight Charts) cu:
- predicție suprapusă + bandă de încredere, indicatori SMA/EMA/Bollinger/Fibonacci,
- tooltip de preț care urmărește mausul (citit direct din scala de preț la
  poziția Y a cursorului, deci coincide mereu exact cu axa — funcționează și
  peste zona de predicție, unde nu există un candle real),
- ticker live de preț (poll la 3s) + variație 24h,
- panou "Fixate" — activele memorate din Scout AI apar direct lângă prețul ETH,
- panou de semnal Cumpără/Vinde din RSI + MACD + trend + sentiment,
- predicții separate pentru orizont 24h și 7 zile.

## Scout AI

`http://localhost:5000/scout.html` — scanează:
- **Underdogs**: cripto rank 30-250 pe CoinGecko + monede „trending", acțiuni
  din screener-ele Yahoo Finance `small_cap_gainers` / `aggressive_small_caps` /
  `undervalued_growth_stocks`. Apar dacă predicția e pozitivă și încrederea
  modelului e ≥ 25/100.
- **Mega-cap (escape hatch)**: top 30 cripto după capitalizare + acțiuni mari
  (≥ $50B) din screener-ul `day_gainers`. Aceste active NU apar automat — au
  nevoie de o predicție de **minim +10% pe 7 zile** cu încredere ≥ 40/100 ca
  să treacă filtrul, marcate cu badge-ul **⭐ Mega-cap**. Scopul: dacă un nume
  mare chiar are un semnal dramatic, tot apare, fără să dilueze restul cu
  zgomot de la mișcări normale ale unor mega-caps.
- Ambele universuri sunt filtrate să conțină **doar active tranzacționabile pe
  Revolut** (`scout/revolut_allowlist.py` — o listă editabilă manual).
- **Pin/Fixare**: orice activ poate fi memorat (☆ → ★), salvat în MongoDB —
  rămâne accesibil chiar dacă iese din scanarea curentă, și apare atât în
  tab-ul „Fixate" din Scout AI, cât și lângă prețul ETH pe dashboard-ul
  principal.
- Click pe un rând → pagină de detaliu per-activ, cu grafic, predicție pe 7
  zile, narativ „de ce și când", știri analizate și preț live.

## API

| Endpoint | Descriere |
|---|---|
| `GET /api/health` | Health check |
| `GET /api/price/current` | Preț curent ETH, volum 24h, variație 24h |
| `GET /api/price/history?interval=15m\|1h\|4h\|1d\|1w&limit=` | Istoric OHLC |
| `GET /api/predict?interval=&steps=&use_sentiment=` | Predicție preț + interval de încredere + scor |
| `GET /api/outlook?use_sentiment=` | Predicție rapidă 24h + 7 zile |
| `GET /api/news?limit=` | Ultimele articole + sentiment |
| `GET /api/sentiment?days=` | Scor agregat de sentiment pe zi |
| `GET /api/summary?interval=` | Predicție + rezumat narativ + știri suport |
| `GET /api/gas` | Gas price on-chain (necesită `ETHERSCAN_API_KEY`) |
| `GET /api/scout?asset_type=crypto\|stock&limit=` | Rezultatele ultimei scanări Scout AI, ranked |
| `GET /api/scout/detail?asset_type=&id=` | Detaliu complet per-activ (grafic, predicție, narativ, știri) |
| `GET /api/scout/price?asset_type=&id=` | Preț live, ușor, pentru pagina de detaliu |
| `GET /api/scout/pins?asset_type=` | Lista activelor fixate |
| `POST /api/scout/pins` | Fixează un activ (`{asset_type, id, symbol, name}`) |
| `DELETE /api/scout/pins?asset_type=&id=` | Scoate un activ de la fixate |

## Modele de predicție

`PREDICTION_BACKEND` din `.env` alege backend-ul:
- `sklearn` (implicit) — ansamblu de `GradientBoostingRegressor` (3 seed-uri
  diferite, mediate) pe randamente logaritmice, cu interval de încredere real
  via regresie quantile.
- `prophet` — necesită `pip install prophet`.
- `lstm` — punct de extensie (necesită `pip install tensorflow`).

Feature-uri folosite la antrenare (`ml/features.py`):
- tehnice: SMA/EMA, RSI(14), volatilitate, MACD, poziție Bollinger, volum
  relativ, 6 lag-uri de preț,
- **taker buy ratio** — ce procent din volumul fiecărui candle a fost
  cumpărare agresivă (direct din breakdown-ul Binance) — proxy real pentru
  cerere vs. ofertă,
- **btc_return_1** — randamentul propriu al BTC pe aceeași grilă de timp —
  majoritatea altcoin-urilor (ETH inclus) se mișcă mai mult odată cu BTC decât
  pe fundamentale proprii,
- **tvl_momentum** — variația zilnică a TVL DeFi pe Ethereum (DeFiLlama) — un
  semnal real de „utilitate a rețelei", nu doar preț și volum,
- sentiment din știri, agregat zilnic.

Toate feature-urile care nu se pot calcula pentru un activ dat (ex.
taker-buy-ul nu există pentru acțiuni, TVL doar pentru ETH) trec la o valoare
neutră în loc să elimine rândul din antrenare.

## Performanță

Antrenarea unui model (ansamblu complet) durează ~6s — inacceptabil pe
request. Soluții aplicate:
- **Cache de predicții** (`api/services.py`) cu TTL 900s, aliniat cu cadența
  reală de colectare a datelor (15 min) — nu are rost să expire mai des decât
  se schimbă datele oricum.
- **Thread de warm-up** (`run_api.py`) — antrenează proactiv, în fundal, la
  pornire și la fiecare 5 min, cele 3 combinații pe care le folosește
  dashboard-ul (1h/1d/1w), ca un request real să găsească mereu cache-ul cald.
- Lock-ul din jurul antrenării acoperă tot ciclul citește-antrenează-scrie, nu
  doar accesul la dicționar — două request-uri concurente pentru aceeași
  cheie (ex. graficul + rezumatul zilnic la refresh) nu mai antrenează
  duplicat.
- Cache dedicat (3s) pentru pagina de detaliu Scout AI — un al doilea click pe
  același activ e instant în loc să reantreneze.

Rezultat măsurat: încărcarea completă a dashboard-ului a scăzut de la
8-15s (rece) la ~1.8s.

## Teste

```bash
pytest tests/ -q
```

Toate testele rulează offline: baza de date e `mongomock`, apelurile HTTP
către Binance/CoinGecko/DeFiLlama sunt interceptate cu `requests-mock` sau
stub-uite explicit (vezi `tests/conftest.py`).

## Extindere la alte criptomonede

Toate funcțiile din `data_collector`, `ml` și `api/services.py` primesc
`coin_id` (id CoinGecko) și, unde e cazul, `symbol` (simbol Binance) ca
parametri opționali cu valoare implicită din `config.py`. Pentru dashboard-ul
principal, e suficient să apelezi endpoint-urile cu `?coin_id=bitcoin` (după
ce extinzi și `BINANCE_SYMBOL`/`COIN_ID` per coin). Scout AI scanează deja
zeci de active simultan, independent de moneda configurată pe dashboard.
