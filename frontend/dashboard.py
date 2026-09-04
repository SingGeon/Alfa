"""Streamlit dashboard: run with `streamlit run frontend/dashboard.py`.

Talks to the Flask API (run_api.py) over HTTP - keeps the frontend fully
decoupled from the ML/DB internals, so it could just as easily be swapped
for the HTML+Chart.js alternative mentioned in the spec.
"""
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config

st.set_page_config(page_title="ETH Price Predictor", page_icon="📈", layout="wide")

API_BASE = config.API_BASE_URL


@st.cache_data(ttl=60)
def api_get(path: str, params: dict | None = None):
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def sentiment_badge(label: str) -> str:
    color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(label, "⚪")
    return f"{color} {label}"


st.title("📈 Ethereum Price Predictor")
st.caption("Date live (CoinGecko/Binance) + predicție ML + sentiment din știri")

with st.sidebar:
    st.header("Setări")
    interval = st.selectbox("Interval istoric", ["1h", "1d", "1w"], index=0)
    steps = st.slider("Orizont predicție (nr. de pași)", min_value=1, max_value=72, value=24)
    use_sentiment = st.checkbox("Include sentiment din știri în model", value=True)
    st.divider()
    st.caption(f"API: {API_BASE}")
    if st.button("🔄 Reîmprospătează"):
        st.cache_data.clear()

col_price, col_conf = st.columns([3, 1])

try:
    current = api_get("/api/price/current")
    with col_price:
        change = current.get("change_24h_pct")
        st.metric(
            label=f"ETH/USD ({current.get('source', '?')})",
            value=f"${current['price']:,.2f}",
            delta=f"{change:.2f}% (24h)" if change is not None else None,
        )
except Exception as exc:
    st.error(f"Nu pot obține prețul curent: {exc}")

try:
    history = api_get("/api/price/history", {"interval": interval, "limit": 300})
    candles = pd.DataFrame(history["candles"])
except Exception as exc:
    st.error(f"Nu pot obține istoricul de prețuri: {exc}")
    candles = pd.DataFrame()

prediction = None
try:
    prediction = api_get(
        "/api/predict",
        {"interval": interval, "steps": steps, "use_sentiment": str(use_sentiment).lower()},
    )
except requests.HTTPError as exc:
    if exc.response is not None and exc.response.status_code == 409:
        st.warning(exc.response.json().get("error", "Date istorice insuficiente pentru predicție."))
    else:
        st.error(f"Predicția a eșuat: {exc}")
except Exception as exc:
    st.error(f"Predicția a eșuat: {exc}")

with col_conf:
    if prediction:
        st.metric("Scor de încredere", f"{prediction['confidence']:.0f}/100")
        st.caption(f"Backend: {prediction['backend']} · sentiment mediu: {prediction['sentiment_avg']:.2f}")

# --- Price + prediction chart -------------------------------------------------

st.subheader("Preț real vs. predicție")

fig = go.Figure()
if not candles.empty:
    candles["timestamp"] = pd.to_datetime(candles["timestamp"])
    fig.add_trace(
        go.Candlestick(
            x=candles["timestamp"],
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"],
            name="Preț real",
        )
    )

if prediction:
    pred_df = pd.DataFrame(prediction["predictions"])
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    fig.add_trace(
        go.Scatter(x=pred_df["timestamp"], y=pred_df["predicted_price"], mode="lines+markers", name="Predicție",
                    line=dict(color="orange"))
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([pred_df["timestamp"], pred_df["timestamp"][::-1]]),
            y=pd.concat([pred_df["upper"], pred_df["lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Interval de încredere",
            showlegend=True,
        )
    )

fig.update_layout(xaxis_title="Timp", yaxis_title="Preț (USD)", height=500, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# --- Narrative summary ---------------------------------------------------------

st.subheader("Rezumat zilnic")
try:
    summary = api_get("/api/summary", {"interval": interval})
    st.info(summary["narrative"])
except requests.HTTPError as exc:
    if exc.response is not None and exc.response.status_code == 409:
        pass  # already warned above about insufficient data
    else:
        st.error(f"Rezumatul a eșuat: {exc}")
except Exception as exc:
    st.error(f"Rezumatul a eșuat: {exc}")

# --- News + sentiment ------------------------------------------------------------

st.subheader("Ultimele știri + sentiment")
try:
    news = api_get("/api/news", {"limit": 15})["news"]
    if not news:
        st.caption("Niciun articol colectat încă. Rulează `python run_scheduler.py` pentru a popula baza de date.")
    for article in news:
        sentiment = article.get("sentiment", {})
        published = article.get("published_at", "")
        st.markdown(
            f"**[{article['title']}]({article['url']})**  \n"
            f"{sentiment_badge(sentiment.get('label', 'neutral'))} "
            f"(scor {sentiment.get('compound', 0):.2f}) · {article.get('source', '')} · {published[:16]}"
        )
        st.divider()
except Exception as exc:
    st.error(f"Nu pot obține știrile: {exc}")

# --- Gas price (bonus, only if ETHERSCAN_API_KEY is configured) ------------------

try:
    gas = api_get("/api/gas")
    if "error" not in gas:
        st.sidebar.divider()
        st.sidebar.subheader("⛽ Gas price (Etherscan)")
        st.sidebar.write(f"Safe: {gas['safe_gwei']} gwei · Propose: {gas['propose_gwei']} gwei · Fast: {gas['fast_gwei']} gwei")
except Exception:
    pass
