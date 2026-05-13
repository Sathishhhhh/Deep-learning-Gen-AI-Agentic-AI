import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8014")
PERIOD_GRANULARITY = {
    "1D": 300,
    "7D": 3600,
    "30D": 3600,
    "90D": 21600,
    "1Y": 86400,
}


st.set_page_config(page_title="BTC-USD Spot Analysis", page_icon="₿", layout="wide")


def get_json(path: str, params: dict | None = None):
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=60)
def load_market_data(period: str, granularity: int):
    params = {"period": period, "granularity": granularity}
    candles = get_json("/candles", params=params)
    analysis = get_json("/analysis", params=params)
    frame = pd.DataFrame(candles)
    if not frame.empty:
        frame["time"] = pd.to_datetime(frame["time"])
    return frame, analysis


def money(value):
    if value is None:
        return "N/A"
    return f"${value:,.2f}"


def make_price_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC-USD",
            increasing_line_color="#0f9f6e",
            decreasing_line_color="#d64545",
        )
    )
    if "sma_20" in df:
        fig.add_trace(go.Scatter(x=df["time"], y=df["sma_20"], name="SMA 20", line=dict(color="#2563eb", width=2)))
    if "sma_50" in df:
        fig.add_trace(go.Scatter(x=df["time"], y=df["sma_50"], name="SMA 50", line=dict(color="#f59e0b", width=2)))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_rsi_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["rsi_14"], name="RSI 14", line=dict(color="#7c3aed", width=2)))
    fig.add_hline(y=70, line_dash="dash", line_color="#d64545")
    fig.add_hline(y=30, line_dash="dash", line_color="#0f9f6e")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=25, b=10), yaxis=dict(range=[0, 100]))
    return fig


st.title("BTC-USD Spot Market Analysis")
st.caption("Live public Coinbase spot data with technical indicators and business-ready market signals.")

with st.sidebar:
    st.header("Controls")
    period = st.selectbox("Analysis period", list(PERIOD_GRANULARITY), index=2)
    granularity = PERIOD_GRANULARITY[period]
    st.caption(f"API: {API_BASE_URL}")
    refresh = st.button("Refresh data", use_container_width=True)
    st.divider()
    st.info("This dashboard is for learning and market analysis. It is not financial advice.")

if refresh:
    st.cache_data.clear()

try:
    df, summary = load_market_data(period, granularity)
except requests.exceptions.ConnectionError:
    st.error(f"Cannot connect to API at {API_BASE_URL}. Start the backend with `uvicorn crypto_backend:app --reload --port 8014`.")
    st.stop()
except requests.HTTPError as exc:
    st.error(f"API error: {exc.response.text}")
    st.stop()
except Exception as exc:
    st.error(f"Unable to load market data: {exc}")
    st.stop()

if df.empty:
    st.warning("No market data returned for the selected period.")
    st.stop()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Latest BTC-USD", money(summary["latest_price"]), f'{summary["price_change_pct"]}%')
metric_2.metric("Trend", summary["trend"], summary["signal"])
metric_3.metric("RSI 14", summary["rsi_14"] if summary["rsi_14"] is not None else "N/A", summary["momentum"])
metric_4.metric("Volatility", f'{summary["annualized_volatility_pct"]}%' if summary["annualized_volatility_pct"] else "N/A")

st.plotly_chart(make_price_chart(df), use_container_width=True)

left, right = st.columns([2, 1])
with left:
    st.subheader("Momentum")
    st.plotly_chart(make_rsi_chart(df), use_container_width=True)

with right:
    st.subheader("Market Levels")
    st.metric("Support", money(summary["support"]))
    st.metric("Resistance", money(summary["resistance"]))
    st.metric("Period High", money(summary["period_high"]))
    st.metric("Period Low", money(summary["period_low"]))

st.subheader("Business Analysis")
insight_cols = st.columns(3)
insight_cols[0].write(f"**Market Signal:** {summary['signal']}")
insight_cols[0].write(f"**Trend State:** {summary['trend']}")
insight_cols[1].write(f"**Period Volume:** {summary['period_volume_btc']:,.2f} BTC")
insight_cols[1].write(f"**Period Open:** {money(summary['period_open'])}")
insight_cols[2].write(f"**SMA 20:** {money(summary['sma_20'])}")
insight_cols[2].write(f"**SMA 50:** {money(summary['sma_50'])}")

with st.expander("Raw candle data"):
    st.dataframe(df.sort_values("time", ascending=False), use_container_width=True, hide_index=True)
