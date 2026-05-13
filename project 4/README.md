# Project 4: BTC-USD Spot Analysis

A business analytics project for Bitcoin spot market analysis using:

- FastAPI backend
- Streamlit dashboard
- Coinbase public BTC-USD spot candles
- Technical indicators: SMA, EMA, MACD, RSI, volatility, support, resistance

## Setup

```bash
cd "project 4"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
uvicorn crypto_backend:app --reload --port 8014
```

Open API docs:

```text
http://localhost:8014/docs
```

## Run Streamlit

Open a second terminal:

```bash
cd "project 4"
source .venv/bin/activate
streamlit run app.py
```

## API Endpoints

- `GET /health`
- `GET /ticker`
- `GET /candles?period=30D&granularity=3600`
- `GET /analysis?period=30D&granularity=3600`

Supported periods: `1D`, `7D`, `30D`, `90D`, `1Y`.

## Notes

This project uses public market data and does not require an API key. The analysis is educational and is not financial advice.
