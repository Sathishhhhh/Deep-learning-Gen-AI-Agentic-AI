from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


PRODUCT_ID = "BTC-USD"
COINBASE_CANDLES_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"
COINBASE_TICKER_URL = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/ticker"
HEADERS = {"User-Agent": "project-4-btc-usd-spot-analysis"}

SUPPORTED_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}

app = FastAPI(
    title="BTC-USD Spot Analysis API",
    description="Public Coinbase market data API for BTC-USD spot analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _fetch_json(url: str, params: dict | None = None):
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Market data provider error: {exc}") from exc


def _period_start(period: str) -> datetime:
    now = datetime.now(timezone.utc)
    periods = {
        "1D": timedelta(days=1),
        "7D": timedelta(days=7),
        "30D": timedelta(days=30),
        "90D": timedelta(days=90),
        "1Y": timedelta(days=365),
    }
    return now - periods.get(period.upper(), timedelta(days=30))


def validate_granularity(granularity: int) -> int:
    if granularity not in SUPPORTED_GRANULARITIES:
        allowed = ", ".join(str(value) for value in sorted(SUPPORTED_GRANULARITIES))
        raise HTTPException(status_code=422, detail=f"granularity must be one of: {allowed}")
    return granularity


def fetch_candles(period: str = "30D", granularity: int = 3600) -> pd.DataFrame:
    granularity = validate_granularity(granularity)
    start = _period_start(period)
    end = datetime.now(timezone.utc)
    max_window = timedelta(seconds=granularity * 300)
    rows = []

    window_start = start
    while window_start < end:
        window_end = min(window_start + max_window, end)
        params = {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "granularity": granularity,
        }
        rows.extend(_fetch_json(COINBASE_CANDLES_URL, params=params))
        window_start = window_end

    if not rows:
        raise HTTPException(status_code=404, detail="No candles returned for the selected period.")

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def serialize_candles(df: pd.DataFrame) -> list[dict]:
    clean = df.copy()
    clean["time"] = clean["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    clean = clean.astype(object).where(pd.notna(clean), None)
    return clean.to_dict(orient="records")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["sma_20"] = result["close"].rolling(20).mean()
    result["sma_50"] = result["close"].rolling(50).mean()
    result["ema_12"] = result["close"].ewm(span=12, adjust=False).mean()
    result["ema_26"] = result["close"].ewm(span=26, adjust=False).mean()
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["returns"] = result["close"].pct_change()

    delta = result["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    result["rsi_14"] = 100 - (100 / (1 + rs))
    return result


def build_analysis(df: pd.DataFrame) -> dict:
    enriched = add_indicators(df)
    latest = enriched.iloc[-1]
    first = enriched.iloc[0]

    price_change = latest["close"] - first["close"]
    price_change_pct = (price_change / first["close"]) * 100
    volatility = enriched["returns"].std() * (365 ** 0.5) * 100
    support = enriched["low"].tail(30).min()
    resistance = enriched["high"].tail(30).max()
    rsi = latest.get("rsi_14")

    trend = "Neutral"
    if pd.notna(latest.get("sma_20")) and pd.notna(latest.get("sma_50")):
        if latest["close"] > latest["sma_20"] > latest["sma_50"]:
            trend = "Bullish"
        elif latest["close"] < latest["sma_20"] < latest["sma_50"]:
            trend = "Bearish"

    momentum = "Neutral"
    if pd.notna(rsi):
        if rsi >= 70:
            momentum = "Overbought"
        elif rsi <= 30:
            momentum = "Oversold"
        elif rsi >= 55:
            momentum = "Positive"
        elif rsi <= 45:
            momentum = "Negative"

    recommendation = "Hold / observe"
    if trend == "Bullish" and momentum in {"Positive", "Neutral"}:
        recommendation = "Bullish bias"
    elif trend == "Bearish" and momentum in {"Negative", "Neutral"}:
        recommendation = "Risk-off bias"
    elif momentum == "Oversold":
        recommendation = "Watch for rebound"
    elif momentum == "Overbought":
        recommendation = "Watch for pullback"

    return {
        "product_id": PRODUCT_ID,
        "latest_time": latest["time"].isoformat(),
        "latest_price": round(float(latest["close"]), 2),
        "period_open": round(float(first["open"]), 2),
        "period_high": round(float(enriched["high"].max()), 2),
        "period_low": round(float(enriched["low"].min()), 2),
        "period_volume_btc": round(float(enriched["volume"].sum()), 4),
        "price_change": round(float(price_change), 2),
        "price_change_pct": round(float(price_change_pct), 2),
        "annualized_volatility_pct": None if pd.isna(volatility) else round(float(volatility), 2),
        "support": round(float(support), 2),
        "resistance": round(float(resistance), 2),
        "sma_20": None if pd.isna(latest.get("sma_20")) else round(float(latest["sma_20"]), 2),
        "sma_50": None if pd.isna(latest.get("sma_50")) else round(float(latest["sma_50"]), 2),
        "rsi_14": None if pd.isna(rsi) else round(float(rsi), 2),
        "trend": trend,
        "momentum": momentum,
        "signal": recommendation,
        "disclaimer": "Educational market analysis only. Not financial advice.",
    }


@app.get("/")
def root():
    return {"message": "BTC-USD Spot Analysis API is running", "product_id": PRODUCT_ID}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/ticker")
def ticker():
    data = _fetch_json(COINBASE_TICKER_URL)
    return {
        "product_id": PRODUCT_ID,
        "price": float(data["price"]),
        "bid": float(data["bid"]),
        "ask": float(data["ask"]),
        "volume": float(data["volume"]),
        "time": data["time"],
    }


@app.get("/candles")
def candles(
    period: str = Query("30D", pattern="^(1D|7D|30D|90D|1Y)$"),
    granularity: int = Query(3600),
):
    df = add_indicators(fetch_candles(period, granularity))
    return serialize_candles(df)


@app.get("/analysis")
def analysis(
    period: str = Query("30D", pattern="^(1D|7D|30D|90D|1Y)$"),
    granularity: int = Query(3600),
):
    return build_analysis(fetch_candles(period, granularity))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8014)
