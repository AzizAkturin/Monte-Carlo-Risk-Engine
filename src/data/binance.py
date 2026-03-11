from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

# Market-data-only host recommended by Binance for public endpoints
BASE_URL = "https://data-api.binance.vision"  # fallback could be https://api.binance.com


@dataclass
class PriceData:
    prices: pd.DataFrame
    returns: pd.DataFrame


def _request_with_backoff(url: str, params: dict, max_retries: int = 6) -> list:
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code in (429, 418):
            # Binance uses Retry-After for 429/418; fall back to exponential wait if missing
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Binance request failed after retries: {url} {params}")


def fetch_klines(
    symbol: str,
    interval: str = "1d",
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    Fetch klines (candles) from Binance Spot public endpoint.
    Returns DataFrame with open_time (ms) and close.
    """
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = int(start_ms)
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    data = _request_with_backoff(url, params)

    # Kline format:
    # [
    #  0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume,
    #  6 close time, 7 quote asset volume, 8 number of trades, ...
    # ]
    df = pd.DataFrame(
        data,
        columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","n_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = df["close"].astype(float)
    return df[["open_time", "close"]].sort_values("open_time")


def fetch_close_series_last_n_days(symbol: str, days: int = 730) -> pd.Series:
    """
    Pull daily closes for the last N days.
    Binance /klines can return max 1000 candles; daily candles for ~2 years fits.
    """
    # 1d candles, request up to 1000 (enough for ~2.7 years)
    df = fetch_klines(symbol, interval="1d", limit=min(1000, max(10, days)))
    # Keep last N rows
    df = df.tail(days)
    s = pd.Series(df["close"].values, index=df["open_time"], name=symbol.upper())
    return s


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    rets = np.log(prices / prices.shift(1))
    return rets.dropna()


def compute_ewma_params(
    returns: pd.DataFrame,
    span: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate mu and cov using exponentially weighted moments.

    Recent observations get higher weight, so the estimates naturally
    reflect the current market regime (bull / bear) without needing
    explicit regime labels.

    Args:
        returns: Daily log-return DataFrame (rows = days, cols = assets).
        span:    EWMA half-life in days. ~20 = very reactive, ~60 = 3-month
                 memory, ~120 = subdued regime sensitivity.

    Returns:
        mu:  1-D array of EWMA daily mean log-returns (one per asset).
        cov: 2-D EWMA covariance matrix.
    """
    ewm = returns.ewm(span=span, adjust=True)
    mu = ewm.mean().iloc[-1].to_numpy()

    ewm_cov_full = ewm.cov()
    last_date = ewm_cov_full.index.get_level_values(0)[-1]
    cov = ewm_cov_full.loc[last_date].to_numpy()

    return mu, cov


def load_binance_price_data(
    symbols: List[str],
    days: int = 730,
) -> PriceData:
    series = []
    for sym in symbols:
        series.append(fetch_close_series_last_n_days(sym, days=days))

    prices = pd.concat(series, axis=1).dropna(how="all")
    returns = compute_log_returns(prices)
    prices = prices.loc[returns.index]
    return PriceData(prices=prices, returns=returns)
