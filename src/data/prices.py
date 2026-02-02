from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

@dataclass
class PriceData:
    prices: pd.DataFrame
    returns: pd.DataFrame

def fetch_prices_yfinance(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "2yr",
    interval: str = "1d",
) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")
    
    if start or end:
        data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    else:
        data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        if "Close" in data.columns:
            prices = data["Close"].to_frame()
        else:
            prices = data.copy()

    prices = prices.dropna(how="all")
    prices.columns = [c.upper() for c in prices.columns]
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()
    returns = np.log(prices/prices.shift(1))
    returns = returns.dropna()
    return returns

def load_price_data(
    tickers: List[str],
    period: str = "2y",
) -> PriceData:
    prices = fetch_prices_yfinance([t.upper() for t in tickers], period=period)
    rets = compute_log_returns(prices)
    prices = prices.loc[rets.index]
    return PriceData(prices=prices, returns=rets)
