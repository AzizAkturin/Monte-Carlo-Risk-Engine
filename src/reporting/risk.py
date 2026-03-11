from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float


def var_cvar(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    """
    pnl: profit/loss distribution (positive = profit, negative = loss)
    VaR is reported as a positive number representing loss threshold.
    """
    losses = -pnl  # convert to losses
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = float(np.mean(tail)) if len(tail) else float(var)
    return float(var), float(cvar)


def compute_risk_metrics(pnl: np.ndarray) -> RiskMetrics:
    var95, cvar95 = var_cvar(pnl, 0.95)
    var99, cvar99 = var_cvar(pnl, 0.99)
    return RiskMetrics(var_95=var95, var_99=var99, cvar_95=cvar95, cvar_99=cvar99)


def compute_rolling_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    window: int = 60,
    horizon_days: int = 30,
    initial_value: float = 1.0,
) -> pd.DataFrame:
    """
    Parametric (Gaussian) rolling VaR scaled to horizon_days, in dollar terms.

    At each date we look back `window` days, compute the rolling mean and
    standard deviation of the portfolio's daily log-return, scale them to
    the simulation horizon, and apply the Gaussian quantile.

    Returns a DataFrame with columns var_95 and var_99 (positive = loss).
    """
    Z_95 = 1.6449
    Z_99 = 2.3263

    port_rets = returns @ weights          # daily portfolio log-return series

    roll_mean  = port_rets.rolling(window).mean()
    roll_std   = port_rets.rolling(window).std()

    mu_h    = roll_mean * horizon_days          # expected P&L over horizon (log scale)
    sigma_h = roll_std  * np.sqrt(horizon_days) # scaled volatility

    var_95 = (-mu_h + Z_95 * sigma_h) * initial_value
    var_99 = (-mu_h + Z_99 * sigma_h) * initial_value

    return pd.DataFrame(
        {"var_95": var_95, "var_99": var_99},
        index=returns.index,
    ).dropna()
