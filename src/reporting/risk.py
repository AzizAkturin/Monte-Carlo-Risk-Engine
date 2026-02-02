from __future__ import annotations

from dataclasses import dataclass
import numpy as np


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
