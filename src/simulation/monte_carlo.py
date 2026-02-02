from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

@dataclass
class MCConfig:
    horizon_days: int = 20
    n_paths: int = 20000
    seed: int = 42

def simulate_correlated_returns(
    mu: np.ndarray,
    cov: np.ndarray,
    config: MCConfig,
) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    n_assets = len(mu)
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal(size = (config.n_paths, config.horizon_days, n_assets))
    correlated = z @ L.T
    simulated = correlated + mu
    return simulated

def portfolio_path_pnl(
    simulated_returns: np.ndarray,
    weights: np.ndarray,
    initial_value: float = 1.0,
) -> np.ndarray:
    port_log_ret = simulated_returns @ weights
    gross = np.exp(port_log_ret)
    values = initial_value * np.cumprod(gross, axis=1)
    terminal = values[:, -1]
    pnl = terminal - initial_value
    return pnl
    