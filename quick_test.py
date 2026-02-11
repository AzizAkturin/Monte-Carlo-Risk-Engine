#!/usr/bin/env python3
"""Ultra-minimal test of the Monte Carlo engine"""

print("Starting quick test...")
print("This will fetch real data from Binance and run a simulation.\n")

import numpy as np
print("✓ NumPy loaded")

import pandas as pd
print("✓ Pandas loaded")

from src.data.binance import load_binance_price_data
from src.simulation.monte_carlo import MCConfig, simulate_correlated_returns, portfolio_path_pnl
from src.reporting.risk import compute_risk_metrics

print("\n1. Fetching data from Binance (BTC + ETH, last 365 days)...")
data = load_binance_price_data(["BTCUSDT", "ETHUSDT"], days=365)
print(f"   ✓ Got {len(data.prices)} days of data")

print("\n2. Computing statistics...")
mu = data.returns.mean().to_numpy()
cov = data.returns.cov().to_numpy()
print(f"   ✓ Daily mean returns: {mu}")
print(f"   ✓ Covariance shape: {cov.shape}")

print("\n3. Running Monte Carlo simulation (5000 paths, 10 days)...")
config = MCConfig(horizon_days=10, n_paths=5000, seed=42)
sim = simulate_correlated_returns(mu=mu, cov=cov, config=config)
weights = np.array([0.5, 0.5])  # 50/50 allocation
pnl = portfolio_path_pnl(sim, weights=weights, initial_value=1.0)
print(f"   ✓ Simulated {config.n_paths} paths")

print("\n4. Computing risk metrics...")
metrics = compute_risk_metrics(pnl)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"VaR 95%:  {metrics.var_95:.4f}")
print(f"CVaR 95%: {metrics.cvar_95:.4f}")
print(f"VaR 99%:  {metrics.var_99:.4f}")
print(f"CVaR 99%: {metrics.cvar_99:.4f}")
print(f"Mean P&L: {pnl.mean():.4f}")
print(f"Std Dev:  {pnl.std():.4f}")
print(f"Prob(Loss): {(pnl < 0).mean():.2%}")
print("="*50)
print("\n✅ Test completed successfully!")
