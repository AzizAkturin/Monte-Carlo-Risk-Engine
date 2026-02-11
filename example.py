#!/usr/bin/env python3
"""
Quick example showing all 5 core components of the Monte Carlo Risk Engine:

1. Get historical close prices
2. Compute returns (log returns)
3. Estimate drift + volatility (with correlation)
4. Simulate paths (GBM-style with correlation)
5. Visualize + compute VaR/CVaR
"""

from __future__ import annotations

import numpy as np
from src.data.binance import load_binance_price_data
from src.simulation.monte_carlo import MCConfig, simulate_correlated_returns, portfolio_path_pnl
from src.reporting.risk import compute_risk_metrics
from src.reporting.visualize import plot_risk_dashboard


def main():
    print("🚀 Monte Carlo Risk Engine - Quick Example\n")
    
    # Component 1: Get historical close prices
    print("1️⃣  Fetching historical prices from Binance...")
    tickers = ["BTCUSDT", "ETHUSDT"]
    data = load_binance_price_data(tickers, days=730)
    print(f"   ✓ Loaded {len(data.prices)} days of data for {', '.join(tickers)}\n")
    
    # Component 2: Compute returns (log returns)
    print("2️⃣  Computing log returns...")
    rets = data.returns
    print(f"   ✓ Computed returns shape: {rets.shape}\n")
    
    # Component 3: Estimate drift + volatility (with correlation)
    print("3️⃣  Estimating drift (μ) and covariance (Σ)...")
    mu = rets.mean().to_numpy()
    cov = rets.cov().to_numpy()
    print(f"   ✓ Daily mean returns: {mu}")
    print(f"   ✓ Correlation matrix:")
    corr = rets.corr().to_numpy()
    print(f"     {corr}\n")
    
    # Component 4: Simulate paths (GBM-style with correlation)
    print("4️⃣  Simulating correlated price paths...")
    n_assets = len(mu)
    weights = np.ones(n_assets) / n_assets  # Equal weights
    
    config = MCConfig(horizon_days=20, n_paths=10000, seed=42)
    sim = simulate_correlated_returns(mu=mu, cov=cov, config=config)
    pnl = portfolio_path_pnl(sim, weights=weights, initial_value=1.0)
    print(f"   ✓ Simulated {config.n_paths:,} paths over {config.horizon_days} days\n")
    
    # Component 5: Visualize + compute VaR/CVaR
    print("5️⃣  Computing risk metrics and visualizing...")
    metrics = compute_risk_metrics(pnl)
    
    print(f"\n📊 Risk Metrics:")
    print(f"   VaR 95%:  {metrics.var_95:.4f}")
    print(f"   CVaR 95%: {metrics.cvar_95:.4f}")
    print(f"   VaR 99%:  {metrics.var_99:.4f}")
    print(f"   CVaR 99%: {metrics.cvar_99:.4f}")
    
    # Generate visualization
    port_log_ret = sim @ weights
    gross = np.exp(port_log_ret)
    portfolio_values = 1.0 * np.cumprod(gross, axis=1)
    
    print("\n📈 Generating risk dashboard...")
    plot_risk_dashboard(
        simulation_df=portfolio_values,
        pnl=pnl,
        var_95=metrics.var_95,
        cvar_95=metrics.cvar_95,
        var_99=metrics.var_99,
        cvar_99=metrics.cvar_99,
        initial_value=1.0,
        save_path="reports/example_dashboard.png",
        show=True,
    )
    
    print("\n✅ Done! All 5 core components executed successfully.")


if __name__ == "__main__":
    main()
