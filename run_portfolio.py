#!/usr/bin/env python3
"""
Custom portfolio Monte Carlo simulation for NVDA, AAPL, META, ETH
"""

from __future__ import annotations

import numpy as np
from src.data.prices import load_price_data
from src.simulation.monte_carlo import MCConfig, simulate_correlated_returns, portfolio_path_pnl
from src.reporting.risk import compute_risk_metrics
from src.reporting.visualize import plot_risk_dashboard


def main():
    # Your custom portfolio: NVIDIA, Apple, Meta, Ethereum
    tickers = ["NVDA", "AAPL", "META", "ETH-USD"]
    
    print("🎲 Monte Carlo Risk Engine - Custom Portfolio")
    print(f"📊 Assets: {', '.join(tickers)}\n")
    
    print("1️⃣  Fetching historical prices (last 2 years)...")
    data = load_price_data(tickers, period="2y")
    print(f"   ✓ Loaded {len(data.prices)} days of data\n")
    
    print("2️⃣  Computing returns and statistics...")
    rets = data.returns
    mu = rets.mean().to_numpy()
    cov = rets.cov().to_numpy()
    
    print(f"   Daily Mean Returns:")
    for ticker, mean_ret in zip(tickers, mu):
        print(f"     {ticker:8s}: {mean_ret:+.6f} ({mean_ret*252*100:.2f}% annualized)")
    
    print(f"\n   Correlation Matrix:")
    corr = rets.corr()
    print(corr.to_string())
    
    print("\n3️⃣  Running Monte Carlo simulation...")
    # Equal weight portfolio (25% each)
    weights = np.ones(len(mu)) / len(mu)
    print(f"   Portfolio weights: {dict(zip(tickers, weights))}")
    
    config = MCConfig(horizon_days=20, n_paths=20000, seed=42)
    sim = simulate_correlated_returns(mu=mu, cov=cov, config=config)
    pnl = portfolio_path_pnl(sim, weights=weights, initial_value=1.0)
    print(f"   ✓ Simulated {config.n_paths:,} paths over {config.horizon_days} days\n")
    
    print("4️⃣  Computing risk metrics...")
    metrics = compute_risk_metrics(pnl)
    
    print("\n" + "="*60)
    print("RISK METRICS")
    print("="*60)
    print(f"VaR 95%:      {metrics.var_95:.4f}  (95% chance loss won't exceed this)")
    print(f"CVaR 95%:     {metrics.cvar_95:.4f}  (Average loss in worst 5% scenarios)")
    print(f"VaR 99%:      {metrics.var_99:.4f}  (99% chance loss won't exceed this)")
    print(f"CVaR 99%:     {metrics.cvar_99:.4f}  (Average loss in worst 1% scenarios)")
    print(f"\nMean P&L:     {pnl.mean():+.4f}")
    print(f"Std Dev P&L:  {pnl.std():.4f}")
    print(f"Prob(Loss):   {(pnl < 0).mean():.2%}")
    print(f"Best Case:    {pnl.max():+.4f}")
    print(f"Worst Case:   {pnl.min():+.4f}")
    print("="*60)
    
    print("\n5️⃣  Generating risk dashboard...")
    # Compute portfolio value paths for visualization
    port_log_ret = sim @ weights
    gross = np.exp(port_log_ret)
    portfolio_values = 1.0 * np.cumprod(gross, axis=1)
    
    plot_risk_dashboard(
        simulation_df=portfolio_values,
        pnl=pnl,
        var_95=metrics.var_95,
        cvar_95=metrics.cvar_95,
        var_99=metrics.var_99,
        cvar_99=metrics.cvar_99,
        initial_value=1.0,
        save_path="reports/portfolio_dashboard.png",
        show=True,
    )
    
    print("\n✅ Complete! Dashboard saved to reports/portfolio_dashboard.png")


if __name__ == "__main__":
    main()
