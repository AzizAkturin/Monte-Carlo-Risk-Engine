#!/usr/bin/env python3
"""
Monte Carlo Risk Analysis for SOL, ETH, BTC, HYPE Portfolio
Using Binance API for historical data
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues

from src.data.binance import load_binance_price_data
from src.simulation.monte_carlo import MCConfig, simulate_correlated_returns, portfolio_path_pnl
from src.reporting.risk import compute_risk_metrics
from src.reporting.visualize import plot_risk_dashboard


def main():
    # Portfolio tokens
    tickers = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "USDCUSDT"]
    token_names = ["Solana", "Ethereum", "Bitcoin", "USDC (Stablecoin)"]
    
    print("🎲 Monte Carlo Risk Engine - Portfolio Analysis")
    print("="*60)
    print(f"📊 Target Portfolio: {', '.join(token_names)}\n")
    
    print("1️⃣  Fetching historical prices from Binance...")
    
    # Try to fetch all tokens, handle if HYPE isn't available
    try:
        data = load_binance_price_data(tickers, days=365)
        print(f"   ✓ Loaded {len(data.prices)} days of data")
        print(f"   ✓ Successfully fetched: {', '.join(data.prices.columns)}")
        print(f"   ✓ Date range: {data.prices.index[0].date()} to {data.prices.index[-1].date()}\n")
    except Exception as e:
        print(f"   ⚠️  Error loading all tokens: {e}")
        print("   Trying without HYPEUSDT (might not be listed on Binance)...\n")
        
        # Fallback: try without HYPE
        tickers = ["SOLUSDT", "ETHUSDT", "BTCUSDT"]
        token_names = ["Solana", "Ethereum", "Bitcoin"]
        
        try:
            data = load_binance_price_data(tickers, days=365)
            print(f"   ✓ Loaded {len(data.prices)} days of data")
            print(f"   ✓ Date range: {data.prices.index[0].date()} to {data.prices.index[-1].date()}\n")
        except Exception as e2:
            print(f"   ❌ Failed to load data: {e2}")
            return
    
    print("2️⃣  Computing returns and statistics...")
    rets = data.returns
    mu = rets.mean().to_numpy()
    cov = rets.cov().to_numpy()
    
    print(f"\n   📈 Daily Mean Returns:")
    for token, ticker, mean_ret in zip(token_names, tickers, mu):
        annual_ret = mean_ret * 252 * 100  # Annualized
        print(f"     {token:12s} ({ticker:10s}): {mean_ret:+.6f} → {annual_ret:+7.2f}% annualized")
    
    print(f"\n   📊 Correlation Matrix:")
    corr = rets.corr()
    corr.index = token_names
    corr.columns = token_names
    print(corr.to_string())
    
    # Volatility analysis
    print(f"\n   📉 Annualized Volatility:")
    vols = rets.std() * np.sqrt(252) * 100
    for token, vol in zip(token_names, vols):
        print(f"     {token:12s}: {vol:.2f}%")
    
    print("\n3️⃣  Setting up portfolio allocation...")
    # Equal weight portfolio
    n_assets = len(mu)
    weights = np.ones(n_assets) / n_assets
    print(f"   Strategy: Equal Weight")
    for token, weight in zip(token_names, weights):
        print(f"     {token:12s}: {weight:.2%}")
    
    print("\n4️⃣  Running Monte Carlo simulation...")
    initial_value = 10000.0
    config = MCConfig(horizon_days=30, n_paths=20000, seed=42)
    sim = simulate_correlated_returns(mu=mu, cov=cov, config=config)
    pnl = portfolio_path_pnl(sim, weights=weights, initial_value=initial_value)
    print(f"   ✓ Simulated {config.n_paths:,} paths over {config.horizon_days} days")
    print(f"   ✓ Initial portfolio value: ${initial_value:,.2f}\n")
    
    print("5️⃣  Computing risk metrics...")
    metrics = compute_risk_metrics(pnl)
    
    print("\n" + "="*60)
    print("PORTFOLIO RISK METRICS (30-Day Horizon)")
    print("="*60)
    print(f"\n💰 Expected Performance:")
    print(f"   Mean P&L:       ${pnl.mean():+.4f}  ({pnl.mean()*100:+.2f}%)")
    print(f"   Median P&L:     ${np.median(pnl):+.4f}")
    print(f"   Std Deviation:  ${pnl.std():.4f}")
    
    print(f"\n⚠️  Downside Risk:")
    print(f"   VaR 95%:        ${metrics.var_95:.4f}  (95% confident loss won't exceed this)")
    print(f"   CVaR 95%:       ${metrics.cvar_95:.4f}  (avg loss in worst 5% scenarios)")
    print(f"   VaR 99%:        ${metrics.var_99:.4f}  (99% confident loss won't exceed this)")
    print(f"   CVaR 99%:       ${metrics.cvar_99:.4f}  (avg loss in worst 1% scenarios)")
    print(f"   Probability of Loss: {(pnl < 0).mean():.2%}")
    
    print(f"\n📊 Extreme Scenarios:")
    print(f"   Best Case:      ${pnl.max():+.4f}  ({pnl.max()*100:+.2f}%)")
    print(f"   Worst Case:     ${pnl.min():+.4f}  ({pnl.min()*100:+.2f}%)")
    print(f"   5th Percentile: ${np.percentile(pnl, 5):+.4f}")
    print(f"   95th Percentile: ${np.percentile(pnl, 95):+.4f}")
    
    # Additional metrics
    print(f"\n📈 Portfolio Metrics:")
    port_ret = pnl.mean() / initial_value
    port_vol = pnl.std() / initial_value
    sharpe_approx = (port_ret / port_vol) * np.sqrt(252/30) if port_vol > 0 else 0
    print(f"   Expected 30-day Return: {port_ret*100:+.2f}%")
    print(f"   30-day Volatility:      {port_vol*100:.2f}%")
    print(f"   Sharpe Ratio (approx):  {sharpe_approx:.2f}")
    
    print("="*60)
    
    # Save results to file
    with open("reports/portfolio_analysis.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("PORTFOLIO RISK ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Tokens: {', '.join(token_names)}\n")
        f.write(f"Data points: {len(data.prices)} days\n")
        f.write(f"Simulation: {config.n_paths:,} paths, {config.horizon_days} days\n\n")
        f.write("RISK METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"VaR 95%:  ${metrics.var_95:.4f}\n")
        f.write(f"CVaR 95%: ${metrics.cvar_95:.4f}\n")
        f.write(f"VaR 99%:  ${metrics.var_99:.4f}\n")
        f.write(f"CVaR 99%: ${metrics.cvar_99:.4f}\n")
        f.write(f"Mean P&L: ${pnl.mean():+.4f}\n")
        f.write(f"Prob(Loss): {(pnl < 0).mean():.2%}\n")
    
    print("\n6️⃣  Generating risk dashboard visualization...")
    # Compute portfolio value paths for visualization
    port_log_ret = sim @ weights
    gross = np.exp(port_log_ret)
    portfolio_values = initial_value * np.cumprod(gross, axis=1)
    
    plot_risk_dashboard(
        simulation_df=portfolio_values,
        pnl=pnl,
        var_95=metrics.var_95,
        cvar_95=metrics.cvar_95,
        var_99=metrics.var_99,
        cvar_99=metrics.cvar_99,
        initial_value=initial_value,
        save_path="reports/portfolio_dashboard.png",
        show=False,  # Save only, don't try to display
    )
    
    print("\n✅ Analysis Complete!")
    print(f"   💡 Tokens analyzed: {', '.join(token_names)}")
    print(f"   🔑 Data source: Binance API")
    print(f"   📁 Results saved to: reports/portfolio_analysis.txt")
    print(f"   📊 Dashboard saved to: reports/portfolio_dashboard.png")


if __name__ == "__main__":
    main()
