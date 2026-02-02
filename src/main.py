from __future__ import annotations

import argparse
import numpy as np
from src.data.binance import load_binance_price_data
from src.simulation.monte_carlo import MCConfig, simulate_correlated_returns, portfolio_path_pnl
from src.reporting.risk import compute_risk_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Monte Carlo Risk Engine (MVP)")
    p.add_argument("--tickers", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Binance symbols like BTCUSDT ETHUSDT")
    p.add_argument("--days_back", type=int, default=730, help="How many daily candles to use for fitting")
    p.add_argument("--period", default="2y", help="yfinance period, e.g. 6mo, 1y, 2y")
    p.add_argument("--days", type=int, default=20, help="Simulation horizon in days")
    p.add_argument("--paths", type=int, default=20000, help="Number of Monte Carlo paths")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--initial", type=float, default=1.0, help="Initial portfolio value")
    return p.parse_args()


def main():
    args = parse_args()

    data = load_binance_price_data(args.tickers, days=args.days_back)
    rets = data.returns

    mu = rets.mean().to_numpy()            # daily mean log return
    cov = rets.cov().to_numpy()            # daily covariance of log returns

    # equal weights
    n_assets = rets.shape[1]
    w = np.ones(n_assets) / n_assets

    cfg = MCConfig(horizon_days=args.days, n_paths=args.paths, seed=args.seed)
    sim = simulate_correlated_returns(mu=mu, cov=cov, config=cfg)
    pnl = portfolio_path_pnl(sim, weights=w, initial_value=args.initial)

    metrics = compute_risk_metrics(pnl)

    print("\n=== Monte Carlo Risk Engine (MVP) ===")
    print(f"Tickers: {', '.join([c for c in rets.columns])}")
    print(f"Paths: {args.paths:,} | Horizon: {args.days} days")
    print(f"VaR 95%:  {metrics.var_95:.4f}")
    print(f"CVaR 95%: {metrics.cvar_95:.4f}")
    print(f"VaR 99%:  {metrics.var_99:.4f}")
    print(f"CVaR 99%: {metrics.cvar_99:.4f}")
    print("====================================\n")


if __name__ == "__main__":
    main()
