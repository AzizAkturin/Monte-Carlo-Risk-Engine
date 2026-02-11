# 🎲 Monte Carlo Risk Engine

A Python-based Monte Carlo simulation tool for portfolio risk analysis. Simulates correlated multi-asset price paths and generates comprehensive risk reports (VaR/CVaR, drawdown analysis, probability of loss).

## 🎯 Core Components

✅ **1. Historical Price Data** - Fetch daily close prices from Binance  
✅ **2. Log Returns** - Compute logarithmic returns for statistical analysis  
✅ **3. Drift & Volatility** - Estimate daily mean returns (μ) and covariance matrix (Σ)  
✅ **4. Path Simulation** - Generate correlated GBM-style price paths using Cholesky decomposition  
✅ **5. Risk Metrics & Visualization** - Calculate VaR/CVaR and create interactive risk dashboards  

## 🚀 Quick Start

### Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with defaults (BTC & ETH, 20k paths, 20-day horizon)
python -m src.main

# Run with visualization
python -m src.main --plot

# Full risk dashboard
python -m src.main --dashboard

# Custom parameters
python -m src.main --tickers BTCUSDT ETHUSDT SOLUSDT --paths 50000 --days 30 --dashboard

# Save plots to disk
python -m src.main --dashboard --save ./reports
```

### Run the Example

```bash
python example.py
```

## 📊 Features

### Current Implementation (MVP)
- ✅ Multi-asset correlation modeling using Cholesky decomposition
- ✅ Monte Carlo simulation with configurable paths and horizons
- ✅ Value at Risk (VaR) at 95% and 99% confidence levels
- ✅ Conditional VaR / Expected Shortfall (CVaR)
- ✅ Interactive visualizations (fan charts, P&L distributions, risk dashboards)
- ✅ Real-time data from Binance API with rate limiting
- ✅ Drawdown analysis

### Roadmap
- 🔲 Regime switching (bull/bear/sideways markets)
- 🔲 Backtesting framework
- 🔲 Custom weight allocation strategies
- 🔲 Additional risk metrics (Sharpe ratio, Sortino ratio)
- 🔲 Web dashboard interface

## 📖 Documentation

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | BTCUSDT ETHUSDT | Space-separated Binance symbols |
| `--days_back` | 730 | Historical data lookback period (days) |
| `--days` | 20 | Simulation horizon (days forward) |
| `--paths` | 20000 | Number of Monte Carlo paths |
| `--initial` | 1.0 | Initial portfolio value |
| `--seed` | 42 | Random seed for reproducibility |
| `--plot` | False | Show individual plots |
| `--dashboard` | False | Show comprehensive risk dashboard |
| `--save` | None | Directory to save plots (e.g., `./reports`) |

### Project Structure

```
Monte-Carlo-Risk-Engine/
├── src/
│   ├── data/
│   │   ├── binance.py      # Binance API integration
│   │   └── prices.py       # yFinance backup (optional)
│   ├── simulation/
│   │   └── monte_carlo.py  # Path simulation engine
│   └── reporting/
│       ├── risk.py         # VaR/CVaR calculations
│       └── visualize.py    # Plotting functions
├── example.py              # Quick demo of all 5 components
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎨 Visualization Examples

The tool generates several types of visualizations:

1. **Simulation Paths (Fan Chart)** - Shows all simulated price trajectories
2. **P&L Distribution** - Histogram with VaR/CVaR markers
3. **Percentile Bands** - 5th, 25th, 50th, 75th, 95th percentile paths
4. **Risk Dashboard** - Comprehensive 6-panel risk analysis
5. **Drawdown Distribution** - Maximum drawdown across all paths

## 📝 Example Output

```
=== Monte Carlo Risk Engine (MVP) ===
Tickers: BTCUSDT, ETHUSDT
Paths: 20,000 | Horizon: 20 days
VaR 95%:  0.1234
CVaR 95%: 0.1567
VaR 99%:  0.2134
CVaR 99%: 0.2456
====================================
```

## 🛠️ Technology Stack

- **Python 3.13+**
- **NumPy** - Matrix operations and Cholesky decomposition
- **Pandas** - Time series data handling
- **Matplotlib** - Visualization
- **SciPy** - Statistical functions
- **Requests** - Binance API calls

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Goal**: Build a production-grade risk management tool for crypto portfolios using Monte Carlo methods.
