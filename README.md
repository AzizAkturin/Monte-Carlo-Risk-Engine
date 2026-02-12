# 🎲 Monte Carlo Risk Engine

A professional Python-based Monte Carlo simulation tool for cryptocurrency portfolio risk analysis. Simulates correlated multi-asset price paths and generates comprehensive risk reports with interactive visualizations.

## 🎯 Core Components

✅ **1. Historical Price Data** - Fetch daily close prices from Binance  
✅ **2. Log Returns** - Compute logarithmic returns for statistical analysis  
✅ **3. Drift & Volatility** - Estimate daily mean returns (μ) and covariance matrix (Σ)  
✅ **4. Path Simulation** - Generate correlated GBM-style price paths using Cholesky decomposition  
✅ **5. Risk Metrics & Visualization** - Calculate VaR/CVaR and create interactive risk dashboards  

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd Monte-Carlo-Risk-Engine

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Run portfolio analysis (SOL, ETH, BTC, USDC)
python run_analysis.py
```

This will:
1. Fetch 1 year of historical data from Binance
2. Run 20,000 Monte Carlo simulations over 30 days
3. Calculate comprehensive risk metrics
4. Generate an interactive dashboard
5. Save results to `reports/` folder

## 📊 Features

- ✅ **Multi-Asset Correlation** - Cholesky decomposition for realistic correlation modeling
- ✅ **Monte Carlo Simulation** - 20,000+ paths with configurable horizons
- ✅ **Risk Metrics** - VaR, CVaR at 95% and 99% confidence levels
- ✅ **Drawdown Analysis** - Peak-to-trough risk measurement
- ✅ **Interactive Dashboard** - 6-panel visualization with clear explanations
- ✅ **Real-Time Data** - Binance API integration with rate limiting
- ✅ **Professional Reports** - Auto-generated analysis reports

## 🔧 Customization

Edit `run_analysis.py` to customize:

**Portfolio Assets:**
```python
tickers = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "USDCUSDT"]
token_names = ["Solana", "Ethereum", "Bitcoin", "USDC"]
```

**Initial Investment:**
```python
initial_value = 10000.0  # $10,000 portfolio
```

**Simulation Parameters:**
```python
config = MCConfig(
    horizon_days=30,    # 30-day forecast
    n_paths=20000,      # 20,000 simulations  
    seed=42
)
```

**Weight Allocation:**
```python
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weight
# or custom: [0.40, 0.30, 0.20, 0.10]  # 40% SOL, 30% ETH, etc.
```

### Project Structure

```
Monte-Carlo-Risk-Engine/
├── src/
│   ├── data/
│   │   ├── binance.py       # Binance API integration
│   │   └── prices.py        # yFinance fallback
│   ├── simulation/
│   │   └── monte_carlo.py   # Monte Carlo engine
│   └── reporting/
│       ├── risk.py          # Risk metrics (VaR/CVaR)
│       └── visualize.py     # Dashboard generation
├── reports/                 # Output folder
│   ├── portfolio_dashboard.png
│   └── portfolio_analysis.txt
├── run_analysis.py          # Main analysis script
├── requirements.txt
└── README.md
```

## 📊 Output

### Dashboard Components

1. **📈 Simulation Paths** - Visual fan chart showing possible futures
2. **💰 P&L Distribution** - Histogram of final outcomes with risk thresholds
3. **📊 Percentile Scenarios** - Best to worst case trajectories
4. **⚠️ Risk Metrics Table** - Key risk numbers with explanations
5. **📉 Drawdown Analysis** - Maximum drop from peak analysis

### Files Generated

- `reports/portfolio_dashboard.png` - Interactive risk dashboard
- `reports/portfolio_analysis.txt` - Detailed metrics report

## 🛠️ Tech Stack

- **Python 3.11+** - Core language
- **NumPy** - Monte Carlo simulations & matrix operations
- **Pandas** - Time series data handling
- **Matplotlib** - Professional visualizations
- **Requests** - Binance API integration

## 📚 Understanding the Output

**VaR (Value at Risk)** - The maximum loss at a given confidence level  
*VaR 95% = $2,327 means: "95% confident you won't lose more than $2,327"*

**CVaR (Conditional VaR)** - Average loss in worst-case scenarios  
*CVaR 95% = $2,760 means: "If you're in the worst 5%, expect $2,760 loss on average"*

**Probability of Loss** - Chance the portfolio loses money  
*58.47% means: "More than half the simulations ended with a loss"*

**Drawdown** - Maximum drop from the highest point  
*26% drawdown means: "Portfolio dropped 26% from its peak"*

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Built for professional crypto portfolio risk analysis using Monte Carlo simulation**
