# US Large Cap Long Only Portfolio Dashboard

An interactive portfolio management dashboard built with **Streamlit** and **Plotly** as part of a Python programming course project at the **University of St. Gallen (HSG)**.

The analytical framework draws on two sources:
- Implementation concepts and library design from the **GS Quant** open-source library by Goldman Sachs (rolling statistics, efficient frontier, VaR, Monte Carlo GBM)
- Theoretical foundations from the HSG course **Asset Allocation and Investment Strategy** (Markowitz mean-variance optimisation, risk parity, factor-based portfolio construction, drawdown analysis)

---

## Features

| Tab | What you get |
|-----|-------------|
| **Overview** | Individual stock cumulative returns with portfolio overlay - Portfolio vs benchmarks (S&P 500, 60/40, US Bonds) - Portfolio metrics table (Sharpe, Sortino, Calmar, Beta, Alpha, VaR, CVaR, Max DD, Win Rate, Skewness, Kurtosis) |
| **Technical Analysis** | Candlestick chart - MA 20/50 - Bollinger Bands - RSI (14) - MACD (12/26/9) - Volume |
| **Correlation** | Pairwise return heatmap - Rolling pairwise correlation - Return scatter with OLS trendline |
| **Portfolio Optimisation** | Backtested cumulative returns with monthly rebalancing - Style performance metrics - Efficient frontier (Monte Carlo) with Capital Market Line - Allocation table for 5 styles - Strategy methodology with formulas |
| **Risk Metrics** | Portfolio-level VaR / CVaR / drawdown / rolling VaR - Individual stock drill-down (adjustable confidence level) - Risk metric definitions |
| **Monte Carlo** | GBM portfolio simulation with percentile bands - Single-stock simulation - Terminal value distribution (adjustable paths and horizon) - Methodology section |

### Portfolio styles compared
- Your Portfolio (equal weight or custom)
- Equal Weight
- Minimum Variance
- Mean-Variance (Maximum Sharpe / Tangency)
- Risk Parity (Inverse Volatility)
- Market Weight (market-cap weighted)

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/GraberTopG/quant-finance-dashboard.git
cd quant-finance-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run dashboard.py
```

Data is fetched automatically via **yfinance** - no API key required. Full history available from 2000 to today.

---

## Ticker universe

~190 S&P 500 equities across 11 GICS sectors. Select up to **10** tickers from the sidebar to build your portfolio.

| Sector | Examples |
|--------|---------|
| Technology | AAPL - MSFT - GOOGL - AMZN - META - NVDA - TSLA - NFLX - AMD |
| Financials | JPM - GS - BAC - MS - BLK - V - MA |
| Healthcare | JNJ - UNH - PFE - ABBV - MRK - LLY |
| Consumer Discretionary | MCD - NKE - HD - SBUX - TGT |
| Consumer Staples | WMT - KO - PG - PEP - COST |
| Industrials | BA - CAT - HON - GE - LMT |
| Energy | XOM - CVX - COP - SLB |
| Communication | T - VZ - CMCSA - DIS |
| Real Estate | AMT - PLD - CCI - EQIX |
| Materials | LIN - APD - SHW - NEM |
| Utilities | NEE - DUK - SO - AEP |

---

## Tech stack

- **Streamlit** - web app framework
- **Plotly** - interactive charts (dark theme, Bloomberg colour palette)
- **pandas / NumPy / SciPy** - data wrangling and optimisation
- **yfinance** - live market data (free, no API key)
- **statsmodels** - OLS trendline in scatter plots

---

## Key analytics

- Annualised return, volatility, Sharpe, Sortino, Calmar ratios
- Beta and Jensen's Alpha vs S&P 500
- Historical and parametric VaR - CVaR / Expected Shortfall
- Max Drawdown, Win Rate, Skewness, Kurtosis
- Markowitz efficient frontier (SLSQP optimisation)
- Risk parity portfolio (inverse-volatility weighting)
- Market-cap weighted portfolio (via yfinance fast_info)
- GBM Monte Carlo simulation (zero-drift, log-normal price paths)
- Custom portfolio weights via sidebar (auto-normalised to 100%)

---

*HSG Master - Programming with Advanced Computer Languages - 2025/26*
