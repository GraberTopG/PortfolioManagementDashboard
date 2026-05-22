# US Large Cap Long Only Portfolio Dashboard

An interactive portfolio management dashboard built with **Streamlit** and **Plotly** as part of a Python programming course project at the **University of St. Gallen (HSG)**.

The analytical framework draws on two sources:
- Implementation concepts and library design from the **GS Quant** open-source library by Goldman Sachs (rolling statistics, efficient frontier, VaR, Monte Carlo GBM)
- Theoretical foundations from the HSG course **Asset Allocation and Investment Strategy** (Markowitz mean-variance optimisation, risk parity, factor-based portfolio construction, drawdown analysis)

---

## Tab structure

Each tab is anchored to a core portfolio question:

| # | Question | Tab |
|---|----------|-----|
| 1 | *What do I own and how has it performed?* | **Overview** |
| 2 | *What are the risks of what I hold?* | **Risk** |
| 3 | *How do my holdings move relative to each other?* | **Correlation** |
| 4 | *How should I construct or improve the portfolio?* | **Optimisation** |
| 5 | *What's the price action on individual names?* | **Technicals** |

---

## Features

### Overview
- Portfolio allocation donut chart with position-level weights
- GICS sector allocation donut chart
- Concentration metrics: HHI, effective number of positions, largest holding
- Sector exposure vs S&P 500 with active weights
- Individual stock cumulative returns with portfolio overlay
- Portfolio vs benchmarks (S&P 500, 60/40, US Bonds)
- Performance metrics table: Ann. Return, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, Beta, Alpha, Tracking Error, Information Ratio, Active Share, VaR, CVaR, Skewness, Kurtosis
- CSV export

### Risk
- Portfolio-level VaR (historical and parametric), CVaR, Max Drawdown
- Return distribution with normal fit overlay
- Drawdown chart
- Rolling VaR
- Individual stock drill-down (adjustable confidence level)
- GBM Monte Carlo simulation with percentile fan chart (zero-drift)
- Terminal value distribution (portfolio and single stock)
- P(loss > 15%) replacing misleading P(value > today)

### Correlation
- Pairwise return heatmap (burgundy - grey - orange scale)
- Rolling pairwise correlation (adjustable window)
- Daily return scatter with OLS trendline

### Optimisation
- Portfolio styles backtested performance (monthly rebalancing)
- Style performance metrics table with CSV export
- Efficient frontier with Capital Market Line
- Optimal portfolio allocations table with CSV export
- Strategy methodology with formulas

**Portfolio styles compared:**
- Your Portfolio (equal weight or custom)
- Equal Weight (1/N)
- Minimum Variance
- Mean-Variance (Maximum Sharpe / Tangency Portfolio)
- Risk Parity (Inverse Volatility)
- Market Weight (buy-and-hold, market-cap weighted)

### Technicals
- Candlestick chart with MA 20/50 and Bollinger Bands
- RSI (14) and MACD (12/26/9) with volume

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

> **Note on data history:** All calculations require every selected ticker to have a price on the same date. The effective start date is therefore determined by the **most recently listed ticker** in your selection. For example, adding META (IPO May 2012) limits the common history to May 2012 onward regardless of the date range you enter. A warning banner is shown whenever the actual data window is shorter than the selected period, identifying which ticker is the bottleneck.

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
- Tracking Error, Information Ratio, Active Share vs market-cap benchmark
- HHI concentration index and effective number of positions
- Sector exposure active weights vs S&P 500 GICS sector weights
- Historical and parametric VaR - CVaR / Expected Shortfall
- Max Drawdown, Win Rate, Skewness, Kurtosis
- Markowitz efficient frontier (SLSQP optimisation)
- Risk parity portfolio (inverse-volatility weighting)
- Market-cap weighted portfolio (via yfinance fast_info)
- GBM Monte Carlo simulation (zero-drift, log-normal price paths)
- Custom portfolio weights via sidebar (auto-normalised to 100%)
- Methodology sections with LaTeX formulas in every tab

---

*HSG Master - Programming with Advanced Computer Languages - 2025/26*
