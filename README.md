# Portfolio Management Dashboard

An interactive portfolio management dashboard built with **Streamlit** and **Plotly** as part of a Python programming course project at the **University of St. Gallen (HSG)**.

The analytics concepts (rolling statistics, portfolio optimisation, risk measures, Monte Carlo simulation) are inspired by the **GS Quant** open-source library from Goldman Sachs.

---

## Features

| Tab | What you get |
|-----|-------------|
| 🏠 **Overview** | Portfolio metrics table vs S&P 500 (Sharpe, Sortino, Calmar, Beta, Alpha, VaR, CVaR, Max DD, Win Rate, Skewness, Kurtosis) · Individual stock cumulative returns · Benchmark comparison chart (S&P 500, 60/40, US Bonds) |
| 📊 **Technical Analysis** | Candlestick chart · MA 20/50 · Bollinger Bands · RSI (14) · MACD (12/26/9) · Volume |
| 🔗 **Correlation** | Pairwise return heatmap · Rolling pairwise correlation · Return scatter with OLS trendline |
| 💼 **Portfolio Optimisation** | Efficient frontier (Monte Carlo) with Capital Market Line · Allocation table for 5 styles · Backtested cumulative returns with monthly rebalancing · Style performance metrics |
| ⚠️ **Risk Metrics** | Portfolio-level VaR / CVaR / drawdown / rolling VaR · Individual stock drill-down (adjustable confidence level) |
| 🎲 **Monte Carlo** | GBM portfolio simulation with percentile bands · Single-stock simulation · Terminal value distribution (adjustable paths & horizon) |

### Portfolio styles compared
- Equal Weight
- Minimum Variance
- Maximum Sharpe
- Risk Parity
- Inverse Volatility

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/quant-finance-dashboard.git
cd quant-finance-dashboard
```

### 2. Install dependencies
```bash
pip install streamlit plotly pandas numpy scipy requests
```

### 3. Run the dashboard
```bash
streamlit run dashboard.py
```

The app loads **pre-generated synthetic market data** from `data_cache/` instantly — no API key needed to explore the dashboard.

### Optional — live market data
Get a free [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key), enter it in the sidebar, and click **⬇ Load Data** to fetch real prices (25 requests/day on the free tier; data is cached locally after the first fetch).

---

## Ticker universe (38 symbols)

| Sector | Tickers |
|--------|---------|
| 🖥️ Technology | AAPL · MSFT · GOOGL · AMZN · META · NVDA · TSLA · NFLX · AMD · ORCL · CRM · ADBE · INTC · QCOM |
| 🏦 Finance | JPM · GS · BAC · MS · BLK · V · MA |
| 🏥 Healthcare | JNJ · UNH · PFE · ABBV |
| 🛒 Consumer | KO · MCD · NKE · WMT · TSCO |
| ⚡ Energy | XOM · CVX |
| 📊 ETFs | SPY · QQQ · IWM · AGG · TLT · GLD |

Select up to **10** tickers from the sidebar to build your portfolio.

---

## Tech stack

- **Streamlit** — web app framework
- **Plotly** — interactive charts (dark theme)
- **pandas / NumPy / SciPy** — data wrangling and optimisation
- **Alpha Vantage** — live market data (free API)
- **GS Quant** concepts — analytics inspiration

---

## Key analytics

- Annualised return, volatility, Sharpe, Sortino, Calmar ratios
- Beta and Jensen's Alpha vs S&P 500
- Historical & parametric VaR; CVaR / Expected Shortfall
- Max Drawdown, Win Rate, Skewness, Kurtosis
- Markowitz efficient frontier (SLSQP optimisation)
- Risk-parity portfolio (equal risk-contribution)
- GBM Monte Carlo simulation (log-normal price paths)

---

*HSG Master · Programming Course Project · 2025/26*
