# US Large Cap Long Only Portfolio Dashboard

An interactive portfolio management dashboard built with **Streamlit** and **Plotly** as part of a Python programming course project at the **University of St. Gallen (HSG)**.

The analytical framework draws on two sources:
- Implementation concepts and library design from the **GS Quant** open-source library by Goldman Sachs (rolling statistics, efficient frontier, VaR, Monte Carlo GBM)
- Theoretical foundations from the HSG course **Asset Allocation and Investment Strategy** (Markowitz mean-variance optimisation, risk parity, factor-based portfolio construction, drawdown analysis)

---

## Tab structure

Each tab answers one core portfolio question:

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
- Portfolio allocation donut chart (position-level weights)
- GICS sector allocation donut chart (with a distinct colour palette)
- Concentration metrics: HHI, effective number of positions, largest holding
- Sector exposure vs S&P 500 with active weights
- Individual stock cumulative returns with portfolio overlay
- Portfolio vs benchmarks (S&P 500, 60/40, US Bonds) — rebased to 0%
- Performance metrics table: Ann. Return, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, Beta, Alpha, Tracking Error, Information Ratio, Active Share, VaR, CVaR, Skewness, Kurtosis
- Data-window disclaimer when the selected period is limited by a ticker's IPO date
- CSV export

### Risk
- Portfolio VaR (historical and parametric), CVaR, Max Drawdown — headline cards
- Return distribution with normal fit overlay and VaR marker
- Drawdown chart
- Rolling VaR (adjustable window)
- **Risk Contribution vs Weight** — bar chart showing each stock's % share of total portfolio variance vs its portfolio weight
- **Rolling 252-day Sharpe Ratio** — for both portfolio and individual stock drill-down
- Individual stock drill-down (VaR, CVaR, Drawdown, Rolling VaR, Rolling Sharpe)
- GBM Monte Carlo simulation (portfolio and single stock) — zero-drift, log-normal paths
- Terminal value distribution with P(loss > 15%) statistic

### Correlation
- Pairwise return heatmap (burgundy–grey–orange diverging scale)
- Rolling pairwise correlation (adjustable window)
- Daily return scatter with OLS trendline

### Optimisation
- Portfolio styles backtested performance chart (walk-forward, monthly rebalancing)
- Style performance metrics table with CSV export
- **Annual Returns by Strategy** — year-by-year colour-coded grid (green/red)
- Efficient frontier with Capital Market Line
- Optimal portfolio allocations table (today's full-period weights) with CSV export
- Strategy methodology with LaTeX formulas

**Portfolio styles compared:**
| Style | Description |
|-------|-------------|
| Your Portfolio | Equal-weight or custom user-defined weights |
| Equal Weight | 1/N, rebalanced monthly |
| Min Variance | Minimises portfolio variance; no return forecasts needed |
| Mean-Variance | Maximises Sharpe ratio using historical ^IRX as risk-free rate |
| Risk Parity | Inverse-volatility weighting; lower-vol stocks get more weight |
| Market Weight | True buy-and-hold from historically estimated starting weights |

### Technicals
- Candlestick chart with MA 20 (gold) and MA 50 (blue)
- Bollinger Bands (±2 std)
- **Dollar volume bars** coloured green/red by price direction — immune to split-adjustment distortion
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

Data is fetched automatically via **yfinance** — no API key required.

> **Note on data history:** All calculations require every selected ticker to have a price on the same date. The effective start date is determined by the **most recently listed ticker** in your selection. For example, adding META (IPO May 2012) limits the common history to May 2012 onward. A disclaimer is shown below the benchmark chart whenever the actual data window is shorter than the selected period, identifying the bottleneck ticker.

---

## Methodology notes

### Walk-forward backtest
The backtest uses an **expanding window**: at each monthly rebalancing date, only data available up to that date is used to estimate means and covariances. No future data leaks into past decisions. The Market Weight strategy is an exception — it requires no rebalancing (see below).

### Market Weight (buy-and-hold)
yfinance only provides current market caps, so historical starting weights are estimated as:

```
implied_shares_i  =  current_mcap_i  /  current_price_i
w_i(t_0)          ∝  implied_shares_i  ×  price_i(t_0)
```

This gives weights proportional to relative market caps on the first day of the backtest rather than today's caps, eliminating the most obvious form of look-ahead bias. The portfolio then drifts freely — no rebalancing.

**Limitation:** `implied_shares` reflects today's share count. Companies have issued new equity (compensation, acquisitions) since the start of the backtest, so the approximation is imperfect — fast-growing companies are slightly overweighted at the start.

### Time-varying risk-free rate
The Max-Sharpe strategy uses the **historical 3-month T-bill rate (^IRX)** at each rebalancing date via `rf_series.asof(rebal)`. Using a fixed rate (e.g. today's 5.25%) for all historical periods would artificially deflate Sharpe estimates in the 2009–2022 near-zero rate environment.

### Risk contribution
Each stock's share of total portfolio variance:

```
RC_i  =  w_i × (Σw)_i  /  w'Σw
```

A stock with a small weight but high RC is a risk concentrator. A stock with a large weight but low RC contributes efficient diversification.

### Dollar volume
Technical Analysis plots **dollar volume** (shares × price) rather than share count. yfinance with `auto_adjust=True` multiplies historical share volumes by the cumulative split factor, making pre-split periods appear to have enormously more trading activity. Dollar volume cancels this distortion: price adjustment and volume adjustment are equal and opposite.

---

## Ticker universe

~190 S&P 500 equities across all 11 GICS sectors. Select up to **10** tickers from the sidebar.

| Sector | Examples |
|--------|---------|
| Technology | AAPL · MSFT · GOOGL · AMZN · META · NVDA · TSLA · NFLX · AMD |
| Financials | JPM · GS · BAC · MS · BLK · V · MA |
| Healthcare | JNJ · UNH · PFE · ABBV · MRK · LLY |
| Consumer Discretionary | MCD · NKE · HD · SBUX · TGT |
| Consumer Staples | WMT · KO · PG · PEP · COST |
| Industrials | BA · CAT · HON · GE · LMT |
| Energy | XOM · CVX · COP · SLB |
| Communication | T · VZ · CMCSA · DIS |
| Real Estate | AMT · PLD · CCI · EQIX |
| Materials | LIN · APD · SHW · NEM |
| Utilities | NEE · DUK · SO · AEP |

---

## Tech stack

| Library | Role |
|---------|------|
| **Streamlit** ≥ 1.35 | Web app framework, UI components, caching |
| **Plotly** ≥ 5.18 | Interactive charts — dark Bloomberg theme |
| **pandas / NumPy** | Data wrangling, matrix operations |
| **SciPy** | SLSQP optimisation (efficient frontier) |
| **yfinance** ≥ 0.2.40 | Live market data, no API key required |
| **statsmodels** | OLS trendline in return scatter plot |

---

## Analytics reference

| Metric | Formula |
|--------|---------|
| Annualised Return | $\bar{r} \times 252$ |
| Annualised Volatility | $\sigma \times \sqrt{252}$ |
| Sharpe Ratio | $(R_p - R_f) / \sigma_p$ |
| Sortino Ratio | $(R_p - R_f) / \sigma_{\text{down}}$ |
| Calmar Ratio | $R_p / |\text{Max DD}|$ |
| Beta | $\text{Cov}(r_p, r_b) / \text{Var}(r_b)$ |
| Alpha | $R_p - R_f - \beta(R_b - R_f)$ |
| Tracking Error | $\sigma(r_p - r_b) \times \sqrt{252}$ |
| Information Ratio | $\overline{(r_p - r_b)} / \text{TE} \times \sqrt{252}$ |
| Active Share | $\frac{1}{2}\sum_i |w_i - b_i|$ |
| HHI | $\sum_i w_i^2$ |
| Historical VaR | $-\text{Percentile}(r, 1-c)$ |
| CVaR / ES | $-E[r \mid r < -\text{VaR}]$ |
| Risk Contribution | $w_i(\Sigma w)_i / w^\top\Sigma w$ |

---

*HSG Master — Programming with Advanced Computer Languages — 2025/26*
