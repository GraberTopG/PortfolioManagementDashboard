# US Large Cap Long Only Portfolio Dashboard

An interactive portfolio management dashboard built with **Streamlit** and **Plotly** as part of a Python programming course project at the **University of St. Gallen (HSG)**.

The analytical framework draws on two sources:
- Implementation concepts and library design from the **GS Quant** open-source library by Goldman Sachs (rolling statistics, efficient frontier, VaR, Monte Carlo GBM)
- Theoretical foundations from the HSG course **Asset Allocation and Investment Strategy** (Markowitz mean-variance optimisation, risk parity, factor-based portfolio construction, drawdown analysis)

---

## Development process

This section documents the step-by-step process followed to design and build the dashboard, from the initial architecture decision to the final refinements.

---

### Step 1 - Project scoping and architecture decision

The first decision was choosing the right framework. The GS Quant library from Goldman Sachs requires access to the Marquee API, which is restricted to institutional clients. The project was therefore rebuilt from scratch using the same analytical concepts (rolling statistics, efficient frontier, VaR, Monte Carlo GBM) but with **yfinance** as the free data source and **Streamlit** as the UI framework.

The core architecture was designed as a single-file application (`dashboard.py`) to keep the course submission self-contained, with a layered structure:

```
Constants & universe  ->  Data layer (yfinance)  ->  Analytics functions
->  Chart builders (Plotly)  ->  Streamlit UI (tabs)
```

Each layer is independent: analytics functions take plain NumPy/pandas inputs, chart functions take analytics outputs, and the UI calls both. This separation makes the code testable and readable.

---

### Step 2 - Data layer and caching

The data layer was built around two cached yfinance wrappers:

- `_yf_close()` - batch-downloads adjusted close prices for all selected tickers plus benchmarks (SPY, AGG) in a single API call, returning a clean `pd.DataFrame`.
- `_yf_ohlcv()` - downloads full OHLCV data for a single ticker for the Technical Analysis tab.

**Key challenge - split-adjustment distortion.** yfinance's `auto_adjust=True` multiplies historical share *volumes* by the cumulative split factor, making pre-split periods appear to have enormously more activity than today. For example, Apple's 7-for-1 (2014) and 4-for-1 (2020) splits produce a 28× inflation of pre-2014 volume. The fix was to plot **dollar volume** (shares × price) instead of share count, which cancels the distortion because the price adjustment and volume adjustment are equal and opposite.

**Key challenge - common start date.** All portfolio calculations require every ticker to have a price on the same date. The effective start date is determined by the most recently listed ticker (e.g. adding META limits history to May 2012). A disclaimer was added to the Overview tab that identifies the bottleneck ticker automatically.

A second cached loader, `load_rf_series()`, downloads the 3-month US T-bill rate (`^IRX`) to provide a time-varying risk-free rate for the backtest optimiser.

---

### Step 3 - Analytics functions

All mathematical functions were written as pure, stateless Python functions operating on `pd.Series` or `np.ndarray` inputs, with no Streamlit dependency. This makes them independently testable and reusable.

The following analytics were implemented in sequence:

| Function | What it computes |
|---|---|
| `log_returns`, `simple_returns` | Daily return series |
| `ann_return`, `ann_vol` | Annualised return and volatility |
| `sharpe`, `sortino`, `calmar` | Risk-adjusted return ratios |
| `hist_var`, `param_var`, `cvar` | Value-at-Risk and Expected Shortfall |
| `max_dd`, `dd_series` | Maximum and rolling drawdown |
| `beta_alpha` | CAPM beta and Jensen's alpha vs benchmark |
| `tracking_error`, `information_ratio` | Active risk and active return efficiency |
| `active_share` | Portfolio overlap vs benchmark |
| `hhi` | Herfindahl-Hirschman concentration index |
| `rolling_vol`, `rolling_sharpe` | Rolling window statistics |
| `bollinger`, `rsi`, `macd` | Technical indicators |
| `_fmt_num` | NaN/inf-safe number formatter for display tables |
| `full_metrics` | Full performance metrics dictionary for the Overview table |

**Key design decision - NaN safety.** Many ratios (Sharpe, Sortino, Calmar, IR) can produce `NaN` or `inf` when denominators are zero or the series is too short. A `_fmt_num(val, fmt)` helper was written to intercept these and return `"-"` rather than displaying `"nan"` in the UI.

---

### Step 4 - Portfolio optimisation

The optimisation module was built around SciPy's SLSQP solver with two hard constraints applied to every strategy:
- **Fully invested**: weights sum to 1
- **Long-only with 40% cap**: `0 ≤ wᵢ ≤ 0.40` - prevents the solver from concentrating into whichever stock dominated the in-sample period

Three optimised strategies were implemented:

1. **Min Variance** (`w_min_var`) - minimises `w'Σw`; uses only the covariance matrix, no return forecasts required.
2. **Max Sharpe** (`w_max_sharpe`) - maximises `(w'μ − rᶠ) / √(w'Σw)`; uses historical means as return estimates (noisy but unbiased).
3. **Inverse Volatility / Risk Parity** (`w_inv_vol`) - weights proportional to `1/σᵢ`; no optimisation required, low turnover.

The **Efficient Frontier** was approximated by sampling 600 Dirichlet-random portfolios and plotting (volatility, return) coloured by Sharpe ratio, with the two optimal portfolios overlaid as labelled markers.

---

### Step 5 - Walk-forward backtest

The backtest was the most methodologically complex component. The key design requirements were:

- **No look-ahead bias**: at each monthly rebalancing date, only data available *up to that date* is used (expanding window).
- **Time-varying risk-free rate**: the Max-Sharpe optimiser uses `rf_series.asof(rebal)` - the T-bill rate on the rebalancing date - rather than a fixed modern rate that would distort Sharpe estimates in the 2009–2022 near-zero rate environment.
- **Realistic Market Weight starting point**: since yfinance only provides current market caps, historical starting weights were approximated as `implied_shares × price(t₀)`, where `implied_shares = current_mcap / current_price`. This removes the most obvious form of look-ahead bias (using today's weights as if they applied historically).
- **Transaction costs**: a turnover-based cost slider was added. At each rebalancing, one-way turnover is computed as `0.5 × Σ|w_new − w_drifted|`, where `w_drifted` are the weights that have naturally drifted from the previous rebalancing due to price moves. The cost `turnover × c` is deducted on the first day of the holding period.

---

### Step 6 - Monte Carlo simulation

GBM paths were implemented with **zero drift** - the simulation models pure volatility/uncertainty without extrapolating the in-sample trend. This avoids falsely projecting a bull-market return into the future.

In a later iteration, the static chart was replaced with a **Plotly animation**: paths start at the current portfolio value and grow left-to-right when the user clicks Play. The animation uses ~60 frames at 40 ms/frame (≈ 2.5-second reveal) and runs entirely in the browser via Plotly's JavaScript engine, requiring no Streamlit re-renders. Up to 50 individual paths are shown alongside the 10th, 50th, and 90th percentile bands.

---

### Step 7 - Streamlit UI and tab structure

The UI was structured around five tabs, each designed to answer a single core portfolio question. The sidebar exposes four controls: ticker multiselect (up to 10 stocks, ~190 S&P 500 universe), date range picker, portfolio weighting mode (equal or custom), and per-ticker weight inputs in Custom mode.

Several UI challenges were solved iteratively:

- **Bloomberg terminal aesthetic** - a custom CSS block applies IBM Plex Serif/Mono fonts, a near-black background, and an orange accent colour throughout all Streamlit components (tabs, buttons, sliders, expanders).
- **KPI metric cards** - custom HTML/CSS `mcard()` components display headline numbers (VaR, Sharpe, Drawdown) in a style matching Bloomberg Terminal function-key screens.
- **Static HTML tables** - Streamlit's `st.dataframe()` renders editable, interactive tables with menus. Custom `render_table()` and `render_annual_returns()` functions generate read-only HTML tables with the correct dark theme.
- **Consistent NaN handling** - all displayed numbers pass through `_fmt_num()` to prevent `"nan"` or `"inf"` appearing in the UI when a calculation has insufficient data.

---

### Step 8 - Refinements and quality fixes

A series of bugs and design issues were identified and corrected after initial implementation:

| Issue | Fix |
|---|---|
| KPI positive/negative colours both orange | `.mpos` set to green `#00C853`, `.mneg` set to red `#E53935` |
| MA20 and MA50 both orange in Technicals | MA50 changed to Bloomberg blue `#00A8E8` |
| Monospace font set to serif family | `_MONO` corrected to `"IBM Plex Mono, Courier New, monospace"` |
| `NaN` displayed in style metrics table | `_fmt_num()` applied consistently to all ratio fields |
| Rolling correlation fill colour wrong | Changed from teal to orange to match accent |
| Volume bars distorted by split adjustments | Switched from share count to dollar volume (`shares × price`) |
| NameError in raw f-string with LaTeX braces | `\text{RC}` -> `\text{{RC}}` in `rf"""..."""` string |
| Sector pie sharing colours with position pie | Separate cool-toned palette assigned to sector donut |
| Market Weight using today's weights | Historical starting weights derived from implied shares × start-date price |

---

### Step 9 - Documentation

The final step was a documentation pass:
- A full module docstring was added to `dashboard.py` explaining the tab structure, key design choices, and dependencies.
- All analytics functions received docstrings explaining inputs, outputs, and methodological notes.
- The README was rewritten to include the tab structure, full feature list, methodology notes (walk-forward backtest, market weight derivation, time-varying RF, risk contribution formula, dollar volume), analytics reference table with LaTeX formulas, tech stack table, and this development process overview.
- A **Critical Limitations** section was added documenting survivorship bias, GBM assumptions, covariance stationarity, noisy sample-mean estimates, backtest execution assumptions, and ten concrete directions for further research.

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
- Portfolio vs benchmarks (S&P 500, 60/40, US Bonds) - rebased to 0%
- Performance metrics table: Ann. Return, Volatility, Sharpe, Sortino, Calmar, Max Drawdown, Beta, Alpha, Tracking Error, Information Ratio, Active Share, VaR, CVaR, Skewness, Kurtosis
- Data-window disclaimer when the selected period is limited by a ticker's IPO date
- CSV export

### Risk
- Portfolio VaR (historical and parametric), CVaR, Max Drawdown - headline cards
- Return distribution with normal fit overlay and VaR marker
- Drawdown chart
- Rolling VaR (adjustable window)
- **Risk Contribution vs Weight** - bar chart showing each stock's % share of total portfolio variance vs its portfolio weight
- **Rolling 252-day Sharpe Ratio** - for both portfolio and individual stock drill-down
- Individual stock drill-down (VaR, CVaR, Drawdown, Rolling VaR, Rolling Sharpe)
- GBM Monte Carlo simulation (portfolio and single stock) - zero-drift, log-normal paths
- Terminal value distribution with P(loss > 15%) statistic

### Correlation
- Pairwise return heatmap (burgundy–grey–orange diverging scale)
- Rolling pairwise correlation (adjustable window)
- Daily return scatter with OLS trendline

### Optimisation
- Portfolio styles backtested performance chart (walk-forward, monthly rebalancing)
- Style performance metrics table with CSV export
- **Annual Returns by Strategy** - year-by-year colour-coded grid (green/red)
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
- **Dollar volume bars** coloured green/red by price direction - immune to split-adjustment distortion
- RSI (14) and MACD (12/26/9) with volume

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/GraberTopG/PortfolioManagementDashboard.git
cd PortfolioManagementDashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run dashboard.py
```

Data is fetched automatically via **yfinance** - no API key required.

> **Note on data history:** All calculations require every selected ticker to have a price on the same date. The effective start date is determined by the **most recently listed ticker** in your selection. For example, adding META (IPO May 2012) limits the common history to May 2012 onward. A disclaimer is shown below the benchmark chart whenever the actual data window is shorter than the selected period, identifying the bottleneck ticker.

---

## Methodology notes

### Walk-forward backtest
The backtest uses an **expanding window**: at each monthly rebalancing date, only data available up to that date is used to estimate means and covariances. No future data leaks into past decisions. The Market Weight strategy is an exception - it requires no rebalancing (see below).

### Market Weight (buy-and-hold)
yfinance only provides current market caps, so historical starting weights are estimated as:

```
implied_shares_i  =  current_mcap_i  /  current_price_i
w_i(t_0)          ∝  implied_shares_i  ×  price_i(t_0)
```

This gives weights proportional to relative market caps on the first day of the backtest rather than today's caps, eliminating the most obvious form of look-ahead bias. The portfolio then drifts freely - no rebalancing.

**Limitation:** `implied_shares` reflects today's share count. Companies have issued new equity (compensation, acquisitions) since the start of the backtest, so the approximation is imperfect - fast-growing companies are slightly overweighted at the start.

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

## Critical limitations

### 1. Data & universe

| Limitation | Detail |
|---|---|
| **Survivorship bias** | The ~190 tickers are all *current* S&P 500 constituents. Companies that were once in the index but went bankrupt or were acquired (e.g. Lehman Brothers, Bear Stearns) are excluded. This systematically overstates historical returns for every strategy. |
| **US large-cap equities only** | No fixed income, commodities, real estate (beyond REITs in universe), international equities, or alternatives. True portfolio construction requires multi-asset allocation. |
| **yfinance data quality** | Corporate action adjustments (dividends, splits) occasionally contain errors, particularly for older data. No cross-validation against a paid data provider is performed. |
| **10-ticker cap** | The sidebar limits selections to 10 stocks. A well-diversified portfolio typically holds 20–30+ positions; with 10 stocks the efficient frontier is artificially constrained and covariance estimates are noisy. |
| **No real-time prices** | yfinance returns end-of-day data with a ~15-minute delay. The dashboard is a research tool, not a live trading system. |

---

### 2. Statistical model assumptions

**GBM / Monte Carlo**
- Assumes **constant volatility** and **i.i.d. log-normal shocks**. Real returns exhibit volatility clustering (GARCH effects), fat tails (excess kurtosis), and occasional jumps (earnings, macro events). The simulation therefore underestimates the probability of extreme outcomes.
- **Zero drift** is used deliberately to avoid extrapolating a bull-market sample, but it ignores the long-run equity risk premium.
- A more realistic model would use GARCH(1,1) for time-varying volatility or a jump-diffusion process (Merton 1976).

**VaR / CVaR**
- **Historical VaR** assumes the observed sample is representative of future tail risk. Short windows (e.g. 1–2 years) that exclude a major crisis will severely underestimate downside risk.
- **Parametric VaR** assumes normality. For equity portfolios with negative skewness and excess kurtosis, the normal approximation consistently underestimates tail losses - a Cornish-Fisher expansion or EVT-based VaR would be more accurate.

**Correlations and covariances**
- Sample covariance is computed from the *same* historical window used for optimisation, which is known to overfit to recent data. No shrinkage estimator (e.g. Ledoit-Wolf) is applied. This inflates apparent portfolio diversification.
- Correlations are treated as **stationary**. In practice, equity correlations spike toward 1 during market crises (the very moment diversification is most needed), a phenomenon known as correlation breakdown.

**Annualised return**
- Reported as $\bar{r} \times 252$ (arithmetic approximation). The geometric compound return $(1+r)^{252/n} - 1$ is lower for volatile assets and is the correct measure for multi-year horizons.

---

### 3. Portfolio optimisation

**Mean-Variance / Max-Sharpe**
- Expected returns are estimated from the *sample mean* of historical returns, which is an extremely noisy estimator - the Sharpe-maximising portfolio effectively bets entirely on whichever stock happened to have the highest in-sample Sharpe ratio. The **Black-Litterman model** addresses this by blending a market-equilibrium prior with the investor's explicit views.
- The 40% single-stock cap is an ad-hoc constraint that improves robustness but is not derived from a principled risk budget.
- Re-estimating the full covariance matrix monthly from an expanding window means early rebalancing periods are dominated by very short, noisy samples.

**Risk Parity (inverse-volatility)**
- The implementation weights by $1/\hat\sigma_i$, which equalises *volatility* contributions but not *variance contributions*. True risk parity requires equalising $w_i \cdot (\Sigma w)_i$ across all assets, which demands solving a convex optimisation problem. The simplified version used here produces a good approximation but deviates from the formal definition, especially when assets are highly correlated.

**Efficient frontier random portfolios**
- The scatter cloud is sampled from a Dirichlet distribution, which places most mass near equal weight. It does not uniformly cover the full feasible set, and extreme low-volatility or high-return corners of the frontier may be under-represented.

---

### 4. Backtest methodology

**Transaction costs**
- The cost model deducts a fixed percentage per unit of one-way turnover. It does not model: bid-ask spread widening during stress periods, market impact for larger orders, short-term capital gains taxes, or the cost of rebalancing in kind versus cash.

**Execution assumptions**
- All trades are assumed to execute at the **daily closing price** with no slippage. In practice, a monthly rebalancing signal generated at close would execute the next morning at open, often at a less favourable price.

**Fixed rebalancing frequency**
- Monthly rebalancing is imposed uniformly. Threshold-based rebalancing (rebalance only when weights drift beyond a band) can reduce turnover costs significantly without sacrificing much of the optimisation benefit.

**Residual look-ahead bias in Market Weight**
- `implied_shares = current_mcap / current_price` uses today's share count, which reflects buybacks and equity issuances since the start of the backtest. Companies that have repurchased large amounts of stock (e.g. Apple) are slightly *over*-weighted at inception.

**No regime conditioning**
- All strategies are estimated on an expanding unconditional window. A regime-aware approach (e.g. Hidden Markov Model to detect bull/bear/high-vol regimes) would dynamically adjust the covariance estimate and potentially improve out-of-sample performance.

---

### 5. Further research directions

| Area | Description |
|------|-------------|
| **Factor models** | Decompose returns into Fama-French factors (market, size, value, profitability, investment) and momentum. Show factor exposures per portfolio strategy and use factor-based expected return estimates in the optimiser. |
| **Black-Litterman** | Replace the sample-mean expected return vector with a Black-Litterman posterior that blends the CAPM equilibrium return with the user's explicit views. Dramatically reduces the estimation-error problem in Mean-Variance optimisation. |
| **GARCH volatility** | Fit a GARCH(1,1) model per asset for the Monte Carlo simulation to capture volatility clustering. Would widen the simulation cone appropriately after high-volatility periods. |
| **CVaR optimisation** | Directly minimise Expected Shortfall (CVaR) as the portfolio objective rather than variance. More robust to tail risk and avoids the normality assumption embedded in Mean-Variance. |
| **Multi-asset allocation** | Extend the universe to include bond ETFs (AGG, TLT), commodity ETFs (GLD, USO), and international equity ETFs (EFA, EEM). Currently the tool is a pure US equity selector. |
| **Stress testing** | Add predefined historical stress scenarios (2000–2002 dot-com bust, 2008–2009 GFC, March 2020 COVID crash) to show how each strategy would have performed in known crises. |
| **Shrinkage estimation** | Apply Ledoit-Wolf covariance shrinkage to reduce estimation error in the optimiser, particularly important when the number of assets approaches the length of the return history. |
| **Rebalancing bands** | Implement threshold-based rebalancing (only rebalance when a weight drifts more than *x*% from target) as an alternative to fixed monthly rebalancing, reducing unnecessary turnover. |
| **True Risk Parity** | Solve the full risk contribution equalisation problem $w_i (\Sigma w)_i = \text{const}$ using sequential quadratic programming, rather than the inverse-volatility approximation. |
| **Tax-loss harvesting** | Simulate the after-tax return improvement from realising capital losses to offset gains, relevant for taxable accounts. |

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
| **Plotly** ≥ 5.18 | Interactive charts - dark Bloomberg theme |
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

*HSG Master - Programming with Advanced Computer Languages - 2025/26*
