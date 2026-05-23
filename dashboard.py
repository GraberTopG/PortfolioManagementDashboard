"""
US Large Cap Long Only Portfolio Dashboard
==========================================
HSG Master – Programming with Advanced Computer Languages – 2025/26

An interactive quantitative finance dashboard built with Streamlit and Plotly.
Analytical framework draws on Markowitz mean-variance optimisation, risk parity,
factor-based portfolio construction, drawdown analysis, and GBM Monte Carlo
simulation. All market data is fetched live via yfinance — no API key required.

Tab structure (each tab answers one core portfolio question):
  1. Overview      – "What do I own and how has it performed?"
                     Allocation pies, sector exposure, benchmark comparison,
                     performance metrics table (Sharpe, Alpha, VaR, etc.)
  2. Risk          – "What are the risks of what I hold?"
                     VaR/CVaR (historical + parametric), drawdown, rolling VaR,
                     risk contribution per stock, rolling Sharpe, GBM Monte Carlo.
  3. Correlation   – "How do my holdings move relative to each other?"
                     Pairwise heatmap, rolling pairwise correlation, OLS scatter.
  4. Optimisation  – "How should I construct or improve the portfolio?"
                     Walk-forward style backtest, annual returns grid, efficient
                     frontier with CML, optimal allocation table per strategy.
  5. Technicals    – "What's the price action on individual names?"
                     Candlestick + MA20/50 + Bollinger Bands, RSI, MACD.

Key design choices:
  - All prices fetched via yfinance with auto_adjust=True (split + dividend adjusted).
  - Dollar volume used in Technicals to remove split-adjustment distortion.
  - Walk-forward backtest: at each monthly rebalancing date only past data is used.
  - Market Weight is a true buy-and-hold using historically estimated starting weights
    (implied shares = current market cap / current price, applied to start-date price).
  - Time-varying risk-free rate from ^IRX (3-month T-bill) used in Max Sharpe backtest.
  - 40% single-stock cap enforced in all SLSQP optimisations.
  - Minimum 63-day (3-month) lookback before covariance matrix is trusted.

Dependencies: streamlit, plotly, pandas, numpy, scipy, yfinance, statsmodels
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
AF = 252
RF = 0.0525
DEFAULT_TICKERS   = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "GS"]
BENCHMARK_TICKERS = ["SPY", "AGG"]

UNIVERSE = {
    "Technology": [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","AMD","ORCL",
        "CRM","ADBE","INTC","QCOM","AVGO","TXN","NOW","PANW","INTU","CSCO",
        "IBM","ACN","AMAT","MU","KLAC","LRCX","SNPS","CDNS","FTNT","HPQ",
    ],
    "Financials": [
        "JPM","GS","BAC","MS","BLK","V","MA","C","WFC","AXP",
        "SCHW","SPGI","ICE","CME","MCO","PGR","AON","MET","TFC","USB",
        "PNC","COF","DFS","FITB","KEY",
    ],
    "Healthcare": [
        "JNJ","UNH","PFE","ABBV","MRK","LLY","BMY","AMGN","GILD","MDT",
        "TMO","ABT","ISRG","SYK","BSX","ELV","CVS","HUM","CI","DHR",
        "VRTX","REGN","ZBH","BDX","IQV",
    ],
    "Consumer Discretionary": [
        "MCD","NKE","TSCO","HD","LOW","SBUX","TGT","CMG","YUM","BKNG",
        "MAR","HLT","F","GM","ORLY","AZO","ROST","TJX","VFC","PVH",
    ],
    "Consumer Staples": [
        "WMT","KO","PG","PEP","COST","CL","KMB","GIS","MO","PM",
        "MDLZ","KHC","SYY","CHD","CAG",
    ],
    "Industrials": [
        "BA","CAT","HON","GE","LMT","RTX","DE","UNP","UPS","FDX",
        "MMM","EMR","ETN","GD","NOC","PH","ROK","TDG","CTAS","RSG",
    ],
    "Energy": [
        "XOM","CVX","COP","SLB","EOG","MPC","OXY","HAL","PSX","VLO",
        "HES","DVN","BKR","FANG","MRO",
    ],
    "Communication": [
        "T","VZ","CMCSA","DIS","CHTR","WBD","PARA","FOXA","OMC","IPG",
    ],
    "Real Estate": [
        "AMT","PLD","CCI","EQIX","SPG","O","WELL","AVB","EQR","PSA",
    ],
    "Materials": [
        "LIN","APD","SHW","NEM","FCX","NUE","PPG","ALB","CF","MOS",
    ],
    "Utilities": [
        "NEE","DUK","SO","AEP","D","EXC","PCG","XEL","ES","AWK",
    ],
}
ALL_TICKERS   = [t for group in UNIVERSE.values() for t in group]
TICKER_SECTOR = {t: sec for sec, tickers in UNIVERSE.items() for t in tickers}

# Approximate S&P 500 GICS sector weights (as of 2025)
SPY_SECTOR_WEIGHTS = {
    "Technology": 0.310, "Financials": 0.134, "Healthcare": 0.118,
    "Consumer Discretionary": 0.104, "Communication": 0.090,
    "Industrials": 0.088, "Consumer Staples": 0.058, "Energy": 0.038,
    "Utilities": 0.024, "Real Estate": 0.022, "Materials": 0.022,
}

CHART_TEMPLATE = "plotly_dark"
# Bloomberg Terminal colour palette
ACCENT = "#FF8C00"   # Bloomberg orange  (primary highlight)
BLUE   = "#00A8E8"   # Bloomberg blue    (secondary)
RED    = "#E53935"   # Bloomberg red     (losses / risk)
GREEN  = "#00C853"   # Bloomberg green   (gains / positive)
GOLD   = "#FFD600"   # Bloomberg yellow  (caution / neutral)
MUTED  = "#78909C"   # blue-grey         (secondary text)

COMPANY_NAMES = {
    # Technology
    "AAPL":"Apple",                 "MSFT":"Microsoft",           "GOOGL":"Alphabet",
    "AMZN":"Amazon",                "META":"Meta Platforms",       "NVDA":"NVIDIA",
    "TSLA":"Tesla",                 "NFLX":"Netflix",              "AMD":"AMD",
    "ORCL":"Oracle",                "CRM":"Salesforce",            "ADBE":"Adobe",
    "INTC":"Intel",                 "QCOM":"Qualcomm",             "AVGO":"Broadcom",
    "TXN":"Texas Instruments",      "NOW":"ServiceNow",            "PANW":"Palo Alto Networks",
    "INTU":"Intuit",                "CSCO":"Cisco",                "IBM":"IBM",
    "ACN":"Accenture",              "AMAT":"Applied Materials",    "MU":"Micron Technology",
    "KLAC":"KLA Corp",              "LRCX":"Lam Research",         "SNPS":"Synopsys",
    "CDNS":"Cadence Design",        "FTNT":"Fortinet",             "HPQ":"HP Inc.",
    # Financials
    "JPM":"JPMorgan Chase",         "GS":"Goldman Sachs",          "BAC":"Bank of America",
    "MS":"Morgan Stanley",          "BLK":"BlackRock",             "V":"Visa",
    "MA":"Mastercard",              "C":"Citigroup",               "WFC":"Wells Fargo",
    "AXP":"American Express",       "SCHW":"Charles Schwab",       "SPGI":"S&P Global",
    "ICE":"Intercontinental Exch.", "CME":"CME Group",             "MCO":"Moody's",
    "PGR":"Progressive",            "AON":"Aon",                   "MET":"MetLife",
    "TFC":"Truist Financial",       "USB":"U.S. Bancorp",          "PNC":"PNC Financial",
    "COF":"Capital One",            "DFS":"Discover Financial",    "FITB":"Fifth Third Bancorp",
    "KEY":"KeyCorp",
    # Healthcare
    "JNJ":"Johnson & Johnson",      "UNH":"UnitedHealth",          "PFE":"Pfizer",
    "ABBV":"AbbVie",                "MRK":"Merck",                 "LLY":"Eli Lilly",
    "BMY":"Bristol-Myers Squibb",   "AMGN":"Amgen",                "GILD":"Gilead Sciences",
    "MDT":"Medtronic",              "TMO":"Thermo Fisher",         "ABT":"Abbott Labs",
    "ISRG":"Intuitive Surgical",    "SYK":"Stryker",               "BSX":"Boston Scientific",
    "ELV":"Elevance Health",        "CVS":"CVS Health",            "HUM":"Humana",
    "CI":"Cigna",                   "DHR":"Danaher",               "VRTX":"Vertex Pharma.",
    "REGN":"Regeneron",             "ZBH":"Zimmer Biomet",         "BDX":"Becton Dickinson",
    "IQV":"IQVIA",
    # Consumer Discretionary
    "MCD":"McDonald's",             "NKE":"Nike",                  "TSCO":"Tractor Supply",
    "HD":"Home Depot",              "LOW":"Lowe's",                "SBUX":"Starbucks",
    "TGT":"Target",                 "CMG":"Chipotle",              "YUM":"Yum! Brands",
    "BKNG":"Booking Holdings",      "MAR":"Marriott",              "HLT":"Hilton",
    "F":"Ford",                     "GM":"General Motors",         "ORLY":"O'Reilly Auto Parts",
    "AZO":"AutoZone",               "ROST":"Ross Stores",          "TJX":"TJX Companies",
    "VFC":"VF Corp",                "PVH":"PVH Corp",
    # Consumer Staples
    "WMT":"Walmart",                "KO":"Coca-Cola",              "PG":"Procter & Gamble",
    "PEP":"PepsiCo",                "COST":"Costco",               "CL":"Colgate-Palmolive",
    "KMB":"Kimberly-Clark",         "GIS":"General Mills",         "MO":"Altria",
    "PM":"Philip Morris",           "MDLZ":"Mondelez",             "KHC":"Kraft Heinz",
    "SYY":"Sysco",                  "CHD":"Church & Dwight",       "CAG":"Conagra Brands",
    # Industrials
    "BA":"Boeing",                  "CAT":"Caterpillar",           "HON":"Honeywell",
    "GE":"GE Aerospace",            "LMT":"Lockheed Martin",       "RTX":"RTX Corp",
    "DE":"Deere & Co",              "UNP":"Union Pacific",         "UPS":"UPS",
    "FDX":"FedEx",                  "MMM":"3M",                    "EMR":"Emerson Electric",
    "ETN":"Eaton",                  "GD":"General Dynamics",       "NOC":"Northrop Grumman",
    "PH":"Parker Hannifin",         "ROK":"Rockwell Automation",   "TDG":"TransDigm",
    "CTAS":"Cintas",                "RSG":"Republic Services",
    # Energy
    "XOM":"ExxonMobil",             "CVX":"Chevron",               "COP":"ConocoPhillips",
    "SLB":"SLB",                    "EOG":"EOG Resources",         "MPC":"Marathon Petroleum",
    "OXY":"Occidental Petroleum",   "HAL":"Halliburton",           "PSX":"Phillips 66",
    "VLO":"Valero Energy",          "HES":"Hess",                  "DVN":"Devon Energy",
    "BKR":"Baker Hughes",           "FANG":"Diamondback Energy",   "MRO":"Marathon Oil",
    # Communication
    "T":"AT&T",                     "VZ":"Verizon",                "CMCSA":"Comcast",
    "DIS":"Walt Disney",            "CHTR":"Charter Comm.",        "WBD":"Warner Bros. Discovery",
    "PARA":"Paramount Global",      "FOXA":"Fox Corp",             "OMC":"Omnicom",
    "IPG":"Interpublic Group",
    # Real Estate
    "AMT":"American Tower",         "PLD":"Prologis",              "CCI":"Crown Castle",
    "EQIX":"Equinix",               "SPG":"Simon Property Group",  "O":"Realty Income",
    "WELL":"Welltower",             "AVB":"AvalonBay Communities", "EQR":"Equity Residential",
    "PSA":"Public Storage",
    # Materials
    "LIN":"Linde",                  "APD":"Air Products",          "SHW":"Sherwin-Williams",
    "NEM":"Newmont",                "FCX":"Freeport-McMoRan",      "NUE":"Nucor",
    "PPG":"PPG Industries",         "ALB":"Albemarle",             "CF":"CF Industries",
    "MOS":"Mosaic",
    # Utilities
    "NEE":"NextEra Energy",         "DUK":"Duke Energy",           "SO":"Southern Company",
    "AEP":"American Electric Power","D":"Dominion Energy",         "EXC":"Exelon",
    "PCG":"PG&E",                   "XEL":"Xcel Energy",           "ES":"Eversource Energy",
    "AWK":"American Water Works",
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER  —  Yahoo Finance (yfinance), no API key required
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _yf_close(tickers_csv: str, start: str, end: str) -> pd.DataFrame:
    """Batch-download adjusted close prices for multiple tickers in one call."""
    tickers = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False)
    except Exception as e:
        st.warning(f"Download failed: {e}")
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Close"].dropna(how="all")
    # Single-ticker download returns flat columns
    return pd.DataFrame({tickers[0]: raw["Close"]}).dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def _yf_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download full OHLCV for a single ticker (Technical Analysis tab)."""
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.xs(ticker, axis=1, level=1)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    return raw[cols].dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def log_returns(p: pd.Series) -> pd.Series:
    """Compute continuously compounded (log) daily returns: ln(P_t / P_{t-1})."""
    return np.log(p / p.shift(1)).dropna()

def simple_returns(p: pd.Series) -> pd.Series:
    """Compute simple daily percentage returns: (P_t - P_{t-1}) / P_{t-1}."""
    return p.pct_change().dropna()

def ann_return(r: pd.Series) -> float:
    """Annualise a daily return series by multiplying the mean by 252.

    Note: this is a first-order approximation. For more precision over long
    horizons use (1+r).prod()^(252/n) - 1, but the difference is small for
    typical daily return magnitudes.
    """
    return r.mean() * AF

def ann_vol(r: pd.Series) -> float:
    """Annualise daily return volatility using the square-root-of-time rule."""
    return r.std() * np.sqrt(AF)

def sharpe(r: pd.Series, rf: float = RF) -> float:
    """Annualised Sharpe ratio: (ann_return - rf) / ann_vol.

    Returns np.nan if annualised volatility is zero (avoids division by zero).
    """
    v = ann_vol(r)
    return (ann_return(r) - rf) / v if v else np.nan

def sortino(r: pd.Series, rf: float = RF) -> float:
    """Annualised Sortino ratio: uses downside deviation (negative returns only).

    Penalises downside volatility only, unlike Sharpe which counts all vol.
    Returns np.nan if there are no negative returns.
    """
    dd = r[r < 0].std() * np.sqrt(AF)
    return (ann_return(r) - rf) / dd if dd else np.nan

def max_dd(prices: pd.Series) -> float:
    """Maximum drawdown: largest peak-to-trough decline in the price series.

    Returns a negative float (e.g. -0.35 means a 35% max drawdown).
    """
    return ((prices - prices.cummax()) / prices.cummax()).min()

def dd_series(prices: pd.Series) -> pd.Series:
    """Rolling drawdown series: current level vs running peak at each date."""
    return (prices - prices.cummax()) / prices.cummax()

def calmar(r: pd.Series, prices: pd.Series) -> float:
    """Calmar ratio: annualised return divided by absolute maximum drawdown.

    Higher is better. Returns np.nan if max drawdown is zero.
    """
    mdd = abs(max_dd(prices))
    return ann_return(r) / mdd if mdd else np.nan

def hist_var(r: pd.Series, c: float = 0.95) -> float:
    """Historical VaR at confidence level c.

    Reads directly from the empirical return distribution: the loss level
    exceeded on only (1-c) of trading days. Returned as a positive number
    (e.g. 0.02 means a 2% daily loss threshold).
    """
    return -np.percentile(r.dropna(), (1 - c) * 100)

def param_var(r: pd.Series, c: float = 0.95) -> float:
    """Parametric (Gaussian) VaR: assumes normally distributed daily returns.

    VaR = -(mu + z_{1-c} * sigma), where z is the standard normal quantile.
    Returned as a positive number. Underestimates tail risk for fat-tailed assets.
    """
    return -(r.mean() + norm.ppf(1 - c) * r.std())

def cvar(r: pd.Series, c: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) at confidence level c.

    Average loss on the days that exceed the VaR threshold. Always >= VaR;
    more sensitive to extreme tail events than VaR alone.
    """
    v = hist_var(r, c)
    tail = r[r < -v]
    return -tail.mean() if len(tail) else v

def beta_alpha(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Compute CAPM beta and Jensen's alpha vs a benchmark return series.

    Beta = Cov(r, bench) / Var(bench).
    Alpha = ann_return(r) - rf - beta * (ann_return(bench) - rf), annualised.
    Returns (nan, nan) if fewer than 10 overlapping observations.
    """
    aligned = pd.concat([r, bench], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan, np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    b = cov[0, 1] / cov[1, 1]
    a = ann_return(aligned.iloc[:, 0]) - RF - b * (ann_return(aligned.iloc[:, 1]) - RF)
    return b, a

def win_rate(r: pd.Series) -> float:
    """Fraction of trading days with a positive return."""
    return (r > 0).mean()

def _fmt_num(val, fmt: str) -> str:
    """Format a numeric scalar using the given format string.

    Returns '-' for NaN, inf, or any non-numeric input, preventing 'nan'
    strings from appearing in display tables.
    """
    try:
        if np.isnan(val) or np.isinf(val):
            return "-"
        return fmt.format(val)
    except (TypeError, ValueError):
        return "-"

def full_metrics(r: pd.Series, prices: pd.Series, bench_r: pd.Series,
                 conf: float = 0.95, port_w: np.ndarray | None = None,
                 bench_w: np.ndarray | None = None) -> dict:
    """Compute the full performance metrics dictionary for display in the Overview table.

    Args:
        r:       Daily simple return series for the portfolio.
        prices:  Cumulative price/value series (used for drawdown and Calmar).
        bench_r: Benchmark daily return series (SPY) for Beta, Alpha, TE, IR.
        conf:    VaR / CVaR confidence level (default 0.95).
        port_w:  Portfolio weight vector — required for Active Share calculation.
        bench_w: Benchmark weight vector — required for Active Share calculation.

    Returns:
        Ordered dict of pre-formatted metric strings ready for table display.
        NaN-safe: all ratios use _fmt_num() to return '-' instead of 'nan'.
    """
    b, a  = beta_alpha(r, bench_r)
    te    = tracking_error(r, bench_r)
    ir    = information_ratio(r, bench_r)
    ash   = active_share(port_w, bench_w) if port_w is not None and bench_w is not None else np.nan
    return {
        "Ann. Return":       f"{ann_return(r):.2%}",
        "Ann. Volatility":   f"{ann_vol(r):.2%}",
        "Sharpe Ratio":      _fmt_num(sharpe(r),  "{:.2f}"),
        "Sortino Ratio":     _fmt_num(sortino(r), "{:.2f}"),
        "Calmar Ratio":      _fmt_num(calmar(r, prices), "{:.2f}"),
        "Max Drawdown":      f"{max_dd(prices):.2%}",
        "Beta (vs SPY)":     _fmt_num(b, "{:.2f}"),
        "Alpha (vs SPY)":    _fmt_num(a, "{:.2%}"),
        # Tracking Error: show '-' when comparing a series to itself (TE = 0)
        "Tracking Error":    _fmt_num(te, "{:.2%}") if te > 1e-10 else "-",
        "Information Ratio": _fmt_num(ir, "{:.2f}"),
        "Active Share":      _fmt_num(ash, "{:.1%}"),
        "Win Rate":          f"{win_rate(r):.1%}",
        f"VaR {conf:.0%}":   f"{hist_var(r, conf):.2%}",
        f"CVaR {conf:.0%}":  f"{cvar(r, conf):.2%}",
        "Skewness":          f"{r.skew():.2f}",
        "Kurtosis":          f"{r.kurtosis():.2f}",
    }

def tracking_error(r: pd.Series, bench_r: pd.Series) -> float:
    """Annualised tracking error: std dev of daily active returns (r - bench)."""
    active = r.subtract(bench_r, fill_value=0)
    return active.std() * np.sqrt(AF)

def information_ratio(r: pd.Series, bench_r: pd.Series) -> float:
    """Information ratio: annualised active return divided by tracking error.

    Measures how consistently the portfolio outperforms its benchmark per unit
    of active risk. Returns np.nan if tracking error is zero.
    """
    active = r.subtract(bench_r, fill_value=0)
    te = tracking_error(r, bench_r)
    return ann_return(active) / te if te else np.nan

def active_share(port_w: np.ndarray, bench_w: np.ndarray) -> float:
    """Active Share vs a benchmark weight vector: 0.5 * sum|w_port - w_bench|.

    Ranges from 0 (identical to benchmark) to 1 (no overlap). Both arrays
    must cover the same asset universe in the same order.
    """
    return 0.5 * np.sum(np.abs(port_w - bench_w))

def hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index: sum of squared portfolio weights.

    HHI = 1 for a single-stock portfolio; 1/N for equal weight.
    Effective number of positions = 1/HHI (inverse HHI).
    """
    return float(np.sum(np.array(weights) ** 2))

def rolling_vol(r: pd.Series, w: int = 21) -> pd.Series:
    """Rolling annualised volatility over a window of w trading days."""
    return r.rolling(w, min_periods=5).std() * np.sqrt(AF)

def rolling_sharpe(r: pd.Series, w: int = 252, rf: float = RF) -> pd.Series:
    """Rolling annualised Sharpe ratio over a trailing window of w trading days.

    min_periods is set to max(21, w//4) so the series starts as soon as
    at least one quarter of the window is filled, avoiding misleading early values.
    """
    roll = r.rolling(w, min_periods=max(21, w // 4))
    return (roll.mean() * AF - rf) / (roll.std() * np.sqrt(AF))

# ── Technical indicators ──────────────────────────────────────────────────────
def bollinger(p, w=20, n=2):
    """Compute Bollinger Bands: (upper, middle, lower) = MA ± n * rolling_std."""
    m = p.rolling(w).mean()
    s = p.rolling(w).std()
    return m + n*s, m, m - n*s

def rsi(p, w=14):
    """Compute RSI (Relative Strength Index) over w periods.

    RSI = 100 - 100 / (1 + avg_gain / avg_loss).
    Values above 70 indicate overbought; below 30 indicate oversold.
    """
    d = p.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + g / l)

def macd(p, fast=12, slow=26, sig=9):
    """Compute MACD line, signal line, and histogram.

    MACD line  = EMA(fast) - EMA(slow)   (default: 12-day minus 26-day)
    Signal line = EMA(MACD, sig)          (default: 9-day)
    Histogram  = MACD line - Signal line  (positive = bullish momentum)

    Returns: (macd_line, signal_line, histogram) as pd.Series.
    """
    ml = p.ewm(span=fast, adjust=False).mean() - p.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

# ── Portfolio optimisation ────────────────────────────────────────────────────
# Maximum weight any single asset may receive in optimised portfolios.
# Prevents the solver from crowding into one stock that happened to dominate
# the historical sample (e.g. NVDA post-2020). 40% is a standard long-only cap.
MAX_SINGLE_W = 0.40

def _port_stats(w, mu, cov, rf=RF):
    """Compute annualised portfolio return, volatility, and Sharpe ratio.

    Args:
        w:   Weight vector (sums to 1).
        mu:  Daily mean return vector.
        cov: Daily covariance matrix.
        rf:  Annualised risk-free rate (default: global RF constant).

    Returns:
        (ann_return, ann_vol, sharpe) — all annualised.
    """
    r = np.dot(w, mu) * AF
    v = np.sqrt(w @ cov @ w) * np.sqrt(AF)
    return r, v, (r - rf) / v if v else np.nan

def _opt(objective, n, extra_constraints=None):
    """Run SLSQP optimisation with full-investment and per-stock cap constraints.

    Args:
        objective:         Callable f(w) to minimise.
        n:                 Number of assets.
        extra_constraints: Optional list of additional scipy constraint dicts.

    Returns:
        Optimal weight vector as a numpy array.

    Constraints enforced:
        - Weights sum to 1 (fully invested, no cash).
        - 0 <= w_i <= MAX_SINGLE_W = 40% (long-only with concentration cap).
    """
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if extra_constraints:
        cons += extra_constraints
    res = minimize(objective, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, MAX_SINGLE_W)]*n, constraints=cons)
    return res.x

def w_min_var(mu, cov):
    """Minimum Variance portfolio: minimise w'Σw subject to full-investment + cap.

    Only the covariance matrix is needed — expected returns are not used,
    which makes this strategy less sensitive to estimation error than Max Sharpe.
    Zero weights are normal: the optimiser excludes high-variance assets.
    """
    return _opt(lambda w: _port_stats(w, mu, cov)[1], len(mu))

def w_max_sharpe(mu, cov, rf=RF):
    """Maximum Sharpe (Tangency) portfolio: maximise (r - rf) / vol.

    Uses the historically correct risk-free rate when called from the backtest,
    eliminating look-ahead bias in the Sharpe optimisation.
    Limitation: sample means are noisy estimators, so the optimiser tends to
    concentrate in whichever stock had the highest in-sample Sharpe ratio.
    """
    return _opt(lambda w: -_port_stats(w, mu, cov, rf)[2], len(mu))

def w_inv_vol(rets_df):
    """Inverse-volatility (Risk Parity) weights: w_i = (1/sigma_i) / sum(1/sigma_j).

    Lower-volatility assets receive higher weights so each position contributes
    roughly equal risk. Volatility is estimated from the expanding window in the
    backtest, so early estimates are noisy but improve as data accumulates.
    No return forecasts are needed — only the per-asset volatility.
    """
    vols = rets_df.std()
    iv = 1 / vols
    return (iv / iv.sum()).values

def w_equal(n):
    """Equal-weight portfolio: w_i = 1/N for all i."""
    return np.ones(n) / n

@st.cache_data(ttl=86400, show_spinner=False)
def w_market_cap(tickers_csv: str) -> np.ndarray:
    """Fetch CURRENT market-cap weights via yfinance fast_info.

    NOTE: This function returns TODAY's market-cap weights. It is used only for
    (1) display in the Optimal Portfolio Allocations table and (2) as the active-
    share benchmark in full_metrics(). It is NOT used as starting weights in the
    backtest — those are derived from historical prices to avoid look-ahead bias.
    Falls back to equal weight (cap=1.0) for any ticker that fails to load.
    Cached for 24 hours (ttl=86400) to limit API calls.
    """
    tickers = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    caps = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            caps.append(getattr(info, "market_cap", None) or 1.0)
        except Exception:
            caps.append(1.0)
    caps = np.array(caps, dtype=float)
    caps = np.where(caps <= 0, 1.0, caps)
    return caps / caps.sum()

@st.cache_data(ttl=3600, show_spinner=False)
def load_rf_series(start: str, end: str) -> pd.Series:
    """Download the 3-month US T-bill rate (^IRX) as a decimal annualised series.

    Used to supply the historically correct risk-free rate to the Max-Sharpe
    optimiser at each backtest rebalancing date. Using a fixed RF rate for all
    periods would introduce look-ahead bias (e.g. 5.25% RF applied to 2010-2020
    when rates were near zero would systematically depress Sharpe estimates).

    At each rebalancing date the backtest calls rf_series.asof(rebal) which
    returns the last known T-bill rate on or before that date — no look-ahead.
    Falls back to the global RF constant if ^IRX data is unavailable.
    Cached for 1 hour (ttl=3600).
    """
    try:
        raw = yf.download("^IRX", start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.Series(dtype=float)
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return (close / 100).dropna()
    except Exception:
        return pd.Series(dtype=float)

def efficient_frontier_mc(mu, cov, n=500):
    """Sample n random portfolios via Dirichlet weights to map the feasible set.

    Dirichlet(ones) produces uniformly distributed weight vectors on the simplex
    (all weights positive, sum to 1). Each portfolio's (vol, return, Sharpe) is
    computed so the scatter cloud can be colour-coded by Sharpe ratio.

    Note: these are NOT constrained by MAX_SINGLE_W — they represent the full
    theoretical feasible set for illustration. The optimised portfolios on the
    efficient frontier DO respect the 40% cap.

    Returns: (vols, returns, sharpes, weights_list) as numpy arrays.
    """
    n_assets = len(mu)
    vols, rets, srs, wts = [], [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = _port_stats(w, mu, cov)
        vols.append(v); rets.append(r); srs.append(s); wts.append(w)
    return np.array(vols), np.array(rets), np.array(srs), wts

# ── Portfolio backtest ────────────────────────────────────────────────────────
def backtest_styles(prices: pd.DataFrame,
                    min_lookback: int = 63,
                    rf_series: pd.Series | None = None,
                    cost_pct: float = 0.0) -> dict[str, pd.Series]:
    """
    Walk-forward backtest with monthly rebalancing for EW, Min Variance,
    Mean-Variance and Risk Parity (inverse-vol). Market Weight is handled
    separately as a true buy-and-hold (no rebalancing) starting from day 1.

    min_lookback (default 63 ≈ 3 months): minimum number of daily returns
    required before the optimiser is trusted. Below this threshold all
    optimised strategies fall back to equal weight.

    rf_series: annualised T-bill rate as a decimal time series. When provided,
    the Max-Sharpe optimiser uses the historically correct RF rate at each
    rebalancing date rather than the fixed global constant.

    cost_pct: one-way transaction cost as a percentage of the amount traded
    at each rebalancing (e.g. 0.05 = 5 bps per unit of one-way turnover).
    At each rebalancing date the one-way turnover is computed as:
        TO = 0.5 × Σ|w_new_i − w_drifted_i|
    where w_drifted are the weights that have naturally drifted from the
    previous rebalancing due to price moves. The cost deducted is:
        cost_hit = TO × cost_pct / 100
    applied as a return reduction on the first day of the holding period.
    Market Weight is exempt (true buy-and-hold, zero rebalancing turnover).
    The first rebalancing date is treated as portfolio inception — no turnover
    is assumed at t=0 (cost only accrues from the second rebalancing onward).
    """
    rets = prices.pct_change().dropna()
    month_starts = rets.resample("MS").first().index.tolist()
    rebal_dates  = [d for d in month_starts if d in rets.index or d > rets.index[0]]

    # Only the four actively managed styles use monthly rebalancing
    style_names = ["Equal Weight", "Min Variance", "Mean-Variance", "Risk Parity"]
    port_rets   = {s: pd.Series(dtype=float) for s in style_names}
    # Track drifted weights at the END of each holding period so we can compute
    # turnover at the NEXT rebalancing.  Initialised to None (= inception, no cost).
    prev_w = {s: None for s in style_names}

    for i, rebal in enumerate(rebal_dates):
        hold_end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rets.index[-1]
        period   = rets.loc[rebal:hold_end].iloc[1:]   # skip same-day row
        if period.empty:
            continue

        # Expanding-window history: only past data, no look-ahead
        hist = rets.loc[:rebal]
        mu   = hist.mean()
        cov  = hist.cov()
        n    = len(rets.columns)

        # Historically correct RF rate at this rebalancing date (no look-ahead:
        # asof() returns the last known value on or before the rebal date)
        if rf_series is not None and not rf_series.empty and rebal >= rf_series.index[0]:
            rf_t = float(rf_series.asof(rebal))
        else:
            rf_t = RF

        if len(hist) >= min_lookback:
            weights = {
                "Equal Weight":  w_equal(n),
                # Min Variance: objective uses only cov, mu is ignored
                "Min Variance":  w_min_var(mu, cov),
                # Mean-Variance: uses hist.mean() as expected returns —
                # no look-ahead, but sample means are noisy estimators.
                # RF rate is the historically correct T-bill rate at rebal date.
                "Mean-Variance": w_max_sharpe(mu, cov, rf=rf_t),
                # Inverse-vol weighting (simplified Risk Parity):
                # vol estimated from expanding window; improves as data accumulates
                "Risk Parity":   w_inv_vol(hist),
            }
        else:
            # Too few observations — fall back to equal weight
            ew = w_equal(n)
            weights = {s: ew for s in style_names}

        for s in style_names:
            new_w = weights[s]
            p_ret = (period * new_w).sum(axis=1).copy()

            # ── Transaction cost ─────────────────────────────────────────────
            # Deduct cost on the first day of each holding period.
            # Inception (prev_w is None) is assumed zero-turnover.
            if cost_pct > 0.0 and prev_w[s] is not None and len(p_ret) > 0:
                one_way_to = 0.5 * float(np.abs(new_w - prev_w[s]).sum())
                p_ret.iloc[0] -= one_way_to * cost_pct / 100.0

            # ── Update drifted weights for next rebalancing ──────────────────
            # Weights drift as individual prices move over the holding period.
            # Drifted weight_i ∝ new_w_i × ∏(1 + r_i) over the holding period.
            period_growth = (1.0 + period).prod()   # cumulative return per stock
            raw = new_w * period_growth.values
            prev_w[s] = raw / raw.sum() if raw.sum() > 0 else new_w.copy()

            port_rets[s] = pd.concat([port_rets[s], p_ret])

    # ── Market Weight: true buy-and-hold, no rebalancing ─────────────────────
    # A MCW portfolio never needs rebalancing: holding each stock proportional
    # to its market cap at t=0 means price appreciation automatically keeps
    # the portfolio at MCW proportions (this is why passive index funds are
    # low-turnover).
    #
    # Historical starting weights: yfinance only provides current market caps,
    # but we can infer implied shares outstanding:
    #   shares_i  ≈  current_mcap_i / current_price_i
    # Then the market-cap weight at the backtest start date is:
    #   w_i(t_0)  ∝  shares_i × price_i(t_0)
    # This removes look-ahead bias — the starting weights reflect relative
    # market caps on the first day of the backtest, not today.
    tickers_list   = rets.columns.tolist()
    mcw_today      = w_market_cap(",".join(tickers_list))       # today's normalised weights ∝ today's caps
    current_prices = prices[tickers_list].iloc[-1].values.astype(float)
    start_prices   = prices[tickers_list].iloc[0].values.astype(float)
    # avoid division by zero
    current_prices = np.where(current_prices > 0, current_prices, 1.0)
    start_prices   = np.where(start_prices   > 0, start_prices,   1.0)
    implied_shares = mcw_today / current_prices          # ∝ actual shares outstanding
    hist_caps_t0   = implied_shares * start_prices       # ∝ market cap at t=0
    hist_caps_t0   = np.where(hist_caps_t0 > 0, hist_caps_t0, 1e-10)
    mcw_w0         = hist_caps_t0 / hist_caps_t0.sum()
    mcw_val = (mcw_w0 * (1 + rets).cumprod()).sum(axis=1)
    port_rets["Market Weight"] = mcw_val.pct_change().dropna()

    return port_rets

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def mc_paths(last_val, daily_rets, n_sim=300, n_days=252):
    # Drift set to zero: simulation models volatility / uncertainty only,
    # not the in-sample historical trend. This avoids falsely extrapolating
    # a bull-market period and produces realistic downside paths.
    sigma = daily_rets.std()
    paths = np.zeros((n_days, n_sim))
    paths[0] = last_val
    shocks = np.random.normal(0, sigma, (n_days - 1, n_sim))
    for t in range(1, n_days):
        paths[t] = paths[t - 1] * np.exp(shocks[t - 1])
    return paths

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_FONT  = "IBM Plex Serif, Georgia, serif"
_MONO  = "IBM Plex Mono, Courier New, monospace"
_SERIF = "IBM Plex Serif, Georgia, serif"
_BG    = "#0A0C10"   # paper / outer background (Bloomberg near-black)
_PLOT  = "#0E1117"   # inner plot area
_GRID  = "#181D24"   # gridlines (very subtle)
_AXIS  = "#263238"   # axis lines
_TICK  = "#546E7A"   # tick labels

def _layout(fig, title="", h=450, **kw):
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT,
        title=dict(text=f"<b>{title}</b>" if title else "",
                   font=dict(size=14, color="#E0E4EA", family=_SERIF),
                   pad=dict(b=10)),
        height=h,
        hovermode="x unified",
        font=dict(family=_FONT, color="#78909C", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#78909C", size=11, family=_MONO)),
        hoverlabel=dict(bgcolor="#111519", bordercolor="#263238",
                        font=dict(family=_MONO, size=12, color="#E0E4EA")),
        **kw
    )
    fig.update_xaxes(gridcolor=_GRID, linecolor=_AXIS, zerolinecolor=_AXIS,
                     tickfont=dict(color=_TICK, size=11, family=_MONO))
    fig.update_yaxes(gridcolor=_GRID, linecolor=_AXIS, zerolinecolor=_AXIS,
                     tickfont=dict(color=_TICK, size=11, family=_MONO))
    return fig


def chart_cumret(series_dict: dict, title="Cumulative Returns (rebased to 0%)") -> go.Figure:
    # Professional Tailwind-based palette — readable on dark backgrounds
    colors = [ACCENT, BLUE, GOLD, GREEN, "#8B5CF6", "#EC4899",
              "#F97316", "#14B8A6", "#A78BFA", "#94A3B8"]
    fig = go.Figure()
    for i, (name, p) in enumerate(series_dict.items()):
        p = p.dropna()
        rebased = (p / p.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased,
                                 name=name, line=dict(color=colors[i % len(colors)], width=2),
                                 hovertemplate="%{y:.2f}%<extra>" + name + "</extra>"))
    return _layout(fig, title, yaxis_title="Return (%)")


def chart_sector_exposure(tickers: list, weights: np.ndarray) -> go.Figure:
    """Grouped bar: portfolio sector weights vs approximate S&P 500 sector weights."""
    sec_w: dict[str, float] = {}
    for t, w in zip(tickers, weights):
        s = TICKER_SECTOR.get(t, "Other")
        sec_w[s] = sec_w.get(s, 0.0) + w

    sectors = sorted(sec_w.keys(), key=lambda s: -sec_w[s])
    port_pct = [sec_w.get(s, 0) * 100 for s in sectors]
    spy_pct  = [SPY_SECTOR_WEIGHTS.get(s, 0) * 100 for s in sectors]
    active   = [p - b for p, b in zip(port_pct, spy_pct)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Your Portfolio", x=sectors, y=port_pct,
                         marker_color=ACCENT, opacity=0.9))
    fig.add_trace(go.Bar(name="S&P 500 (SPY)", x=sectors, y=spy_pct,
                         marker_color=MUTED, opacity=0.7))
    fig.add_trace(go.Scatter(name="Active Weight", x=sectors, y=active,
                             mode="markers",
                             marker=dict(color=[GREEN if v >= 0 else RED
                                                for v in active], size=9,
                                         symbol="diamond"),
                             hovertemplate="%{y:+.1f}%<extra>Active</extra>"))
    fig.update_layout(barmode="group")
    return _layout(fig, "Sector Exposure vs S&P 500",
                   yaxis_title="Weight (%)", h=420)


def chart_benchmark_comparison(port_r, bench_prices, label="Equal-Weight Portfolio") -> go.Figure:
    """
    Build a benchmark comparison chart.
    Passes raw cumulative-product series to chart_cumret, which handles the
    rebase-to-0% step.  (Previously the rebase was done here *and* again inside
    chart_cumret, causing division by zero that flattened all lines.)
    """
    raw: dict[str, pd.Series] = {label: (1 + port_r).cumprod()}

    if "SPY" in bench_prices.columns:
        spy_r = bench_prices["SPY"].pct_change().dropna()
        raw["S&P 500 (SPY)"] = (1 + spy_r).cumprod()

    if "SPY" in bench_prices.columns and "AGG" in bench_prices.columns:
        spy_r2 = bench_prices["SPY"].pct_change().dropna()
        agg_r  = bench_prices["AGG"].pct_change().dropna()
        bal_r  = (0.6 * spy_r2 + 0.4 * agg_r).dropna()
        raw["60/40 (SPY+AGG)"] = (1 + bal_r).cumprod()

    if "AGG" in bench_prices.columns:
        agg_r2 = bench_prices["AGG"].pct_change().dropna()
        raw["US Bonds (AGG)"] = (1 + agg_r2).cumprod()

    # Align all series to a common start date (latest of first valid dates)
    valid = {n: s.dropna() for n, s in raw.items() if not s.dropna().empty}
    if not valid:
        return go.Figure()
    common_start = max(s.index[0] for s in valid.values())
    aligned = {n: s[s.index >= common_start] for n, s in valid.items()}

    # chart_cumret will rebase each series to 0% — pass raw cumulative products
    return chart_cumret(aligned, "Portfolio vs Benchmarks (rebased to 0%)")


def chart_price_ta(ohlcv: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=ohlcv.index, open=ohlcv["Open"], high=ohlcv["High"],
        low=ohlcv["Low"], close=ohlcv["Close"], name="OHLC",
        increasing_line_color=ACCENT, decreasing_line_color=RED,
    ), row=1, col=1)
    for period, color in [(20, GOLD), (50, BLUE)]:
        ma = ohlcv["Close"].rolling(period).mean()
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ma, name=f"MA{period}",
                                 line=dict(color=color, width=1.2)), row=1, col=1)
    upper, mid, lower = bollinger(ohlcv["Close"])
    for band, name in [(upper, "BB Upper"), (lower, "BB Lower")]:
        fig.add_trace(go.Scatter(x=ohlcv.index, y=band, name=name,
                                 line=dict(color="rgba(150,150,255,0.5)", dash="dot", width=1),
                                 showlegend=True), row=1, col=1)
    # Dollar volume (shares × price) — immune to split-adjustment inflation.
    # yfinance auto_adjust multiplies historical share counts by the cumulative
    # split factor, making pre-split volume look enormous vs today. Dollar volume
    # cancels this out and is a more meaningful measure of daily liquidity.
    dollar_vol = ohlcv["Volume"] * ohlcv["Close"]
    vol_colors = [GREEN if c >= o else RED
                  for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
    fig.add_trace(go.Bar(x=ohlcv.index, y=dollar_vol, name="Dollar Volume",
                         marker_color=vol_colors, opacity=0.65), row=2, col=1)
    fig.update_yaxes(tickprefix="$", tickformat=".2s", row=2, col=1)
    fig.update_layout(template=CHART_TEMPLATE,
                      paper_bgcolor=_BG, plot_bgcolor=_PLOT,
                      title=dict(text=f"<b>{ticker} — Price & Volume</b>",
                                 font=dict(size=14, color="#E0E4EA", family=_SERIF)),
                      xaxis_rangeslider_visible=False, height=600,
                      font=dict(family=_FONT, color="#78909C"),
                      hoverlabel=dict(bgcolor="#111519", bordercolor="#263238",
                                      font=dict(family=_MONO, size=12)),
                      legend=dict(orientation="h", font=dict(color="#78909C", family=_MONO)))
    fig.update_xaxes(gridcolor=_GRID, linecolor=_AXIS, tickfont=dict(color=_TICK, family=_MONO))
    fig.update_yaxes(gridcolor=_GRID, linecolor=_AXIS, tickfont=dict(color=_TICK, family=_MONO))
    return fig


def chart_rsi_macd(prices: pd.Series, ticker: str) -> go.Figure:
    r = rsi(prices)
    ml, sl, hist = macd(prices)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["RSI (14)", "MACD (12/26/9)"],
                        vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=prices.index, y=r, name="RSI",
                             line=dict(color=ACCENT, width=1.5)), row=1, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED,  opacity=0.08, row=1, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GREEN, opacity=0.08, row=1, col=1)
    for level, color in [(70, RED), (30, GREEN)]:
        fig.add_hline(y=level, line=dict(color=color, dash="dash", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=ml, name="MACD",
                             line=dict(color=ACCENT, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=sl, name="Signal",
                             line=dict(color=GOLD, width=1.5)), row=2, col=1)
    fig.add_trace(go.Bar(x=prices.index, y=hist, name="Histogram",
                         marker_color=[ACCENT if v >= 0 else RED for v in hist],
                         opacity=0.6), row=2, col=1)
    fig.update_layout(template=CHART_TEMPLATE,
                      paper_bgcolor=_BG, plot_bgcolor=_PLOT,
                      height=500, font=dict(family=_FONT, color="#78909C"),
                      hoverlabel=dict(bgcolor="#111519", bordercolor="#263238",
                                      font=dict(family=_MONO, size=12)),
                      legend=dict(orientation="h", font=dict(color="#78909C", family=_MONO)))
    fig.update_xaxes(gridcolor=_GRID, linecolor=_AXIS, tickfont=dict(color=_TICK, family=_MONO))
    fig.update_yaxes(gridcolor=_GRID, linecolor=_AXIS, tickfont=dict(color=_TICK, family=_MONO))
    return fig


def chart_ef(mu, cov, tickers, user_weights=None, user_label="Your Portfolio") -> go.Figure:
    vols, rets, srs, wts = efficient_frontier_mc(mu, cov, n=600)
    hover = ["<br>".join(f"{t}: {w:.1%}" for t, w in zip(tickers, ws)) for ws in wts]

    # Pre-compute optimal portfolios before y_max so their returns are included
    w_mv = w_min_var(mu, cov)
    r_mv, v_mv, _ = _port_stats(w_mv, mu.values, cov.values)
    w_ms = w_max_sharpe(mu, cov)
    r_ms, v_ms, _ = _port_stats(w_ms, mu.values, cov.values)

    # y_max: 98th-pct of random scatter OR the Mean-Variance return (whichever
    # is larger) — ensures the orange marker is never clipped off the chart
    y_max = max(np.percentile(rets, 98) * 1.25, r_ms * 1.15)
    y_min = min(min(rets) * 1.1, 0)

    fig = go.Figure()

    # ── Scatter cloud (random portfolios) ────────────────────────────────────
    # Sequential scale: dark navy (low Sharpe) → Bloomberg blue (high Sharpe).
    # Gives the cloud depth and encodes information without competing with the
    # orange Mean-Variance and blue Min-Variance marker circles.
    fig.add_trace(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(
            color=srs,
            # Warm amber gradient: near-black (low Sharpe) → Bloomberg amber (high Sharpe)
            # Distinct from both the blue Min Variance and orange Mean-Variance markers
            colorscale=[[0.0, "#0A0800"], [0.5, "#7A4800"], [1.0, "#CC7000"]],
            reversescale=False,
            size=5, opacity=0.75,
            colorbar=dict(title="Sharpe", thickness=10, len=0.55,
                          tickformat=".1f",
                          tickfont=dict(color=_TICK, family=_FONT),
                          title_font=dict(color=_TICK, family=_FONT)),
        ),
        text=hover,
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>%{text}"
                      "<extra>Random Portfolio</extra>",
        name="Random Portfolios",
    ))

    # ── Capital Market Line (subtle) ──────────────────────────────────────────
    slope = (r_ms - RF) / v_ms
    cml_x = [0, max(vols) * 1.05]
    cml_y = [RF, RF + slope * cml_x[1]]
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", dash="dot", width=1.5),
        name="Capital Market Line", hoverinfo="skip",
    ))

    # ── Risk-free rate dot ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0], y=[RF], mode="markers+text",
        marker=dict(size=8, color=MUTED),
        text=["Rf"], textposition="middle right",
        textfont=dict(color=MUTED, size=10),
        showlegend=False,
        hovertemplate=f"Risk-Free Rate: {RF:.2%}<extra></extra>",
    ))

    # ── Individual stocks (only those within chart bounds) ────────────────────
    for t in tickers:
        idx  = list(tickers).index(t)
        sv   = np.sqrt(cov.iloc[idx, idx]) * np.sqrt(AF)
        rv   = mu.iloc[idx] * AF
        if rv > y_max:          # skip extreme outliers
            continue
        alloc_t = f"Vol: {sv:.2%}  |  Return: {rv:.2%}"
        fig.add_trace(go.Scatter(
            x=[sv], y=[rv], mode="markers+text",
            marker=dict(symbol="circle", size=8,
                        color="rgba(255,255,255,0.12)",
                        line=dict(color="rgba(255,255,255,0.6)", width=1.2)),
            text=[t], textposition="top center",
            textfont=dict(size=9, color="rgba(255,255,255,0.65)"),
            showlegend=False,
            hovertemplate=f"<b>{t}</b><br>{alloc_t}<extra></extra>",
        ))

    # ── Optimal portfolios (solid circles + direct text labels) ──────────────
    for label, r_opt, v_opt, w_opt, color in [
        ("Min Variance",  r_mv, v_mv, w_mv, "#E0E4EA"),   # near-white — stands out from amber cloud
        ("Mean-Variance", r_ms, v_ms, w_ms, ACCENT),       # Bloomberg orange
    ]:
        alloc = "<br>".join(f"{t}: {wi:.1%}" for t, wi in zip(tickers, w_opt))
        fig.add_trace(go.Scatter(
            x=[v_opt], y=[r_opt], mode="markers+text",
            marker=dict(symbol="circle", size=10, color=color,
                        line=dict(color="white", width=1)),
            text=[label], textposition="top right",
            textfont=dict(size=10, color=color),
            name=label,
            hovertemplate=(f"<b>{label}</b><br>Vol: {v_opt:.2%}<br>"
                           f"Return: {r_opt:.2%}<br><br>{alloc}<extra></extra>"),
        ))

    # ── User's custom portfolio (gold diamond — only shown when Custom weights set) ──
    if user_weights is not None:
        r_u, v_u, _ = _port_stats(np.array(user_weights), mu.values, cov.values)
        if r_u <= y_max:
            alloc_u = "<br>".join(f"{t}: {wi:.1%}" for t, wi in zip(tickers, user_weights))
            fig.add_trace(go.Scatter(
                x=[v_u], y=[r_u], mode="markers+text",
                marker=dict(symbol="circle", size=10, color=GOLD,
                            line=dict(color="white", width=1)),
                text=[user_label], textposition="top right",
                textfont=dict(size=10, color=GOLD),
                name=user_label,
                hovertemplate=(f"<b>{user_label}</b><br>Vol: {v_u:.2%}<br>"
                               f"Return: {r_u:.2%}<br><br>{alloc_u}<extra></extra>"),
            ))

    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=_BG, plot_bgcolor=_PLOT,
        height=580,
        title=dict(text="<b>Efficient Frontier</b>",
                   font=dict(size=14, color="#E0E4EA", family=_SERIF)),
        font=dict(family=_FONT, color="#78909C"),
        legend=dict(orientation="v", yanchor="top", y=0.98,
                    xanchor="left", x=0.01,
                    bgcolor="rgba(10,12,16,0.75)",
                    bordercolor="#263238", borderwidth=1,
                    font=dict(color="#78909C", size=11, family=_MONO)),
        hoverlabel=dict(bgcolor="#111519", bordercolor="#263238",
                        font=dict(family=_MONO, size=12, color="#E0E4EA")),
        annotations=[dict(
            x=0.99, y=0.02, xref="paper", yref="paper", showarrow=False,
            text="Each dot represents one randomly weighted portfolio.",
            align="right", font=dict(size=10, color=_TICK, family=_MONO),
        )],
    )
    # Fix axes separately to avoid deprecated titlefont key
    fig.update_xaxes(title_text="Annualised Volatility", tickformat=".0%",
                     gridcolor=_GRID, linecolor=_AXIS,
                     tickfont=dict(color=_TICK, family=_MONO),
                     title_font=dict(color=_TICK, family=_FONT))
    fig.update_yaxes(title_text="Annualised Return", tickformat=".0%",
                     range=[y_min, y_max],
                     gridcolor=_GRID, linecolor=_AXIS,
                     tickfont=dict(color=_TICK, family=_MONO),
                     title_font=dict(color=_TICK, family=_FONT))
    return fig


def chart_styles_cumret(style_rets: dict) -> go.Figure:
    series = {s: (1 + r).cumprod() for s, r in style_rets.items() if not r.empty}
    colors_map = {
        "Your Portfolio": "#E0E4EA",   # near-white — matches EF custom marker
        "Equal Weight":   ACCENT,
        "Min Variance":   BLUE,
        "Mean-Variance":  GOLD,
        "Risk Parity":    GREEN,
        "Market Weight":  RED,
    }
    fig = go.Figure()
    for name, p in series.items():
        p = p.dropna()
        rebased = (p / p.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased, name=name,
                                 line=dict(color=colors_map.get(name, "#fff"), width=2),
                                 hovertemplate="%{y:.2f}%<extra>" + name + "</extra>"))
    return _layout(fig, "Portfolio Styles — Cumulative Return",
                   yaxis_title="Return (%)")


def chart_dist(r: pd.Series, label: str) -> go.Figure:
    mu, sigma = r.mean(), r.std()
    x = np.linspace(r.min(), r.max(), 200)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60, histnorm="probability density",
                               marker_color=ACCENT, opacity=0.7, name="Actual"))
    fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mu, sigma), name="Normal fit",
                             line=dict(color=RED, width=2)))
    v = hist_var(r)
    fig.add_vline(x=-v, line=dict(color=GOLD, dash="dash"),
                  annotation_text="VaR 95%", annotation_position="top right")
    return _layout(fig, f"{label} — Return Distribution",
                   xaxis_title="Daily Return", yaxis_title="Density", h=400)


def chart_dd(prices: pd.Series, label: str) -> go.Figure:
    dd = dd_series(prices) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                             line=dict(color=RED, width=1.2),
                             fillcolor="rgba(255,75,75,0.18)"))
    return _layout(fig, f"{label} — Drawdown (%)", yaxis_title="Drawdown (%)", h=340)


def chart_rolling_var(r: pd.Series, label: str, w: int, c: float) -> go.Figure:
    rv = r.rolling(w).apply(lambda x: hist_var(pd.Series(x), c), raw=False) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv.index, y=rv, fill="tozeroy",
                             line=dict(color=RED, width=1.5),
                             fillcolor="rgba(255,75,75,0.15)",
                             name=f"Rolling {c:.0%} VaR"))
    return _layout(fig, f"{label} — Rolling {w}d Historical VaR ({c:.0%})",
                   yaxis_title="VaR (%)", h=360)


def chart_mc(paths: np.ndarray, label: str, last_val: float) -> go.Figure:
    """Animated Monte Carlo chart with Plotly Play / Pause buttons.

    Paths grow left-to-right when the user presses Play — all animation is
    client-side (browser JS), so no Streamlit re-renders are needed.

    Architecture:
      - Traces 0 … n_display-1  : individual GBM paths (thin, semi-transparent)
      - Traces n_display … +2   : 90th / 50th / 10th percentile bands
    Each Plotly frame advances the x-range by frame_step days, updating every
    trace with the extended data.  ~60 frames keeps the JSON small while
    producing a smooth ~2.5-second animation for a 252-day horizon.
    """
    n_days, n_sim = paths.shape
    n_display  = min(50, n_sim)          # individual paths shown (stats use all)
    frame_step = max(1, n_days // 60)    # target ~60 frames
    frame_indices = list(range(0, n_days, frame_step))
    if frame_indices[-1] != n_days - 1:
        frame_indices.append(n_days - 1)

    pct_specs = [
        (90, "90th pct (bull)", ACCENT),
        (50, "Median",           GOLD),
        (10, "10th pct (bear)",  RED),
    ]

    # Fixed y-axis range computed from the full paths so axes never jump mid-animation
    y_lo = paths.min() * 0.97
    y_hi = paths.max() * 1.03

    # ── Initial state: just the starting dot (day 0) ──────────────────────────
    fig = go.Figure()

    for i in range(n_display):
        fig.add_trace(go.Scatter(
            x=[0], y=[paths[0, i]], mode="lines",
            line=dict(width=0.6, color="rgba(100,200,255,0.28)"),
            showlegend=False, hoverinfo="skip",
        ))
    for pct, name, color in pct_specs:
        fig.add_trace(go.Scatter(
            x=[0], y=[float(np.percentile(paths[0], pct))],
            name=name, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate="%{y:.4f}<extra>" + name + "</extra>",
        ))

    # ── Animation frames ──────────────────────────────────────────────────────
    frames = []
    for k in frame_indices:
        x = list(range(k + 1))
        frame_data = []
        for i in range(n_display):
            frame_data.append(go.Scatter(x=x, y=list(paths[:k + 1, i])))
        for pct, _, _ in pct_specs:
            frame_data.append(go.Scatter(
                x=x,
                y=list(np.percentile(paths[:k + 1, :], pct, axis=1)),
            ))
        frames.append(go.Frame(data=frame_data, name=str(k)))
    fig.frames = frames

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=_BG, plot_bgcolor=_PLOT,
        height=500,
        title=dict(
            text=f"<b>{label} — Monte Carlo ({n_days}d forecast)</b>",
            font=dict(size=14, color="#E0E4EA", family=_SERIF),
            pad=dict(b=10),
        ),
        xaxis=dict(
            title="Trading Days", range=[0, n_days - 1],
            gridcolor=_GRID, linecolor=_AXIS,
            tickfont=dict(color=_TICK, family=_MONO),
            title_font=dict(color=_TICK, family=_FONT),
        ),
        yaxis=dict(
            title="Value", range=[y_lo, y_hi],
            gridcolor=_GRID, linecolor=_AXIS,
            tickfont=dict(color=_TICK, family=_MONO),
            title_font=dict(color=_TICK, family=_FONT),
        ),
        font=dict(family=_FONT, color="#78909C"),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#78909C", size=11, family=_MONO),
        ),
        hoverlabel=dict(bgcolor="#111519", bordercolor="#263238",
                        font=dict(family=_MONO, size=12, color="#E0E4EA")),
        # Current-value reference line as a layout shape (persists across frames)
        shapes=[dict(
            type="line", x0=0, x1=n_days - 1, y0=last_val, y1=last_val,
            line=dict(color="rgba(255,255,255,0.45)", dash="dash", width=1),
        )],
        annotations=[dict(
            x=n_days - 1, y=last_val, xanchor="right", yanchor="bottom",
            text="Current value", showarrow=False,
            font=dict(color="rgba(255,255,255,0.45)", size=10, family=_MONO),
        )],
        # ── Play / Pause buttons ──────────────────────────────────────────────
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            x=0.0, xanchor="left",
            y=1.13, yanchor="top",
            pad=dict(r=8, t=0),
            bgcolor="#111519",
            bordercolor=ACCENT,
            borderwidth=1,
            font=dict(color=ACCENT, family=_MONO, size=12),
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=40, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate",
                    )],
                ),
                dict(
                    label="⏸  Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
    )
    return fig


def chart_risk_contribution(w: np.ndarray, cov: pd.DataFrame, tickers: list) -> go.Figure:
    """Grouped bar: % portfolio weight vs % risk contribution per asset."""
    w = np.array(w, dtype=float)
    sigma_w  = cov.values @ w
    port_var = float(w @ sigma_w)
    if port_var <= 0:
        return go.Figure()
    rc_pct = w * sigma_w / port_var * 100   # % contribution to total variance
    wt_pct = w * 100
    order     = np.argsort(rc_pct)[::-1]
    t_sorted  = [tickers[i] for i in order]
    rc_sorted = rc_pct[order]
    wt_sorted = wt_pct[order]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=t_sorted, y=wt_sorted, name="Weight %",
        marker_color=BLUE, opacity=0.85,
        hovertemplate="%{x}: %{y:.1f}%<extra>Weight</extra>",
    ))
    fig.add_trace(go.Bar(
        x=t_sorted, y=rc_sorted, name="Risk Contribution %",
        marker_color=RED, opacity=0.85,
        hovertemplate="%{x}: %{y:.1f}%<extra>Risk</extra>",
    ))
    return _layout(fig, "Risk Contribution vs Weight (%)",
                   yaxis_title="% of Total", h=380, barmode="group")


def chart_rolling_sharpe(r: pd.Series, label: str, w: int = 252) -> go.Figure:
    """Rolling annualised Sharpe ratio (252-day window by default)."""
    rs = rolling_sharpe(r, w)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rs.index, y=rs, name=f"Rolling {w}d Sharpe",
        line=dict(color=ACCENT, width=1.5),
        fill="tozeroy", fillcolor="rgba(255,140,0,0.10)",
        hovertemplate="%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, dash="dash", width=1))
    return _layout(fig, f"{label} — Rolling {w}d Sharpe Ratio",
                   yaxis_title="Sharpe Ratio", h=340)


def render_annual_returns(style_rets: dict) -> None:
    """Year-by-year returns table — green for positive, red for negative."""
    styles = [s for s, r in style_rets.items() if not r.empty]
    if not styles:
        return
    years = sorted({y for s in styles for y in style_rets[s].index.year.unique()})
    ann: dict[str, dict] = {}
    for s in styles:
        r = style_rets[s].dropna()
        ann[s] = {y: float((1 + r[r.index.year == y]).prod() - 1)
                  for y in years if not r[r.index.year == y].empty}

    th = (f"background:#1C2128;color:#E0E4EA;padding:9px 14px;text-align:center;"
          f"border-bottom:2px solid #263238;font-family:{_FONT};"
          f"font-size:0.82rem;font-weight:600;letter-spacing:0.04em;")
    idx_s = (f"background:#111519;color:#E0E4EA;padding:8px 14px;"
             f"border-bottom:1px solid #1C2128;font-family:{_FONT};"
             f"font-size:0.82rem;font-weight:600;")
    header = "".join(f'<th style="{th}">{y}</th>' for y in years)
    rows = ""
    for s in styles:
        cells = ""
        for y in years:
            v = ann[s].get(y)
            if v is None or np.isnan(v):
                col, txt = "#546E7A", "—"
            elif v >= 0:
                col, txt = "#00C853", f"+{v:.1%}"
            else:
                col, txt = "#E53935", f"{v:.1%}"
            td = (f"background:#111519;color:{col};padding:8px 12px;"
                  f"border-bottom:1px solid #1C2128;text-align:center;"
                  f"font-family:{_MONO};font-size:0.82rem;font-variant-numeric:tabular-nums;")
            cells += f'<td style="{td}">{txt}</td>'
        rows += f'<tr><td style="{idx_s}">{s}</td>{cells}</tr>'

    st.markdown(
        f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
        f'<thead><tr><th style="{th}; text-align:left;">Strategy</th>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def chart_corr_heatmap(prices: pd.DataFrame) -> go.Figure:
    corr = prices.pct_change().dropna().corr()
    # Diverging scale: dark burgundy (negative) → steel grey (zero) → deep navy (positive)
    bbg_div = [[0.0, "#B71C1C"], [0.5, "#37474F"], [1.0, "#FF8C00"]]
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=bbg_div,
                    zmin=-1, zmax=1, title="Pairwise Return Correlation")
    fig.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=_BG, plot_bgcolor=_PLOT,
        height=460,
        font=dict(family=_FONT, color="#78909C"),
        title=dict(font=dict(size=14, color="#E0E4EA", family=_SERIF)),
        coloraxis_colorbar=dict(tickfont=dict(color=_TICK, family=_MONO)),
    )
    fig.update_xaxes(tickfont=dict(color=_TICK, family=_MONO))
    fig.update_yaxes(tickfont=dict(color=_TICK, family=_MONO))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="US Large Cap Long Only Portfolio Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ── IBM Plex fonts (scientific / financial terminal aesthetic) ───────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, span, label, li, td, th {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
}

/* ── Typography — headings use IBM Plex Serif for gravitas ──────────────── */
h1 {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-size: 2rem !important; font-weight: 700 !important;
    letter-spacing: -0.01em !important; color: #E0E4EA !important;
}
h2, h3 {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-weight: 600 !important; letter-spacing: -0.01em !important;
    color: #C8CDD6 !important;
}

/* ── KPI metric cards — Bloomberg terminal style ─────────────────────────── */
.mcard {
    background: #111519;
    border-top: 1px solid #263238;
    border-bottom: 1px solid #263238;
    padding: 14px 18px;
    margin-bottom: 0px;
}
.mlabel {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-size: 0.62rem;
    color: #546E7A;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 500;
    margin-bottom: 6px;
}
.mvalue {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-size: 1.35rem;
    font-weight: 600;
    color: #FF8C00;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.01em;
}
.mpos { color: #00C853 !important; }
.mneg { color: #E53935 !important; }

/* ── Sidebar collapse button — hide raw icon text fallback ───────────────── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] { overflow: hidden; }
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span { font-size: 0 !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid #1C2128;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FF8C00 !important;
    color: #0A0C10 !important;
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 2px !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #E07800 !important;
}

/* ── Tabs — Bloomberg function-key style ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #1C2128 !important;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #546E7A !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #FF8C00 !important;
    border-bottom: 2px solid #FF8C00 !important;
    background: rgba(255,140,0,0.06) !important;
}

/* ── DataFrames ──────────────────────────────────────────────────────────── */
.stDataFrame {
    border: 1px solid #1C2128 !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Serif', Georgia, serif !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    color: #78909C !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: #1C2128 !important; opacity: 1 !important; }

</style>""", unsafe_allow_html=True)


def render_table(df: pd.DataFrame) -> None:
    """Bloomberg-styled static HTML table — no menus, no dropdowns."""
    th_style = (
        "background:#1C2128;color:#E0E4EA;padding:9px 14px;"
        "text-align:left;border-bottom:2px solid #263238;"
        f"font-family:{_FONT};font-size:0.82rem;font-weight:600;letter-spacing:0.04em;"
    )
    idx_style = (
        "background:#111519;color:#E0E4EA;padding:8px 14px;"
        "border-bottom:1px solid #1C2128;"
        f"font-family:{_FONT};font-size:0.82rem;font-weight:600;"
    )
    td_style = (
        "background:#111519;color:#FF8C00;padding:8px 14px;"
        "border-bottom:1px solid #1C2128;"
        f"font-family:{_FONT};font-size:0.82rem;"
    )
    header = "".join(f'<th style="{th_style}">{c}</th>' for c in df.columns)
    rows   = "".join(
        f'<tr><td style="{idx_style}">{idx}</td>'
        + "".join(f'<td style="{td_style}">{str(v).replace("—", "-")}</td>' for v in row)
        + "</tr>"
        for idx, row in df.iterrows()
    )
    st.markdown(
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr><th style="{th_style}"></th>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )


def mcard(label, value, pos=None):
    cls = "mpos" if pos is True else ("mneg" if pos is False else "")
    st.markdown(
        f'<div class="mcard">'
        f'<div class="mlabel">{label}</div>'
        f'<div class="mvalue {cls}">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("US Large Cap Long Only Portfolio Dashboard")
    st.caption("Programming with Advanced Computer Languages")
    st.divider()
    st.markdown("**Select Portfolio Tickers** (max 10)")
    tickers = st.multiselect(
        "Choose up to 10 S&P 500 stocks",
        options=ALL_TICKERS,
        default=DEFAULT_TICKERS,
        max_selections=10,
        format_func=lambda t: f"{t} ({COMPANY_NAMES.get(t, '')})",
    )
    if not tickers:
        tickers = DEFAULT_TICKERS

    st.divider()
    st.markdown("**Period**")
    col_s, col_e = st.columns(2)
    start_dt = col_s.date_input("From",
        value=pd.Timestamp.today() - pd.Timedelta(days=365 * 3),
        min_value=pd.Timestamp("2000-01-01"),
        max_value=pd.Timestamp.today())
    end_dt = col_e.date_input("To",
        value=pd.Timestamp.today(),
        min_value=pd.Timestamp("2000-01-01"),
        max_value=pd.Timestamp.today())

    st.divider()
    st.markdown("**Portfolio Weighting**")
    weight_mode = st.radio(
        "wm", ["Equal Weight", "Custom"],
        horizontal=True, label_visibility="collapsed", key="weight_mode",
    )
    custom_pcts: dict[str, float] = {}
    if weight_mode == "Custom" and tickers:
        st.caption("Enter each position size in %. Total should equal 100%.")
        n_t = len(tickers)
        default_pct = round(100.0 / n_t, 1)
        ca, cb = st.columns(2)
        for idx, t in enumerate(tickers):
            col = ca if idx % 2 == 0 else cb
            custom_pcts[t] = col.number_input(
                f"{t} (%)", min_value=0.0, max_value=100.0,
                value=default_pct, step=0.5, format="%.1f",
                key=f"w_{t}",
            )
        total_pct = sum(custom_pcts.values())
        if abs(total_pct - 100.0) > 1.0:
            st.warning(f"Total: {total_pct:.1f}% — will be auto-normalised to 100%")
        else:
            st.success(f"Total: {total_pct:.1f}%")

    st.divider()

# ── Fetch all data in one batch (portfolio tickers + SPY + AGG) ───────────────
_all = list(dict.fromkeys(tickers + BENCHMARK_TICKERS))   # dedup, preserve order
with st.spinner("Loading market data…"):
    _close = _yf_close(",".join(_all), str(start_dt), str(end_dt))

if _close.empty:
    st.error("No data returned. Check your internet connection or try a different date range.")
    st.stop()

avail = [t for t in tickers if t in _close.columns and _close[t].notna().sum() > 10]
if not avail:
    st.error("None of the selected tickers returned data for this period.")
    st.stop()

prices       = _close[avail].dropna(how="all")
bench_prices = _close[[t for t in BENCHMARK_TICKERS if t in _close.columns]].dropna(how="all")

# Historical 3-month T-bill rate for time-varying RF in the backtest optimiser
rf_hist = load_rf_series(str(start_dt), str(end_dt))

# Slider values are defined inside their tabs but read here first via session state
confidence = st.session_state.get("var_conf",  0.95)
mc_sims    = st.session_state.get("mc_paths", 300)

# ── Derived objects used across tabs ──────────────────────────────────────────
rets_df  = prices.pct_change().dropna()
n_assets = len(avail)

# ── Resolve user-defined portfolio weights ────────────────────────────────────
if weight_mode == "Custom" and custom_pcts:
    raw_w  = np.array([custom_pcts.get(t, 0.0) for t in avail])
    user_w = raw_w / raw_w.sum() if raw_w.sum() > 0 else w_equal(n_assets)
    port_label = "Custom Portfolio"
else:
    user_w     = w_equal(n_assets)
    port_label = "Equal-Weight Portfolio"

port_r   = (rets_df * user_w).sum(axis=1)
port_val = (1 + port_r).cumprod()
spy_r    = bench_prices["SPY"].pct_change().dropna() if "SPY" in bench_prices.columns else pd.Series(dtype=float)

# ── Header ────────────────────────────────────────────────────────────────────
# rets_df starts from the first date ALL tickers simultaneously have returns.
# Identify the bottleneck ticker (latest individual data start).
_ticker_starts  = {t: _close[t].dropna().index[0] for t in avail}
_common_start   = rets_df.index[0]
_limiting_t     = max(_ticker_starts, key=lambda t: _ticker_starts[t])

st.title("US Large Cap Long Only Portfolio Dashboard")
st.caption(
    f"Universe: {', '.join(avail)}  ·  "
    f"{_common_start.strftime('%d %b %Y')} to {prices.index[-1].strftime('%d %b %Y')}  ·  "
    f"{len(rets_df)} trading days"
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_risk, tab_corr, tab_port, tab_tech = st.tabs([
    "Overview",
    "Risk",
    "Correlation",
    "Optimisation",
    "Technicals",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.caption("*What do I own and how has it performed?*")
    # ── Portfolio allocation pie + concentration cards ────────────────────────
    st.subheader("Portfolio Allocation")
    _pie_colors = [
        "#C97A2F",  # muted amber
        "#3D6B8A",  # muted steel blue
        "#5A8A5A",  # muted sage green
        "#7A5A8A",  # muted slate purple
        "#8A5A5A",  # muted brick
        "#4A7A7A",  # muted teal
        "#8A7A4A",  # muted olive gold
        "#5A6A8A",  # muted navy
        "#7A8A5A",  # muted moss
        "#6A5A7A",  # muted mauve
    ]
    fig_pie = go.Figure(go.Pie(
        labels=[f"{t} ({COMPANY_NAMES.get(t, t)})" for t in avail],
        values=[round(w * 100, 2) for w in user_w],
        hole=0.38,
        marker=dict(
            colors=_pie_colors[:len(avail)],
            line=dict(color=_BG, width=2),
        ),
        texttemplate="%{label}<br><b>%{percent}</b>",
        textfont=dict(family=_FONT, size=12, color="#E0E4EA"),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
        insidetextorientation="radial",
    ))
    fig_pie.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT,
        font=dict(family=_FONT, color="#78909C"),
        height=460,
        showlegend=False,
        margin=dict(t=30, b=20, l=20, r=20),
        annotations=[dict(
            text=port_label.replace(" Portfolio", ""),
            x=0.5, y=0.5, font=dict(size=13, color="#E0E4EA", family=_FONT),
            showarrow=False,
        )],
    )

    # ── Sector allocation pie ─────────────────────────────────────────────────
    _sec_w: dict[str, float] = {}
    for t, w in zip(avail, user_w):
        s = TICKER_SECTOR.get(t, "Other")
        _sec_w[s] = _sec_w.get(s, 0.0) + w
    _sec_sorted = sorted(_sec_w.items(), key=lambda x: -x[1])
    _sec_labels = [s for s, _ in _sec_sorted]
    _sec_values = [round(v * 100, 2) for _, v in _sec_sorted]

    # Sector pie uses a distinct cool-toned palette so it doesn't mirror
    # the position pie when the portfolio is concentrated in one sector.
    _sec_colors = [
        "#2D6A8A",  # steel blue
        "#5A8A6A",  # sage green
        "#8A6A2D",  # warm tan
        "#6A2D8A",  # slate purple
        "#2D8A6A",  # teal
        "#8A2D6A",  # mauve
        "#6A8A2D",  # olive
        "#2D5A8A",  # navy
        "#8A5A2D",  # bronze
        "#4A2D8A",  # violet
        "#2D8A5A",  # emerald
    ]
    fig_sec_pie = go.Figure(go.Pie(
        labels=_sec_labels,
        values=_sec_values,
        hole=0.38,
        marker=dict(
            colors=_sec_colors[:len(_sec_labels)],
            line=dict(color=_BG, width=2),
        ),
        texttemplate="%{label}<br><b>%{percent}</b>",
        textfont=dict(family=_FONT, size=12, color="#E0E4EA"),
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.1f}%<extra></extra>",
        insidetextorientation="radial",
    ))
    fig_sec_pie.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_PLOT,
        font=dict(family=_FONT, color="#78909C"),
        height=460,
        showlegend=False,
        margin=dict(t=30, b=20, l=20, r=20),
        annotations=[dict(
            text="GICS Sectors",
            x=0.5, y=0.5, font=dict(size=13, color="#E0E4EA", family=_FONT),
            showarrow=False,
        )],
    )

    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_pie2:
        st.plotly_chart(fig_sec_pie, use_container_width=True)

    # ── Concentration metrics ─────────────────────────────────────────────────
    _hhi = hhi(user_w)
    _eff_n = 1 / _hhi if _hhi > 0 else n_assets
    _max_w = float(np.max(user_w))
    c1, c2, c3 = st.columns(3)
    with c1: mcard("HHI (concentration)", f"{_hhi:.3f}")
    with c2: mcard("Effective positions", f"{_eff_n:.1f}")
    with c3: mcard("Largest position", f"{_max_w:.1%}")

    st.divider()

    # ── Sector exposure ───────────────────────────────────────────────────────
    st.subheader("Sector Exposure vs S&P 500")
    st.plotly_chart(chart_sector_exposure(avail, user_w), use_container_width=True)

    st.divider()

    # ── Individual stock cumulative returns + portfolio overlay ───────────────
    st.subheader("Individual Stock Cumulative Returns")
    fig_cr = chart_cumret({t: prices[t] for t in avail})
    port_rebased = (port_val / port_val.iloc[0] - 1) * 100
    fig_cr.add_trace(go.Scatter(
        x=port_rebased.index, y=port_rebased,
        name=port_label,
        line=dict(color="#E0E4EA", width=2.5, dash="dash"),
        hovertemplate="%{y:.2f}%<extra>" + port_label + "</extra>",
    ))
    st.plotly_chart(fig_cr, use_container_width=True)

    st.divider()

    # ── Benchmark comparison ──────────────────────────────────────────────────
    st.subheader("Portfolio vs Benchmarks")
    if bench_prices.empty:
        st.info("Benchmark data (SPY / AGG) not loaded.")
    else:
        st.plotly_chart(chart_benchmark_comparison(port_r, bench_prices, label=port_label), use_container_width=True)

    if _common_start > pd.Timestamp(start_dt):
        st.markdown(
            f"<p style='font-size:0.78rem;color:#546E7A;margin-top:-8px;'>"
            f"<em>Disclaimer: charts and metrics cover {_common_start.strftime('%d %b %Y')} – "
            f"{prices.index[-1].strftime('%d %b %Y')} ({len(rets_df)} trading days). "
            f"The selected period starts from {start_dt}, but <strong>{_limiting_t}</strong> "
            f"has no price history before {_ticker_starts[_limiting_t].strftime('%d %b %Y')}. "
            f"All tickers must share a common date range. "
            f"Remove {_limiting_t} to access an earlier start date.</em></p>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Metrics table (bottom) ────────────────────────────────────────────────
    st.subheader("Portfolio Performance vs S&P 500")
    # Market-cap weights used as benchmark proxy for active share calculation
    _mcw_bench = w_market_cap(",".join(avail))
    port_metrics = full_metrics(port_r, port_val, spy_r, confidence,
                                port_w=user_w, bench_w=_mcw_bench)
    spy_metrics  = full_metrics(spy_r, (1+spy_r).cumprod(), spy_r, confidence) if not spy_r.empty else {}

    rows = list(port_metrics.keys())
    table_data = {"Metric": rows,
                  "Your Portfolio": [port_metrics[k] for k in rows]}
    if spy_metrics:
        table_data["S&P 500 (SPY)"] = [spy_metrics.get(k, "-") for k in rows]

    df_table = pd.DataFrame(table_data).set_index("Metric")
    render_table(df_table)
    st.download_button("Download as CSV", df_table.to_csv(),
                       file_name="portfolio_metrics.csv", mime="text/csv")

    st.divider()
    st.subheader("Methodology")
    st.markdown(r"""
**Returns** — Daily simple returns, compounded to cumulative and rebased to 0% at period start.
Portfolio return is the weighted sum of stock returns: $r_p = \sum_i w_i r_i$.

**Concentration (HHI)** — $\text{HHI} = \sum w_i^2$; Effective N $= 1/\text{HHI}$.
Higher HHI means more concentration; Effective N is the equivalent number of equal-weight positions.

**Sector exposure** — Active weight = portfolio sector weight minus approximate S&P 500 GICS weight. Positive = overweight vs the index.

**Benchmark metrics**

| Metric | Formula |
|--------|---------|
| Tracking Error | $\sigma(r_p - r_b)\cdot\sqrt{252}$ |
| Information Ratio | $\overline{(r_p - r_b)}/\text{TE}\cdot\sqrt{252}$ |
| Active Share | $\tfrac{1}{2}\sum_i \|w_i - b_i\|$ |
| Beta | $\text{Cov}(r_p,r_b)/\text{Var}(r_b)$ |
| Alpha | $\bar{r}_p - r_f - \beta(\bar{r}_b - r_f)$, annualised |
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_tech:
    st.caption("*What's the price action on individual names?*")
    st.subheader("Technical Analysis")
    primary = st.selectbox("Select ticker", avail, key="ta_ticker")
    ohlcv = _yf_ohlcv(primary, str(start_dt), str(end_dt))

    if ohlcv.empty:
        st.warning("No OHLCV data available.")
    else:
        st.plotly_chart(chart_price_ta(ohlcv, primary), use_container_width=True)
        st.plotly_chart(chart_rsi_macd(ohlcv["Close"], primary), use_container_width=True)

    st.divider()
    st.subheader("Methodology")
    st.markdown(r"""
**MA 20 / 50** — Simple moving averages over 20 and 50 days. Price above MA = bullish bias; below = bearish.

**Bollinger Bands** — MA20 ± 2 standard deviations. Bands widen in high-volatility periods; prices near the edges signal statistically extended moves.

**RSI (14d)** — Momentum oscillator on a 0–100 scale. Above 70 = overbought; below 30 = oversold.
$$\text{RSI} = 100 - \frac{100}{1 + \bar{G}_{14}/\bar{L}_{14}}$$

**MACD (12/26/9)** — Difference between a 12-day and 26-day EMA. The 9-day signal line smooths it; crossovers indicate momentum shifts.

*Technical indicators are descriptive only and do not predict future returns.*
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
    st.caption("*How do my holdings move relative to each other?*")
    st.subheader("Correlation Analysis")
    st.plotly_chart(chart_corr_heatmap(prices), use_container_width=True)

    st.subheader("Rolling Pairwise Correlation")
    c1, c2, c3 = st.columns(3)
    t1 = c1.selectbox("Asset A", avail, index=0, key="ca")
    t2 = c2.selectbox("Asset B", avail, index=min(1, len(avail)-1), key="cb")
    rw = c3.slider("Window (days)", 10, 63, 21, key="cw")

    if t1 != t2:
        rc = rets_df[t1].rolling(rw).corr(rets_df[t2])
        fig_rc = go.Figure()
        fig_rc.add_trace(go.Scatter(x=rc.index, y=rc, fill="tozeroy",
                                    line=dict(color=ACCENT, width=1.5),
                                    fillcolor="rgba(255,140,0,0.10)",
                                    name=f"ρ({t1},{t2})"))
        fig_rc.add_hline(y=0, line=dict(color="white", dash="dash", width=1))
        _layout(fig_rc, f"Rolling {rw}d Correlation: {t1} vs {t2}",
                yaxis=dict(range=[-1, 1]))
        st.plotly_chart(fig_rc, use_container_width=True)

        combined = pd.concat([rets_df[t1], rets_df[t2]], axis=1).dropna()
        fig_sc = px.scatter(combined, x=t1, y=t2, opacity=0.45, trendline="ols",
                            color_discrete_sequence=[ACCENT],
                            title=f"Daily Returns Scatter: {t1} vs {t2}")
        fig_sc.update_layout(template=CHART_TEMPLATE, height=400,
                             paper_bgcolor=_BG, plot_bgcolor=_PLOT,
                             font=dict(family=_FONT, color="#78909C"))
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Select two different assets.")

    st.divider()
    st.subheader("Methodology")
    st.markdown(r"""
**Pearson correlation** — Measures linear co-movement of daily returns, ranging from −1 to +1.
$$\rho_{i,j} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \cdot \sigma_j}$$
High correlation between two holdings reduces diversification benefit. Correlations tend to spike toward +1 during market stress, precisely when diversification matters most.

**Rolling correlation** — Shows how the pairwise relationship changes through time. A pair that looks uncorrelated over the full period may have been tightly correlated during drawdowns.

**Return scatter** — Each point is one trading day. The OLS trendline fits $r_j = \alpha + \beta r_i$; wider scatter around the line means more residual diversification benefit.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_port:
    st.caption("*How should I construct or improve the portfolio?*")
    if n_assets < 2:
        st.info("Select at least 2 tickers.")
    else:
        mu  = rets_df.mean()
        cov = rets_df.cov()

        # ── Portfolio styles backtest ─────────────────────────────────────────
        st.subheader("Portfolio Styles — Backtested Performance")

        cost_pct = st.slider(
            "Monthly rebalancing costs per unit of turnover (%)",
            min_value=0.00, max_value=0.50, value=0.00, step=0.01,
            format="%.2f%%",
            key="cost_pct",
        )

        with st.spinner("Backtesting portfolio styles…"):
            style_rets = backtest_styles(prices, rf_series=rf_hist, cost_pct=cost_pct)
        # Prepend the user's portfolio so it appears first in the legend
        style_rets = {"Your Portfolio": (rets_df * user_w).sum(axis=1)} | style_rets
        # In Equal Weight mode Your Portfolio == Equal Weight — drop the duplicate
        if weight_mode != "Custom":
            style_rets.pop("Equal Weight", None)

        st.plotly_chart(chart_styles_cumret(style_rets), use_container_width=True)

        # ── Style metrics table ───────────────────────────────────────────────
        st.subheader("Style Performance Metrics")
        rows_s = []
        for s_name, s_ret in style_rets.items():
            if s_ret.empty:
                continue
            s_ret = s_ret.dropna()
            s_val = (1 + s_ret).cumprod()
            b, a  = beta_alpha(s_ret, spy_r)
            rows_s.append({
                "Style":          s_name,
                "Ann. Return":    f"{ann_return(s_ret):.2%}",
                "Ann. Volatility":f"{ann_vol(s_ret):.2%}",
                "Sharpe":         _fmt_num(sharpe(s_ret),        "{:.2f}"),
                "Sortino":        _fmt_num(sortino(s_ret),       "{:.2f}"),
                "Calmar":         _fmt_num(calmar(s_ret, s_val), "{:.2f}"),
                "Max Drawdown":   f"{max_dd(s_val):.2%}",
                "Beta":           _fmt_num(b, "{:.2f}"),
                "Alpha":          _fmt_num(a, "{:.2%}"),
                f"VaR {confidence:.0%}": f"{hist_var(s_ret, confidence):.2%}",
            })
        _df_styles = pd.DataFrame(rows_s).set_index("Style")
        render_table(_df_styles)
        st.download_button("Download as CSV", _df_styles.to_csv(),
                           file_name="style_performance.csv", mime="text/csv")

        st.subheader("Annual Returns by Strategy")
        render_annual_returns(style_rets)

        st.divider()

        # ── Efficient Frontier ────────────────────────────────────────────────
        st.subheader("Efficient Frontier")
        with st.spinner("Simulating portfolios…"):
            _uw = user_w if weight_mode == "Custom" else None
            st.plotly_chart(chart_ef(mu, cov, avail, user_weights=_uw,
                                     user_label=port_label), use_container_width=True)

        # ── Optimal portfolio weights table ───────────────────────────────────
        st.subheader("Optimal Portfolio Allocations")
        mcw = w_market_cap(",".join(avail))
        wts_dict: dict[str, np.ndarray] = {
            "Your Portfolio": user_w,
            "Min Variance":   w_min_var(mu, cov),
            "Mean-Variance":  w_max_sharpe(mu, cov),
            "Risk Parity":    w_inv_vol(rets_df),
            "Market Weight":  mcw,
        }
        # Show Equal Weight as a separate column only when user has chosen Custom weights
        if weight_mode == "Custom":
            wts_dict = {"Your Portfolio": wts_dict["Your Portfolio"],
                        "Equal Weight":   w_equal(n_assets),
                        **{k: v for k, v in wts_dict.items() if k != "Your Portfolio"}}
        wts_df = pd.DataFrame(wts_dict, index=avail).map(lambda x: f"{x:.1%}")
        render_table(wts_df)
        st.download_button("Download as CSV", wts_df.to_csv(),
                           file_name="portfolio_allocations.csv", mime="text/csv")

        st.divider()

        # ── Strategy methodology ──────────────────────────────────────────────
        st.subheader("Methodology")
        st.markdown(f"""
All strategies are constrained to full investment, long-only, and a 40% single-stock cap.
The backtest is walk-forward: at each monthly rebalancing date only data up to that date is used.
Data starts from **{_common_start.strftime('%d %b %Y')}** (earliest common date across all selected tickers).

**Transaction costs** — The slider above applies a one-way cost per unit of portfolio turnover at each
monthly rebalancing. Strategies that change their weights substantially from month to month (such as
Mean-Variance) incur higher costs than low-turnover strategies (such as Risk Parity or Equal Weight).
Market Weight (buy-and-hold) and *Your Portfolio* are shown without cost. At rebalancing date $t$, the one-way turnover is:
""")
        st.latex(r"\mathrm{TO}_t = \tfrac{1}{2}\sum_i \left|w_i^{\mathrm{new}} - w_i^{\mathrm{drifted}}\right|")
        st.markdown(
            r"where $w^{\mathrm{drifted}}$ are the weights that have naturally drifted from the "
            r"previous rebalancing due to price changes. The cost deducted from the portfolio is:"
        )
        st.latex(r"\text{cost}_t = \mathrm{TO}_t \times c")
        st.markdown(
            r"with $c$ set by the slider above (one-way cost per unit of turnover). "
            r"Market Weight (buy-and-hold) has zero turnover and incurs no cost. "
            r"*Your Portfolio* is also shown without cost as a buy-and-hold baseline."
        )

        st.markdown("**1. Equal Weight**")
        st.latex(r"w_i = \frac{1}{N}")
        st.markdown("Rebalanced monthly to restore equal weights as prices drift. No estimation required.")

        st.markdown("**2. Minimum Variance**")
        st.latex(r"\min_w \; w^\top \Sigma\, w \quad \text{s.t.} \; \mathbf{1}^\top w=1,\; 0\le w_i\le 0.4")
        st.markdown("Only the covariance matrix $\\Sigma$ is needed — no return forecasts. Zero-weight allocations are normal.")

        st.markdown("**3. Mean-Variance (Max Sharpe)**")
        st.latex(r"\max_w \; \frac{w^\top\mu - r_f}{\sqrt{w^\top\Sigma\, w}} \quad \text{s.t.} \; \mathbf{1}^\top w=1,\; 0\le w_i\le 0.4")
        st.markdown("The backtest uses the historical 3-month T-bill rate (^IRX) at each rebalancing date. "
                    "*Limitation: sample means are noisy estimators, so the optimiser tends to concentrate in whichever stock performed best in-sample.*")

        st.markdown("**4. Risk Parity (Inverse Volatility)**")
        st.latex(r"w_i = \frac{1/\hat\sigma_i}{\sum_j 1/\hat\sigma_j}")
        st.markdown("Lower-volatility assets receive higher weights so each stock contributes roughly equal risk. "
                    "Volatility is estimated from the expanding window. "
                    "*Note: this is inverse-vol weighting, a simplified form of true risk parity.*")

        st.markdown("**5. Market Weight (Buy-and-Hold)**")
        st.latex(r"w_i(0) = \frac{\hat{S}_i \cdot P_i(t_0)}{\sum_j \hat{S}_j \cdot P_j(t_0)}, \quad \hat{S}_i = \frac{\mathrm{MCap}_i^{\,\text{today}}}{P_i^{\,\text{today}}}, \qquad V(t)=\sum_i w_i(0)\prod_{s=1}^t(1+r_i^s)")
        st.markdown("Held without rebalancing: price appreciation naturally keeps weights at market-cap proportions. "
                    "Starting weights are derived from implied shares outstanding "
                    "(current market cap ÷ current price) multiplied by the stock price on the first day of the backtest — "
                    "so the initial allocation reflects historical market caps, not today's.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk:
    st.caption("*What are the risks of what I hold?*")
    confidence = st.slider(
        "VaR confidence level", 0.90, 0.99,
        st.session_state.get("var_conf", 0.95), 0.01,
        key="var_conf",
        help="Percentile used for all VaR / CVaR calculations in this tab",
    )
    st.subheader(f"Portfolio Risk — {port_label}")

    # ── Portfolio headline metrics ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard(f"Hist. VaR ({confidence:.0%})",  f"{hist_var(port_r, confidence):.2%}", False)
    with c2: mcard(f"Param. VaR ({confidence:.0%})", f"{param_var(port_r, confidence):.2%}", False)
    with c3: mcard(f"CVaR / ES ({confidence:.0%})",  f"{cvar(port_r, confidence):.2%}",  False)
    with c4: mcard("Max Drawdown", f"{max_dd(port_val):.2%}", False)

    st.plotly_chart(chart_dist(port_r, port_label), use_container_width=True)
    st.plotly_chart(chart_dd(port_val, port_label), use_container_width=True)

    var_win = st.slider("Rolling VaR window (days)", 20, 63, 30, key="rv_win")
    st.plotly_chart(chart_rolling_var(port_r, port_label, var_win, confidence),
                    use_container_width=True)

    st.plotly_chart(chart_rolling_sharpe(port_r, port_label),
                    use_container_width=True)

    st.divider()

    # ── Risk contribution ─────────────────────────────────────────────────────
    st.subheader("Risk Contribution per Stock")
    st.caption("How much of total portfolio variance each position contributes, vs its weight.")
    st.plotly_chart(chart_risk_contribution(user_w, rets_df.cov(), avail),
                    use_container_width=True)

    st.divider()

    # ── Individual stock drill-down ───────────────────────────────────────────
    st.subheader("Individual Stock Risk")
    risk_t = st.selectbox("Select ticker", avail, key="risk_t")
    r_t = rets_df[risk_t]
    p_t = prices[risk_t].dropna()

    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard(f"Hist. VaR ({confidence:.0%})",  f"{hist_var(r_t, confidence):.2%}", False)
    with c2: mcard(f"Param. VaR ({confidence:.0%})", f"{param_var(r_t, confidence):.2%}", False)
    with c3: mcard(f"CVaR / ES ({confidence:.0%})",  f"{cvar(r_t, confidence):.2%}",  False)
    with c4: mcard("Max Drawdown", f"{max_dd(p_t):.2%}", False)

    st.plotly_chart(chart_dist(r_t, risk_t), use_container_width=True)
    st.plotly_chart(chart_dd(p_t, risk_t),   use_container_width=True)
    st.plotly_chart(chart_rolling_var(r_t, risk_t, var_win, confidence),
                    use_container_width=True)
    st.plotly_chart(chart_rolling_sharpe(r_t, risk_t), use_container_width=True)

    st.divider()

    # ── Monte Carlo Simulation ────────────────────────────────────────────────
    st.subheader("Monte Carlo Simulation")
    col_mc1, col_mc2 = st.columns(2)
    mc_sims = col_mc1.slider(
        "Monte Carlo paths", 100, 800, 300, 100,
        key="mc_paths",
        help="Number of simulated price paths (more = smoother but slower)",
    )
    mc_days = col_mc2.slider("Forecast horizon (trading days)", 21, 504, 252, 21)
    np.random.seed(42)

    # ── Portfolio simulation ──────────────────────────────────────────────────
    st.subheader("Portfolio Monte Carlo Simulation")
    port_log_r = np.log(port_val / port_val.shift(1)).dropna()
    paths_port = mc_paths(float(port_val.iloc[-1]), port_log_r, mc_sims, mc_days)
    st.plotly_chart(chart_mc(paths_port, port_label, float(port_val.iloc[-1])),
                    use_container_width=True)

    term_port = paths_port[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard("Median terminal value",  f"{np.percentile(term_port,50):.3f}")
    with c2: mcard("10th pct (bear)",        f"{np.percentile(term_port,10):.3f}", False)
    with c3: mcard("90th pct (bull)",        f"{np.percentile(term_port,90):.3f}", True)
    with c4:
        p_loss15 = (term_port < float(port_val.iloc[-1]) * 0.85).mean()
        mcard("P(loss > 15%)", f"{p_loss15:.1%}", False)

    fig_th = go.Figure()
    fig_th.add_trace(go.Histogram(x=term_port, nbinsx=60,
                                  marker_color=ACCENT, opacity=0.75))
    fig_th.add_vline(x=float(port_val.iloc[-1]),
                     line=dict(color="white", dash="dash"),
                     annotation_text="Current")
    _layout(fig_th, f"{port_label} — Terminal Value Distribution ({mc_days}d)",
            xaxis_title="Portfolio Value", yaxis_title="Frequency", h=360)
    st.plotly_chart(fig_th, use_container_width=True)

    st.divider()

    # ── Single-stock simulation ───────────────────────────────────────────────
    st.subheader("Single Stock Monte Carlo Simulation")
    mc_t = st.selectbox("Select ticker", avail, key="mc_t")
    mc_prices = prices[mc_t].dropna()
    mc_log_r  = np.log(mc_prices / mc_prices.shift(1)).dropna()
    last_px   = float(mc_prices.iloc[-1])
    paths_s   = mc_paths(last_px, mc_log_r, mc_sims, mc_days)
    st.plotly_chart(chart_mc(paths_s, mc_t, last_px), use_container_width=True)

    term_s = paths_s[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard("Median terminal price", f"${np.percentile(term_s,50):,.2f}")
    with c2: mcard("10th pct (bear)",       f"${np.percentile(term_s,10):,.2f}", False)
    with c3: mcard("90th pct (bull)",       f"${np.percentile(term_s,90):,.2f}", True)
    with c4:
        p_loss15_s = (term_s < last_px * 0.85).mean()
        mcard("P(loss > 15%)", f"{p_loss15_s:.1%}", False)

    fig_th2 = go.Figure()
    fig_th2.add_trace(go.Histogram(x=term_s, nbinsx=60,
                                   marker_color=ACCENT, opacity=0.75))
    fig_th2.add_vline(x=last_px, line=dict(color="white", dash="dash"),
                      annotation_text="Current price")
    _layout(fig_th2, f"{mc_t} — Terminal Price Distribution ({mc_days}d)",
            xaxis_title="Price (USD)", yaxis_title="Frequency", h=360)
    st.plotly_chart(fig_th2, use_container_width=True)

    st.divider()
    st.subheader("Methodology")
    st.markdown(rf"""
**VaR ({confidence:.0%})** — The loss level exceeded on only {1-confidence:.0%} of trading days.
- *Historical*: read directly from the return distribution — $\text{{VaR}} = -\text{{Percentile}}(r,\, {(1-confidence)*100:.0f}\%)$
- *Parametric*: assumes normally distributed returns — $\text{{VaR}} = -(\mu + z_{{1-\alpha}}\,\sigma)$

**CVaR** — Average loss on days VaR is breached. Always $\geq$ VaR; more sensitive to tail events.

**Drawdown** — Decline from the most recent peak: $\text{{DD}}_t = (P_t - \max_{{s\leq t}} P_s)\,/\,\max_{{s\leq t}} P_s$.

**Risk Contribution** — Each stock's share of total portfolio variance: $\text{{RC}}_i = w_i\,(\Sigma w)_i\,/\,w^\top\Sigma w$. A stock that is a small position but dominates risk is a candidate for trimming; one that is a large position but low-risk adds diversification efficiently.

**Rolling VaR** — VaR recomputed over a sliding window. Spikes indicate periods of elevated risk.

**Rolling Sharpe** — Annualised Sharpe ratio over a trailing 252-day window. Falls below zero when the portfolio underperforms the risk-free rate on a risk-adjusted basis.

**Monte Carlo (GBM)** — Simulates future price paths using Geometric Brownian Motion with zero drift:
$$P_t = P_{{t-1}}\cdot e^{{\varepsilon_t}},\qquad \varepsilon_t\sim\mathcal{{N}}(0,\,\hat{{\sigma}})$$
Zero drift is used instead of historical mean returns to avoid extrapolating a bull-market bias into the forecast. The fan chart shows the 10th–90th percentile range of simulated paths.

**P(loss > 15%)** — Share of simulated paths ending more than 15% below today's value. More informative than P(value > today), which is upward-biased by the log-normal distribution.

*Limitations: GBM assumes constant volatility, normal shocks, and independent daily returns. Real markets exhibit volatility clustering, fat tails, and serial correlation.*
""")
