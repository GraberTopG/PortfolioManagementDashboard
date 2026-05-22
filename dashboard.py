"""
Quantitative Finance Dashboard
================================
HSG Master – Programming with Advanced Computer Languages

Tabs:
  1. Overview        – portfolio metrics vs S&P 500, cumulative returns, benchmark comparison
  2. Technical       – candlestick, MAs, Bollinger, RSI, MACD  (ticker selector inside)
  3. Correlation     – heatmap + rolling pairwise correlation
  4. Portfolio       – efficient frontier, style comparison (EW / MinVar / MeanVariance / RiskParity)
  5. Risk Metrics    – portfolio VaR/CVaR/drawdown, individual stock drill-down, GBM Monte Carlo simulation

Data: Yahoo Finance via yfinance — no API key required, full history from 2000.
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
    return np.log(p / p.shift(1)).dropna()

def simple_returns(p: pd.Series) -> pd.Series:
    return p.pct_change().dropna()

def ann_return(r: pd.Series) -> float:
    return r.mean() * AF

def ann_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(AF)

def sharpe(r: pd.Series, rf: float = RF) -> float:
    v = ann_vol(r)
    return (ann_return(r) - rf) / v if v else np.nan

def sortino(r: pd.Series, rf: float = RF) -> float:
    dd = r[r < 0].std() * np.sqrt(AF)
    return (ann_return(r) - rf) / dd if dd else np.nan

def max_dd(prices: pd.Series) -> float:
    return ((prices - prices.cummax()) / prices.cummax()).min()

def dd_series(prices: pd.Series) -> pd.Series:
    return (prices - prices.cummax()) / prices.cummax()

def calmar(r: pd.Series, prices: pd.Series) -> float:
    mdd = abs(max_dd(prices))
    return ann_return(r) / mdd if mdd else np.nan

def hist_var(r: pd.Series, c: float = 0.95) -> float:
    return -np.percentile(r.dropna(), (1 - c) * 100)

def param_var(r: pd.Series, c: float = 0.95) -> float:
    return -(r.mean() + norm.ppf(1 - c) * r.std())

def cvar(r: pd.Series, c: float = 0.95) -> float:
    v = hist_var(r, c)
    tail = r[r < -v]
    return -tail.mean() if len(tail) else v

def beta_alpha(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([r, bench], axis=1).dropna()
    if len(aligned) < 10:
        return np.nan, np.nan
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    b = cov[0, 1] / cov[1, 1]
    a = ann_return(aligned.iloc[:, 0]) - RF - b * (ann_return(aligned.iloc[:, 1]) - RF)
    return b, a

def win_rate(r: pd.Series) -> float:
    return (r > 0).mean()

def full_metrics(r: pd.Series, prices: pd.Series, bench_r: pd.Series,
                 conf: float = 0.95, port_w: np.ndarray | None = None,
                 bench_w: np.ndarray | None = None) -> dict:
    b, a  = beta_alpha(r, bench_r)
    te    = tracking_error(r, bench_r)
    ir    = information_ratio(r, bench_r)
    ash   = active_share(port_w, bench_w) if port_w is not None and bench_w is not None else np.nan
    return {
        "Ann. Return":       f"{ann_return(r):.2%}",
        "Ann. Volatility":   f"{ann_vol(r):.2%}",
        "Sharpe Ratio":      f"{sharpe(r):.2f}",
        "Sortino Ratio":     f"{sortino(r):.2f}",
        "Calmar Ratio":      f"{calmar(r, prices):.2f}",
        "Max Drawdown":      f"{max_dd(prices):.2%}",
        "Beta (vs SPY)":     f"{b:.2f}",
        "Alpha (vs SPY)":    f"{a:.2%}",
        "Tracking Error":    f"{te:.2%}",
        "Information Ratio": f"{ir:.2f}",
        "Active Share":      f"{ash:.1%}" if not np.isnan(ash) else "-",
        "Win Rate":          f"{win_rate(r):.1%}",
        f"VaR {conf:.0%}":   f"{hist_var(r, conf):.2%}",
        f"CVaR {conf:.0%}":  f"{cvar(r, conf):.2%}",
        "Skewness":          f"{r.skew():.2f}",
        "Kurtosis":          f"{r.kurtosis():.2f}",
    }

def tracking_error(r: pd.Series, bench_r: pd.Series) -> float:
    active = r.subtract(bench_r, fill_value=0)
    return active.std() * np.sqrt(AF)

def information_ratio(r: pd.Series, bench_r: pd.Series) -> float:
    active = r.subtract(bench_r, fill_value=0)
    te = tracking_error(r, bench_r)
    return ann_return(active) / te if te else np.nan

def active_share(port_w: np.ndarray, bench_w: np.ndarray) -> float:
    """Active share vs a benchmark weight vector (same asset universe)."""
    return 0.5 * np.sum(np.abs(port_w - bench_w))

def hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index — sum of squared weights."""
    return float(np.sum(np.array(weights) ** 2))

def rolling_vol(r: pd.Series, w: int = 21) -> pd.Series:
    return r.rolling(w, min_periods=5).std() * np.sqrt(AF)

# ── Technical indicators ──────────────────────────────────────────────────────
def bollinger(p, w=20, n=2):
    m = p.rolling(w).mean()
    s = p.rolling(w).std()
    return m + n*s, m, m - n*s

def rsi(p, w=14):
    d = p.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    return 100 - 100 / (1 + g / l)

def macd(p, fast=12, slow=26, sig=9):
    ml = p.ewm(span=fast, adjust=False).mean() - p.ewm(span=slow, adjust=False).mean()
    sl = ml.ewm(span=sig, adjust=False).mean()
    return ml, sl, ml - sl

# ── Portfolio optimisation ────────────────────────────────────────────────────
def _port_stats(w, mu, cov):
    r = np.dot(w, mu) * AF
    v = np.sqrt(w @ cov @ w) * np.sqrt(AF)
    return r, v, (r - RF) / v if v else np.nan

def _opt(objective, n, extra_constraints=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if extra_constraints:
        cons += extra_constraints
    res = minimize(objective, np.ones(n)/n, method="SLSQP",
                   bounds=[(0, 1)]*n, constraints=cons)
    return res.x

def w_min_var(mu, cov):
    return _opt(lambda w: _port_stats(w, mu, cov)[1], len(mu))

def w_max_sharpe(mu, cov):
    return _opt(lambda w: -_port_stats(w, mu, cov)[2], len(mu))

def w_risk_parity(cov):
    n = len(cov)
    def obj(w):
        sv = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / sv
        return np.sum((rc - rc.mean())**2)
    return _opt(obj, n, [{"type": "ineq", "fun": lambda w: w - 0.01}])

def w_inv_vol(rets_df):
    vols = rets_df.std()
    iv = 1 / vols
    return (iv / iv.sum()).values

def w_equal(n):
    return np.ones(n) / n

@st.cache_data(ttl=86400, show_spinner=False)
def w_market_cap(tickers_csv: str) -> np.ndarray:
    """Fetch current market caps via yfinance and return normalised weights."""
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

def efficient_frontier_mc(mu, cov, n=500):
    n_assets = len(mu)
    vols, rets, srs, wts = [], [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = _port_stats(w, mu, cov)
        vols.append(v); rets.append(r); srs.append(s); wts.append(w)
    return np.array(vols), np.array(rets), np.array(srs), wts

# ── Portfolio backtest with monthly rebalancing ───────────────────────────────
def backtest_styles(prices: pd.DataFrame, min_lookback: int = 5) -> dict[str, pd.Series]:
    rets = prices.pct_change().dropna()
    month_starts = rets.resample("MS").first().index.tolist()
    # append end of series
    rebal_dates = [d for d in month_starts if d in rets.index or d > rets.index[0]]

    style_names = ["Equal Weight", "Min Variance", "Mean-Variance", "Risk Parity", "Market Weight"]
    port_rets = {s: pd.Series(dtype=float) for s in style_names}

    for i, rebal in enumerate(rebal_dates):
        # holding period: from this rebal to the next (or end)
        hold_end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else rets.index[-1]
        period = rets.loc[rebal:hold_end].iloc[1:]   # skip first row (same-day rebal)
        if period.empty:
            continue

        hist = rets.loc[:rebal]
        mu = hist.mean()
        cov = hist.cov()
        n = len(rets.columns)

        mcw = w_market_cap(",".join(rets.columns.tolist()))
        if len(hist) >= min_lookback:
            weights = {
                "Equal Weight":  w_equal(n),
                "Min Variance":  w_min_var(mu, cov),
                "Mean-Variance": w_max_sharpe(mu, cov),   # tangency portfolio
                "Risk Parity":   w_inv_vol(hist),          # inverse-volatility weighting
                "Market Weight": mcw,
            }
        else:
            ew = w_equal(n)
            weights = {s: ew for s in style_names[:-1]} | {"Market Weight": mcw}

        for s in style_names:
            p_ret = (period * weights[s]).sum(axis=1)
            port_rets[s] = pd.concat([port_rets[s], p_ret])

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
_MONO  = "IBM Plex Serif, Georgia, serif"
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
    for period, color in [(20, GOLD), (50, "#ff8c00")]:
        ma = ohlcv["Close"].rolling(period).mean()
        fig.add_trace(go.Scatter(x=ohlcv.index, y=ma, name=f"MA{period}",
                                 line=dict(color=color, width=1.2)), row=1, col=1)
    upper, mid, lower = bollinger(ohlcv["Close"])
    for band, name in [(upper, "BB Upper"), (lower, "BB Lower")]:
        fig.add_trace(go.Scatter(x=ohlcv.index, y=band, name=name,
                                 line=dict(color="rgba(150,150,255,0.5)", dash="dot", width=1),
                                 showlegend=True), row=1, col=1)
    fig.add_trace(go.Bar(x=ohlcv.index, y=ohlcv["Volume"], name="Volume",
                         marker_color="rgba(100,149,237,0.4)"), row=2, col=1)
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
        title=dict(text="<b>Efficient Frontier — Monte Carlo Simulation</b>",
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


def chart_mc(paths, label, last_val) -> go.Figure:
    n_days = paths.shape[0]
    x = list(range(n_days))
    fig = go.Figure()
    for i in range(min(80, paths.shape[1])):
        fig.add_trace(go.Scatter(x=x, y=paths[:, i], mode="lines",
                                 line=dict(width=0.4, color="rgba(100,200,255,0.12)"),
                                 showlegend=False))
    for pct, name, color in [(90, "90th pct (bull)", ACCENT),
                              (50, "Median",           GOLD),
                              (10, "10th pct (bear)",  RED)]:
        fig.add_trace(go.Scatter(x=x, y=np.percentile(paths, pct, axis=1),
                                 name=name, line=dict(color=color, width=2)))
    fig.add_hline(y=last_val, line=dict(color="white", dash="dash", width=1),
                  annotation_text="Current value")
    return _layout(fig, f"{label} — Monte Carlo ({n_days}d forecast)",
                   xaxis_title="Trading Days", yaxis_title="Value", h=480)


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
.mpos { color: #FF8C00 !important; }
.mneg { color: #FF8C00 !important; }

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
st.title("US Large Cap Long Only Portfolio Dashboard")
st.caption(f"Universe: {', '.join(avail)}  ·  "
           f"{prices.index[0].strftime('%d %b %Y')} to {prices.index[-1].strftime('%d %b %Y')}  ·  "
           f"{len(prices)} trading days")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_tech, tab_corr, tab_port, tab_risk = st.tabs([
    "Overview",
    "Technical Analysis",
    "Correlation",
    "Portfolio Optimisation",
    "Risk Metrics",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
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
    st.plotly_chart(fig_pie, use_container_width=True)

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
**Returns and compounding**

All return series use daily log-close prices adjusted for splits and dividends.
Cumulative returns are computed as the compounded product of daily simple returns,
rebased to 0% at the start of the selected period:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}, \qquad \text{Cumulative}_t = \prod_{s=1}^{t}(1 + r_s) - 1$$

**Portfolio return**

The portfolio daily return is the weighted sum of individual stock returns,
where weights $w_i$ are either equal ($1/N$) or user-defined and normalised to sum to 1:

$$r_p = \sum_{i=1}^{N} w_i \cdot r_i$$

**Benchmark-relative metrics**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Tracking Error | $\sigma(r_p - r_b) \cdot \sqrt{252}$ | Annualised volatility of active returns |
| Information Ratio | $\frac{\overline{r_p - r_b}}{\text{TE}} \cdot \sqrt{252}$ | Active return per unit of tracking risk |
| Active Share | $\frac{1}{2}\sum_i |w_i - b_i|$ | Portfolio divergence from market-cap benchmark; 0% = index clone, 100% = fully active |
| Beta | $\beta = \frac{\text{Cov}(r_p, r_b)}{\text{Var}(r_b)}$ | Sensitivity to benchmark moves |
| Alpha | $\alpha = \bar{r}_p - r_f - \beta(\bar{r}_b - r_f)$ | Annualised excess return above what Beta predicts |

**Concentration (HHI)**

The Herfindahl-Hirschman Index measures portfolio concentration.
Effective N translates it into an equivalent number of equal-weight positions:

$$\text{HHI} = \sum_{i=1}^{N} w_i^2, \qquad \text{Effective } N = \frac{1}{\text{HHI}}$$

An equal-weight 8-stock portfolio has HHI = 0.125 and Effective N = 8.
A portfolio with 70% in one name has HHI ≈ 0.50 and Effective N ≈ 2.

**Sector exposure**

Portfolio sector weights are compared against approximate S&P 500 GICS sector
weights (2025). The active weight (diamond markers) is the difference:
$w_{\text{sector}}^{\text{portfolio}} - w_{\text{sector}}^{\text{S\&P 500}}$.
A positive active weight signals an overweight relative to the index.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_tech:
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
**Moving Averages (MA 20 / MA 50)**

Simple moving averages smooth price noise to reveal the underlying trend.
The 20-day MA captures short-term momentum; the 50-day MA captures the medium-term trend.
A price crossing above its MA is conventionally read as bullish; crossing below as bearish.

$$\text{MA}_t(n) = \frac{1}{n}\sum_{k=0}^{n-1} P_{t-k}$$

**Bollinger Bands**

Bollinger Bands place a volatility envelope around the 20-day MA.
The bands expand during high-volatility periods and contract when volatility is low.
Prices touching the upper band are statistically extended; touching the lower band may signal oversold conditions.

$$\text{Upper} = \text{MA}_{20} + 2\sigma_{20}, \qquad \text{Lower} = \text{MA}_{20} - 2\sigma_{20}$$

where $\sigma_{20}$ is the rolling 20-day standard deviation of prices.

**RSI — Relative Strength Index (14 days)**

RSI measures the speed and magnitude of recent price changes on a 0–100 scale.
Readings above 70 are conventionally considered overbought; below 30 oversold.

$$\text{RSI} = 100 - \frac{100}{1 + \frac{\overline{G}_{14}}{\overline{L}_{14}}}$$

where $\overline{G}_{14}$ and $\overline{L}_{14}$ are the 14-day average gain and average loss respectively.

**MACD — Moving Average Convergence/Divergence (12/26/9)**

MACD captures momentum by comparing a fast and slow exponential moving average.
The signal line (9-day EMA of MACD) smooths the indicator; the histogram shows the gap between them.
A MACD crossing above its signal line is a conventional buy signal; crossing below is a sell signal.

$$\text{MACD} = \text{EMA}_{12} - \text{EMA}_{26}, \qquad \text{Signal} = \text{EMA}_9(\text{MACD})$$

**Important caveat**

Technical indicators are descriptive tools derived solely from price and volume history.
They do not predict future returns and should not be used in isolation for investment decisions.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_corr:
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
                                    fillcolor="rgba(0,212,170,0.1)",
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
**Pearson correlation coefficient**

The heatmap shows the pairwise Pearson correlation of daily log returns over the selected period.
It measures the linear co-movement between two return series, ranging from −1 (perfect inverse) to +1 (perfect co-movement):

$$\rho_{i,j} = \frac{\text{Cov}(r_i,\, r_j)}{\sigma_i \cdot \sigma_j}$$

A high positive correlation between two holdings means they tend to move together —
adding both to a portfolio provides less diversification benefit than their individual risk suggests.
For a long-only portfolio, avoiding highly correlated positions is the primary lever for reducing idiosyncratic concentration risk.

**Rolling correlation**

The rolling window (default 21 trading days ≈ 1 month) shows how the pairwise relationship
changes through time. Correlations are not stable: they tend to spike toward +1 during market
stress (the "correlation breakdown" effect), which is precisely when diversification is most needed.
A pair that looks uncorrelated over the full period may have been tightly correlated during every drawdown.

**Return scatter and OLS trendline**

Each point represents one trading day. The OLS (Ordinary Least Squares) trendline fits:

$$r_j = \alpha + \beta \cdot r_i + \varepsilon$$

The slope $\beta$ is equivalent to the Pearson correlation scaled by the ratio of volatilities.
The scatter of points around the line shows the residual (idiosyncratic) component that is
not explained by the linear relationship — the wider the scatter, the more diversification benefit remains.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_port:
    if n_assets < 2:
        st.info("Select at least 2 tickers.")
    else:
        mu  = rets_df.mean()
        cov = rets_df.cov()

        # ── Portfolio styles backtest ─────────────────────────────────────────
        st.subheader("Portfolio Styles — Backtested Performance")

        with st.spinner("Backtesting portfolio styles…"):
            style_rets = backtest_styles(prices)
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
                "Sharpe":         f"{sharpe(s_ret):.2f}",
                "Sortino":        f"{sortino(s_ret):.2f}",
                "Calmar":         f"{calmar(s_ret, s_val):.2f}",
                "Max Drawdown":   f"{max_dd(s_val):.2%}",
                "Beta":           f"{b:.2f}",
                "Alpha":          f"{a:.2%}",
                f"VaR {confidence:.0%}": f"{hist_var(s_ret, confidence):.2%}",
            })
        _df_styles = pd.DataFrame(rows_s).set_index("Style")
        render_table(_df_styles)
        st.download_button("Download as CSV", _df_styles.to_csv(),
                           file_name="style_performance.csv", mime="text/csv")

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
        st.markdown("""
All optimisations are solved subject to **full investment** (weights sum to 1)
and **long-only** constraints (no short selling). $N$ denotes the number of
assets, $\\Sigma$ the covariance matrix of returns, and $\\mu$ the vector of
mean returns.
""")

        st.markdown("**1. Equal Weight (1/N)**")
        st.markdown("*Invest an equal proportion in every asset.*")
        st.latex(r"w_i = \frac{1}{N} \qquad \forall\; i = 1, \ldots, N")
        st.markdown("""
No estimation of expected returns or covariances is required. Research
(DeMiguel et al., 2009) shows it is surprisingly hard to beat out-of-sample —
it benefits from maximum diversification and has zero estimation error.
""")

        st.markdown("**2. Minimum Variance**")
        st.markdown("*Minimise portfolio volatility regardless of expected returns.*")
        st.latex(r"""
\min_{w} \;\; w^\top \Sigma\, w \qquad
\text{subject to} \;\; \mathbf{1}^\top w = 1,\;\; w_i \ge 0
""")
        st.markdown("""
The solution concentrates in low-volatility, low-correlation assets. It is the
leftmost point on the efficient frontier and requires only the covariance matrix
— no return forecasts needed.
""")

        st.markdown("**3. Mean-Variance (Tangency Portfolio)**")
        st.markdown("*Find the portfolio with the best return per unit of risk.*")
        st.markdown(r"""
The general Mean-Variance framework (Markowitz, 1952) maximises expected
portfolio return for a given level of variance:
""")
        st.latex(r"""
\max_{w} \;\; \mathbb{E}[R_p] - \frac{k}{2}\,\mathrm{var}(R_p)
\qquad \text{subject to} \;\; \mathbf{1}^\top w = 1,\;\; w_i \ge 0
""")
        st.markdown(r"""
Different values of the risk-aversion parameter $k$ trace out the entire
efficient frontier. The **tangency portfolio** is the specific point where the
Capital Market Line touches the frontier — equivalently, it **maximises the
Sharpe ratio**:
""")
        st.latex(r"""
\max_{w} \;\; \frac{w^\top \mu - r_f}{\sqrt{w^\top \Sigma\, w}}
\qquad \text{subject to} \;\; \mathbf{1}^\top w = 1,\;\; w_i \ge 0
""")
        st.markdown(r"""
$r_f = 5.25\%$ is the risk-free rate. This is the portfolio implemented here:
**Mean-Variance optimisation solved as a Maximum Sharpe problem.**
""")

        st.markdown("**4. Risk Parity**")
        st.markdown("*Weight each asset inversely proportional to its volatility.*")
        st.latex(r"""
w_i = \frac{1/\sigma_i}{\displaystyle\sum_{j=1}^{N} 1/\sigma_j}
""")
        st.markdown(r"""
$\sigma_i$ is the annualised volatility of asset $i$. Assets with high
volatility receive smaller weights; low-volatility assets receive larger
weights. The goal is for every asset to contribute **equal risk** to the
portfolio rather than equal capital. This rule requires no matrix inversion and
no return forecasts.
""")

        st.markdown("**5. Market Weight Portfolio**")
        st.markdown("*Hold every asset in proportion to its market capitalisation.*")
        st.latex(r"""
w_i = \frac{\mathrm{Market\;cap}_i}{\displaystyle\sum_{j=1}^{N} \mathrm{Market\;cap}_j}
""")
        st.markdown(r"""
Larger companies receive larger weights, mirroring how broad index funds such
as the S&P 500 are constructed. No optimisation is required — weights are
determined entirely by the market's collective valuation. Current market caps
are sourced from Yahoo Finance and held fixed across the backtest period.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · RISK METRICS
# ─────────────────────────────────────────────────────────────────────────────
with tab_risk:
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
**Value at Risk (VaR)**

VaR answers the question: *what is the maximum loss I should expect on a typical bad day?*
At a {confidence:.0%} confidence level, VaR is the loss threshold that is exceeded on only {1-confidence:.0%} of trading days.

Two approaches are implemented here:

- **Historical VaR** reads the loss directly from the empirical return distribution — no distributional assumption required:

$$\text{{VaR}}_{{1-\alpha}} = -\text{{Percentile}}(r,\, (1-\alpha) \times 100)$$

- **Parametric VaR** assumes returns are normally distributed and uses the sample mean $\mu$ and standard deviation $\sigma$:

$$\text{{VaR}}_{{1-\alpha}} = -(\mu + z_{{1-\alpha}} \cdot \sigma)$$

where $z_{{1-\alpha}}$ is the standard normal quantile (e.g. $-1.645$ at 95%).

**CVaR / Expected Shortfall (ES)**

CVaR goes further than VaR — it measures the *average* loss on the days that VaR is breached.
It is a coherent risk measure and better captures tail risk:

$$\text{{CVaR}}_{{1-\alpha}} = -\,\mathbb{{E}}\!\left[r \;\middle|\; r < -\text{{VaR}}_{{1-\alpha}}\right]$$

CVaR is always $\geq$ VaR and is more sensitive to extreme outcomes, making it preferred by risk committees.

**Drawdown**

Drawdown measures the decline from a historical peak. Maximum Drawdown is the worst such decline over the period:

$$\text{{DD}}_t = \frac{{P_t - \max_{{s \leq t}} P_s}}{{\max_{{s \leq t}} P_s}}, \qquad \text{{Max DD}} = \min_t \text{{DD}}_t$$

**Rolling VaR**

The rolling VaR shows how the portfolio's downside risk has evolved through time using a sliding window.
Spikes in rolling VaR correspond to periods of elevated volatility (e.g. market drawdowns, macro shocks).
A stable rolling VaR indicates a consistent risk profile; a rising trend signals deteriorating conditions.

**Geometric Brownian Motion (GBM)**

Monte Carlo simulation generates a large number of plausible future price paths by repeatedly sampling random shocks.
Each path evolves according to Geometric Brownian Motion — the standard continuous-time model for asset prices:

$$P_t = P_{{t-1}} \cdot e^{{\varepsilon_t}}, \qquad \varepsilon_t \sim \mathcal{{N}}(0,\, \hat{{\sigma}})$$

where $\hat{{\sigma}}$ is the daily volatility estimated from the historical return series of the selected period.
After $N$ simulations, the distribution of terminal values gives a probabilistic view of the range of outcomes.

**Why zero drift?**

The drift term is set to zero rather than using the historical mean return. Extrapolating past returns — especially from a recent bull-market period — would bias almost all paths upward and mask genuine downside risk. Zero drift means paths spread symmetrically around today's value, driven purely by volatility. This is the conservative and intellectually honest choice for a risk tool.

**Reading the fan chart**

The shaded bands show the 10th, 25th, 75th and 90th percentiles of simulated paths at each point in time.
The spread of the fan widens with the horizon — reflecting that uncertainty compounds over time.
A high-volatility portfolio produces a much wider fan than a low-volatility one, even with identical expected returns.

**P(loss > 15%)**

Rather than the misleading P(value > today) — which is biased upward by the log-normal distribution's positive skew — this dashboard reports the probability that the terminal value falls more than 15% below today's level:

$$P(\text{{loss}} > 15\%) = \frac{{1}}{{N}}\sum_{{i=1}}^{{N}} \mathbf{{1}}\!\left[P_T^{{(i)}} < 0.85 \cdot P_0\right]$$

This is the question a risk committee actually asks: *what is the probability of a material loss?*

**Model limitations**

- **Constant volatility** — GBM does not model volatility clustering (GARCH effects). Real volatility spikes during crises, so tail events are likely underestimated.
- **Normal shocks** — Real return distributions have fat tails and negative skew. Extreme losses occur more frequently than the normal distribution implies.
- **Independent increments** — Serial correlation, momentum, and mean-reversion are not captured.
- **Single-factor** — All randomness comes from one source. Macro regime changes, liquidity crises, and correlation breakdowns are outside the model's scope.
""")
