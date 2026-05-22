"""
Quantitative Finance Dashboard
================================
HSG Master – Programming Course Project

Tabs:
  1. Overview        – portfolio metrics vs S&P 500, cumulative returns, benchmark comparison
  2. Technical       – candlestick, MAs, Bollinger, RSI, MACD  (ticker selector inside)
  3. Correlation     – heatmap + rolling pairwise correlation
  4. Portfolio       – efficient frontier, style comparison (EW / MinVar / MeanVariance / RiskParity)
  5. Risk Metrics    – portfolio VaR/CVaR/drawdown, then individual stock drill-down
  6. Monte Carlo     – portfolio GBM simulation, then single-stock simulation

Data sources (tried in order):
  1. Local CSV cache  — instant, works offline, survives restarts
  2. Alpha Vantage    — free API key (25 req/day), used on first load or refresh
"""

import os, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import requests
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
    "ETFs": [
        "SPY","QQQ","IWM","AGG","TLT","GLD","VTI","EFA","EEM","HYG",
        "XLF","XLK","XLE","XLV","XLI","XLP",
    ],
}
ALL_TICKERS = [t for group in UNIVERSE.values() for t in group]
AV_BASE        = "https://www.alphavantage.co/query"
CACHE_DIR      = os.path.join(os.path.dirname(__file__), "data_cache")
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
    # ETFs
    "SPY":"S&P 500 ETF",            "QQQ":"NASDAQ-100 ETF",        "IWM":"Russell 2000 ETF",
    "AGG":"US Bond ETF",            "TLT":"20yr Treasury ETF",     "GLD":"Gold ETF",
    "VTI":"Total Market ETF",       "EFA":"Intl Developed ETF",    "EEM":"Emerging Markets ETF",
    "HYG":"High Yield Bond ETF",    "XLF":"Financials Sector ETF", "XLK":"Technology Sector ETF",
    "XLE":"Energy Sector ETF",      "XLV":"Healthcare Sector ETF", "XLI":"Industrials Sector ETF",
    "XLP":"Staples Sector ETF",
}

os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER  —  CSV cache  →  Alpha Vantage fallback
# ══════════════════════════════════════════════════════════════════════════════

def _csv_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.csv")


def _load_csv(ticker: str) -> pd.DataFrame | None:
    """Return cached DataFrame if the CSV exists, else None."""
    path = _csv_path(ticker)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _save_csv(ticker: str, df: pd.DataFrame) -> None:
    df.to_csv(_csv_path(ticker))


@st.cache_data(ttl=86400)
def _av_daily(ticker: str, api_key: str) -> pd.DataFrame:
    """Fetch compact daily OHLCV from Alpha Vantage and persist to CSV."""
    r = requests.get(AV_BASE, params={
        "function": "TIME_SERIES_DAILY", "symbol": ticker,
        "outputsize": "compact", "apikey": api_key, "datatype": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "Time Series (Daily)" not in data:
        raise ValueError(data.get("Information") or data.get("Note") or str(data))
    df = pd.DataFrame(data["Time Series (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().rename(columns={
        "1. open": "Open", "2. high": "High",
        "3. low": "Low",  "4. close": "Close", "5. volume": "Volume",
    })
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    _save_csv(ticker, df)   # persist immediately
    return df


def _get_ohlcv(ticker: str, api_key: str, bar=None, i: int = 0, n: int = 1) -> pd.DataFrame:
    """Return OHLCV: CSV cache first, Alpha Vantage second."""
    cached = _load_csv(ticker)
    if cached is not None:
        if bar:
            bar.progress((i + 1) / n, text=f"Loaded {ticker} from cache ({i+1}/{n})")
        return cached
    # Not cached → fetch from API
    if not api_key:
        if bar:
            bar.progress((i + 1) / n, text=f"Skipped {ticker} — no API key ({i+1}/{n})")
        return pd.DataFrame()
    try:
        df = _av_daily(ticker, api_key)
        if bar:
            bar.progress((i + 1) / n, text=f"Fetched {ticker} from API ({i+1}/{n})")
        time.sleep(1.2)
        return df
    except Exception as e:
        st.warning(f"Could not fetch {ticker}: {e}")
        if bar:
            bar.progress((i + 1) / n, text=f"Failed: {ticker} ({i+1}/{n})")
        return pd.DataFrame()


def fetch_prices(tickers: list[str], api_key: str) -> pd.DataFrame:
    bar = st.progress(0, text="Loading portfolio data…")
    frames = {}
    for i, t in enumerate(tickers):
        df = _get_ohlcv(t, api_key, bar, i, len(tickers))
        if not df.empty:
            frames[t] = df["Close"]
    bar.empty()
    return pd.DataFrame(frames).dropna(how="all") if frames else pd.DataFrame()


def fetch_benchmarks(api_key: str) -> pd.DataFrame:
    bar = st.progress(0, text="Loading benchmark data…")
    frames = {}
    for i, t in enumerate(BENCHMARK_TICKERS):
        df = _get_ohlcv(t, api_key, bar, i, len(BENCHMARK_TICKERS))
        if not df.empty:
            frames[t] = df["Close"]
    bar.empty()
    return pd.DataFrame(frames).dropna(how="all") if frames else pd.DataFrame()


def fetch_single(ticker: str, api_key: str) -> pd.DataFrame:
    df = _get_ohlcv(ticker, api_key)
    if df.empty:
        return df
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[cols].dropna()


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
                 conf: float = 0.95) -> dict:
    b, a = beta_alpha(r, bench_r)
    return {
        "Ann. Return":    f"{ann_return(r):.2%}",
        "Ann. Volatility":f"{ann_vol(r):.2%}",
        "Sharpe Ratio":   f"{sharpe(r):.2f}",
        "Sortino Ratio":  f"{sortino(r):.2f}",
        "Calmar Ratio":   f"{calmar(r, prices):.2f}",
        "Max Drawdown":   f"{max_dd(prices):.2%}",
        "Beta (vs SPY)":  f"{b:.2f}",
        "Alpha (vs SPY)": f"{a:.2%}",
        "Win Rate":       f"{win_rate(r):.1%}",
        f"VaR {conf:.0%}":  f"{hist_var(r, conf):.2%}",
        f"CVaR {conf:.0%}": f"{cvar(r, conf):.2%}",
        "Skewness":       f"{r.skew():.2f}",
        "Kurtosis":       f"{r.kurtosis():.2f}",
    }

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

def efficient_frontier_mc(mu, cov, n=500):
    n_assets = len(mu)
    vols, rets, srs, wts = [], [], [], []
    for _ in range(n):
        w = np.random.dirichlet(np.ones(n_assets))
        r, v, s = _port_stats(w, mu, cov)
        vols.append(v); rets.append(r); srs.append(s); wts.append(w)
    return np.array(vols), np.array(rets), np.array(srs), wts

# ── Portfolio backtest with monthly rebalancing ───────────────────────────────
def backtest_styles(prices: pd.DataFrame, min_lookback: int = 30) -> dict[str, pd.Series]:
    rets = prices.pct_change().dropna()
    month_starts = rets.resample("MS").first().index.tolist()
    # append end of series
    rebal_dates = [d for d in month_starts if d in rets.index or d > rets.index[0]]

    style_names = ["Equal Weight", "Min Variance", "Mean-Variance", "Risk Parity"]
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

        if len(hist) >= min_lookback:
            weights = {
                "Equal Weight":  w_equal(n),
                "Min Variance":  w_min_var(mu, cov),
                "Mean-Variance": w_max_sharpe(mu, cov),   # tangency portfolio
                "Risk Parity":   w_inv_vol(hist),          # inverse-volatility weighting
            }
        else:
            ew = w_equal(n)
            weights = {s: ew for s in style_names}

        for s in style_names:
            p_ret = (period * weights[s]).sum(axis=1)
            port_rets[s] = pd.concat([port_rets[s], p_ret])

    return port_rets

# ── Monte Carlo ───────────────────────────────────────────────────────────────
def mc_paths(last_val, daily_rets, n_sim=300, n_days=252):
    mu, sigma = daily_rets.mean(), daily_rets.std()
    paths = np.zeros((n_days, n_sim))
    paths[0] = last_val
    shocks = np.random.normal(mu, sigma, (n_days - 1, n_sim))
    for t in range(1, n_days):
        paths[t] = paths[t - 1] * np.exp(shocks[t - 1])
    return paths

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_FONT  = "IBM Plex Sans, Helvetica Neue, Arial, sans-serif"
_MONO  = "IBM Plex Mono, Courier New, monospace"   # axes / numbers
_SERIF = "IBM Plex Serif, Georgia, serif"           # titles
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
            colorscale=[[0.0, "#0D1B2A"], [0.5, "#1A3A5C"], [1.0, "#00A8E8"]],
            reversescale=False,
            size=5, opacity=0.65,
            colorbar=dict(title="Sharpe", thickness=10, len=0.55,
                          tickformat=".1f",
                          tickfont=dict(color=_TICK, family=_MONO)),
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
        marker=dict(size=8, color=GOLD),
        text=["Rf"], textposition="middle right",
        textfont=dict(color=GOLD, size=10),
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
        ("Min Variance",  r_mv, v_mv, w_mv, BLUE),
        ("Mean-Variance", r_ms, v_ms, w_ms, ACCENT),
    ]:
        alloc = "<br>".join(f"{t}: {wi:.1%}" for t, wi in zip(tickers, w_opt))
        fig.add_trace(go.Scatter(
            x=[v_opt], y=[r_opt], mode="markers+text",
            marker=dict(symbol="circle", size=18, color=color,
                        line=dict(color="white", width=2)),
            text=[label], textposition="top right",
            textfont=dict(size=11, color=color),
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
                marker=dict(symbol="diamond", size=16, color=GOLD,
                            line=dict(color="white", width=1.5)),
                text=[user_label], textposition="top right",
                textfont=dict(size=11, color=GOLD),
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
        "Equal Weight":  ACCENT,
        "Min Variance":  BLUE,
        "Mean-Variance": GOLD,
        "Risk Parity":   GREEN,
    }
    fig = go.Figure()
    for name, p in series.items():
        p = p.dropna()
        rebased = (p / p.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=rebased.index, y=rebased, name=name,
                                 line=dict(color=colors_map.get(name, "#fff"), width=2),
                                 hovertemplate="%{y:.2f}%<extra>" + name + "</extra>"))
    return _layout(fig, "Portfolio Styles — Cumulative Return (monthly rebalancing)",
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
    # Diverging scale: red (negative) → blue-grey (zero) → Bloomberg blue (positive)
    # Mid-point #37474F matches the scatter-cloud grey used on the efficient frontier
    bbg_div = [[0.0, "#E53935"], [0.5, "#37474F"], [1.0, "#00A8E8"]]
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

st.set_page_config(page_title="Portfolio Management Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* ── IBM Plex fonts (scientific / financial terminal aesthetic) ───────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, span, label, li, td, th {
    font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif !important;
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
    border-radius: 2px;
    padding: 16px 20px;
    border: 1px solid #1C2128;
    border-left: 3px solid #FF8C00;
    margin-bottom: 8px;
}
.mlabel {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.62rem;
    color: #546E7A;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 500;
}
.mvalue {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.45rem;
    font-weight: 600;
    color: #E0E4EA;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.02em;
}
.mpos { color: #00C853 !important; }
.mneg { color: #E53935 !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid #1C2128;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #FF8C00 !important;
    color: #0A0C10 !important;
    font-family: 'IBM Plex Mono', monospace !important;
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
    font-family: 'IBM Plex Mono', monospace !important;
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
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    color: #78909C !important;
}

/* ── Dividers ────────────────────────────────────────────────────────────── */
hr { border-color: #1C2128 !important; opacity: 1 !important; }
</style>""", unsafe_allow_html=True)

def mcard(label, value, pos=None):
    cls   = "mpos" if pos is True else ("mneg" if pos is False else "")
    # accent border colour: green for positive, red for negative, blue for neutral
    border = "#00C853" if pos is True else ("#E53935" if pos is False else "#FF8C00")
    st.markdown(
        f'<div class="mcard" style="border-left-color:{border}">'
        f'<div class="mlabel">{label}</div>'
        f'<div class="mvalue {cls}">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Portfolio Management Dashboard")
    st.caption("HSG Master · Programming Project")
    st.divider()
    st.markdown("**Alpha Vantage API Key**")
    st.markdown("[Get free key →](https://www.alphavantage.co/support/#api-key)")
    api_key = st.text_input("API key", placeholder="e.g. ABCDE12345")
    load_btn = st.button("Load Data", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Select Portfolio Tickers** (max 15)")
    tickers = st.multiselect(
        "Choose up to 10 stocks / ETFs",
        options=ALL_TICKERS,
        default=DEFAULT_TICKERS,
        format_func=lambda t: f"{t} ({COMPANY_NAMES.get(t, '')})",
    )
    if len(tickers) > 15:
        st.warning("Maximum 15 tickers — only the first 15 will be used.")
        tickers = tickers[:15]
    if not tickers:
        tickers = DEFAULT_TICKERS

    # Group labels as helper
    with st.expander("Available tickers by sector"):
        for sector, members in UNIVERSE.items():
            st.caption(f"{sector}:  {' · '.join(members)}")

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

    col1, col2 = st.columns(2)
    start_dt = col1.date_input("Start", value=pd.Timestamp.today() - pd.Timedelta(days=120))
    end_dt   = col2.date_input("End",   value=pd.Timestamp.today())
    st.divider()
    st.caption("Data: Alpha Vantage · Built on gs_quant concepts")

# ── Auth guard ────────────────────────────────────────────────────────────────
if not api_key:
    st.info("### Enter your Alpha Vantage API key\n\n"
            "1. Visit https://www.alphavantage.co/support/#api-key\n"
            "2. Enter name & email — key appears instantly (free, no credit card)\n"
            "3. Paste it into the sidebar and click **Load Data**")
    st.stop()

if not load_btn and "data_loaded" not in st.session_state:
    st.info("API key entered — click **Load Data** to fetch market data.")
    st.stop()

# ── Fetch portfolio data ──────────────────────────────────────────────────────
with st.spinner("Loading portfolio data…"):
    prices_all = fetch_prices(tickers, api_key)

avail = [t for t in tickers if t in prices_all.columns and prices_all[t].notna().sum() > 20]
if not avail:
    st.error("No data returned. Check your API key and tickers.")
    st.stop()

prices = prices_all[avail].loc[str(start_dt):str(end_dt)]

# ── Fetch benchmark data (SPY + AGG) ─────────────────────────────────────────
if "bench_prices" not in st.session_state or load_btn:
    with st.spinner("Loading benchmark data (SPY, AGG)…"):
        bench_prices = fetch_benchmarks(api_key)
    st.session_state["bench_prices"] = bench_prices
else:
    bench_prices = st.session_state["bench_prices"]

bench_prices = bench_prices.loc[str(start_dt):str(end_dt)] if not bench_prices.empty else bench_prices
st.session_state["data_loaded"] = True

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
st.title("Portfolio Management Dashboard")
st.caption(f"Universe: {', '.join(avail)}  ·  Period: {start_dt} → {end_dt}  ·  "
           f"{len(prices)} trading days")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_tech, tab_corr, tab_port, tab_risk, tab_mc = st.tabs([
    "Overview",
    "Technical Analysis",
    "Correlation",
    "Portfolio Optimisation",
    "Risk Metrics",
    "Monte Carlo",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Portfolio Performance vs S&P 500")

    # ── Metrics table ────────────────────────────────────────────────────────
    port_metrics = full_metrics(port_r, port_val, spy_r, confidence)
    spy_metrics  = full_metrics(spy_r, (1+spy_r).cumprod(), spy_r, confidence) if not spy_r.empty else {}

    rows = list(port_metrics.keys())
    table_data = {"Metric": rows,
                  "Your Portfolio": [port_metrics[k] for k in rows]}
    if spy_metrics:
        table_data["S&P 500 (SPY)"] = [spy_metrics.get(k, "—") for k in rows]

    df_table = pd.DataFrame(table_data).set_index("Metric")
    st.dataframe(df_table, use_container_width=True)

    st.divider()

    # ── Cumulative return of portfolio ────────────────────────────────────────
    st.subheader("Individual Stock Cumulative Returns")
    st.plotly_chart(chart_cumret({t: prices[t] for t in avail}), use_container_width=True)

    st.divider()

    # ── Benchmark comparison ──────────────────────────────────────────────────
    st.subheader("Portfolio vs Benchmarks")
    if bench_prices.empty:
        st.info("Benchmark data (SPY / AGG) not loaded — re-click Load Data to fetch.")
    else:
        st.plotly_chart(chart_benchmark_comparison(port_r, bench_prices, label=port_label), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_tech:
    st.subheader("Technical Analysis")
    primary = st.selectbox("Select ticker", avail, key="ta_ticker")
    ohlcv = fetch_single(primary, api_key)
    ohlcv = ohlcv.loc[str(start_dt):str(end_dt)] if not ohlcv.empty else ohlcv

    if ohlcv.empty:
        st.warning("No OHLCV data available.")
    else:
        st.plotly_chart(chart_price_ta(ohlcv, primary), use_container_width=True)
        st.plotly_chart(chart_rsi_macd(ohlcv["Close"], primary), use_container_width=True)
        with st.expander("Indicator guide"):
            st.markdown("""
| Indicator | Signal |
|-----------|--------|
| **MA 20 / 50** | Price above both MAs → uptrend; below both → downtrend |
| **Bollinger Bands** | Price near upper band → potentially overbought; near lower → oversold |
| **RSI > 70** | Overbought — possible pullback |
| **RSI < 30** | Oversold — possible bounce |
| **MACD crossover ↑** | Bullish momentum signal |
| **MACD crossover ↓** | Bearish momentum signal |
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

        with st.expander("Return scatter"):
            combined = pd.concat([rets_df[t1], rets_df[t2]], axis=1).dropna()
            fig_sc = px.scatter(combined, x=t1, y=t2, opacity=0.45, trendline="ols",
                                color_discrete_sequence=[ACCENT],
                                title=f"Daily Returns Scatter: {t1} vs {t2}")
            fig_sc.update_layout(template=CHART_TEMPLATE, height=400)
            st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Select two different assets.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_port:
    if n_assets < 2:
        st.info("Select at least 2 tickers.")
    else:
        mu  = rets_df.mean()
        cov = rets_df.cov()

        # ── Efficient Frontier ────────────────────────────────────────────────
        st.subheader("Efficient Frontier")
        with st.spinner("Simulating portfolios…"):
            _uw = user_w if weight_mode == "Custom" else None
            st.plotly_chart(chart_ef(mu, cov, avail, user_weights=_uw,
                                     user_label=port_label), use_container_width=True)

        # ── Optimal portfolio weights table ───────────────────────────────────
        st.subheader("Optimal Portfolio Allocations")
        wts_dict: dict[str, np.ndarray] = {}
        if weight_mode == "Custom":
            wts_dict["Your Portfolio"] = user_w
        wts_dict.update({
            "Equal Weight":  w_equal(n_assets),
            "Min Variance":  w_min_var(mu, cov),
            "Mean-Variance": w_max_sharpe(mu, cov),   # tangency portfolio = Max Sharpe
            "Risk Parity":   w_inv_vol(rets_df),       # inverse-volatility weighting
        })
        wts_df = pd.DataFrame(wts_dict, index=avail).map(lambda x: f"{x:.1%}")
        st.dataframe(wts_df, use_container_width=True)

        st.divider()

        # ── Portfolio styles backtest ─────────────────────────────────────────
        # ── Strategy methodology ──────────────────────────────────────────────
        with st.expander("Strategy methodology — formulas and intuition"):
            st.markdown("""
Four portfolio construction strategies are compared. All optimisations are
solved subject to **full investment** (weights sum to 1) and **long-only**
constraints (no short selling). $N$ denotes the number of assets, $\\Sigma$
the covariance matrix of returns, and $\\mu$ the vector of mean returns.

---
#### 1. Equal Weight (1/N)
*Invest an equal proportion in every asset.*
""")
            st.latex(r"w_i = \frac{1}{N} \qquad \forall\; i = 1, \ldots, N")
            st.markdown("""
No estimation of expected returns or covariances is required. Research
(DeMiguel et al., 2009) shows it is surprisingly hard to beat out-of-sample —
it benefits from maximum diversification and has zero estimation error.

---
#### 2. Minimum Variance
*Minimise portfolio volatility regardless of expected returns.*
""")
            st.latex(r"""
\min_{w} \;\; w^\top \Sigma\, w \qquad
\text{subject to} \;\; \mathbf{1}^\top w = 1,\;\; w_i \ge 0
""")
            st.markdown(r"""
The solution concentrates in low-volatility, low-correlation assets. It is
the leftmost point on the efficient frontier and requires only the covariance
matrix (no return forecasts needed).

---
#### 3. Mean-Variance (Tangency Portfolio)
*Find the portfolio with the best return per unit of risk.*

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

---
#### 4. Risk Parity
*Weight each asset inversely proportional to its volatility.*

Each asset $i$ receives a weight proportional to $1/\sigma_i$, normalised so
weights sum to one:
""")
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

        st.divider()
        st.subheader("Portfolio Styles — Backtested Performance")
        st.caption("Monthly rebalancing · in-sample · Equal Weight used for periods with < 30 days of history")

        with st.spinner("Backtesting portfolio styles…"):
            style_rets = backtest_styles(prices)

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
        st.dataframe(pd.DataFrame(rows_s).set_index("Style"), use_container_width=True)

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

    with st.expander("Risk metric definitions"):
        st.markdown(f"""
| Metric | Definition |
|--------|-----------|
| **Historical VaR** | Loss not exceeded on {confidence:.0%} of days, read directly from the return distribution |
| **Parametric VaR** | Assumes normally distributed returns; uses μ and σ to derive the {confidence:.0%} quantile |
| **CVaR / ES** | Expected loss *given* VaR is breached — a coherent, tail-sensitive measure |
| **Max Drawdown** | Largest peak-to-trough decline in the portfolio value over the period |
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
with tab_mc:
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
        p_up = (term_port > float(port_val.iloc[-1])).mean()
        mcard("P(value > today)", f"{p_up:.1%}", p_up > 0.5)

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
        p_up_s = (term_s > last_px).mean()
        mcard("P(price > today)", f"{p_up_s:.1%}", p_up_s > 0.5)

    fig_th2 = go.Figure()
    fig_th2.add_trace(go.Histogram(x=term_s, nbinsx=60,
                                   marker_color=ACCENT, opacity=0.75))
    fig_th2.add_vline(x=last_px, line=dict(color="white", dash="dash"),
                      annotation_text="Current price")
    _layout(fig_th2, f"{mc_t} — Terminal Price Distribution ({mc_days}d)",
            xaxis_title="Price (USD)", yaxis_title="Frequency", h=360)
    st.plotly_chart(fig_th2, use_container_width=True)
