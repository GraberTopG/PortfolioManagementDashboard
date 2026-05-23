"""
analytics.py - Pure quantitative analytics functions.
=======================================================
Contains return calculations, risk metrics (VaR, CVaR, drawdown),
factor metrics (beta, alpha, tracking error), technical indicators
(Bollinger Bands, RSI, MACD), and portfolio optimisation routines.
No I/O, no Streamlit calls - all functions are pure Python/numpy/scipy.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from config import AF, RF, MAX_SINGLE_W

# ── Return calculations ───────────────────────────────────────────────────────

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
        port_w:  Portfolio weight vector  -  required for Active Share calculation.
        bench_w: Benchmark weight vector  -  required for Active Share calculation.

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

def _port_stats(w, mu, cov, rf=RF):
    """Compute annualised portfolio return, volatility, and Sharpe ratio.

    Args:
        w:   Weight vector (sums to 1).
        mu:  Daily mean return vector.
        cov: Daily covariance matrix.
        rf:  Annualised risk-free rate (default: global RF constant).

    Returns:
        (ann_return, ann_vol, sharpe)  -  all annualised.
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

    Only the covariance matrix is needed  -  expected returns are not used,
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
    No return forecasts are needed  -  only the per-asset volatility.
    """
    vols = rets_df.std()
    iv = 1 / vols
    return (iv / iv.sum()).values

def w_equal(n):
    """Equal-weight portfolio: w_i = 1/N for all i."""
    return np.ones(n) / n

def efficient_frontier_mc(mu, cov, n=500):
    """Sample n random portfolios via Dirichlet weights to map the feasible set.

    Dirichlet(ones) produces uniformly distributed weight vectors on the simplex
    (all weights positive, sum to 1). Each portfolio's (vol, return, Sharpe) is
    computed so the scatter cloud can be colour-coded by Sharpe ratio.

    Note: these are NOT constrained by MAX_SINGLE_W  -  they represent the full
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
