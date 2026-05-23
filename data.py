"""
data.py - Market data loaders, backtest engine, and Monte Carlo paths.
========================================================================
Fetches price data from Yahoo Finance via yfinance, runs the walk-forward
portfolio style backtest, and generates GBM Monte Carlo simulation paths.
All yfinance calls are decorated with @st.cache_data to avoid redundant
network requests during interactive Streamlit sessions.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from config import RF
from analytics import w_equal, w_min_var, w_max_sharpe, w_inv_vol

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER   -   Yahoo Finance (yfinance), no API key required
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


@st.cache_data(ttl=86400, show_spinner=False)
def w_market_cap(tickers_csv: str) -> np.ndarray:
    """Fetch CURRENT market-cap weights via yfinance fast_info.

    NOTE: This function returns TODAY's market-cap weights. It is used only for
    (1) display in the Optimal Portfolio Allocations table and (2) as the active-
    share benchmark in full_metrics(). It is NOT used as starting weights in the
    backtest  -  those are derived from historical prices to avoid look-ahead bias.
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
    returns the last known T-bill rate on or before that date  -  no look-ahead.
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
    The first rebalancing date is treated as portfolio inception  -  no turnover
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
                # Mean-Variance: uses hist.mean() as expected returns  -
                # no look-ahead, but sample means are noisy estimators.
                # RF rate is the historically correct T-bill rate at rebal date.
                "Mean-Variance": w_max_sharpe(mu, cov, rf=rf_t),
                # Inverse-vol weighting (simplified Risk Parity):
                # vol estimated from expanding window; improves as data accumulates
                "Risk Parity":   w_inv_vol(hist),
            }
        else:
            # Too few observations  -  fall back to equal weight
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
    # This removes look-ahead bias  -  the starting weights reflect relative
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
