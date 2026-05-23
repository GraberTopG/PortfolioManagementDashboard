"""
charts.py - Plotly chart builders and Streamlit UI helper components.
======================================================================
All chart-rendering functions return go.Figure objects or render directly
into Streamlit (render_table, mcard, render_annual_returns). No business
logic lives here - inputs are already computed arrays/series from analytics.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st

from config import (
    AF, RF, ACCENT, BLUE, RED, GREEN, GOLD, MUTED,
    CHART_TEMPLATE, TICKER_SECTOR, SPY_SECTOR_WEIGHTS,
    _FONT, _MONO, _SERIF, _BG, _PLOT, _GRID, _AXIS, _TICK,
)
from analytics import (
    hist_var, dd_series, bollinger, rsi, macd,
    rolling_sharpe, efficient_frontier_mc, w_min_var, w_max_sharpe,
    _port_stats,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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
    # Professional Tailwind-based palette  -  readable on dark backgrounds
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

    # chart_cumret will rebase each series to 0%  -  pass raw cumulative products
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
    # Dollar volume (shares × price)  -  immune to split-adjustment inflation.
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
                      title=dict(text=f"<b>{ticker}  -  Price & Volume</b>",
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
    # is larger)  -  ensures the orange marker is never clipped off the chart
    y_max = max(np.percentile(rets, 98) * 1.25, r_ms * 1.15)
    y_min = min(min(rets) * 1.1, 0)

    fig = go.Figure()

    # ── Scatter cloud (random portfolios) ────────────────────────────────────
    # Sequential scale: dark navy (low Sharpe) -> Bloomberg blue (high Sharpe).
    # Gives the cloud depth and encodes information without competing with the
    # orange Mean-Variance and blue Min-Variance marker circles.
    fig.add_trace(go.Scatter(
        x=vols, y=rets, mode="markers",
        marker=dict(
            color=srs,
            # Warm amber gradient: near-black (low Sharpe) -> Bloomberg amber (high Sharpe)
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
        ("Min Variance",  r_mv, v_mv, w_mv, "#E0E4EA"),   # near-white  -  stands out from amber cloud
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

    # ── User's custom portfolio (gold diamond  -  only shown when Custom weights set) ──
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
        "Your Portfolio": "#E0E4EA",   # near-white  -  matches EF custom marker
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
    return _layout(fig, "Portfolio Styles  -  Cumulative Return",
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
    return _layout(fig, f"{label}  -  Return Distribution",
                   xaxis_title="Daily Return", yaxis_title="Density", h=400)


def chart_dd(prices: pd.Series, label: str) -> go.Figure:
    dd = dd_series(prices) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                             line=dict(color=RED, width=1.2),
                             fillcolor="rgba(255,75,75,0.18)"))
    return _layout(fig, f"{label}  -  Drawdown (%)", yaxis_title="Drawdown (%)", h=340)


def chart_rolling_var(r: pd.Series, label: str, w: int, c: float) -> go.Figure:
    rv = r.rolling(w).apply(lambda x: hist_var(pd.Series(x), c), raw=False) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv.index, y=rv, fill="tozeroy",
                             line=dict(color=RED, width=1.5),
                             fillcolor="rgba(255,75,75,0.15)",
                             name=f"Rolling {c:.0%} VaR"))
    return _layout(fig, f"{label}  -  Rolling {w}d Historical VaR ({c:.0%})",
                   yaxis_title="VaR (%)", h=360)


def chart_mc(paths: np.ndarray, label: str, last_val: float) -> go.Figure:
    """Animated Monte Carlo chart with Plotly Play / Pause buttons.

    Paths grow left-to-right when the user presses Play  -  all animation is
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
            text=f"<b>{label}  -  Monte Carlo ({n_days}d forecast)</b>",
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
    return _layout(fig, f"{label}  -  Rolling {w}d Sharpe Ratio",
                   yaxis_title="Sharpe Ratio", h=340)


def chart_corr_heatmap(prices: pd.DataFrame) -> go.Figure:
    corr = prices.pct_change().dropna().corr()
    # Diverging scale: dark burgundy (negative) -> steel grey (zero) -> deep navy (positive)
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


def render_table(df: pd.DataFrame) -> None:
    """Bloomberg-styled static HTML table  -  no menus, no dropdowns."""
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
        + "".join(f'<td style="{td_style}">{str(v).replace(" - ", "-")}</td>' for v in row)
        + "</tr>"
        for idx, row in df.iterrows()
    )
    st.markdown(
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch">'
        f'<table style="width:100%;border-collapse:collapse;min-width:400px">'
        f'<thead><tr><th style="{th_style}"></th>{header}</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>',
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


def render_annual_returns(style_rets: dict) -> None:
    """Year-by-year returns table  -  green for positive, red for negative."""
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
                col, txt = "#546E7A", " - "
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
