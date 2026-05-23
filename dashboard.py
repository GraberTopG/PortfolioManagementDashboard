"""
US Large Cap Long Only Portfolio Dashboard
==========================================
Streamlit UI shell. All analytics live in analytics.py, data fetching in
data.py, chart builders in charts.py, and constants in config.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

from config import (
    ALL_TICKERS, DEFAULT_TICKERS, BENCHMARK_TICKERS,
    COMPANY_NAMES, TICKER_SECTOR, SPY_SECTOR_WEIGHTS,
    ACCENT, BLUE, RED, GREEN, GOLD, MUTED,
    _FONT, _MONO, _SERIF, _BG, _PLOT, _GRID, _AXIS, _TICK,
    AF, RF,
)
from analytics import (
    ann_return, ann_vol, sharpe, sortino, calmar,
    max_dd, hist_var, param_var, cvar, beta_alpha, hhi, rolling_sharpe,
    rolling_vol, w_equal, w_min_var, w_max_sharpe, w_inv_vol,
    active_share, win_rate, _fmt_num, full_metrics, tracking_error,
    information_ratio,
)
from data import (
    _yf_close, _yf_ohlcv, w_market_cap, load_rf_series,
    backtest_styles, mc_paths,
)
from charts import (
    chart_cumret, chart_sector_exposure,
    chart_benchmark_comparison, chart_price_ta, chart_rsi_macd,
    chart_ef, chart_styles_cumret, chart_dist, chart_dd,
    chart_rolling_var, chart_mc, chart_risk_contribution,
    chart_rolling_sharpe, chart_corr_heatmap,
    render_table, mcard, render_annual_returns,
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="US Large Cap Long Only Portfolio Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="auto")

st.markdown("""
<style>
/* ── IBM Plex fonts (scientific / financial terminal aesthetic) ───────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@300;400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, span, label, li, td, th {
    font-family: 'IBM Plex Serif', Georgia, serif !important;
}

/* ── Typography  -  headings use IBM Plex Serif for gravitas ──────────────── */
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

/* ── KPI metric cards  -  Bloomberg terminal style ─────────────────────────── */
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

/* ── Sidebar collapse button  -  hide raw icon text fallback ───────────────── */
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

/* ── Tabs  -  Bloomberg function-key style ─────────────────────────────────── */
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

/* ══════════════════════════════════════════════════════════════════════════
   MOBILE  (≤ 768 px)
   ══════════════════════════════════════════════════════════════════════════ */
@media screen and (max-width: 768px) {

    /* Main container - tighter padding on small screens */
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }

    /* Headings - scale down so they don't dominate the viewport */
    h1 {
        font-size: 1.35rem !important;
        letter-spacing: 0em !important;
    }
    h2, h3 {
        font-size: 1.0rem !important;
    }

    /* st.columns - stack vertically instead of side by side.
       Targets the flex row that Streamlit generates for every st.columns() call. */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0 !important;
    }
    div[data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* KPI metric cards - tighter padding, smaller value text */
    .mcard {
        padding: 10px 12px !important;
        margin-bottom: 4px !important;
    }
    .mvalue { font-size: 1.05rem !important; }
    .mlabel {
        font-size: 0.57rem !important;
        letter-spacing: 0.07em !important;
    }

    /* Tabs - scroll horizontally so all five labels fit without wrapping */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none !important; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.60rem !important;
        padding: 6px 10px !important;
        letter-spacing: 0.05em !important;
        white-space: nowrap !important;
    }

    /* Captions and body text - slightly smaller */
    .stCaption p, .stMarkdown p { font-size: 0.80rem !important; }

    /* Sliders - prevent overflow on narrow screens */
    div[data-testid="stSlider"] { width: 100% !important; }

    /* Sidebar - collapses natively on mobile; just ensure no min-width forces overflow */
    section[data-testid="stSidebar"] { min-width: 0 !important; }

    /* Selectboxes and multiselects - full width */
    div[data-testid="stSelectbox"],
    div[data-testid="stMultiSelect"] { width: 100% !important; }
}

</style>""", unsafe_allow_html=True)


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
            st.warning(f"Total: {total_pct:.1f}%  -  will be auto-normalised to 100%")
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
        template="plotly_dark",
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
        template="plotly_dark",
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
**Returns**  -  Daily simple returns, compounded to cumulative and rebased to 0% at period start.
Portfolio return is the weighted sum of stock returns: $r_p = \sum_i w_i r_i$.

**Concentration (HHI)**  -  $\text{HHI} = \sum w_i^2$; Effective N $= 1/\text{HHI}$.
Higher HHI means more concentration; Effective N is the equivalent number of equal-weight positions.

**Sector exposure**  -  Active weight = portfolio sector weight minus approximate S&P 500 GICS weight. Positive = overweight vs the index.

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
**MA 20 / 50**  -  Simple moving averages over 20 and 50 days. Price above MA = bullish bias; below = bearish.

**Bollinger Bands**  -  MA20 ± 2 standard deviations. Bands widen in high-volatility periods; prices near the edges signal statistically extended moves.

**RSI (14d)**  -  Momentum oscillator on a 0–100 scale. Above 70 = overbought; below 30 = oversold.
$$\text{RSI} = 100 - \frac{100}{1 + \bar{G}_{14}/\bar{L}_{14}}$$

**MACD (12/26/9)**  -  Difference between a 12-day and 26-day EMA. The 9-day signal line smooths it; crossovers indicate momentum shifts.

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
        from charts import _layout
        _layout(fig_rc, f"Rolling {rw}d Correlation: {t1} vs {t2}",
                yaxis=dict(range=[-1, 1]))
        st.plotly_chart(fig_rc, use_container_width=True)

        import plotly.express as px
        combined = pd.concat([rets_df[t1], rets_df[t2]], axis=1).dropna()
        fig_sc = px.scatter(combined, x=t1, y=t2, opacity=0.45, trendline="ols",
                            color_discrete_sequence=[ACCENT],
                            title=f"Daily Returns Scatter: {t1} vs {t2}")
        fig_sc.update_layout(template="plotly_dark", height=400,
                             paper_bgcolor=_BG, plot_bgcolor=_PLOT,
                             font=dict(family=_FONT, color="#78909C"))
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Select two different assets.")

    st.divider()
    st.subheader("Methodology")
    st.markdown(r"""
**Pearson correlation**  -  Measures linear co-movement of daily returns, ranging from −1 to +1.
$$\rho_{i,j} = \frac{\text{Cov}(r_i, r_j)}{\sigma_i \cdot \sigma_j}$$
High correlation between two holdings reduces diversification benefit. Correlations tend to spike toward +1 during market stress, precisely when diversification matters most.

**Rolling correlation**  -  Shows how the pairwise relationship changes through time. A pair that looks uncorrelated over the full period may have been tightly correlated during drawdowns.

**Return scatter**  -  Each point is one trading day. The OLS trendline fits $r_j = \alpha + \beta r_i$; wider scatter around the line means more residual diversification benefit.
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
        st.subheader("Portfolio Styles  -  Backtested Performance")

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
        # In Equal Weight mode Your Portfolio == Equal Weight  -  drop the duplicate
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

**Transaction costs**  -  The slider above applies a one-way cost per unit of portfolio turnover at each
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
        st.markdown("Only the covariance matrix $\\Sigma$ is needed  -  no return forecasts. Zero-weight allocations are normal.")

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
                    "(current market cap ÷ current price) multiplied by the stock price on the first day of the backtest  -  "
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
    st.subheader(f"Portfolio Risk  -  {port_label}")

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
    from charts import _layout
    _layout(fig_th, f"{port_label}  -  Terminal Value Distribution ({mc_days}d)",
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
    _layout(fig_th2, f"{mc_t}  -  Terminal Price Distribution ({mc_days}d)",
            xaxis_title="Price (USD)", yaxis_title="Frequency", h=360)
    st.plotly_chart(fig_th2, use_container_width=True)

    st.divider()
    st.subheader("Methodology")
    st.markdown(rf"""
**VaR ({confidence:.0%})**  -  The loss level exceeded on only {1-confidence:.0%} of trading days.
- *Historical*: read directly from the return distribution  -  $\text{{VaR}} = -\text{{Percentile}}(r,\, {(1-confidence)*100:.0f}\%)$
- *Parametric*: assumes normally distributed returns  -  $\text{{VaR}} = -(\mu + z_{{1-\alpha}}\,\sigma)$

**CVaR**  -  Average loss on days VaR is breached. Always $\geq$ VaR; more sensitive to tail events.

**Drawdown**  -  Decline from the most recent peak: $\text{{DD}}_t = (P_t - \max_{{s\leq t}} P_s)\,/\,\max_{{s\leq t}} P_s$.

**Risk Contribution**  -  Each stock's share of total portfolio variance: $\text{{RC}}_i = w_i\,(\Sigma w)_i\,/\,w^\top\Sigma w$. A stock that is a small position but dominates risk is a candidate for trimming; one that is a large position but low-risk adds diversification efficiently.

**Rolling VaR**  -  VaR recomputed over a sliding window. Spikes indicate periods of elevated risk.

**Rolling Sharpe**  -  Annualised Sharpe ratio over a trailing 252-day window. Falls below zero when the portfolio underperforms the risk-free rate on a risk-adjusted basis.

**Monte Carlo (GBM)**  -  Simulates future price paths using Geometric Brownian Motion with zero drift:
$$P_t = P_{{t-1}}\cdot e^{{\varepsilon_t}},\qquad \varepsilon_t\sim\mathcal{{N}}(0,\,\hat{{\sigma}})$$
Zero drift is used instead of historical mean returns to avoid extrapolating a bull-market bias into the forecast. The fan chart shows the 10th–90th percentile range of simulated paths.

**P(loss > 15%)**  -  Share of simulated paths ending more than 15% below today's value. More informative than P(value > today), which is upward-biased by the log-normal distribution.

*Limitations: GBM assumes constant volatility, normal shocks, and independent daily returns. Real markets exhibit volatility clustering, fat tails, and serial correlation.*
""")
