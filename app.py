# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from spx_options_flow_v2 import (
    run_full_analysis,
    UnusualConfig,
)

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Options Flow & Gamma Dashboard",
    layout="wide",
)

st.title("Options Flow & Gamma Dashboard")

st.markdown(
    """
This app wraps your options-analysis engine in a simple UI:

- Choose a ticker (SPY / QQQ / etc.)
- Adjust how aggressive the **unusual activity** filters are
- Inspect price, gamma walls, and flagged options flow
"""
)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Underlying ticker", value="SPY").upper()

max_expiries = st.sidebar.selectbox(
    "Number of expiries to analyze",
    options=[1, 2, 3, 4, 5],
    index=2,
)

rate = st.sidebar.number_input(
    "Risk-free rate (annual)",
    value=0.04,
    step=0.005,
    format="%.3f",
)

div_yield = st.sidebar.number_input(
    "Dividend yield (annual)",
    value=0.012,
    step=0.002,
    format="%.3f",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Unusual activity filters (advanced)")

notional_pct = st.sidebar.slider(
    "Top notional percentile (1 - threshold)",
    min_value=0.90,
    max_value=0.995,
    value=0.97,
    step=0.005,
    help="0.97 ≈ top 3% by notional volume intraday",
)

volume_pct = st.sidebar.slider(
    "Top volume percentile (1 - threshold)",
    min_value=0.90,
    max_value=0.995,
    value=0.97,
    step=0.005,
)

vol_oi_pct = st.sidebar.slider(
    "Top vol/OI percentile (1 - threshold)",
    min_value=0.90,
    max_value=0.995,
    value=0.97,
    step=0.005,
)

min_notional = st.sidebar.number_input(
    "Min $ notional per line (USD)",
    value=1_000_000,
    step=500_000,
)

min_vol_vs_hist = st.sidebar.number_input(
    "Min volume vs history (x)",
    value=5.0,
    step=1.0,
    help="Only kicks in once you have history snapshots stored.",
)

min_oi_change_ratio = st.sidebar.number_input(
    "Min OI change ratio",
    value=0.5,
    step=0.1,
    help="0.5 = OI up at least 50% vs yesterday.",
)

max_dte_short = st.sidebar.number_input(
    "Max DTE for 'short-dated' filter",
    value=7,
    step=1,
)

st.sidebar.markdown("---")
history_lookback = st.sidebar.number_input(
    "History lookback (days)",
    value=60,
    step=10,
    help="Used for volume-vs-history & IV rank (once snapshots accumulate).",
)

run_button = st.sidebar.button("Run analysis")

# ---------------------------------------------------------
# Helper to call your engine (no caching to keep it simple)
# ---------------------------------------------------------
def run_analysis(
    ticker: str,
    max_expiries: int,
    rate: float,
    div_yield: float,
    history_lookback: int,
    notional_pct: float,
    volume_pct: float,
    vol_oi_pct: float,
    min_notional: float,
    min_vol_vs_hist: float,
    min_oi_change_ratio: float,
    max_dte_short: int,
):
    config = UnusualConfig(
        notional_pct=notional_pct,
        volume_pct=volume_pct,
        vol_oi_pct=vol_oi_pct,
        min_vol_vs_hist=min_vol_vs_hist,
        min_oi_change_ratio=min_oi_change_ratio,
        min_dollar_notional=min_notional,
        max_dte_short=max_dte_short,
    )

    result = run_full_analysis(
        ticker=ticker,
        price_period="6mo",
        price_interval="1d",
        expiries=None,  # nearest expiries
        max_expiries=max_expiries,
        rate=rate,
        dividend_yield=div_yield,
        history_lookback_days=history_lookback,
        unusual_config=config,
    )
    return result

# ---------------------------------------------------------
# Main app body
# ---------------------------------------------------------
if run_button:
    with st.spinner("Fetching data and running analysis..."):
        try:
            result = run_analysis(
                ticker=ticker,
                max_expiries=max_expiries,
                rate=rate,
                div_yield=div_yield,
                history_lookback=history_lookback,
                notional_pct=notional_pct,
                volume_pct=volume_pct,
                vol_oi_pct=vol_oi_pct,
                min_notional=min_notional,
                min_vol_vs_hist=min_vol_vs_hist,
                min_oi_change_ratio=min_oi_change_ratio,
                max_dte_short=max_dte_short,
            )
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.stop()

    spot = result["spot"]
    price_df = result["price_df"]
    chain_df = result["chain_df"]
    pcr_info = result["pcr_info"]
    gex_table = result["gex_table"]
    gamma_info = result["gamma_info"]
    clusters = result["clusters"]
    unusual_df = result["unusual_df"]
    narrative = result["narrative"]

    st.subheader(f"Overview for {ticker} (spot ≈ {spot:.2f})")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Price & moving averages (robust to MultiIndex)
    # -----------------------------------------------------
    with col1:
        st.markdown("**Price & Moving Averages (6M)**")

        price_plot = price_df.copy()

        # 1) Extract a 1-D close series regardless of MultiIndex/DataFrame
        close_obj = price_plot["Close"]
        if isinstance(close_obj, pd.DataFrame):
            close_series = close_obj.iloc[:, 0]
        else:
            close_series = close_obj

        # 2) Build plotting DataFrame
        plot_df = pd.DataFrame(
            {
                "Date": price_plot.index,
                "Price": close_series.values,
                "SMA20": price_plot["SMA20"].values,
                "SMA50": price_plot["SMA50"].values,
                "SMA200": price_plot["SMA200"].values,
            }
        )

        fig_price = px.line(
            plot_df,
            x="Date",
            y=["Price", "SMA20", "SMA50", "SMA200"],
            labels={"value": "Price", "variable": "Series"},
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # -----------------------------------------------------
    # Put/Call ratios & narrative
    # -----------------------------------------------------
    with col2:
        st.markdown("**Put/Call Ratios & Narrative**")

        st.write("**Overall Put/Call Ratios:**")
        st.json(pcr_info["overall"])

        st.markdown("**Summary & Interpretation:**")
        # Use text so it preserves line breaks nicely
        st.text(narrative)

    st.markdown("---")

    # -----------------------------------------------------
    # Gamma exposure
    # -----------------------------------------------------
    st.subheader("Gamma Exposure by Strike")

    if not gex_table.empty:
        fig_gex = px.bar(
            gex_table,
            x="strike",
            y="gex",
            labels={"strike": "Strike", "gex": "Gamma Exposure"},
            title="Aggregated Gamma Exposure (all expiries)",
        )
        st.plotly_chart(fig_gex, use_container_width=True)

        st.dataframe(
            gex_table.sort_values("gex", ascending=True).tail(20).style.format(
                {
                    "gex": "{:,.0f}",
                    "total_oi": "{:,.0f}",
                    "call_oi": "{:,.0f}",
                    "put_oi": "{:,.0f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("No gamma exposure data available.")

    st.markdown("---")

    # -----------------------------------------------------
    # Unusual options activity
    # -----------------------------------------------------
    st.subheader("Flagged Unusual Options Activity")

    if unusual_df.empty:
        st.info("No contracts passed the advanced unusual-activity filters with the current settings.")
    else:
        fig_unusual = px.scatter(
            unusual_df,
            x="strike",
            y="dollar_notional",
            color="type",
            size="dollar_notional",
            hover_data=["expiry", "dte", "delta_bucket", "reason"],
            labels={"dollar_notional": "Notio_
