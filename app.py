# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from spx_options_flow_v2 import (
    run_full_analysis,
    UnusualConfig,
)

st.set_page_config(
    page_title="Options Flow & Gamma Dashboard",
    layout="wide",
)

st.title("Options Flow & Gamma Dashboard")

st.markdown(
    """
This app wraps your analysis engine around a simple UI.

- Choose a ticker (SPY / QQQ / etc.)
- Adjust how aggressive the **unusual activity** filter is
- Inspect price, gamma walls, and flagged options flow
"""
)

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Underlying ticker", value="SPY").upper()

max_expiries = st.sidebar.selectbox(
    "Number of expiries to analyze",
    options=[1, 2, 3, 4, 5],
    index=2,
)

rate = st.sidebar.number_input("Risk-free rate (annual)", value=0.04, step=0.005, format="%.3f")
div_yield = st.sidebar.number_input("Dividend yield (annual)", value=0.012, step=0.002, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.subheader("Unusual activity filters (advanced)")

notional_pct = st.sidebar.slider(
    "Top notional percentile (1 - threshold)",
    min_value=0.90,
    max_value=0.995,
    value=0.97,
    step=0.005,
    help="0.97 = top 3% by notional volume intraday",
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
    help="Only used once you have a few days of snapshots",
)

min_oi_change_ratio = st.sidebar.number_input(
    "Min OI change ratio",
    value=0.5,
    step=0.1,
    help="0.5 = OI up at least 50% vs yesterday",
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
    help="Used for vol vs history & IV rank (once snapshots accumulate).",
)

run_button = st.sidebar.button("Run analysis")


# ------------------------------
# Main logic (with caching)
# ------------------------------

@st.cache_data(show_spinner=True)
def run_analysis_cached(
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


if run_button:
    with st.spinner("Fetching data and running analysis..."):
        try:
            result = run_analysis_cached(
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

    # ------------------------------
    # Layout sections
    # ------------------------------
    st.subheader(f"Overview for {ticker} (spot ≈ {spot:.2f})")

    col1, col2 = st.columns(2)

    # --- Price & MAs chart ---
    with col1:
        st.markdown("**Price & Moving Averages (6M)**")

        # Flatten MultiIndex if necessary
        price_plot = price_df.copy()
        if isinstance(price_plot.columns, pd.MultiIndex):
            price_plot.columns = ["_".join([str(c) for c in col]).strip("_") for col in price_plot.columns]

        # Build a tidy frame for plotting
        price_plot = price_plot[["Close", "SMA20", "SMA50", "SMA200"]]
        price_plot = price_plot.rename(columns={"Close": "Price"})

        fig_price = px.line(
            price_plot.reset_index(),
            x="Date",
            y=["Price", "SMA20", "SMA50", "SMA200"],
            labels={"value": "Price", "variable": "Series"},
        )
        st.plotly_chart(fig_price, use_container_width=True)

    # --- PCR & narrative ---
    with col2:
        st.markdown("**Put/Call Ratios & Narrative**")

        st.write("**Overall Put/Call Ratios:**")
        st.json(pcr_info["overall"])

        st.markdown("**Summary & Interpretation:**")
        st.text(narrative)

    st.markdown("---")

    # ------------------------------
    # Gamma exposure
    # ------------------------------
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
                {"gex": "{:,.0f}", "total_oi": "{:,.0f}", "call_oi": "{:,.0f}", "put_oi": "{:,.0f}"}
            ),
            use_container_width=True,
        )
    else:
        st.info("No gamma exposure data available.")

    st.markdown("---")

    # ------------------------------
    # Unusual activity
    # ------------------------------
    st.subheader("Flagged Unusual Options Activity")

    if unusual_df.empty:
        st.info("No contracts passed the advanced unusual-activity filters with current settings.")
    else:
        # Small chart: notional by strike vs type
        fig_unusual = px.scatter(
            unusual_df,
            x="strike",
            y="dollar_notional",
            color="type",
            size="dollar_notional",
            hover_data=["expiry", "dte", "delta_bucket", "reason"],
            labels={"dollar_notional": "Notional ($)", "strike": "Strike"},
            title="Unusual Trades by Strike and Notional",
        )
        st.plotly_chart(fig_unusual, use_container_width=True)

        # Data table
        st.dataframe(
            unusual_df.head(200).style.format(
                {
                    "dollar_notional": "{:,.0f}",
                    "volume": "{:,.0f}",
                    "openInterest": "{:,.0f}",
                    "vol_oi_ratio": "{:,.2f}",
                    "iv": "{:.3f}",
                    "delta": "{:.3f}",
                    "gamma": "{:.3f}",
                    "vega": "{:.3f}",
                    "theta": "{:.3f}",
                    "moneyness": "{:.3%}",
                    "vol_vs_hist": "{:.2f}",
                    "oi_change_ratio": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("---")

    # ------------------------------
    # Clusters
    # ------------------------------
    st.subheader("Notional Clusters by Expiry & Moneyness")

    if not clusters.empty:
        st.dataframe(
            clusters.head(50).style.format(
                {
                    "total_notional": "{:,.0f}",
                    "call_notional": "{:,.0f}",
                    "put_notional": "{:,.0f}",
                    "net_call_minus_put": "{:,.0f}",
                    "avg_moneyness": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        fig_clusters = px.bar(
            clusters.head(20),
            x="moneyness_band",
            y="total_notional",
            color="expiry",
            labels={"moneyness_band": "Moneyness band (% from spot)", "total_notional": "Total notional ($)"},
            title="Top Notional Clusters (by expiry & moneyness band)",
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.info("No cluster information available.")
else:
    st.info("Set your parameters in the sidebar and click **Run analysis**.")
