import streamlit as st
import pandas as pd
import plotly.express as px

from spx_options_flow_v2 import (
    run_full_analysis,
    UnusualConfig,
    vol_complex_summary,
    macro_tape_summary,
)

# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Options Flow & Gamma Dashboard (v2)",
    layout="wide",
)

st.title("Options Flow & Gamma Dashboard")

st.markdown(
    """
This app wraps your options-analysis engine in a simple UI:

- Choose a ticker (SPY / QQQ / etc.)
- Adjust how aggressive the **unusual activity** filters are
- Inspect price, gamma walls, flagged options flow
- Overlay **volatility regime** (VIX, RV vs IV, skew) and **macro tape** (rates, credit, dollar)
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
# Helper to call your engine
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

        # Extract a 1-D close series regardless of MultiIndex/DataFrame
        close_obj = price_plot["Close"]
        if isinstance(close_obj, pd.DataFrame):
            close_series = close_obj.iloc[:, 0]
        else:
            close_series = close_obj

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
        st.text(narrative)

    # -----------------------------------------------------
    # Volatility Complex (Pillar 1)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Volatility Complex – Regime View")

    vol_info = vol_complex_summary(price_df, chain_df)

    rv20 = vol_info["rv20"]
    atm_iv_30d = vol_info["atm_iv_30d"]
    vix = vol_info["vix"]
    vix3m = vol_info["vix3m"]
    iv_rv_label = vol_info["iv_rv_label"]
    vix_term_label = vol_info["vix_term_label"]
    skew_label = vol_info["skew_label"]
    vix_long = vol_info["vix_long_df"]

    colv1, colv2 = st.columns([1, 2])

    with colv1:
        st.markdown("**Key metrics (approx 30D):**")
        if rv20 is not None:
            st.write(f"- 20D realized vol: **{rv20:.1f}%**")
        else:
            st.write("- 20D realized vol: *(insufficient data)*")

        if atm_iv_30d is not None:
            st.write(f"- 30D ATM implied vol: **{atm_iv_30d:.1f}%**")
        else:
            st.write("- 30D ATM implied vol: *(could not infer)*")

        st.write(f"- IV vs RV: **{iv_rv_label}**")

        if vix is not None:
            st.write(f"- Spot VIX: **{vix:.1f}**")
        if vix3m is not None:
            st.write(f"- 3M VIX: **{vix3m:.1f}**")

        st.write(f"- Term structure: **{vix_term_label}**")
        st.write(f"- 30D skew (25Δ put – call): **{skew_label}**")

        st.caption(
            "Heuristic: IV >> RV + steep negative skew = rich downside protection "
            "→ better for selling vol / being picky buying gamma. "
            "IV ~ RV + mild skew = optionality relatively cheap."
        )

    with colv2:
        # VIX term-structure chart
        if vix_long is not None and not vix_long.empty:
            fig_vix = px.line(
                vix_long,
                x="Date",
                y="Level",
                color="Index",
                title="VIX Term Structure (6M)",
                labels={"Level": "Index level"},
            )
            st.plotly_chart(fig_vix, use_container_width=True)

        # IV vs RV bar chart
        bars = []
        vals = []
        if rv20 is not None:
            bars.append("20D Realized")
            vals.append(rv20)
        if atm_iv_30d is not None:
            bars.append("30D ATM Implied")
            vals.append(atm_iv_30d)

        if bars:
            iv_rv_df = pd.DataFrame({"Metric": bars, "Vol": vals})
            fig_ivrv = px.bar(
                iv_rv_df,
                x="Metric",
                y="Vol",
                title="Realized vs Implied Vol (approx)",
                labels={"Vol": "Annualized vol (%)"},
            )
            st.plotly_chart(fig_ivrv, use_container_width=True)

    # -----------------------------------------------------
    # Macro Tape – Rates, Credit, Dollar (Pillar 4)
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Macro Tape – Rates, Credit, Dollar (6M)")

    macro = macro_tape_summary(period="6mo", interval="1d")
    metrics = macro["metrics"]
    yc_df = macro["yield_curve_df"]
    credit_df = macro["credit_df"]
    dollar_df = macro["dollar_df"]

    colm1, colm2 = st.columns([1, 2])

    with colm1:
        st.markdown("**Summary:**")

        rates = metrics.get("rates")
        credit = metrics.get("credit")
        dollar = metrics.get("dollar")

        if rates:
            st.write(
                f"- 10Y: **{rates['last_10y']:.2f}%**, 3M: **{rates['last_3m']:.2f}%**, "
                f"slope: **{rates['slope']:.2f}pp** → {rates['regime']}"
            )
        else:
            st.write("- Rates: *(data unavailable)*")

        if credit:
            st.write(
                f"- Credit (HYG/LQD): level **{credit['last_ratio']:.3f}**, "
                f"z-score **{credit['zscore']:.2f}** → {credit['regime']}"
            )
        else:
            st.write("- Credit: *(data unavailable)*")

        if dollar:
            st.write(
                f"- Dollar (UUP): last **{dollar['last']:.2f}**, regime: {dollar['regime']}"
            )
        else:
            st.write("- Dollar: *(data unavailable)*")

        st.caption(
            "Heuristic: deep curve inversion + credit risk-off + strong dollar "
            "→ macro headwind for SPX. The opposite is macro tailwind."
        )

    with colm2:
        # Yield curve plot
        if yc_df is not None and not yc_df.empty:
            fig_yc = px.line(
                yc_df,
                x="Date",
                y="Yield",
                color="Tenor",
                title="US Yield Curve (10Y vs 3M)",
                labels={"Yield": "Yield (%)"},
            )
            st.plotly_chart(fig_yc, use_container_width=True)

        # Credit ratio plot
        if credit_df is not None and not credit_df.empty:
            fig_cr = px.line(
                credit_df,
                x="Date",
                y="HYG/LQD",
                title="Credit Risk Appetite (HYG/LQD)",
            )
            st.plotly_chart(fig_cr, use_container_width=True)

        # Dollar trend plot
        if dollar_df is not None and not dollar_df.empty:
            fig_uup = px.line(
                dollar_df,
                x="Date",
                y="UUP",
                title="Dollar ETF (UUP)",
            )
            st.plotly_chart(fig_uup, use_container_width=True)

    # -----------------------------------------------------
    # Gamma exposure
    # -----------------------------------------------------
    st.markdown("---")
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

    # -----------------------------------------------------
    # Unusual options activity
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Flagged Unusual Options Activity")

    if unusual_df.empty:
        st.info(
            "No contracts passed the advanced unusual-activity filters with the current settings."
        )
    else:
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

    # -----------------------------------------------------
    # Notional clusters
    # -----------------------------------------------------
    st.markdown("---")
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
            labels={
                "moneyness_band": "Moneyness band (% from spot)",
                "total_notional": "Total notional ($)",
            },
            title="Top Notional Clusters (by expiry & moneyness band)",
        )
        st.plotly_chart(fig_clusters, use_container_width=True)
    else:
        st.info("No cluster information available.")
else:
    st.info("Set your parameters in the sidebar and click **Run analysis**.")
