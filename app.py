import streamlit as st
import pandas as pd
import plotly.express as px

from spx_options_flow_v2 import (
    run_full_analysis,
    UnusualConfig,
    vol_complex_summary,   # imported for completeness / potential future use
    macro_tape_summary,    # idem
)

st.set_page_config(page_title="Options Flow & Gamma Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Helper to call the analysis engine with UI parameters
# ---------------------------------------------------------------------------


def run_analysis(
    ticker: str,
    max_expiries: int,
    history_lookback: int,
    config: UnusualConfig,
):
    """
    Wrapper that forwards UI parameters into the analysis engine.
    """
    result = run_full_analysis(
        ticker=ticker,
        price_period="6mo",
        price_interval="1d",
        max_expiries=max_expiries,
        history_lookback_days=history_lookback,
        unusual_config=config,
    )
    return result


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


st.sidebar.header("Controls")

ticker = st.sidebar.text_input("Underlying ticker (index or ETF)", value="SPY")

max_expiries = st.sidebar.slider(
    "Max expiries to load",
    min_value=1,
    max_value=6,
    value=3,
    step=1,
    help="Number of closest expiries to pull from the options chain.",
)

st.sidebar.markdown("### Unusual activity filters (advanced)")

notional_pct = st.sidebar.slider(
    "Top notional percentile (1 − threshold)",
    min_value=0.90,
    max_value=0.999,
    value=0.97,
    step=0.001,
)
volume_pct = st.sidebar.slider(
    "Top volume percentile (1 − threshold)",
    min_value=0.90,
    max_value=0.999,
    value=0.97,
    step=0.001,
)
vol_oi_pct = st.sidebar.slider(
    "Top vol/OI percentile (1 − threshold)",
    min_value=0.90,
    max_value=0.999,
    value=0.97,
    step=0.001,
)
min_notional = st.sidebar.number_input(
    "Min $ notional per line (USD)",
    min_value=0.0,
    value=1_000_000.0,
    step=100_000.0,
    format="%.0f",
)
min_vol_vs_hist = st.sidebar.slider(
    "Min volume vs history (x)",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
)
min_oi_change_ratio = st.sidebar.slider(
    "Min OI change ratio",
    min_value=0.0,
    max_value=2.0,
    value=0.5,
    step=0.1,
)
max_dte_short = st.sidebar.slider(
    "Max DTE for 'short-dated' filter",
    min_value=1,
    max_value=30,
    value=7,
    step=1,
)
history_lookback = st.sidebar.slider(
    "History lookback (days)",
    min_value=10,
    max_value=120,
    value=60,
    step=5,
)

run_button = st.sidebar.button("Run analysis")

config = UnusualConfig(
    notional_pct=notional_pct,
    volume_pct=volume_pct,
    vol_oi_pct=vol_oi_pct,
    min_vol_vs_hist=min_vol_vs_hist,
    min_oi_change_ratio=min_oi_change_ratio,
    min_dollar_notional=min_notional,
    max_dte_short=max_dte_short,
)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


st.title("Options Flow & Gamma Dashboard")
st.caption(
    "Intraday snapshot of SPX/ETF options flow, gamma positioning, volatility regime "
    "and macro tape. Use this as a risk compass for short-term trading."
)

st.markdown(
    """
This app wraps your options-analysis engine in a simple UI:

- Choose a ticker (SPY / QQQ / index ETF / etc.)
- Adjust how aggressive the **unusual activity** filters are
- Inspect price, gamma walls, and flagged options flow
- Overlay **volatility regime** (VIX, IV vs RV, skew) and **macro tape** (rates, credit, dollar)
"""
)


if run_button:
    with st.spinner("Fetching data and running analysis…"):
        try:
            result = run_analysis(ticker, max_expiries, history_lookback, config)
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
    vol_info = result["vol_info"]
    macro = result["macro"]

    tab1, tab2, tab3 = st.tabs(
        [
            "Options Flow & Gamma",
            "Volatility Complex – Regime View",
            "Macro Tape – Rates, Credit, Dollar (6M)",
        ]
    )

    # ------------------------------------------------------------------
    # Tab 1 – options flow & gamma
    # ------------------------------------------------------------------
    with tab1:
        st.subheader(f"Overview for {ticker} (spot ≈ {spot:.2f})")

        st.markdown(
            """
**How to use this view**

- Check whether options flow is **call- or put-heavy**.
- Locate key **gamma levels** (zero-gamma, call/put walls) around spot.
- Scan for **large, short-dated trades** clustered in strike / expiry.
"""
        )

        col_price, col_pcr = st.columns([2.5, 1.5])

        with col_price:
            st.markdown("#### Price & Moving Averages (6M)")
            px_price = price_df.reset_index()
            if "Date" not in px_price.columns:
                px_price = px_price.rename(columns={px_price.columns[0]: "Date"})

            fig_price = px.line(
                px_price,
                x="Date",
                y=["Close", "SMA20", "SMA50", "SMA200"],
                labels={"value": "Price", "variable": "Series"},
            )
            fig_price.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_price, use_container_width=True)

        with col_pcr:
            st.markdown("#### Put/Call Ratios & Narrative")
            overall = pcr_info["overall"]
            pcr_df = pd.DataFrame(
                {
                    "call_vol": [overall["call_vol"]],
                    "put_vol": [overall["put_vol"]],
                    "call_oi": [overall["call_oi"]],
                    "put_oi": [overall["put_oi"]],
                    "pcr_vol": [overall["pcr_vol"]],
                    "pcr_oi": [overall["pcr_oi"]],
                }
            ).T
            pcr_df.columns = ["value"]
            st.dataframe(pcr_df.style.format("{:,.2f}"))

            st.markdown("**Summary & Interpretation:**")
            st.write(narrative)

        st.markdown("---")
        st.markdown("### Gamma Exposure by Strike")

        if gex_table.empty:
            st.info("No gamma information available.")
        else:
            fig_gex = px.bar(
                gex_table,
                x="strike",
                y="gex",
                labels={"strike": "Strike", "gex": "Gamma exposure"},
            )
            st.plotly_chart(fig_gex, use_container_width=True)
            st.dataframe(
                gex_table.sort_values("gex", ascending=False),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Flagged Unusual Options Activity")

        if unusual_df.empty:
            st.info("No trades matched your unusual-flow filters.")
        else:
            bubble_df = unusual_df.copy()
            fig_bubble = px.scatter(
                bubble_df,
                x="strike",
                y="dollar_notional",
                size="dollar_notional",
                color="type",
                hover_data=["contractSymbol", "expiry", "dte", "reason"],
                labels={"dollar_notional": "Notional ($)", "strike": "Strike"},
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

            st.dataframe(
                unusual_df,
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Notional Clusters by Expiry & Moneyness")

        if clusters.empty:
            st.info("No notional cluster data.")
        else:
            st.dataframe(
                clusters.sort_values("total_notional", ascending=False),
                use_container_width=True,
            )

            top_clusters = clusters.sort_values("total_notional", ascending=False).head(15)
            fig_clusters = px.bar(
                top_clusters,
                x="moneyness_band",
                y="total_notional",
                color="expiry",
                labels={
                    "moneyness_band": "Moneyness band (% from spot)",
                    "total_notional": "Total notional ($)",
                },
            )
            st.plotly_chart(fig_clusters, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 2 – volatility complex
    # ------------------------------------------------------------------
    with tab2:
        st.markdown("### Volatility Complex – Regime View")

        rv20 = vol_info["rv20"]
        atm_iv_30d = vol_info["atm_iv_30d"]
        iv_rv_label = vol_info["iv_rv_label"]
        vix = vol_info["vix"]
        vix3m = vol_info["vix3m"]
        vix_term_label = vol_info["vix_term_label"]
        skew_label = vol_info["skew_label"]
        vix_long = vol_info["vix_long_df"]

        st.markdown("#### Key metrics (approx 30D):")
        bullet_lines = [
            f"- 20D realized vol: **{rv20:.1f}%**" if rv20 == rv20 else "- 20D realized vol: n/a",
            f"- 30D ATM implied vol: **{atm_iv_30d:.1f}%**" if atm_iv_30d == atm_iv_30d else "- 30D ATM implied vol: n/a",
            f"- IV vs RV: **{iv_rv_label}**",
            f"- Spot VIX: **{vix:.1f}**, 3M VIX: **{vix3m:.1f}**" if vix == vix and vix3m == vix3m else "- VIX data: n/a",
            f"- Term structure: **{vix_term_label}**",
            f"- 30D skew (25Δ put – call): **{skew_label}**",
        ]
        st.markdown("\n".join(bullet_lines))

        col_vix, col_bar = st.columns([2.2, 1.8])

        with col_vix:
            st.markdown("#### VIX Term Structure (6M)")
            if vix_long.empty:
                st.info("No VIX data.")
            else:
                fig_vix = px.line(
                    vix_long,
                    x="Date",
                    y="Level",
                    color="Index",
                    labels={"Level": "Index level"},
                )
                st.plotly_chart(fig_vix, use_container_width=True)

        with col_bar:
            st.markdown("#### Realized vs Implied Vol (approx)")
            vol_bar = pd.DataFrame(
                {
                    "Metric": ["20D Realized", "30D ATM Implied"],
                    "Value": [rv20, atm_iv_30d],
                }
            )
            fig_bar = px.bar(vol_bar, x="Metric", y="Value", labels={"Value": "Annualised vol (%)"})
            st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 3 – macro tape
    # ------------------------------------------------------------------
    with tab3:
        st.markdown("### Macro Tape – Rates, Credit, Dollar (6M)")

        metrics = macro["metrics"]
        yc_df = macro["yield_curve_df"]
        credit_df = macro["credit_df"]
        dollar_df = macro["dollar_df"]

        st.markdown("#### Summary:")
        st.markdown(
            f"""
- 10Y: **{metrics['rates']['last_10y']*100:.2f}%**, 3M: **{metrics['rates']['last_3m']*100:.2f}%**, slope: **{metrics['rates']['slope']*100:.2f}pp** → {metrics['rates']['regime']}
- Credit (HYG/LQD): level **{metrics['credit']['last_ratio']:.3f}**, z-score **{metrics['credit']['zscore']:.2f}** → {metrics['credit']['regime']}
- Dollar (UUP): last **{metrics['dollar']['last']:.2f}** → {metrics['dollar']['regime']}
"""
        )

        col_yc, col_credit, col_dollar = st.columns(3)

        with col_yc:
            st.markdown("#### US Yield Curve (10Y vs 3M)")
            if yc_df.empty:
                st.info("No yield data.")
            else:
                fig_yc = px.line(
                    yc_df,
                    x="Date",
                    y="Yield",
                    color="Tenor",
                    labels={"Yield": "Yield (%)"},
                )
                st.plotly_chart(fig_yc, use_container_width=True)

        with col_credit:
            st.markdown("#### Credit Risk Appetite (HYG/LQD)")
            if credit_df.empty:
                st.info("No credit data.")
            else:
                fig_credit = px.line(
                    credit_df,
                    x="Date",
                    y="HYG/LQD",
                    labels={"HYG/LQD": "HYG / LQD"},
                )
                st.plotly_chart(fig_credit, use_container_width=True)

        with col_dollar:
            st.markdown("#### Dollar ETF (UUP)")
            if dollar_df.empty:
                st.info("No dollar data.")
            else:
                fig_dollar = px.line(
                    dollar_df,
                    x="Date",
                    y="UUP",
                    labels={"UUP": "UUP"},
                )
                st.plotly_chart(fig_dollar, use_container_width=True)

else:
    st.info("Set your filters in the sidebar and click **Run analysis**.")
