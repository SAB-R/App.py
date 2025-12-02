"""
Options flow + gamma analysis engine for the Streamlit dashboard.

- Fetches price history & adds moving averages
- Downloads options chains for chosen expiries
- Computes IV & Greeks (Black–Scholes)
- Computes gamma exposure (GEX) & zero-gamma level
- Computes put/call ratios
- Saves daily snapshots to options_history/
- Loads history to compute vol_vs_hist, OI change, IV rank
- Flags unusual options activity
- Builds notional clusters
- Produces a narrative summary

Designed to work with the Streamlit app in app.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------

CONTRACT_SIZE = 100
DATA_DIR = Path("options_history")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Utility: normal pdf / cdf without scipy
# ---------------------------------------------------------------------


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    # Abramowitz & Stegun approximation via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------
# Price history + moving averages
# ---------------------------------------------------------------------


def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No price data for {ticker} ({period}, {interval})")

    close = df["Close"].copy()
    df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA50"] = close.rolling(window=50, min_periods=1).mean()
    df["SMA200"] = close.rolling(window=200, min_periods=1).mean()
    return df


# ---------------------------------------------------------------------
# Black–Scholes pricing & Greeks
# ---------------------------------------------------------------------


def _bs_price(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    div_yield: float,
    vol: float,
    is_call: bool,
) -> float:
    if time <= 0 or vol <= 0:
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        return intrinsic
    sqrt_t = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * time) / (
        vol * sqrt_t
    )
    d2 = d1 - vol * sqrt_t
    disc_r = math.exp(-rate * time)
    disc_q = math.exp(-div_yield * time)

    if is_call:
        return disc_q * spot * _norm_cdf(d1) - disc_r * strike * _norm_cdf(d2)
    else:
        return disc_r * strike * _norm_cdf(-d2) - disc_q * spot * _norm_cdf(-d1)


def _bs_greeks(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    div_yield: float,
    vol: float,
    is_call: bool,
) -> Dict[str, float]:
    if time <= 0 or vol <= 0:
        # Very close to expiry: approximate with intrinsic-style greeks
        intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
        delta = 1.0 if (is_call and intrinsic > 0) else (-1.0 if (not is_call and intrinsic > 0) else 0.0)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    sqrt_t = math.sqrt(time)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol * vol) * time) / (
        vol * sqrt_t
    )
    d2 = d1 - vol * sqrt_t
    disc_q = math.exp(-div_yield * time)
    disc_r = math.exp(-rate * time)

    pdf_d1 = _norm_pdf(d1)

    if is_call:
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -disc_q * spot * pdf_d1 * vol / (2 * sqrt_t)
            - rate * disc_r * strike * _norm_cdf(d2)
            + div_yield * disc_q * spot * _norm_cdf(d1)
        )
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -disc_q * spot * pdf_d1 * vol / (2 * sqrt_t)
            + rate * disc_r * strike * _norm_cdf(-d2)
            - div_yield * disc_q * spot * _norm_cdf(-d1)
        )

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = disc_q * spot * pdf_d1 * sqrt_t

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def _implied_vol_bisection(
    price: float,
    spot: float,
    strike: float,
    time: float,
    rate: float,
    div_yield: float,
    is_call: bool,
    tol: float = 1e-4,
    max_iter: int = 60,
    vol_low: float = 1e-4,
    vol_high: float = 5.0,
) -> float:
    """
    Simple bisection IV solver. Returns None if price is not compatible.
    """
    intrinsic = max(0.0, spot - strike) if is_call else max(0.0, strike - spot)
    if price < intrinsic:
        return None

    low = vol_low
    high = vol_high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = _bs_price(spot, strike, time, rate, div_yield, mid, is_call)
        diff = mid_price - price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return mid


# ---------------------------------------------------------------------
# Options chain fetch & enrichment
# ---------------------------------------------------------------------


def list_expiries(ticker: str) -> List[str]:
    tk = yf.Ticker(ticker)
    try:
        exps = tk.options
    except Exception:
        exps = []
    return list(exps or [])


def fetch_options_chain(
    ticker: str,
    expiries: Sequence[str],
    spot: float,
    rate: float,
    dividend_yield: float,
    assumed_vol: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch options chains for given expiries and compute IV & Greeks.
    """
    tk = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()

    frames = []
    for expiry in expiries:
        try:
            oc = tk.option_chain(expiry)
        except Exception:
            continue

        calls = oc.calls.copy()
        puts = oc.puts.copy()

        if calls.empty and puts.empty:
            continue

        if not calls.empty:
            calls["type"] = "call"
        if not puts.empty:
            puts["type"] = "put"

        df_exp = pd.concat([calls, puts], ignore_index=True)
        df_exp["expiry"] = pd.to_datetime(expiry)

        # DTE & time to expiry
        df_exp["dte"] = (df_exp["expiry"].dt.normalize() - today).dt.days.clip(lower=1)
        df_exp["time_to_expiry"] = df_exp["dte"] / 365.0

        # Moneyness
        df_exp["moneyness"] = df_exp["strike"] / spot - 1.0

        # Choose price for greeks/notional
        mid = (df_exp["bid"].fillna(0) + df_exp["ask"].fillna(0)) / 2
        df_exp["price_used"] = df_exp["lastPrice"].where(
            df_exp["lastPrice"] > 0, mid
        )
        df_exp["price_used"] = df_exp["price_used"].fillna(mid)

        # Implied vol: use Yahoo's if present, otherwise invert
        iv_yahoo = df_exp.get("impliedVolatility")
        iv = iv_yahoo.copy() if iv_yahoo is not None else pd.Series(index=df_exp.index, dtype=float)

        iv = iv.astype(float)
        iv_invalid = (iv <= 0) | iv.isna()

        for idx in iv_invalid[iv_invalid].index:
            row = df_exp.loc[idx]
            if row["price_used"] <= 0 or row["time_to_expiry"] <= 0:
                iv.loc[idx] = assumed_vol
                continue
            is_call = row["type"].lower() == "call"
            est = _implied_vol_bisection(
                price=float(row["price_used"]),
                spot=float(spot),
                strike=float(row["strike"]),
                time=float(row["time_to_expiry"]),
                rate=rate,
                div_yield=dividend_yield,
                is_call=is_call,
            )
            iv.loc[idx] = est if est is not None and est > 0 else assumed_vol

        df_exp["iv"] = iv.clip(lower=1e-4)

        # Greeks
        greeks = {"delta": [], "gamma": [], "vega": [], "theta": []}
        for _, row in df_exp.iterrows():
            is_call = row["type"].lower() == "call"
            g = _bs_greeks(
                spot=float(spot),
                strike=float(row["strike"]),
                time=float(row["time_to_expiry"]),
                rate=rate,
                div_yield=dividend_yield,
                vol=float(row["iv"]),
                is_call=is_call,
            )
            for k in greeks:
                greeks[k].append(g[k])

        for k in greeks:
            df_exp[k] = greeks[k]

        # Notional & vol/OI
        df_exp["dollar_notional"] = (
            df_exp["price_used"].fillna(0)
            * df_exp["volume"].fillna(0)
            * CONTRACT_SIZE
        )
        df_exp["openInterest"] = df_exp["openInterest"].fillna(0)
        df_exp["vol_oi_ratio"] = df_exp["volume"].fillna(0) / (
            df_exp["openInterest"].abs() + 1e-9
        )

        frames.append(df_exp)

    if not frames:
        raise ValueError(f"No options data for {ticker} at expiries {list(expiries)}")

    chain = pd.concat(frames, ignore_index=True)
    return chain


# ---------------------------------------------------------------------
# History: save/load snapshots & enrich features
# ---------------------------------------------------------------------


def save_daily_snapshot(
    ticker: str,
    chain_df: pd.DataFrame,
    spot: float,
    asof_date: Optional[pd.Timestamp] = None,
) -> None:
    if asof_date is None:
        asof_date = pd.Timestamp.today().normalize()

    df = chain_df.copy()
    df["ticker_underlying"] = ticker
    df["spot_underlying"] = spot
    df["asof_date"] = asof_date

    path = DATA_DIR / f"{ticker}_options_{asof_date.strftime('%Y%m%d')}.parquet"
    df.to_parquet(path, index=False)
    print(f"[INFO] Saved snapshot to {path}")


def load_history_snapshots(
    ticker: str,
    lookback_days: int = 60,
) -> Optional[pd.DataFrame]:
    if not DATA_DIR.exists():
        return None

    files = sorted(DATA_DIR.glob(f"{ticker}_options_*.parquet"))
    if not files:
        return None

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "asof_date" not in df.columns:
            continue
        df["asof_date"] = pd.to_datetime(df["asof_date"]).dt.normalize()
        df = df[df["asof_date"] >= cutoff]
        if not df.empty:
            frames.append(df)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def add_history_features(
    today_chain: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    lookback_days: int = 20,
) -> pd.DataFrame:
    """
    Add:
        - avg_volume_recent, vol_vs_hist
        - previous OI, OI change, OI change ratio
        - IV rank over lookback window for each contractSymbol

    If there is no history yet, columns are created and left NaN.
    """
    df = today_chain.copy()

    # Guarantee these columns exist, even if we return early
    base_hist_cols = [
        "avg_volume_recent",
        "vol_vs_hist",
        "openInterest_prev",
        "oi_change",
        "oi_change_ratio",
        "iv_rank_recent",
    ]
    for col in base_hist_cols:
        if col not in df.columns:
            df[col] = np.nan

    if (
        history_df is None
        or history_df.empty
        or "contractSymbol" not in df.columns
    ):
        print(
            "[WARN] No history snapshots available yet – "
            "history-based features will be treated as 0."
        )
        return df

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    h = history_df.copy()
    h["asof_date"] = pd.to_datetime(h["asof_date"]).dt.normalize()
    h = h[(h["asof_date"] >= start) & (h["asof_date"] < end)]

    if h.empty:
        print(
            "[WARN] History available but nothing within lookback window – "
            "history-based features will be treated as 0."
        )
        return df

    # Average volume over lookback
    vol_stats = (
        h.groupby("contractSymbol")
        .agg(avg_volume_recent=("volume", "mean"))
        .reset_index()
    )

    # IV rank over lookback per contract
    h = h.sort_values(["contractSymbol", "asof_date"])
    h["iv_rank_recent"] = h.groupby("contractSymbol")["iv"].rank(pct=True)
    iv_stats = (
        h.groupby("contractSymbol")
        .agg(iv_rank_recent=("iv_rank_recent", "max"))
        .reset_index()
    )

    stats = vol_stats.merge(iv_stats, on="contractSymbol", how="outer")
    df = df.merge(stats, on="contractSymbol", how="left")

    # volume vs history
    df["vol_vs_hist"] = df["volume"].fillna(0) / (
        df["avg_volume_recent"].fillna(0) + 1e-9
    )

    # yesterday's OI
    yday = end - pd.Timedelta(days=1)
    h_yday = history_df.copy()
    h_yday["asof_date"] = pd.to_datetime(h_yday["asof_date"]).dt.normalize()
    h_yday = h_yday[h_yday["asof_date"] == yday]
    if not h_yday.empty:
        oi_prev = h_yday[["contractSymbol", "openInterest"]].rename(
            columns={"openInterest": "openInterest_prev"}
        )
        df = df.merge(oi_prev, on="contractSymbol", how="left")

    df["oi_change"] = df["openInterest"] - df["openInterest_prev"].fillna(0)
    df["oi_change_ratio"] = df["oi_change"] / (
        df["openInterest_prev"].abs() + 1e-9
    )

    return df


# ---------------------------------------------------------------------
# Gamma exposure & gamma-level summary
# ---------------------------------------------------------------------


def compute_gamma_exposure(
    chain: pd.DataFrame, spot: float, contract_size: int = CONTRACT_SIZE
) -> pd.DataFrame:
    df = chain.copy()
    if "gamma" not in df.columns or "openInterest" not in df.columns:
        raise ValueError("Chain must have 'gamma' and 'openInterest' columns")

    df["gamma"] = df["gamma"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)

    sign = np.where(df["type"].str.lower().str.startswith("c"), 1.0, -1.0)
    df["gex_contract"] = (
        df["gamma"]
        * df["openInterest"]
        * contract_size
        * (spot ** 2)
        * 0.01
        * sign
    )

    grouped = (
        df.groupby("strike")
        .agg(
            gex=("gex_contract", "sum"),
            total_oi=("openInterest", "sum"),
            call_oi=("openInterest", lambda x: x[df.loc[x.index, "type"] == "call"].sum()),
            put_oi=("openInterest", lambda x: x[df.loc[x.index, "type"] == "put"].sum()),
        )
        .reset_index()
        .sort_values("strike")
    )

    grouped["cum_gex"] = grouped["gex"].cumsum()
    return grouped


def summarize_gamma_levels(gex_table: pd.DataFrame) -> Dict[str, Any]:
    if gex_table is None or gex_table.empty:
        return {
            "zero_gamma_level": None,
            "net_gex": 0.0,
            "call_wall": pd.DataFrame(),
            "put_wall": pd.DataFrame(),
            "oi_wall": pd.DataFrame(),
        }

    # zero gamma: where cum_gex crosses 0
    zg = None
    strikes = gex_table["strike"].values
    cum = gex_table["cum_gex"].values
    for i in range(1, len(strikes)):
        if cum[i - 1] == 0:
            zg = strikes[i - 1]
            break
        if cum[i - 1] * cum[i] < 0:
            # linear interpolation between strikes[i-1] and strikes[i]
            w = abs(cum[i - 1]) / (abs(cum[i - 1]) + abs(cum[i]))
            zg = strikes[i - 1] * (1 - w) + strikes[i] * w
            break

    net_gex = float(gex_table["gex"].sum())

    # call/put walls by GEX and OI
    call_wall = (
        gex_table.sort_values("gex", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    put_wall = (
        gex_table.sort_values("gex", ascending=True)
        .head(10)
        .reset_index(drop=True)
    )
    oi_wall = (
        gex_table.sort_values("total_oi", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "zero_gamma_level": zg,
        "net_gex": net_gex,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "oi_wall": oi_wall,
    }


# ---------------------------------------------------------------------
# Put/Call ratio
# ---------------------------------------------------------------------


def compute_pcr(chain: pd.DataFrame) -> Dict[str, Any]:
    df = chain.copy()
    df["volume"] = df["volume"].fillna(0)
    df["openInterest"] = df["openInterest"].fillna(0)

    calls = df[df["type"] == "call"]
    puts = df[df["type"] == "put"]

    call_vol = float(calls["volume"].sum())
    put_vol = float(puts["volume"].sum())
    call_oi = float(calls["openInterest"].sum())
    put_oi = float(puts["openInterest"].sum())

    pcr_vol = put_vol / call_vol if call_vol > 0 else None
    pcr_oi = put_oi / call_oi if call_oi > 0 else None

    return {
        "overall": {
            "call_vol": call_vol,
            "put_vol": put_vol,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "pcr_vol": pcr_vol,
            "pcr_oi": pcr_oi,
        }
    }


# ---------------------------------------------------------------------
# Unusual activity config & logic
# ---------------------------------------------------------------------


@dataclass
class UnusualConfig:
    notional_pct: float = 0.97
    volume_pct: float = 0.97
    vol_oi_pct: float = 0.97
    min_vol_vs_hist: float = 5.0
    min_oi_change_ratio: float = 0.5
    min_dollar_notional: float = 1_000_000.0
    max_dte_short: int = 7


def _delta_bucket(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.2:
        return "lottery"
    if ad < 0.4:
        return "otm"
    if ad < 0.7:
        return "atm"
    return "deep_itm"


def flag_unusual_activity_advanced(
    chain: pd.DataFrame,
    config: UnusualConfig,
) -> pd.DataFrame:
    """
    Use intraday percentiles + history-based metrics to flag unusual lines.
    """
    df = chain.copy()

    # Ensure history fields exist & fill NaN with 0 for logic
    for col in ["avg_volume_recent", "vol_vs_hist", "iv_rank_recent", "oi_change_ratio"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    vol_vs_hist = df["vol_vs_hist"]
    iv_rank_recent = df["iv_rank_recent"]
    oi_change_ratio = df["oi_change_ratio"]

    # Base filters
    big_notional = df["dollar_notional"] >= config.min_dollar_notional
    short_dated = df["dte"] <= config.max_dte_short

    high_notional_rank = df["notional_pct"] >= config.notional_pct
    high_volume_rank = df["volume_pct"] >= config.volume_pct
    high_vol_oi_rank = df["vol_oi_pct"] >= config.vol_oi_pct

    big_vol_vs_hist = vol_vs_hist >= config.min_vol_vs_hist
    big_oi_change = oi_change_ratio >= config.min_oi_change_ratio
    high_iv_rank = iv_rank_recent >= 0.9

    # Pathways
    cond1 = big_notional & high_notional_rank & short_dated
    cond2 = high_volume_rank & high_vol_oi_rank & big_oi_change & short_dated
    cond3 = big_vol_vs_hist & high_iv_rank & short_dated

    mask = (cond1 | cond2 | cond3) & big_notional

    unusual = df.loc[mask].copy()
    if unusual.empty:
        return unusual

    # Reason strings
    reasons = []
    for _, row in unusual.iterrows():
        r = []
        if row["dollar_notional"] >= config.min_dollar_notional:
            r.append("big_notional")
        if row["notional_pct"] >= config.notional_pct:
            r.append(f"top_notional_{int((1 - config.notional_pct)*100)}%")
        if row["volume_pct"] >= config.volume_pct:
            r.append(f"top_volume_{int((1 - config.volume_pct)*100)}%")
        if row["vol_oi_pct"] >= config.vol_oi_pct:
            r.append(f"top_volOI_{int((1 - config.vol_oi_pct)*100)}%")
        if row["vol_vs_hist"] >= config.min_vol_vs_hist:
            r.append(f"vol>{config.min_vol_vs_hist}x_hist")
        if row["oi_change_ratio"] >= config.min_oi_change_ratio:
            r.append("new_OI")
        if row["iv_rank_recent"] >= 0.9:
            r.append("IV_rank>90%")
        if row["dte"] <= config.max_dte_short:
            r.append(f"{int(row['dte'])}DTE")
        reasons.append("|".join(r))

    unusual["reason"] = reasons
    unusual["delta_bucket"] = unusual["delta"].apply(_delta_bucket)

    cols = [
        "contractSymbol",
        "type",
        "expiry",
        "dte",
        "strike",
        "price_used",
        "volume",
        "openInterest",
        "dollar_notional",
        "vol_oi_ratio",
        "notional_pct",
        "volume_pct",
        "vol_oi_pct",
        "avg_volume_recent",
        "vol_vs_hist",
        "openInterest_prev",
        "oi_change",
        "oi_change_ratio",
        "iv",
        "iv_rank_recent",
        "delta",
        "delta_bucket",
        "gamma",
        "vega",
        "theta",
        "moneyness",
        "reason",
    ]
    existing = [c for c in cols if c in unusual.columns]
    return unusual[existing].sort_values("dollar_notional", ascending=False)


# ---------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------


def build_notional_clusters(chain: pd.DataFrame) -> pd.DataFrame:
    df = chain.copy()
    df["moneyness_band"] = (df["moneyness"] * 100).round().astype(int)

    grouped = (
        df.groupby(["expiry", "moneyness_band"])
        .agg(
            total_notional=("dollar_notional", "sum"),
            avg_moneyness=("moneyness", "mean"),
            call_notional=(
                "dollar_notional",
                lambda x: x[df.loc[x.index, "type"] == "call"].sum(),
            ),
            put_notional=(
                "dollar_notional",
                lambda x: x[df.loc[x.index, "type"] == "put"].sum(),
            ),
        )
        .reset_index()
    )

    grouped["net_call_minus_put"] = (
        grouped["call_notional"] - grouped["put_notional"]
    )
    grouped = grouped.sort_values("total_notional", ascending=False)
    return grouped


# ---------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------


def summarize_unusual_narrative(
    spot: float,
    pcr_info: Dict[str, Any],
    gamma_info: Dict[str, Any],
    gex_table: pd.DataFrame,
    unusual_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    overall = pcr_info["overall"]

    pcr_vol = overall["pcr_vol"]
    pcr_oi = overall["pcr_oi"]

    # 1) Put/Call tone
    tone_today = "balanced"
    if pcr_vol is not None and not np.isnan(pcr_vol):
        if pcr_vol < 0.9:
            tone_today = "call-dominant"
        elif pcr_vol > 1.1:
            tone_today = "put-dominant"

    tone_position = "balanced positioning"
    if pcr_oi is not None and not np.isnan(pcr_oi):
        if pcr_oi < 0.9:
            tone_position = "call-heavy positioning"
        elif pcr_oi > 1.1:
            tone_position = "put-heavy positioning"

    lines.append(
        f"Spot ≈ {spot:.2f}. Intraday options flow is {tone_today} "
        f"(PCR_vol ≈ {pcr_vol:.2f} if finite), while outstanding positioning is "
        f"{tone_position} (PCR_OI ≈ {pcr_oi:.2f} if finite)."
    )

    # 2) Gamma regime & levels
    zg = gamma_info.get("zero_gamma_level", None)
    net_gex = gamma_info.get("net_gex", 0.0)
    gamma_regime = "neutral"
    if net_gex < 0:
        gamma_regime = "net short gamma (moves can be amplified)"
    elif net_gex > 0:
        gamma_regime = "net long gamma (moves more damped)"

    if zg is not None:
        rel = "above" if spot > zg else "below"
        lines.append(
            f"Approx zero-gamma level: {zg:.1f}. Spot trades {rel} this pivot with {gamma_regime} overall."
        )
    else:
        lines.append(
            "No zero-gamma crossing across analysed strikes; cumulative GEX stays "
            f"one-sided, indicating a persistent {gamma_regime} profile."
        )

    call_wall = gamma_info.get("call_wall", pd.DataFrame())
    put_wall = gamma_info.get("put_wall", pd.DataFrame())
    key_levels = []

    def _closest_wall(wall_df: pd.DataFrame, label: str):
        if wall_df is None or wall_df.empty:
            return None
        tmp = wall_df.copy()
        tmp["dist"] = (tmp["strike"] - spot).abs()
        row = tmp.sort_values("dist").iloc[0]
        return label, float(row["strike"])

    cw = _closest_wall(call_wall, "call_wall")
    pw = _closest_wall(put_wall, "put_wall")
    if cw:
        key_levels.append(f"{cw[0]} near {cw[1]:.1f}")
    if pw:
        key_levels.append(f"{pw[0]} near {pw[1]:.1f}")
    if key_levels:
        lines.append("Key gamma levels to monitor intraday: " + ", ".join(key_levels) + ".")

    # 3) Unusual flow
    if not unusual_df.empty:
        top = unusual_df.head(20)
        total_notional = top["dollar_notional"].sum() / 1e6
        calls_notional = top.loc[top["type"] == "call", "dollar_notional"].sum() / 1e6
        puts_notional = top.loc[top["type"] == "put", "dollar_notional"].sum() / 1e6

        lines.append(
            f"Top 20 unusual lines carry about ${total_notional:.1f}M notional: "
            f"${calls_notional:.1f}M calls vs ${puts_notional:.1f}M puts."
        )

        dte_desc = (
            top["dte"].apply(lambda x: f"{int(x)}D").value_counts().head(3).to_dict()
        )
        lines.append(f"Unusual flow is concentrated in DTE buckets: {dte_desc}.")

        if "delta_bucket" in top.columns:
            db = top["delta_bucket"].value_counts().to_dict()
            lines.append(f"Directional tilt (by delta bucket): {db}.")

        near_spot = top[top["moneyness"].abs() < 0.02]
        if not near_spot.empty:
            n_near = near_spot.shape[0]
            n_total = top.shape[0]
            lines.append(
                f"{n_near}/{n_total} flagged lines are within ±2% of spot, "
                "so much of the unusual activity is effectively betting around current levels."
            )
    else:
        lines.append("No contracts passed the advanced unusual-activity filters today.")

    # 4) Clusters
    if not cluster_df.empty:
        top_clusters = cluster_df.head(3)
        lines.append("Largest notional clusters (expiry / moneyness / tilt):")
        for _, c in top_clusters.iterrows():
            band = c["moneyness_band"]
            exp = c["expiry"].date()
            bias = "call-tilted" if c["net_call_minus_put"] > 0 else "put-tilted"
            lines.append(
                f"  • {exp}, ~{band:.0f}% moneyness, {bias}, "
                f"≈ ${c['total_notional']/1e6:.1f}M notional"
            )

    # 5) Action-oriented watchlist
    watchlist: List[str] = []

    if gamma_regime.startswith("net short gamma"):
        watchlist.append(
            "Expect faster tape and bigger intraday swings – short gamma regime "
            "means sharp moves can extend as dealers hedge with the move."
        )
    else:
        watchlist.append(
            "Gamma profile is not aggressively short; dealer hedging may dampen extreme moves."
        )

    if cw or pw:
        watchlist.append(
            "Use nearby put/call walls as tactical reference: look for intraday reversals, "
            "pinning, or acceleration when spot tests those strikes."
        )

    if pcr_vol is not None and not np.isnan(pcr_vol):
        if pcr_vol < 0.9:
            watchlist.append(
                "Call volume outpacing puts intraday – watch for upside follow-through, "
                "especially near short-dated call clusters."
            )
        elif pcr_vol > 1.1:
            watchlist.append(
                "Put volume dominating – monitor for downside hedging or outright bearish bets, "
                "especially if clustered below spot."
            )

    if not unusual_df.empty:
        watchlist.append(
            "Track price action around strikes/expiries with largest unusual blocks – "
            "if price respects them, they can act as short-term support/resistance magnets."
        )

    if watchlist:
        lines.append("\nWhat to watch / how to use this:")
        for item in watchlist:
            lines.append(f"- {item}")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------


def run_full_analysis(
    ticker: str,
    price_period: str = "6mo",
    price_interval: str = "1d",
    expiries: Optional[Sequence[str]] = None,
    max_expiries: int = 3,
    rate: float = 0.04,
    dividend_yield: float = 0.012,
    history_lookback_days: int = 60,
    unusual_config: Optional[UnusualConfig] = None,
) -> Dict[str, Any]:
    if unusual_config is None:
        unusual_config = UnusualConfig()

    # 1) Price history
    price_df = get_price_history(ticker, period=price_period, interval=price_interval)
    # spot = last close
    spot = float(price_df["Close"].iloc[-1])

    # 2) Expiries to analyse
    all_exps = list_expiries(ticker)
    if expiries is not None and len(expiries) > 0:
        chosen_exps = [e for e in expiries if e in all_exps]
        if not chosen_exps:
            chosen_exps = all_exps[:max_expiries]
    else:
        chosen_exps = all_exps[:max_expiries]
    if not chosen_exps:
        raise ValueError(f"No expiries available for {ticker}")

    # 3) Options chain
    chain_df = fetch_options_chain(
        ticker=ticker,
        expiries=chosen_exps,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    # 4) History features
    history_df = load_history_snapshots(ticker, lookback_days=history_lookback_days)
    chain_df = add_history_features(chain_df, history_df, lookback_days=min(20, history_lookback_days))

    # 5) Intraday percentiles
    chain_df["notional_pct"] = chain_df["dollar_notional"].rank(pct=True)
    chain_df["volume_pct"] = chain_df["volume"].rank(pct=True)
    chain_df["vol_oi_pct"] = chain_df["vol_oi_ratio"].rank(pct=True)

    # 6) Put/Call ratio
    pcr_info = compute_pcr(chain_df)

    # 7) Gamma exposure
    gex_table = compute_gamma_exposure(chain_df, spot=spot)
    gamma_info = summarize_gamma_levels(gex_table)

    # 8) Unusual activity
    unusual_df = flag_unusual_activity_advanced(chain_df, unusual_config)

    # 9) Clusters
    clusters = build_notional_clusters(chain_df)

    # 10) Narrative
    narrative = summarize_unusual_narrative(
        spot=spot,
        pcr_info=pcr_info,
        gamma_info=gamma_info,
        gex_table=gex_table,
        unusual_df=unusual_df,
        cluster_df=clusters,
    )

    # 11) Save snapshot for future history
    save_daily_snapshot(ticker, chain_df, spot)

    return {
        "spot": spot,
        "price_df": price_df,
        "chain_df": chain_df,
        "pcr_info": pcr_info,
        "gex_table": gex_table,
        "gamma_info": gamma_info,
        "clusters": clusters,
        "unusual_df": unusual_df,
        "narrative": narrative,
    }


if __name__ == "__main__":
    # Simple local test
    cfg = UnusualConfig()
    res = run_full_analysis(
        ticker="SPY",
        price_period="6mo",
        price_interval="1d",
        expiries=None,
        max_expiries=3,
        rate=0.04,
        dividend_yield=0.012,
        history_lookback_days=60,
        unusual_config=cfg,
    )
    print("[OK] Analysis ran. Spot:", res["spot"])
    print(res["narrative"])
