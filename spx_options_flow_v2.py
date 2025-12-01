#!/usr/bin/env python3
"""
SPX/SPY options-flow style analysis (Unusual-activity oriented), v2.

Additions vs v1:
    - Intraday percentile ranks (notional, volume, vol/OI)
    - Optional history-based features: vol vs past avg, OI change, IV rank
    - Delta buckets
    - Cluster-level stats (by expiry, by strike band)
    - Advanced unusual-activity flagging using percentiles (data-driven)
    - Narrative summary
    - Snapshot saving/loading to enable your own backtests & calibration

Requirements:
    pip install yfinance pandas numpy pyarrow
"""

import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# Config & Data Structures
# =============================================================================

UNDERLYING_DEFAULT = "SPY"   # e.g. "SPY", "^SPX"
CONTRACT_SIZE = 100

OptionType = Literal["call", "put"]

DATA_DIR = Path("options_history")  # where snapshots are stored locally
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class MarketParams:
    spot: float
    rate: float = 0.0
    dividend_yield: float = 0.0
    time_to_expiry_fallback: float = 30 / 365.0
    assumed_vol: float = 0.2


@dataclass
class UnusualConfig:
    # Percentiles for intraday ranks
    notional_pct: float = 0.97
    volume_pct: float = 0.97
    vol_oi_pct: float = 0.97
    # History-based conditions
    min_vol_vs_hist: float = 5.0      # today's vol >= 5x avg
    min_oi_change_ratio: float = 0.5  # OI up >= 50%
    # Baseline filters
    min_dollar_notional: float = 1_000_000.0
    max_dte_short: int = 7


# =============================================================================
# Price Data & Moving Averages
# =============================================================================

def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"No price data returned for {ticker}")
    return data


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError("Expected one 'Close' column; got multiple.")
        close = close.iloc[:, 0]

    df["SMA20"] = sma(close, 20)
    df["SMA50"] = sma(close, 50)
    df["SMA200"] = sma(close, 200)
    return df


# =============================================================================
# Black–Scholes Greeks & Implied Vol
# =============================================================================

def _d1_d2(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    t: float,
) -> (float, float):
    if vol <= 0 or t <= 0:
        raise ValueError("Volatility and time to expiry must be positive")
    num = math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol**2) * t
    den = vol * math.sqrt(t)
    d1 = num / den
    d2 = d1 - vol * math.sqrt(t)
    return d1, d2


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    t: float,
) -> float:
    d1, d2 = _d1_d2(spot, strike, rate, dividend_yield, vol, t)
    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend_yield * t)

    if option_type == "call":
        return spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    else:
        return strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)


def bs_delta(
    option_type: OptionType,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    t: float,
) -> float:
    d1, _ = _d1_d2(spot, strike, rate, dividend_yield, vol, t)
    disc_q = math.exp(-dividend_yield * t)
    if option_type == "call":
        return disc_q * _norm_cdf(d1)
    else:
        return -disc_q * _norm_cdf(-d1)


def bs_gamma(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    t: float,
) -> float:
    d1, _ = _d1_d2(spot, strike, rate, dividend_yield, vol, t)
    disc_q = math.exp(-dividend_yield * t)
    return disc_q * _norm_pdf(d1) / (spot * vol * math.sqrt(t))


def bs_vega(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    t: float,
) -> float:
    d1, _ = _d1_d2(spot, strike, rate, dividend_yield, vol, t)
    disc_q = math.exp(-dividend_yield * t)
    return spot * disc_q * _norm_pdf(d1) * math.sqrt(t)


def bs_theta(
    option_type: OptionType,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    vol: float,
    t: float,
) -> float:
    d1, d2 = _d1_d2(spot, strike, rate, dividend_yield, vol, t)
    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend_yield * t)

    first = -(spot * disc_q * _norm_pdf(d1) * vol) / (2.0 * math.sqrt(t))
    if option_type == "call":
        second = -rate * strike * disc_r * _norm_cdf(d2)
        third = dividend_yield * spot * disc_q * _norm_cdf(d1)
    else:
        second = rate * strike * disc_r * _norm_cdf(-d2)
        third = -dividend_yield * spot * disc_q * _norm_cdf(-d1)
    return first + second + third


def implied_vol_bisection(
    option_type: OptionType,
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    t: float,
    vol_low: float = 1e-4,
    vol_high: float = 5.0,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Optional[float]:
    if market_price <= 0:
        return None

    low, high = vol_low, vol_high
    price_low = bs_price(option_type, spot, strike, rate, dividend_yield, low, t)
    price_high = bs_price(option_type, spot, strike, rate, dividend_yield, high, t)

    if (price_low - market_price) * (price_high - market_price) > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price_mid = bs_price(option_type, spot, strike, rate, dividend_yield, mid, t)
        diff = price_mid - market_price
        if abs(diff) < tol:
            return mid
        if (price_low - market_price) * diff < 0:
            high = mid
            price_high = price_mid
        else:
            low = mid
            price_low = price_mid

    return 0.5 * (low + high)


# =============================================================================
# Options Chain Fetch
# =============================================================================

def list_expiries(ticker: str) -> List[str]:
    tk = yf.Ticker(ticker)
    return list(tk.options or [])


def get_options_chain_yf(
    ticker: str,
    expiries: Optional[List[str]] = None,
    max_expiries: int = 3,
) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    all_exps = tk.options
    if not all_exps:
        raise ValueError(f"No options expiries available for {ticker}")

    if expiries is None:
        expiries = sorted(all_exps)[:max_expiries]

    chains = []
    for expiry in expiries:
        if expiry not in all_exps:
            continue
        oc = tk.option_chain(expiry)
        calls = oc.calls.copy()
        puts = oc.puts.copy()
        calls["type"] = "call"
        puts["type"] = "put"

        tmp = pd.concat([calls, puts], ignore_index=True)
        tmp["expiry"] = pd.to_datetime(expiry)
        chains.append(tmp)

    if not chains:
        raise ValueError("No valid chains collected.")
    chain = pd.concat(chains, ignore_index=True)
    return chain


def days_to_years(days: float) -> float:
    return days / 365.0


def add_iv_and_greeks_to_chain(
    chain: pd.DataFrame,
    market_params: MarketParams,
) -> pd.DataFrame:
    df = chain.copy()
    spot = market_params.spot
    r = market_params.rate
    q = market_params.dividend_yield

    expiry_dates = pd.to_datetime(df["expiry"])
    today = pd.Timestamp.today().normalize()
    days_to_expiry = (expiry_dates - today).dt.days.clip(lower=1)
    df["dte"] = days_to_expiry
    df["time_to_expiry"] = df["dte"].apply(days_to_years)

    iv_list, delta_list, gamma_list, vega_list, theta_list = [], [], [], [], []

    for _, row in df.iterrows():
        k = float(row["strike"])
        opt_type: OptionType = "call" if str(row["type"]).lower().startswith("c") else "put"
        t = float(row["time_to_expiry"])

        iv = row.get("impliedVolatility", np.nan)
        if pd.isna(iv) or iv <= 0:
            bid = float(row.get("bid", np.nan))
            ask = float(row.get("ask", np.nan))
            mid = (bid + ask) / 2.0 if not (pd.isna(bid) or pd.isna(ask)) else float("nan")
            if not pd.isna(mid) and mid > 0:
                iv = implied_vol_bisection(
                    opt_type,
                    market_price=mid,
                    spot=spot,
                    strike=k,
                    rate=r,
                    dividend_yield=q,
                    t=t,
                )
        if iv is None or pd.isna(iv) or iv <= 0:
            iv = market_params.assumed_vol

        try:
            d = bs_delta(opt_type, spot, k, r, q, iv, t)
            g = bs_gamma(spot, k, r, q, iv, t)
            v = bs_vega(spot, k, r, q, iv, t)
            th = bs_theta(opt_type, spot, k, r, q, iv, t)
        except Exception:
            d, g, v, th = np.nan, np.nan, np.nan, np.nan

        iv_list.append(iv)
        delta_list.append(d)
        gamma_list.append(g)
        vega_list.append(v)
        theta_list.append(th)

    df["iv"] = iv_list
    df["delta"] = delta_list
    df["gamma"] = gamma_list
    df["vega"] = vega_list
    df["theta"] = theta_list

    return df


# =============================================================================
# Gamma Exposure & Walls
# =============================================================================

def compute_gamma_exposure(
    chain: pd.DataFrame,
    spot: float,
    contract_size: int = CONTRACT_SIZE,
) -> pd.DataFrame:
    df = chain.copy()
    if "gamma" not in df.columns or "openInterest" not in df.columns:
        raise ValueError("Chain must have 'gamma' and 'openInterest' columns")

    sign = np.where(df["type"].str.lower().str.startswith("c"), 1.0, -1.0)

    df["gex"] = (
        df["gamma"].astype(float)
        * df["openInterest"].astype(float)
        * contract_size
        * (spot ** 2)
        * 0.01
        * sign
    )

    agg = (
        df.groupby("strike", as_index=False)
        .agg(
            gex=("gex", "sum"),
            total_oi=("openInterest", "sum"),
            call_oi=("openInterest", lambda x: x[df.loc[x.index, "type"] == "call"].sum()),
            put_oi=("openInterest", lambda x: x[df.loc[x.index, "type"] == "put"].sum()),
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    return agg


def summarize_gamma_levels(
    gex_table: pd.DataFrame,
    top_n: int = 5,
) -> Dict[str, Any]:
    gex_table = gex_table.sort_values("strike").reset_index(drop=True)
    gex_table["cum_gex"] = gex_table["gex"].cumsum()

    zero_level = None
    for i in range(1, len(gex_table)):
        prev_val = gex_table.loc[i - 1, "cum_gex"]
        curr_val = gex_table.loc[i, "cum_gex"]
        if prev_val == 0:
            zero_level = gex_table.loc[i - 1, "strike"]
            break
        if prev_val * curr_val < 0:
            s1 = gex_table.loc[i - 1, "strike"]
            s2 = gex_table.loc[i, "strike"]
            w = abs(prev_val) / (abs(prev_val) + abs(curr_val))
            zero_level = s1 + w * (s2 - s1)
            break

    pos = gex_table[gex_table["gex"] > 0].copy()
    call_wall = pos.sort_values("gex", ascending=False).head(top_n)

    neg = gex_table[gex_table["gex"] < 0].copy()
    put_wall = neg.sort_values("gex", ascending=True).head(top_n)

    oi_wall = gex_table.sort_values("total_oi", ascending=False).head(top_n)

    return {
        "zero_gamma_level": zero_level,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "oi_wall": oi_wall,
    }


# =============================================================================
# Flow Metrics & History Features
# =============================================================================

def enrich_flow_metrics(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    df = chain.copy()
    df["volume"] = df["volume"].fillna(0).astype(float)
    df["openInterest"] = df["openInterest"].fillna(0).astype(float)

    last = df.get("lastPrice", np.nan).astype(float)
    bid = df.get("bid", np.nan).astype(float)
    ask = df.get("ask", np.nan).astype(float)
    mid = (bid + ask) / 2.0

    df["price_used"] = last.where(last > 0, mid)
    df["price_used"] = df["price_used"].fillna(mid).fillna(0.0)

    df["dollar_notional"] = df["price_used"] * df["volume"] * CONTRACT_SIZE
    df["vol_oi_ratio"] = df["volume"] / (df["openInterest"] + 1e-9)
    df["moneyness"] = df["strike"].astype(float) / spot - 1.0

    return df


def delta_bucket(d: float) -> str:
    ad = abs(d)
    if ad < 0.20:
        return "lottery"
    elif ad < 0.40:
        return "otm"
    elif ad < 0.70:
        return "atm"
    else:
        return "deep_itm"


def add_intraday_percentiles(chain: pd.DataFrame) -> pd.DataFrame:
    df = chain.copy()
    df["notional_pct"] = df["dollar_notional"].rank(pct=True)
    df["volume_pct"] = df["volume"].rank(pct=True)
    df["vol_oi_pct"] = df["vol_oi_ratio"].rank(pct=True)
    return df


# ---- Snapshot Saving / Loading (for your own backtests) ----------------------

def snapshot_path(ticker: str, asof_date: pd.Timestamp) -> Path:
    date_str = asof_date.strftime("%Y%m%d")
    return DATA_DIR / f"{ticker}_options_{date_str}.parquet"


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

    path = snapshot_path(ticker, asof_date)
    df.to_parquet(path, index=False)
    print(f"[INFO] Saved snapshot to {path}")


def load_history_snapshots(
    ticker: str,
    lookback_days: int = 60,
) -> Optional[pd.DataFrame]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    frames = []
    for fname in os.listdir(DATA_DIR):
        if not fname.startswith(f"{ticker}_options_") or not fname.endswith(".parquet"):
            continue
        date_str = fname.split("_options_")[-1].replace(".parquet", "")
        try:
            asof = pd.to_datetime(date_str, format="%Y%m%d")
        except ValueError:
            continue
        if start <= asof <= end:
            df = pd.read_parquet(DATA_DIR / fname)
            frames.append(df)

    if not frames:
        return None
    hist = pd.concat(frames, ignore_index=True)
    return hist


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

    If there is no history yet (e.g. first run), fields stay NaN and a warning
    is printed. The unusual-activity logic later will safely treat these as 0.
    """
    df = today_chain.copy()

    df["avg_volume_recent"] = np.nan
    df["vol_vs_hist"] = np.nan
    df["openInterest_prev"] = np.nan
    df["oi_change"] = np.nan
    df["oi_change_ratio"] = np.nan
    df["iv_rank_recent"] = np.nan

    if history_df is None or history_df.empty or "contractSymbol" not in df.columns:
        print("[WARN] No history snapshots available yet – "
              "vol_vs_hist, oi_change, iv_rank_recent will be treated as 0 "
              "for unusual-activity logic.")
        return df

    # restrict to lookback window
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    h = history_df.copy()
    h = h[(h["asof_date"] >= start) & (h["asof_date"] < end)]

    if h.empty:
        print("[WARN] History available but nothing within lookback window – "
              "history-based features will be 0.")
        return df

    # avg volume over lookback
    vol_stats = (
        h.groupby("contractSymbol")
        .agg(
            avg_volume_recent=("volume", "mean"),
        )
        .reset_index()
    )

    # IV rank within lookback per contract
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
    df["vol_vs_hist"] = df["volume"] / (df["avg_volume_recent"] + 1e-9)

    # yesterday's OI
    yday = end - pd.Timedelta(days=1)
    h_yday = history_df[history_df["asof_date"] == yday]
    if not h_yday.empty:
        oi_prev = h_yday[["contractSymbol", "openInterest"]].rename(
            columns={"openInterest": "openInterest_prev"}
        )
        df = df.merge(oi_prev, on="contractSymbol", how="left")

    df["oi_change"] = df["openInterest"] - df["openInterest_prev"].fillna(0)
    df["oi_change_ratio"] = df["oi_change"] / (df["openInterest_prev"].abs() + 1e-9)

    return df


# =============================================================================
# Put/Call Ratios & Clusters
# =============================================================================

def compute_put_call_ratios(chain: pd.DataFrame) -> Dict[str, Any]:
    df = chain.copy()
    df["volume"] = df["volume"].fillna(0).astype(float)
    df["openInterest"] = df["openInterest"].fillna(0).astype(float)

    is_call = df["type"].str.lower().str.startswith("c")
    is_put = df["type"].str.lower().str.startswith("p")

    total_call_vol = df.loc[is_call, "volume"].sum()
    total_put_vol = df.loc[is_put, "volume"].sum()
    total_call_oi = df.loc[is_call, "openInterest"].sum()
    total_put_oi = df.loc[is_put, "openInterest"].sum()

    overall_pcr_vol = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan
    overall_pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan

    by_expiry = (
        df.groupby(["expiry", "type"])
        .agg(
            volume=("volume", "sum"),
            openInterest=("openInterest", "sum"),
        )
        .reset_index()
    )

    expiry_stats = {}
    for expiry, grp in by_expiry.groupby("expiry"):
        calls = grp.loc[grp["type"].str.lower().str.startswith("c")]
        puts = grp.loc[grp["type"].str.lower().str.startswith("p")]

        call_vol = float(calls["volume"].sum())
        put_vol = float(puts["volume"].sum())
        call_oi = float(calls["openInterest"].sum())
        put_oi = float(puts["openInterest"].sum())

        pcr_vol = put_vol / call_vol if call_vol > 0 else np.nan
        pcr_oi = put_oi / call_oi if call_oi > 0 else np.nan

        expiry_stats[pd.to_datetime(expiry)] = {
            "call_vol": call_vol,
            "put_vol": put_vol,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "pcr_vol": pcr_vol,
            "pcr_oi": pcr_oi,
        }

    return {
        "overall": {
            "call_vol": total_call_vol,
            "put_vol": total_put_vol,
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "pcr_vol": overall_pcr_vol,
            "pcr_oi": overall_pcr_oi,
        },
        "by_expiry": expiry_stats,
    }


def summarize_clusters(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Cluster by expiry & 1% moneyness band, summarise notional and net call/put tilt.
    """
    df = chain.copy()
    df["moneyness_band"] = (df["moneyness"] * 100).round()  # in percentage points
    grp = (
        df.groupby(["expiry", "moneyness_band"])
        .agg(
            total_notional=("dollar_notional", "sum"),
            avg_moneyness=("moneyness", "mean"),
            call_notional=("dollar_notional",
                           lambda x: x[df.loc[x.index, "type"] == "call"].sum()),
            put_notional=("dollar_notional",
                          lambda x: x[df.loc[x.index, "type"] == "put"].sum()),
        )
        .reset_index()
    )
    grp["net_call_minus_put"] = grp["call_notional"] - grp["put_notional"]
    return grp.sort_values("total_notional", ascending=False)


# =============================================================================
# Advanced Unusual Activity
# =============================================================================

def flag_unusual_activity_advanced(
    chain: pd.DataFrame,
    config: UnusualConfig,
) -> pd.DataFrame:
    """
    Use intraday percentiles + history-based metrics to flag unusual lines.

    Pathways (OR):
        1) big_notional AND high notional percentile AND short-dated
        2) high volume percentile AND high vol/OI percentile AND big OI increase AND short-dated
        3) volume >> own history AND high IV rank AND short-dated

    All with a hard floor on min_dollar_notional.
    """
    df = chain.copy()

    # --- History fields: fill NaNs with 0 so logic is robust ---
    for col in ["vol_vs_hist", "iv_rank_recent", "oi_change_ratio"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    vol_vs_hist = df["vol_vs_hist"]
    iv_rank_recent = df["iv_rank_recent"]
    oi_change_ratio = df["oi_change_ratio"]

    # --- Base filters ---
    big_notional = df["dollar_notional"] >= config.min_dollar_notional
    short_dated = df["dte"] <= config.max_dte_short

    high_notional_rank = df["notional_pct"] >= config.notional_pct
    high_volume_rank = df["volume_pct"] >= config.volume_pct
    high_vol_oi_rank = df["vol_oi_pct"] >= config.vol_oi_pct

    big_vol_vs_hist = vol_vs_hist >= config.min_vol_vs_hist
    big_oi_change = oi_change_ratio >= config.min_oi_change_ratio
    high_iv_rank = iv_rank_recent >= 0.9

    # --- Pathways ---
    cond1 = big_notional & high_notional_rank & short_dated
    cond2 = high_volume_rank & high_vol_oi_rank & big_oi_change & short_dated
    cond3 = big_vol_vs_hist & high_iv_rank & short_dated

    mask = (cond1 | cond2 | cond3) & big_notional

    unusual = df.loc[mask].copy()

    # Reason strings
    reasons = []
    for _, row in unusual.iterrows():
        r = []
        if row["dollar_notional"] >= config.min_dollar_notional:
            r.append("big_notional")
        if row["notional_pct"] >= config.notional_pct:
            r.append(f"top_notional_{int((1-config.notional_pct)*100)}%")
        if row["volume_pct"] >= config.volume_pct:
            r.append(f"top_volume_{int((1-config.volume_pct)*100)}%")
        if row["vol_oi_pct"] >= config.vol_oi_pct:
            r.append(f"top_volOI_{int((1-config.vol_oi_pct)*100)}%")
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

    # Delta bucket
    unusual["delta_bucket"] = unusual["delta"].apply(delta_bucket)

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

# =============================================================================
# Narrative Summary
# =============================================================================

def summarize_unusual_narrative(
    spot: float,
    pcr_info: Dict[str, Any],
    gamma_info: Dict[str, Any],
    gex_table: pd.DataFrame,
    unusual_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> str:
    lines = []
    overall = pcr_info["overall"]

    pcr_vol = overall["pcr_vol"]
    pcr_oi = overall["pcr_oi"]

    # === 1) Put/Call tone ===
    tone_today = "balanced"
    if pcr_vol is not None and not np.isnan(pcr_vol):
        if pcr_vol < 0.9:
            tone_today = "call-dominant"
        elif pcr_vol > 1.1:
            tone_today = "put-dominant"

    tone_position = "balanced"
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

    # === 2) Gamma regime & key levels ===
    zg = gamma_info.get("zero_gamma_level", None)
    net_gex = float(gex_table["gex"].sum()) if not gex_table.empty else np.nan
    gamma_regime = "neutral"
    if not np.isnan(net_gex):
        if net_gex < 0:
            gamma_regime = "net short gamma (moves can be amplified)"
        elif net_gex > 0:
            gamma_regime = "net long gamma (moves more damped)"

    if zg is not None:
        rel = "above" if spot > zg else "below"
        lines.append(
            f"Approx zero-gamma level: {zg:.1f}. Spot trades {rel} this pivot with "
            f"{gamma_regime} overall."
        )
    else:
        lines.append(
            f"No zero-gamma crossing across analysed strikes; cumulative GEX stays "
            f"one-sided, indicating a persistent {gamma_regime} profile."
        )

    # Call/put walls closest to spot
    call_wall = gamma_info.get("call_wall", pd.DataFrame())
    put_wall = gamma_info.get("put_wall", pd.DataFrame())
    key_levels = []

    def _closest_wall(wall_df, label):
        if wall_df is None or wall_df.empty:
            return None
        tmp = wall_df.copy()
        tmp["dist"] = (tmp["strike"] - spot).abs()
        row = tmp.sort_values("dist").iloc[0]
        return label, float(row["strike"]), float(row["gex"])

    cw = _closest_wall(call_wall, "call_wall")
    pw = _closest_wall(put_wall, "put_wall")
    if cw:
        label, s, g = cw
        key_levels.append(f"{label} near {s:.1f}")
    if pw:
        label, s, g = pw
        key_levels.append(f"{label} near {s:.1f}")

    if key_levels:
        lines.append(
            "Key gamma levels to monitor intraday: " + ", ".join(key_levels) + "."
        )

    # === 3) Unusual flow summary ===
    if not unusual_df.empty:
        top = unusual_df.head(20)
        total_notional = top["dollar_notional"].sum() / 1e6

        calls_notional = top.loc[top["type"] == "call", "dollar_notional"].sum() / 1e6
        puts_notional = top.loc[top["type"] == "put", "dollar_notional"].sum() / 1e6

        lines.append(
            f"Top 20 unusual lines carry about ${total_notional:.1f}M notional: "
            f"${calls_notional:.1f}M calls vs ${puts_notional:.1f}M puts."
        )

        # DTE distribution
        dte_desc = (
            top["dte"]
            .apply(lambda x: f"{int(x)}D")
            .value_counts()
            .head(3)
            .to_dict()
        )
        lines.append(f"Unusual flow is concentrated in DTE buckets: {dte_desc}.")

        # Delta buckets
        if "delta_bucket" in top.columns:
            db = top["delta_bucket"].value_counts().to_dict()
            lines.append(f"Directional tilt (by delta bucket): {db}.")

        # Moneyness band summary
        near_spot = top[top["moneyness"].abs() < 0.02]
        if not near_spot.empty:
            n_near = near_spot.shape[0]
            n_total = top.shape[0]
            lines.append(
                f"{n_near}/{n_total} flagged lines are within ±2% of spot, "
                "so a lot of the unusual activity is effectively betting around current levels."
            )

    else:
        lines.append("No contracts passed the advanced unusual-activity filters today.")

    # === 4) Cluster-level context ===
    if not cluster_df.empty:
        top_clusters = cluster_df.head(3)
        cluster_lines = []
        for _, c in top_clusters.iterrows():
            band = c["moneyness_band"]
            exp = c["expiry"].date()
            bias = "call-tilted" if c["net_call_minus_put"] > 0 else "put-tilted"
            cluster_lines.append(
                f"{exp}, ~{band:.0f}% moneyness, {bias}, "
                f"≈ ${c['total_notional']/1e6:.1f}M notional"
            )
        lines.append("Largest notional clusters (expiry / moneyness / tilt):")
        for cl in cluster_lines:
            lines.append(f"  • {cl}")

    # === 5) Action-oriented watchlist ===
    watchlist = []

    # 1) Gamma regime & walls
    if gamma_regime.startswith("net short gamma"):
        watchlist.append(
            "Expect faster tape and bigger intraday swings – short gamma regime. "
            "Sharp moves can extend as dealers chase rather than dampen price."
        )
    else:
        watchlist.append(
            "Gamma profile is not aggressively short; dealer hedging may dampen extremes."
        )

    if pw or cw:
        watchlist.append(
            "Use nearby put/call walls as tactical reference: look for intraday "
            "reversals, pinning, or acceleration when spot tests those strikes."
        )

    # 2) Flow tilt
    if pcr_vol is not None and not np.isnan(pcr_vol):
        if pcr_vol < 0.9:
            watchlist.append(
                "Call volume outpacing puts intraday – watch for follow-through "
                "in upside plays, especially near short-dated call clusters."
            )
        elif pcr_vol > 1.1:
            watchlist.append(
                "Put volume dominating – monitor for downside hedging or outright "
                "short bias, particularly if clustered below spot."
            )

    # 3) Unusuals
    if not unusual_df.empty:
        watchlist.append(
            "Track how price reacts around strikes/expiries with largest unusual "
            "put/call blocks – if price respects them, they can act as short-term "
            "support/resistance magnets."
        )

    if watchlist:
        lines.append("\nWhat to watch / how to use this:")
        for item in watchlist:
            lines.append(f"- {item}")

    return "\n".join(lines)

# =============================================================================
# High-level wrapper
# =============================================================================

def run_full_analysis(
    ticker: str = UNDERLYING_DEFAULT,
    price_period: str = "6mo",
    price_interval: str = "1d",
    expiries: Optional[List[str]] = None,
    max_expiries: int = 3,
    rate: float = 0.04,
    dividend_yield: float = 0.012,
    history_lookback_days: int = 60,
    unusual_config: Optional[UnusualConfig] = None,
) -> Dict[str, Any]:
    if unusual_config is None:
        unusual_config = UnusualConfig()

    # 1) Price + MAs
    price_df = get_price_history(ticker, period=price_period, interval=price_interval)
    price_df = add_moving_averages(price_df)

    close_last = price_df["Close"].iloc[-1]
    if isinstance(close_last, pd.Series):
        spot = float(close_last.iloc[0])
    else:
        spot = float(close_last)
    print(f"[INFO] {ticker} last close: {spot:.2f}")

    # 2) Options chain
    chain = get_options_chain_yf(ticker, expiries=expiries, max_expiries=max_expiries)

    # 3) IV & Greeks
    mp = MarketParams(
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
    )
    chain = add_iv_and_greeks_to_chain(chain, mp)

    # 4) Flow metrics & intraday percentiles
    chain = enrich_flow_metrics(chain, spot=spot)
    chain = add_intraday_percentiles(chain)

    # 5) Load history and add history-based features
    history_df = load_history_snapshots(ticker, lookback_days=history_lookback_days)
    chain = add_history_features(chain, history_df, lookback_days=min(20, history_lookback_days))

    # 6) Put/Call ratios
    pcr_info = compute_put_call_ratios(chain)

    # 7) Gamma exposure
    gex_table = compute_gamma_exposure(chain, spot=spot, contract_size=CONTRACT_SIZE)
    gamma_info = summarize_gamma_levels(gex_table)

    # 8) Clusters
    clusters = summarize_clusters(chain, spot=spot)

    # 9) Advanced unusual activity
    unusual_df = flag_unusual_activity_advanced(chain, unusual_config)

    # 10) Save snapshot for future backtests
    save_daily_snapshot(ticker, chain, spot)

    # 11) Narrative summary
    narrative = summarize_unusual_narrative(
        spot=spot,
        pcr_info=pcr_info,
        gamma_info=gamma_info,
        gex_table=gex_table,
        unusual_df=unusual_df,
        cluster_df=clusters,
    )

    return {
        "spot": spot,
        "price_df": price_df,
        "chain_df": chain,
        "pcr_info": pcr_info,
        "gex_table": gex_table,
        "gamma_info": gamma_info,
        "clusters": clusters,
        "unusual_df": unusual_df,
        "narrative": narrative,
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    result = run_full_analysis(
        ticker=UNDERLYING_DEFAULT,
        price_period="6mo",
        price_interval="1d",
        expiries=None,       # None -> nearest `max_expiries`
        max_expiries=3,
        rate=0.04,
        dividend_yield=0.012,
        history_lookback_days=60,
        unusual_config=UnusualConfig(
            notional_pct=0.97,
            volume_pct=0.97,
            vol_oi_pct=0.97,
            min_vol_vs_hist=5.0,
            min_oi_change_ratio=0.5,
            min_dollar_notional=1_000_000.0,
            max_dte_short=7,
        ),
    )

    spot = result["spot"]
    print("\n=== PRICE SNAPSHOT (last row) ===")
    print(result["price_df"].tail(1).T)

    print("\n=== OVERALL PUT/CALL RATIO ===")
    print(result["pcr_info"]["overall"])

    print("\n=== APPROX ZERO-GAMMA LEVEL ===")
    print(result["gamma_info"]["zero_gamma_level"])

    print("\n=== TOP CALL-WALL STRIKES BY GEX ===")
    print(result["gamma_info"]["call_wall"].head(10))

    print("\n=== TOP PUT-WALL STRIKES BY GEX ===")
    print(result["gamma_info"]["put_wall"].head(10))

    print("\n=== TOP OI STRIKES ===")
    print(result["gamma_info"]["oi_wall"].head(10))

    print("\n=== UNUSUAL OPTIONS ACTIVITY (head) ===")
    print(result["unusual_df"].head(20))

    print("\n=== TOP CLUSTERS (head) ===")
    print(result["clusters"].head(10))

    print("\n=== NARRATIVE SUMMARY ===")
    print(result["narrative"])
