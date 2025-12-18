from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Core data-fetch helpers
# ---------------------------------------------------------------------------


def get_price_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download underlying price history and add simple moving averages.
    Returns a DataFrame indexed by date with at least: Open, High, Low, Close, Volume,
    plus SMA20 / SMA50 / SMA200.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if df.empty:
        return pd.DataFrame()

    # Some yfinance versions return a multi-index with "Adj Close"
    if isinstance(df.columns, pd.MultiIndex):
        wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        cols = [c for c in df.columns if c[0] in wanted]
        df = df[cols]
        df.columns = [c[0] for c in df.columns]

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} in price history for {ticker}")

    close = df["Close"]
    df["SMA20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA50"] = close.rolling(window=50, min_periods=1).mean()
    df["SMA200"] = close.rolling(window=200, min_periods=1).mean()

    return df


def list_option_expiries(ticker: str) -> List[str]:
    tk = yf.Ticker(ticker)
    return list(getattr(tk, "options", []) or [])


def fetch_option_chain(
    ticker: str,
    expiries: Optional[Sequence[str]] = None,
    max_expiries: int = 3,
) -> pd.DataFrame:
    """
    Fetch option chains for the requested expiries and return a single tidy DataFrame.
    """
    tk = yf.Ticker(ticker)
    all_exps: List[str] = list(getattr(tk, "options", []) or [])

    if not all_exps:
        return pd.DataFrame()

    if expiries is None:
        expiries = all_exps[:max_expiries]
    else:
        # keep only those that actually exist
        expiries = [e for e in expiries if e in all_exps]
        if not expiries:
            raise ValueError(f"No expiries available for {ticker}")

    frames: List[pd.DataFrame] = []

    for exp in expiries:
        chain = tk.option_chain(exp)
        for opt_type, df_part in (("call", chain.calls), ("put", chain.puts)):
            if df_part is None or df_part.empty:
                continue
            df_tmp = df_part.copy()
            df_tmp["type"] = opt_type
            df_tmp["expiry"] = pd.to_datetime(exp)
            frames.append(df_tmp)

    if not frames:
        return pd.DataFrame()

    chain_df = pd.concat(frames, ignore_index=True)

    # Make sure key columns exist
    for col in ["volume", "openInterest"]:
        if col not in chain_df.columns:
            chain_df[col] = 0

    if "impliedVolatility" not in chain_df.columns:
        chain_df["impliedVolatility"] = np.nan

    if "strike" not in chain_df.columns:
        raise ValueError("Option chain missing strike column")

    # DTE
    today = datetime.now(timezone.utc).date()
    chain_df["expiry"] = pd.to_datetime(chain_df["expiry"]).dt.date
    chain_df["dte"] = (
        pd.to_datetime(chain_df["expiry"]) - pd.Timestamp(today)
    ).dt.days.clip(lower=0)

    # Mid price (fallback to lastPrice if bid/ask missing)
    def _mid(row):
        b = row.get("bid", np.nan)
        a = row.get("ask", np.nan)
        lp = row.get("lastPrice", np.nan)
        if not math.isnan(b) and not math.isnan(a) and (a - b) >= 0:
            return 0.5 * (a + b)
        if not math.isnan(lp):
            return lp
        return max(b, a) if not (math.isnan(b) and math.isnan(a)) else np.nan

    chain_df["price_used"] = chain_df.apply(_mid, axis=1)

    return chain_df


# ---------------------------------------------------------------------------
# Black–Scholes greeks
# ---------------------------------------------------------------------------


def _bs_d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float]:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan"), float("nan")
    num = math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T
    den = sigma * math.sqrt(T)
    d1 = num / den
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> Tuple[float, float, float, float]:
    """
    Return (delta, gamma, vega, theta_annual) for a call or put.
    Theta is per year; we'll divide by 365 later if we want per-day.
    """
    d1, d2 = _bs_d1_d2(S, K, T, r, q, sigma)
    if math.isnan(d1):
        return float("nan"), 0.0, 0.0, 0.0

    phi_d1 = 1.0 / math.sqrt(2.0 * math.pi) * math.exp(-0.5 * d1 * d1)

    if option_type == "call":
        delta = math.exp(-q * T) * _norm_cdf(d1)
    else:
        delta = -math.exp(-q * T) * _norm_cdf(-d1)

    gamma = math.exp(-q * T) * phi_d1 / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * phi_d1 * math.sqrt(T) / 100.0  # per 1 vol point

    if option_type == "call":
        theta = (
            -0.5 * S * math.exp(-q * T) * phi_d1 * sigma / math.sqrt(T)
            - r * math.exp(-r * T) * K * _norm_cdf(d2)
            + q * math.exp(-q * T) * S * _norm_cdf(d1)
        )
    else:
        theta = (
            -0.5 * S * math.exp(-q * T) * phi_d1 * sigma / math.sqrt(T)
            + r * math.exp(-r * T) * K * _norm_cdf(-d2)
            - q * math.exp(-q * T) * S * _norm_cdf(-d1)
        )
    return delta, gamma, vega, theta


def enrich_with_greeks(
    chain_df: pd.DataFrame,
    spot: float,
    rate: float = 0.04,
    dividend_yield: float = 0.012,
) -> pd.DataFrame:
    if chain_df.empty:
        return chain_df

    df = chain_df.copy()
    df["iv"] = df["impliedVolatility"].astype(float).clip(lower=1e-4, upper=5.0)

    # Time to expiry in years
    df["T"] = df["dte"].astype(float) / 365.0

    deltas = []
    gammas = []
    vegas = []
    thetas = []

    for _, row in df.iterrows():
        S = float(spot)
        K = float(row["strike"])
        T = float(row["T"])
        sigma = float(row["iv"])
        if T <= 0 or sigma <= 0:
            deltas.append(0.0)
            gammas.append(0.0)
            vegas.append(0.0)
            thetas.append(0.0)
            continue
        d, g, v, th = black_scholes_greeks(
            S=S,
            K=K,
            T=T,
            r=rate,
            q=dividend_yield,
            sigma=sigma,
            option_type=row["type"],
        )
        deltas.append(d)
        gammas.append(g)
        thetas.append(th / 365.0)  # per day
        vegas.append(v)

    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas

    df["moneyness"] = df["strike"] / float(spot) - 1.0
    return df


# ---------------------------------------------------------------------------
# Put/call ratios and gamma exposure
# ---------------------------------------------------------------------------


def compute_put_call_ratios(chain_df: pd.DataFrame) -> Dict[str, Any]:
    if chain_df.empty:
        return {
            "call_vol": 0.0,
            "put_vol": 0.0,
            "call_oi": 0.0,
            "put_oi": 0.0,
            "pcr_vol": float("nan"),
            "pcr_oi": float("nan"),
        }

    calls = chain_df[chain_df["type"] == "call"]
    puts = chain_df[chain_df["type"] == "put"]

    call_vol = float(calls["volume"].sum())
    put_vol = float(puts["volume"].sum())
    call_oi = float(calls["openInterest"].sum())
    put_oi = float(puts["openInterest"].sum())

    pcr_vol = put_vol / call_vol if call_vol > 0 else float("nan")
    pcr_oi = put_oi / call_oi if call_oi > 0 else float("nan")

    return {
        "call_vol": call_vol,
        "put_vol": put_vol,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "pcr_vol": pcr_vol,
        "pcr_oi": pcr_oi,
    }


def compute_gamma_exposure_by_strike(
    chain_df: pd.DataFrame,
    spot: float,
    contract_size: int = 100,
) -> pd.DataFrame:
    """
    Aggregate gamma exposure by strike across all expiries.
    Positive = long gamma for the street, negative = short.
    """
    if chain_df.empty:
        return pd.DataFrame(columns=["strike", "gex", "total_oi", "call_oi", "put_oi", "cum_gex"])

    df = chain_df.copy()
    df["gex_raw"] = df["gamma"] * df["openInterest"] * contract_size * (spot ** 2)

    # Calls positive gamma exposure, puts negative (dealer perspective)
    df.loc[df["type"] == "put", "gex_raw"] *= -1.0

    grouped = df.groupby("strike").agg(
        gex=("gex_raw", "sum"),
        total_oi=("openInterest", "sum"),
        call_oi=("openInterest", lambda x: x[df.loc[x.index, "type"] == "call"].sum()),
        put_oi=("openInterest", lambda x: x[df.loc[x.index, "type"] == "put"].sum()),
    )
    grouped = grouped.sort_index().reset_index()

    grouped["cum_gex"] = grouped["gex"].cumsum()

    return grouped


def summarize_gamma_levels(gex_table: pd.DataFrame, spot: float) -> Dict[str, Any]:
    if gex_table.empty:
        return {
            "zero_gamma": None,
            "call_walls": pd.DataFrame(),
            "put_walls": pd.DataFrame(),
        }

    # Approx zero-gamma: strike where |cum_gex| is minimal
    idx_z = gex_table["cum_gex"].abs().idxmin()
    zero_gamma = float(gex_table.loc[idx_z, "strike"])

    # Call wall = highest positive gex; put wall = most negative
    call_walls = gex_table.sort_values("gex", ascending=False).head(5)
    put_walls = gex_table.sort_values("gex").head(5)

    return {
        "zero_gamma": zero_gamma,
        "call_walls": call_walls,
        "put_walls": put_walls,
    }


# ---------------------------------------------------------------------------
# Unusual flow detection
# ---------------------------------------------------------------------------


@dataclass
class UnusualConfig:
    notional_pct: float = 0.97
    volume_pct: float = 0.97
    vol_oi_pct: float = 0.97
    min_vol_vs_hist: float = 2.0
    min_oi_change_ratio: float = 0.5
    min_dollar_notional: float = 1_000_000.0
    max_dte_short: int = 7


def _bucket_delta(delta: float) -> str:
    if math.isnan(delta):
        return "unknown"
    a = abs(delta)
    if a >= 0.75:
        return "deep_itm"
    if 0.35 <= a < 0.75:
        return "atm"
    if 0.15 <= a < 0.35:
        return "otm"
    return "lottery"


def flag_unusual_activity(
    chain_df: pd.DataFrame,
    config: UnusualConfig,
) -> pd.DataFrame:
    """
    Flag large / concentrated trades using only *today's* cross-section.
    This avoids needing a persistent history on Streamlit.
    """
    if chain_df.empty:
        return pd.DataFrame()

    df = chain_df.copy()

    # Restrict to short-dated flow
    df = df[df["dte"] <= config.max_dte_short].copy()
    if df.empty:
        return df

    # Dollar notional
    df["dollar_notional"] = df["price_used"].fillna(0.0) * df["volume"].fillna(0.0) * 100.0

    # Basic ratios
    df["vol_oi_ratio"] = (
        df["volume"].astype(float)
        / df["openInterest"].replace(0, np.nan).astype(float)
    )
    df["vol_oi_ratio"] = df["vol_oi_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Cross-sectional percentiles (per snapshot)
    df["notional_pct"] = df["dollar_notional"].rank(pct=True)
    df["volume_pct"] = df["volume"].rank(pct=True)
    df["vol_oi_pct"] = df["vol_oi_ratio"].rank(pct=True)

    # Volume vs "history": use median per expiry & type as a stand-in baseline
    grp_key = ["expiry", "type"]
    med_vol = df.groupby(grp_key)["volume"].transform("median").replace(0, np.nan)
    df["vol_vs_hist"] = (df["volume"] / med_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # We don't have real OI changes without a history database, so keep this neutral
    df["oi_change_ratio"] = 1.0

    # Filters
    mask_big_notional = df["dollar_notional"] >= config.min_dollar_notional
    mask_percentiles = (
        (df["notional_pct"] >= config.notional_pct)
        | (df["volume_pct"] >= config.volume_pct)
        | (df["vol_oi_pct"] >= config.vol_oi_pct)
    )
    mask_oi_change = df["oi_change_ratio"] >= config.min_oi_change_ratio

    # For now we require big notional AND percentile filters AND neutral OI change
    mask = mask_big_notional & mask_percentiles & mask_oi_change

    unusual = df.loc[mask].copy()
    if unusual.empty:
        return unusual

    # Categorise delta & build "reason" string
    unusual["delta_bucket"] = unusual["delta"].astype(float).apply(_bucket_delta)

    def _reason(row: pd.Series) -> str:
        reasons = ["big_notional"]
        if row["notional_pct"] >= config.notional_pct:
            reasons.append("notional_pct>thr")
        if row["volume_pct"] >= config.volume_pct:
            reasons.append("vol_pct>thr")
        if row["vol_oi_pct"] >= config.vol_oi_pct:
            reasons.append("volOI_pct>thr")
        if row["vol_vs_hist"] >= config.min_vol_vs_hist:
            reasons.append("vol_vs_hist>thr")
        if row["oi_change_ratio"] >= config.min_oi_change_ratio:
            reasons.append("OI_change>thr")
        reasons.append(f"{int(row['dte'])}DTE")
        return "|".join(reasons)

    unusual["reason"] = unusual.apply(_reason, axis=1)

    # Nice ordering of columns for the app
    cols_order = [
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
        "vol_vs_hist",
        "oi_change_ratio",
        "iv",
        "delta",
        "gamma",
        "vega",
        "theta",
        "moneyness",
        "delta_bucket",
        "reason",
    ]
    existing_cols = [c for c in cols_order if c in unusual.columns]
    unusual = unusual[existing_cols].sort_values("dollar_notional", ascending=False)

    return unusual


def build_notional_clusters(chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Aggregate notional by expiry and moneyness band (% from spot).
    """
    if chain_df.empty:
        return pd.DataFrame()

    df = chain_df.copy()
    df["dollar_notional"] = df["price_used"].fillna(0.0) * df["openInterest"].fillna(0.0) * 100.0
    df["moneyness_band"] = ((df["strike"] / spot - 1.0) * 100).round().astype(int)

    grouped = (
        df.groupby(["expiry", "moneyness_band"])
        .agg(
            total_notional=("dollar_notional", "sum"),
            avg_moneyness=("moneyness_band", "mean"),
            call_notional=("dollar_notional", lambda x: x[df.loc[x.index, "type"] == "call"].sum()),
            put_notional=("dollar_notional", lambda x: x[df.loc[x.index, "type"] == "put"].sum()),
        )
        .reset_index()
    )

    grouped["net_call_minus_put"] = grouped["call_notional"] - grouped["put_notional"]
    return grouped


# ---------------------------------------------------------------------------
# Volatility complex (IV vs RV, VIX, skew)
# ---------------------------------------------------------------------------


def compute_realized_vol(price_df: pd.DataFrame, window: int = 20) -> float:
    if price_df.empty or price_df.shape[0] < window + 1:
        return float("nan")
    close = price_df["Close"].astype(float)
    ret = np.log(close / close.shift(1)).dropna()
    rv = ret.rolling(window=window).std().iloc[-1] * math.sqrt(252.0) * 100.0
    return float(rv)


def compute_atm_iv_30d(chain_df: pd.DataFrame, spot: float) -> float:
    """
    Very simple proxy: average IV of options with 20–40 DTE and |moneyness| < 2%.
    """
    if chain_df.empty:
        return float("nan")

    df = chain_df.copy()
    mask = (df["dte"].between(20, 40)) & (df["moneyness"].abs() <= 0.02)
    subset = df.loc[mask]
    if subset.empty:
        return float("nan")
    return float(subset["iv"].mean() * 100.0)


def compute_skew_30d(chain_df: pd.DataFrame, spot: float) -> float:
    """
    30D 25Δ skew = (25Δ put IV – 25Δ call IV) in vol points.
    We'll approximate 25Δ by bucket on delta.
    """
    if chain_df.empty:
        return float("nan")

    df = chain_df.copy()
    mask_30d = df["dte"].between(20, 40)
    df = df.loc[mask_30d]
    if df.empty:
        return float("nan")

    puts = df[(df["type"] == "put") & (df["delta"].between(-0.35, -0.15))]
    calls = df[(df["type"] == "call") & (df["delta"].between(0.15, 0.35))]

    if puts.empty or calls.empty:
        return float("nan")

    put_iv = puts["iv"].mean() * 100.0
    call_iv = calls["iv"].mean() * 100.0
    return float(put_iv - call_iv)


def fetch_vix_term_structure(period: str = "6mo") -> pd.DataFrame:
    """
    Fetch ^VIX and ^VIX3M history.
    Returns DataFrame indexed by date with columns '^VIX' and '^VIX3M'.
    """
    tickers = ["^VIX", "^VIX3M"]
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw.xs("Adj Close", axis=1, level=0, drop_level=True)
    else:
        adj = raw[["Adj Close"]]

    # In the multi-index case columns will be tickers; in the single-index case we only have one
    if isinstance(adj, pd.DataFrame) and set(tickers).issubset(adj.columns):
        df = adj[tickers].copy()
    else:
        # Fallback: best-effort rename if possible
        df = adj.copy()
    df.index.name = "Date"
    return df.dropna(how="all")


def fetch_yield_curve(period: str = "6mo") -> pd.DataFrame:
    """
    Fetch 10Y and 3M US yields (TNX / IRX). Returned in %.
    """
    tickers = ["^TNX", "^IRX"]
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw.xs("Adj Close", axis=1, level=0, drop_level=True)
    else:
        adj = raw[["Adj Close"]]

    df = adj.copy()

    col_map = {}
    for col in df.columns:
        if "TNX" in str(col).upper():
            col_map[col] = "10Y"
        elif "IRX" in str(col).upper():
            col_map[col] = "3M"
    if col_map:
        df = df.rename(columns=col_map)
    if "10Y" not in df.columns or "3M" not in df.columns:
        return pd.DataFrame()

    df["10Y"] = df["10Y"] / 100.0
    df["3M"] = df["3M"] / 100.0
    df["slope"] = df["10Y"] - df["3M"]
    df.index.name = "Date"
    return df.dropna(how="any")


def fetch_credit_series(period: str = "6mo") -> tuple[pd.DataFrame, float, float]:
    """
    Credit risk appetite via HYG/LQD.
    Returns (df, last_level, last_zscore).
    """
    raw = yf.download(["HYG", "LQD"], period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame(), float("nan"), float("nan")

    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw.xs("Adj Close", axis=1, level=0, drop_level=True)
    else:
        adj = raw[["Adj Close"]]

    df = adj.copy()
    col_map = {}
    for col in df.columns:
        if "HYG" in str(col).upper():
            col_map[col] = "HYG"
        elif "LQD" in str(col).upper():
            col_map[col] = "LQD"
    if col_map:
        df = df.rename(columns=col_map)

    if "HYG" not in df.columns or "LQD" not in df.columns:
        return pd.DataFrame(), float("nan"), float("nan")

    ratio = (df["HYG"] / df["LQD"]).rename("HYG_LQD")
    out = ratio.to_frame()
    level = float(ratio.iloc[-1])

    if ratio.std(ddof=0) > 0:
        z = (ratio - ratio.mean()) / ratio.std(ddof=0)
        z_last = float(z.iloc[-1])
        out["z"] = z
    else:
        z_last = float("nan")
        out["z"] = np.nan

    out.index.name = "Date"
    return out, level, z_last


def fetch_dollar_series(period: str = "6mo") -> tuple[pd.DataFrame, float, str]:
    """
    Dollar via UUP ETF. Returns (df, last_level, regime_label).
    """
    raw = yf.download("UUP", period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        return pd.DataFrame(), float("nan"), "n/a"

    if isinstance(raw.columns, pd.MultiIndex):
        adj = raw.xs("Adj Close", axis=1, level=0, drop_level=True)
        if isinstance(adj, pd.DataFrame):
            if "UUP" in adj.columns:
                df = adj[["UUP"]].copy()
            else:
                df = adj.iloc[:, [0]].copy()
        else:
            df = adj.to_frame(name="UUP")
    else:
        df = raw[["Adj Close"]].copy()
        df = df.rename(columns={"Adj Close": "UUP"})

    df.index.name = "Date"

    if df.shape[0] < 5:
        last = float(df["UUP"].iloc[-1])
        return df, last, "n/a"

    last = float(df["UUP"].iloc[-1])
    # Simple trend classification: slope of last 20d
    series = df["UUP"].astype(float)
    x = np.arange(len(series[-20:]))
    y = series[-20:].values
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
    else:
        slope = 0.0

    if slope > 0:
        regime = "uptrend / stronger dollar"
    elif slope < 0:
        regime = "downtrend / benign dollar"
    else:
        regime = "flat / neutral"

    return df, last, regime


def vol_complex_summary(price_df: pd.DataFrame, chain_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Pull together IV vs RV, VIX term structure and skew into a compact summary.
    """
    if price_df.empty:
        rv20 = float("nan")
    else:
        rv20 = compute_realized_vol(price_df, window=20)

    spot = float(price_df["Close"].iloc[-1]) if not price_df.empty else float("nan")
    atm_iv_30d = compute_atm_iv_30d(chain_df, spot)
    skew_30d = compute_skew_30d(chain_df, spot)

    iv_rv_ratio = (
        atm_iv_30d / rv20
        if rv20 and not math.isnan(rv20) and rv20 != 0 and atm_iv_30d and not math.isnan(atm_iv_30d)
        else float("nan")
    )

    if math.isnan(iv_rv_ratio):
        iv_label = "n/a"
    elif iv_rv_ratio > 1.15:
        iv_label = "rich (IV >> RV)"
    elif iv_rv_ratio < 0.85:
        iv_label = "cheap (IV << RV)"
    else:
        iv_label = "fair (IV/RV ≈ 1)"

    # VIX term structure
    vix_df = fetch_vix_term_structure(period="6mo")
    if not vix_df.empty and "^VIX" in vix_df.columns and "^VIX3M" in vix_df.columns:
        spot_vix = float(vix_df["^VIX"].iloc[-1])
        spot_vix3m = float(vix_df["^VIX3M"].iloc[-1])
    else:
        spot_vix = float("nan")
        spot_vix3m = float("nan")

    term_label = "n/a"
    if not math.isnan(spot_vix) and not math.isnan(spot_vix3m):
        spread = spot_vix3m - spot_vix
        if spread > 3.0:
            term_label = f"calm contango (3M–spot ≈ +{spread:.1f} vol pts)"
        elif spread < 0:
            term_label = f"backwardation / stress (3M–spot ≈ {spread:.1f} vol pts)"
        else:
            term_label = f"flat / mildly upward (3M–spot ≈ +{spread:.1f} vol pts)"

    # Skew label
    if math.isnan(skew_30d):
        skew_label = "n/a"
    elif skew_30d > 3.0:
        skew_label = "steep downside skew (puts rich vs calls)"
    elif skew_30d < 0.5:
        skew_label = "flat / inverted skew (little downside premium)"
    else:
        skew_label = "mild downside skew"

    # Tidy VIX for plotting
    if vix_df.empty:
        vix_long = pd.DataFrame(columns=["Date", "Index", "Level"])
    else:
        rename_map = {}
        for c in vix_df.columns:
            if "VIX3M" in str(c).upper():
                rename_map[c] = "VIX3M"
            elif "VIX" in str(c).upper():
                rename_map[c] = "VIX"
        vix_plot = vix_df.rename(columns=rename_map)
        cols = [c for c in vix_plot.columns if c in ("VIX", "VIX3M")]
        vix_plot = vix_plot[cols]
        vix_long = (
            vix_plot.reset_index(names="Date")
            .melt(id_vars="Date", var_name="Index", value_name="Level")
        )

    return {
        "rv20": rv20,
        "atm_iv_30d": atm_iv_30d,
        "iv_rv_ratio": iv_rv_ratio,
        "iv_vs_rv_label": iv_label,
        "iv_rv_label": iv_label,  # alias for the app
        "skew_30d": skew_30d,
        "skew_label": skew_label,
        "vix": spot_vix,
        "vix3m": spot_vix3m,
        "vix_term_label": term_label,
        "vix_df": vix_df,
        "vix_long_df": vix_long,
    }


# ---------------------------------------------------------------------------
# Macro tape (rates, credit, dollar)
# ---------------------------------------------------------------------------


def macro_tape_summary(period: str = "6mo") -> Dict[str, Any]:
    """
    Summarise rates (10Y vs 3M), credit (HYG/LQD) and dollar (UUP)
    into a compact macro "tape".
    """
    yc_df = fetch_yield_curve(period=period)
    credit_df, credit_level, credit_z = fetch_credit_series(period=period)
    uup_df, uup_last, uup_regime = fetch_dollar_series(period=period)

    # Rates metrics
    if yc_df.empty:
        last_10y = last_3m = yc_slope = float("nan")
        rates_regime = "n/a"
    else:
        last_row = yc_df.iloc[-1]
        last_10y = float(last_row["10Y"])
        last_3m = float(last_row["3M"])
        yc_slope = float(last_row["slope"])
        if yc_slope < -0.005:
            rates_regime = "deep inversion / late-cycle risk"
        elif yc_slope < 0.0:
            rates_regime = "flat / mild inversion"
        else:
            rates_regime = "steepening / macro tailwind"

    # Credit metrics
    if credit_df.empty or math.isnan(credit_level):
        credit_regime = "n/a"
    else:
        if credit_z < -1.0:
            credit_regime = "risk-off / stress"
        elif credit_z > 1.0:
            credit_regime = "risk-on / stretched"
        else:
            credit_regime = "neutral"

    # Dollar metrics (already labelled in helper)
    if uup_df.empty or math.isnan(uup_last):
        dollar_regime = "n/a"
    else:
        dollar_regime = uup_regime

    # Tidy dataframes for plotting
    if yc_df.empty:
        yc_tidy = pd.DataFrame(columns=["Date", "Tenor", "Yield"])
    else:
        yc_tidy = (
            yc_df[["10Y", "3M"]]
            .reset_index(names="Date")
            .melt(id_vars="Date", var_name="Tenor", value_name="Yield")
        )

    if credit_df.empty:
        credit_tidy = pd.DataFrame(columns=["Date", "HYG/LQD"])
    else:
        credit_tidy = (
            credit_df[["HYG_LQD"]]
            .rename(columns={"HYG_LQD": "HYG/LQD"})
            .reset_index(names="Date")
        )

    if uup_df.empty:
        dollar_tidy = pd.DataFrame(columns=["Date", "UUP"])
    else:
        dollar_tidy = uup_df.reset_index(names="Date")

    metrics = {
        "rates": {
            "last_10y": last_10y,
            "last_3m": last_3m,
            "slope": yc_slope,
            "regime": rates_regime,
        },
        "credit": {
            "last_ratio": credit_level,
            "zscore": credit_z,
            "regime": credit_regime,
        },
        "dollar": {
            "last": uup_last,
            "regime": dollar_regime,
        },
    }

    return {
        "metrics": metrics,
        "yield_curve_df": yc_tidy,
        "credit_df": credit_tidy,
        "dollar_df": dollar_tidy,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_full_analysis(
    ticker: str,
    price_period: str = "6mo",
    price_interval: str = "1d",
    expiries: Optional[Sequence[str]] = None,
    max_expiries: int = 3,
    rate: float = 0.04,
    dividend_yield: float = 0.012,
    history_lookback_days: int = 60,  # kept for compatibility with the app
    unusual_config: Optional[UnusualConfig] = None,
) -> Dict[str, Any]:
    """
    High-level orchestration: fetch price + options, compute all metrics.
    Returns a dict to be consumed by the Streamlit app.
    """
    if unusual_config is None:
        unusual_config = UnusualConfig()

    price_df = get_price_history(ticker, period=price_period, interval=price_interval)
    if price_df.empty:
        raise ValueError(f"No price data for {ticker}")

    spot = float(price_df["Close"].iloc[-1])

    chain_df = fetch_option_chain(ticker, expiries=expiries, max_expiries=max_expiries)
    if chain_df.empty:
        raise ValueError(f"No option chain data for {ticker}")

    # Greeks & derived fields
    chain_df = enrich_with_greeks(chain_df, spot=spot, rate=rate, dividend_yield=dividend_yield)

    # Put/call ratios
    pcr_info = {"overall": compute_put_call_ratios(chain_df)}

    # Gamma exposure
    gex_table = compute_gamma_exposure_by_strike(chain_df, spot=spot)
    gamma_info = summarize_gamma_levels(gex_table, spot=spot)

    # Unusual activity & notional clusters
    unusual_df = flag_unusual_activity(chain_df, unusual_config)
    clusters_df = build_notional_clusters(chain_df, spot=spot)

    # Narrative summary for the main page
    overall = pcr_info["overall"]
    pcr_vol = overall["pcr_vol"]
    pcr_oi = overall["pcr_oi"]
    zero_gamma = gamma_info["zero_gamma"]
    call_wall = gamma_info["call_walls"]["strike"].iloc[0] if not gamma_info["call_walls"].empty else None
    put_wall = gamma_info["put_walls"]["strike"].iloc[0] if not gamma_info["put_walls"].empty else None

    narrative_lines = [f"Spot ≈ {spot:.2f}. "]
    if not math.isnan(pcr_vol):
        narrative_lines.append(
            f"Intraday flow is "
            f"{'put-heavy' if pcr_vol > 1.0 else 'call-biased' if pcr_vol < 0.8 else 'balanced'} "
            f"(PCR_vol ≈ {pcr_vol:.2f}). "
        )
    if not math.isnan(pcr_oi):
        narrative_lines.append(
            f"Outstanding positioning is "
            f"{'put-heavy' if pcr_oi > 1.2 else 'balanced'} "
            f"(PCR_OI ≈ {pcr_oi:.2f}). "
        )
    if zero_gamma is not None:
        narrative_lines.append(f"Approx zero-gamma level near {zero_gamma:.1f}. ")
    if call_wall is not None and put_wall is not None:
        narrative_lines.append(
            f"Key gamma levels: call_wall ≈ {call_wall:.1f}, put_wall ≈ {put_wall:.1f}."
        )

    narrative = "".join(narrative_lines)

    # Volatility complex & macro tape
    vol_info = vol_complex_summary(price_df, chain_df)
    macro = macro_tape_summary(period=price_period)

    return {
        "spot": spot,
        "price_df": price_df,
        "chain_df": chain_df,
        "pcr_info": pcr_info,
        "gex_table": gex_table,
        "gamma_info": gamma_info,
        "unusual_df": unusual_df,
        "clusters": clusters_df,
        "narrative": narrative,
        "vol_info": vol_info,
        "macro": macro,
    }
