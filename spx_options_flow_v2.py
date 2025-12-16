"""
Options flow + gamma + volatility analytics engine for the Streamlit dashboard.

Stage 2 version: includes
- price history with moving averages
- options chain download (yfinance) with Greeks
- gamma exposure (GEX) / zero-gamma / walls
- put/call ratios
- unusual options activity (intraday, percentile-based)
- notional clusters by expiry & moneyness band
- implied distribution / expected moves from ATM IV
- per-expiry vol surface snapshot (ATM, 25Δ RR & butterfly) with simple history

External API:
    - UnusualConfig dataclass
    - run_full_analysis(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datetime as dt
import math

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# Basic math helpers
# ---------------------------------------------------------------------


def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------
# Black–Scholes greeks
# ---------------------------------------------------------------------


def bs_greeks(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    div_yield: float,
    vol: float,
    is_call: bool,
) -> Dict[str, float]:
    """
    Black–Scholes greeks for a European option.

    time: in years
    vol: annualized volatility (e.g. 0.20 for 20%)
    """
    if time <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

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
    vega = disc_q * spot * pdf_d1 * sqrt_t  # per 1.0 vol (not %)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
    }


# ---------------------------------------------------------------------
# Price history
# ---------------------------------------------------------------------


def get_price_history(
    ticker: str, period: str = "6mo", interval: str = "1d"
) -> pd.DataFrame:
    """
    Download OHLCV history from yfinance and compute moving averages.
    """
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    if data.empty:
        raise RuntimeError(f"No price data for {ticker}")

    df = data.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    return df


# ---------------------------------------------------------------------
# Options chain download & greeks
# ---------------------------------------------------------------------


def _all_expiries(ticker: str) -> List[str]:
    t = yf.Ticker(ticker)
    return list(t.options)


def _select_expiries(
    ticker: str, expiries: Optional[Sequence[str]], max_expiries: int
) -> List[str]:
    if expiries:
        return list(expiries)[:max_expiries]
    all_exp = _all_expiries(ticker)
    return all_exp[:max_expiries]


def get_options_chain(
    ticker: str,
    expiries: Optional[Sequence[str]] = None,
    max_expiries: int = 3,
    rate: float = 0.04,
    dividend_yield: float = 0.012,
) -> pd.DataFrame:
    """
    Download options chain for selected expiries and compute greeks.
    Uses yfinance's impliedVolatility when available.
    """
    t = yf.Ticker(ticker)
    chosen_exp = _select_expiries(ticker, expiries, max_expiries)
    if not chosen_exp:
        raise RuntimeError(f"No options expiries available for {ticker}")

    rows: List[Dict[str, Any]] = []
    spot = float(t.history(period="1d")["Close"].iloc[-1])
    today = dt.date.today()

    for exp_str in chosen_exp:
        opt = t.option_chain(exp_str)
        expiry = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()

        for opt_type, df_side in (("call", opt.calls), ("put", opt.puts)):
            if df_side.empty:
                continue
            for _, row in df_side.iterrows():
                strike = float(row["strike"])
                dte = (expiry - today).days
                time = max(dte, 0) / 365.0

                # Price proxy: mid of bid/ask if possible, else lastPrice
                bid = float(row.get("bid", np.nan))
                ask = float(row.get("ask", np.nan))
                last = float(row.get("lastPrice", np.nan))
                if not math.isnan(bid) and not math.isnan(ask) and ask > 0:
                    price_used = 0.5 * (bid + ask)
                elif not math.isnan(last):
                    price_used = last
                else:
                    price_used = np.nan

                iv_raw = float(row.get("impliedVolatility", np.nan))
                if math.isnan(iv_raw) or iv_raw <= 0:
                    # Fallback if missing – rough 20%
                    iv_raw = 0.20
                vol = iv_raw  # 0.xx

                greeks = bs_greeks(
                    spot=spot,
                    strike=strike,
                    time=time,
                    rate=rate,
                    div_yield=dividend_yield,
                    vol=vol,
                    is_call=(opt_type == "call"),
                )

                volume = float(row.get("volume", 0.0))
                open_interest = float(row.get("openInterest", 0.0))

                rows.append(
                    {
                        "contractSymbol": row.get("contractSymbol"),
                        "type": opt_type,
                        "expiry": expiry,
                        "dte": dte,
                        "strike": strike,
                        "lastPrice": last,
                        "bid": bid,
                        "ask": ask,
                        "price_used": price_used,
                        "volume": volume,
                        "openInterest": open_interest,
                        "iv": iv_raw * 100.0,  # store in %
                        "delta": greeks["delta"],
                        "gamma": greeks["gamma"],
                        "vega": greeks["vega"],
                        "theta": greeks["theta"],
                    }
                )

    chain = pd.DataFrame(rows)
    if chain.empty:
        raise RuntimeError("No options data loaded")

    chain["moneyness"] = chain["strike"] / spot - 1.0
    chain["dollar_notional"] = chain["price_used"].fillna(0.0) * chain["volume"] * 100.0
    chain["vol_oi_ratio"] = chain["volume"] / chain["openInterest"].replace(0, np.nan)
    return chain


# ---------------------------------------------------------------------
# Put / call ratios
# ---------------------------------------------------------------------


def compute_put_call_ratios(chain_df: pd.DataFrame) -> Dict[str, float]:
    call_vol = chain_df.loc[chain_df["type"] == "call", "volume"].sum()
    put_vol = chain_df.loc[chain_df["type"] == "put", "volume"].sum()
    call_oi = chain_df.loc[chain_df["type"] == "call", "openInterest"].sum()
    put_oi = chain_df.loc[chain_df["type"] == "put", "openInterest"].sum()

    pcr_vol = float(put_vol / call_vol) if call_vol > 0 else np.nan
    pcr_oi = float(put_oi / call_oi) if call_oi > 0 else np.nan

    return {
        "call_vol": float(call_vol),
        "put_vol": float(put_vol),
        "call_oi": float(call_oi),
        "put_oi": float(put_oi),
        "pcr_vol": pcr_vol,
        "pcr_oi": pcr_oi,
    }


# ---------------------------------------------------------------------
# Gamma exposure / walls / zero-gamma
# ---------------------------------------------------------------------


def compute_gex(chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Approximate per-strike gamma exposure.

    gex ≈ gamma * spot^2 * openInterest * contract_multiplier (100)
    sign: calls positive, puts negative.
    """
    df = chain_df.copy()
    df["gamma_dollar"] = df["gamma"] * (spot ** 2) * df["openInterest"] * 100.0
    df.loc[df["type"] == "put", "gamma_dollar"] *= -1.0

    grouped = (
        df.groupby("strike")
        .agg(
            gex=("gamma_dollar", "sum"),
            total_oi=("openInterest", "sum"),
            call_oi=(
                "openInterest",
                lambda x: x[df.loc[x.index, "type"] == "call"].sum(),
            ),
            put_oi=(
                "openInterest",
                lambda x: x[df.loc[x.index, "type"] == "put"].sum(),
            ),
        )
        .reset_index()
        .sort_values("strike")
    )

    grouped["cum_gex"] = grouped["gex"].cumsum()
    return grouped


def summarize_gamma(grouped_gex: pd.DataFrame) -> Dict[str, Any]:
    if grouped_gex.empty:
        return {"call_wall": None, "put_wall": None, "zero_gamma": None}

    call_row = grouped_gex.loc[grouped_gex["gex"].idxmax()]
    put_row = grouped_gex.loc[grouped_gex["gex"].idxmin()]

    call_wall = float(call_row["strike"])
    put_wall = float(put_row["strike"])

    # Zero-gamma: where cum_gex crosses zero
    cg = grouped_gex["cum_gex"].values
    strikes = grouped_gex["strike"].values
    zero_level: Optional[float] = None
    for i in range(1, len(cg)):
        if cg[i - 1] <= 0 <= cg[i] or cg[i - 1] >= 0 >= cg[i]:
            x0, x1 = strikes[i - 1], strikes[i]
            y0, y1 = cg[i - 1], cg[i]
            if y1 != y0:
                zero_level = float(x0 + (x1 - x0) * (-y0) / (y1 - y0))
            else:
                zero_level = float(x0)
            break

    return {"call_wall": call_wall, "put_wall": put_wall, "zero_gamma": zero_level}


# ---------------------------------------------------------------------
# Unusual options activity (intraday, percentile-based)
# ---------------------------------------------------------------------


@dataclass
class UnusualConfig:
    notional_pct: float = 0.97
    volume_pct: float = 0.97
    vol_oi_pct: float = 0.97
    # kept for compatibility with your UI – currently not used in logic
    min_vol_vs_hist: float = 5.0
    min_oi_change_ratio: float = 0.5
    min_dollar_notional: float = 1_000_000.0
    max_dte_short: int = 7


def flag_unusual_activity(chain_df: pd.DataFrame, config: UnusualConfig) -> pd.DataFrame:
    df = chain_df.copy()

    df["notional_pct"] = df["dollar_notional"].rank(pct=True)
    df["volume_pct"] = df["volume"].rank(pct=True)
    df["vol_oi_pct"] = df["vol_oi_ratio"].rank(pct=True)

    cond_big = df["dollar_notional"] >= config.min_dollar_notional
    cond_notional = df["notional_pct"] >= config.notional_pct
    cond_volume = df["volume_pct"] >= config.volume_pct
    cond_vol_oi = df["vol_oi_pct"] >= config.vol_oi_pct
    cond_short = df["dte"] <= config.max_dte_short

    mask = cond_big & cond_short & (cond_notional | cond_volume | cond_vol_oi)
    unusual = df.loc[mask].copy()

    if unusual.empty:
        return unusual

    reasons: List[str] = []
    for _, row in unusual.iterrows():
        r: List[str] = []
        if row["dollar_notional"] >= config.min_dollar_notional:
            r.append("big_notional")
        if row["notional_pct"] >= config.notional_pct:
            r.append("top_notional")
        if row["volume_pct"] >= config.volume_pct:
            r.append("top_volume")
        if row["vol_oi_pct"] >= config.vol_oi_pct:
            r.append("top_volOI")
        if row["dte"] <= config.max_dte_short:
            r.append(f"{int(row['dte'])}DTE")
        reasons.append("|".join(r))

    unusual["reason"] = reasons
    unusual = unusual.sort_values("dollar_notional", ascending=False)
    return unusual


# ---------------------------------------------------------------------
# Notional clusters by expiry & moneyness
# ---------------------------------------------------------------------


def build_notional_clusters(chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    df = chain_df.copy()
    if df.empty:
        return df

    # moneyness band in % (rounded)
    df["moneyness_band"] = (100.0 * (df["strike"] / spot - 1.0)).round().astype(int)

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

    grouped["net_call_minus_put"] = grouped["call_notional"] - grouped["put_notional"]
    grouped = grouped.sort_values("total_notional", ascending=False).reset_index(drop=True)
    return grouped


# ---------------------------------------------------------------------
# Implied distribution / expected moves (from ATM IV)
# ---------------------------------------------------------------------


def pick_reference_expiry(
    chain_df: pd.DataFrame, target_dte: int = 30, max_dte: int = 60
) -> Optional[pd.Timestamp]:
    if "expiry" not in chain_df.columns or "dte" not in chain_df.columns:
        return None
    exp = chain_df[["expiry", "dte"]].drop_duplicates().sort_values("dte")
    if exp.empty:
        return None

    # Prefer expiries with dte >= target
    candidates = exp[exp["dte"] >= target_dte]
    if not candidates.empty:
        return pd.to_datetime(candidates.iloc[0]["expiry"])

    return pd.to_datetime(exp.iloc[-1]["expiry"])


def atm_iv_for_expiry(
    chain_df: pd.DataFrame, expiry: pd.Timestamp, spot: float
) -> Optional[float]:
    df = chain_df.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    df = df[df["expiry"] == expiry]
    if df.empty or "iv" not in df.columns:
        return None

    rows: List[Dict[str, float]] = []
    for strike, g in df.groupby("strike"):
        call_iv = g.loc[g["type"] == "call", "iv"].dropna()
        put_iv = g.loc[g["type"] == "put", "iv"].dropna()
        if call_iv.empty and put_iv.empty:
            continue
        if call_iv.empty:
            iv_mid = float(put_iv.iloc[0])
        elif put_iv.empty:
            iv_mid = float(call_iv.iloc[0])
        else:
            iv_mid = float(0.5 * (call_iv.iloc[0] + put_iv.iloc[0]))
        rows.append({"strike": strike, "iv_mid": iv_mid})

    if not rows:
        return None

    tmp = pd.DataFrame(rows)
    row_atm = tmp.iloc[(tmp["strike"] - spot).abs().argmin()]
    return float(row_atm["iv_mid"])


def implied_move_stats(
    spot: float, atm_iv_annual_pct: float, horizons_days: Sequence[int] = (1, 5)
) -> Dict[int, Dict[str, float]]:
    if atm_iv_annual_pct is None or math.isnan(atm_iv_annual_pct):
        return {}

    vol_annual = atm_iv_annual_pct / 100.0
    vol_daily = vol_annual / math.sqrt(252.0)

    out: Dict[int, Dict[str, float]] = {}
    for h in horizons_days:
        sigma_h = vol_daily * math.sqrt(float(h))
        move_1sigma_pts = spot * sigma_h
        move_2sigma_pts = spot * 2.0 * sigma_h
        out[int(h)] = {
            "sigma_ret": sigma_h,
            "move_1sigma_pct": sigma_h * 100.0,
            "move_1sigma_pts": move_1sigma_pts,
            "move_2sigma_pct": 2.0 * sigma_h * 100.0,
            "move_2sigma_pts": move_2sigma_pts,
            "p_gt_1sigma": 2.0 * (1.0 - _norm_cdf(1.0)),
            "p_gt_2sigma": 2.0 * (1.0 - _norm_cdf(2.0)),
        }
    return out


def compute_implied_distribution(
    chain_df: pd.DataFrame, spot: float, target_dte: int = 30
) -> Dict[str, Any]:
    ref_expiry = pick_reference_expiry(chain_df, target_dte=target_dte)
    if ref_expiry is None:
        return {}
    atm_iv = atm_iv_for_expiry(chain_df, ref_expiry, spot)
    if atm_iv is None:
        return {}
    moves = implied_move_stats(spot, atm_iv, horizons_days=(1, 5))
    dte = int(chain_df.loc[chain_df["expiry"] == ref_expiry, "dte"].iloc[0])
    return {"ref_expiry": ref_expiry, "dte": dte, "atm_iv": atm_iv, "moves": moves}


# ---------------------------------------------------------------------
# Vol surface snapshot (ATM + 25Δ RR & butterfly) with simple history
# ---------------------------------------------------------------------


def _find_delta_option(
    df: pd.DataFrame, opt_type: str, target_delta: float, band: float = 0.15
) -> Optional[float]:
    if "delta" not in df.columns or "iv" not in df.columns:
        return None
    if opt_type == "call":
        sub = df[
            (df["type"] == "call")
            & (df["delta"] >= target_delta - band)
            & (df["delta"] <= target_delta + band)
        ]
    else:
        sub = df[
            (df["type"] == "put")
            & (df["delta"] <= target_delta + band)
            & (df["delta"] >= target_delta - band)
        ]
    if sub.empty:
        return None
    row = sub.iloc[(sub["delta"] - target_delta).abs().argmin()]
    return float(row["iv"])


def vol_surface_snapshot(chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    df = chain_df.copy()
    df["expiry"] = pd.to_datetime(df["expiry"])
    out_rows: List[Dict[str, Any]] = []
    for expiry, g in df.groupby("expiry"):
        dte = int(g["dte"].iloc[0])
        atm = atm_iv_for_expiry(df, expiry, spot)
        if atm is None:
            continue
        call25_iv = _find_delta_option(g, "call", 0.25)
        put25_iv = _find_delta_option(g, "put", -0.25)
        if call25_iv is None or put25_iv is None:
            rr_25 = None
            bf_25 = None
        else:
            rr_25 = float(put25_iv - call25_iv)
            bf_25 = float(0.5 * (put25_iv + call25_iv) - atm)
        out_rows.append(
            {
                "expiry": expiry,
                "dte": dte,
                "atm_iv": atm,
                "call25_iv": call25_iv,
                "put25_iv": put25_iv,
                "rr_25": rr_25,
                "bf_25": bf_25,
            }
        )
    if not out_rows:
        return pd.DataFrame(
            columns=["expiry", "dte", "atm_iv", "call25_iv", "put25_iv", "rr_25", "bf_25"]
        )
    res = pd.DataFrame(out_rows).sort_values("dte").reset_index(drop=True)
    return res


def update_vol_surface_history(
    snapshot_df: pd.DataFrame, history_path: str = "vol_surface_history.parquet"
) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()
    today = pd.Timestamp.today().normalize()
    snap = snapshot_df.copy()
    snap["asof_date"] = today
    path = Path(history_path)
    if path.exists():
        try:
            hist = pd.read_parquet(path)
        except Exception:
            hist = pd.DataFrame()
    else:
        hist = pd.DataFrame()
    if not hist.empty:
        hist = hist[hist["asof_date"] != today]
        hist_all = pd.concat([hist, snap], ignore_index=True)
    else:
        hist_all = snap
    hist_all.to_parquet(path, index=False)
    return hist_all


def add_vol_surface_ranks(
    snapshot_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    lookback_days: int = 90,
    dte_tolerance: int = 7,
) -> pd.DataFrame:
    snap = snapshot_df.copy()
    snap["atm_iv_rank"] = np.nan
    snap["rr_25_rank"] = np.nan
    snap["bf_25_rank"] = np.nan

    if history_df is None or history_df.empty:
        return snap
    hist = history_df.copy()
    hist["asof_date"] = pd.to_datetime(hist["asof_date"]).dt.normalize()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=lookback_days)
    hist = hist[hist["asof_date"] >= cutoff]
    if hist.empty:
        return snap
    for idx, row in snap.iterrows():
        dte = row["dte"]
        mask = hist["dte"].between(dte - dte_tolerance, dte + dte_tolerance)
        h = hist[mask]
        if h.empty:
            continue
        for col, rank_col in [
            ("atm_iv", "atm_iv_rank"),
            ("rr_25", "rr_25_rank"),
            ("bf_25", "bf_25_rank"),
        ]:
            vals = h[col].dropna().values
            if vals.size == 0 or pd.isna(row[col]):
                continue
            rank = (vals < row[col]).mean()
            snap.at[idx, rank_col] = rank
    return snap


# ---------------------------------------------------------------------
# Narrative summary
# ---------------------------------------------------------------------


def build_narrative(
    spot: float,
    pcr_info: Dict[str, float],
    gamma_summary: Dict[str, Any],
    unusual_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    implied_dist: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"Spot ≈ {spot:.2f}.")
    if pcr_info.get("pcr_vol") is not None and not math.isnan(pcr_info["pcr_vol"]):
        lines.append(
            f" Intraday options flow put/call volume ratio ≈ {pcr_info['pcr_vol']:.2f}, "
            f"open-interest PCR ≈ {pcr_info['pcr_oi']:.2f}."
        )
    if gamma_summary.get("zero_gamma") is not None:
        zg = gamma_summary["zero_gamma"]
        lines.append(
            f" Approx zero-gamma level around {zg:.1f} with call-wall near "
            f"{gamma_summary['call_wall']:.1f} and put-wall near {gamma_summary['put_wall']:.1f}."
        )
    if implied_dist:
        mv1 = implied_dist["moves"][1]["move_1sigma_pts"]
        lines.append(
            f" Options-implied 1D 1σ move is about ±{mv1:.1f} points from spot."
        )
    if not unusual_df.empty:
        top_notional = unusual_df["dollar_notional"].nlargest(3).sum() / 1e6
        lines.append(
            f" Top flagged unusual lines sum to roughly ${top_notional:.1f}M notional."
        )
    if not clusters_df.empty:
        big_cluster = clusters_df.iloc[0]
        lines.append(
            f" Largest notional cluster sits at expiry {big_cluster['expiry'].date()} "
            f"around moneyness band {big_cluster['moneyness_band']}%."
        )
    return "".join(lines)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


def run_full_analysis(
    ticker: str,
    price_period: str = "6mo",
    price_interval: str = "1d",
    expiries: Optional[Sequence[str]] = None,
    max_expiries: int = 3,
    rate: float = 0.04,
    dividend_yield: float = 0.012,
    unusual_config: Optional[UnusualConfig] = None,
) -> Dict[str, Any]:
    """
    High-level orchestration: fetch price + options, compute all metrics.
    Returns a dict to be consumed by the Streamlit app.
    """
    if unusual_config is None:
        unusual_config = UnusualConfig()

    # price
    price_df = get_price_history(ticker, period=price_period, interval=price_interval)
    spot = float(price_df["Close"].iloc[-1])

    # options chain
    chain_df = get_options_chain(
        ticker=ticker,
        expiries=expiries,
        max_expiries=max_expiries,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    # pcr
    pcr_info = compute_put_call_ratios(chain_df)

    # gamma
    gex_table = compute_gex(chain_df, spot)
    gamma_info = summarize_gamma(gex_table)

    # unusual
    unusual_df = flag_unusual_activity(chain_df, unusual_config)

    # clusters
    clusters_df = build_notional_clusters(chain_df, spot)

    # implied distribution
    implied_dist = compute_implied_distribution(chain_df, spot)

    # vol surface + history
    vol_surf_today = vol_surface_snapshot(chain_df, spot)
    vol_hist = update_vol_surface_history(vol_surf_today)
    vol_surf_today = add_vol_surface_ranks(vol_surf_today, vol_hist)

    # narrative
    narrative = build_narrative(
        spot=spot,
        pcr_info=pcr_info,
        gamma_summary=gamma_info,
        unusual_df=unusual_df,
        clusters_df=clusters_df,
        implied_dist=implied_dist,
    )

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
        # new advanced blocks (you can wire these into the app UI later)
        "implied_dist": implied_dist,
        "vol_surface": vol_surf_today,
        "vol_surface_history": vol_hist,
    }
