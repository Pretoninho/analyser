"""
engine/microstructure/stop_cascade.py

Stop-loss cascade and robust stop execution utilities.

This module provides reusable, backtest-friendly functions to model:
- stop-market conversion slippage (VWAP sweep)
- quote fading and liquidity-vacuum detection
- convex impact during depleted-book regimes
- Hawkes-like cascade pressure from clustered triggers
- robust dynamic stop execution plan (delay + dynamic limit + slicing)

Expected LOB columns (single snapshot / row-wise):
    bid_px_1..N, bid_sz_1..N, ask_px_1..N, ask_sz_1..N

Optional time columns:
    ts or timestamp
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass(frozen=True)
class LOBCols:
    bid_px: list[str]
    bid_sz: list[str]
    ask_px: list[str]
    ask_sz: list[str]


def _sorted_level_cols(columns: Iterable[str], prefix: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    out: list[tuple[int, str]] = []
    for c in columns:
        m = pat.match(c)
        if m:
            out.append((int(m.group(1)), c))
    out.sort(key=lambda x: x[0])
    return [c for _, c in out]


def detect_lob_cols(df: pd.DataFrame) -> LOBCols:
    bid_px = _sorted_level_cols(df.columns, "bid_px_")
    bid_sz = _sorted_level_cols(df.columns, "bid_sz_")
    ask_px = _sorted_level_cols(df.columns, "ask_px_")
    ask_sz = _sorted_level_cols(df.columns, "ask_sz_")

    n = min(len(bid_px), len(bid_sz), len(ask_px), len(ask_sz))
    if n == 0:
        raise ValueError("No LOB levels found. Expected bid_px_i/bid_sz_i/ask_px_i/ask_sz_i")

    return LOBCols(
        bid_px=bid_px[:n],
        bid_sz=bid_sz[:n],
        ask_px=ask_px[:n],
        ask_sz=ask_sz[:n],
    )


def _infer_dt(df: pd.DataFrame) -> np.ndarray:
    if "ts" in df.columns:
        t = pd.to_numeric(df["ts"], errors="coerce").fillna(method="ffill").fillna(0.0).to_numpy(dtype="float64")
    elif "timestamp" in df.columns:
        ts_ns = pd.to_datetime(df["timestamp"], utc=True).astype("int64")
        t = ts_ns.to_numpy(dtype="float64") / 1e9
    else:
        return np.ones(len(df), dtype="float64")

    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1.0
    return dt


def stop_market_vwap_sell(
    order_size: float,
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
) -> dict[str, float]:
    """
    Simulate a stop-market SELL sweeping bid levels.

    Returns:
        vwap_exec: executed VWAP
        filled_qty: quantity actually filled (<= order_size)
        slippage_abs: trigger - vwap (requires trigger externally)
        levels_used: number of levels consumed
    """
    q = max(float(order_size), 0.0)
    px = np.asarray(bid_prices, dtype="float64")
    sz = np.maximum(np.asarray(bid_sizes, dtype="float64"), 0.0)

    if q <= 0 or len(px) == 0:
        return {"vwap_exec": np.nan, "filled_qty": 0.0, "levels_used": 0.0}

    rem = q
    notional = 0.0
    filled = 0.0
    levels_used = 0

    for i in range(len(px)):
        take = min(rem, sz[i])
        if take > 0:
            notional += take * px[i]
            filled += take
            rem -= take
            levels_used = i + 1
        if rem <= _EPS:
            break

    vwap = notional / (filled + _EPS) if filled > 0 else np.nan
    return {
        "vwap_exec": float(vwap),
        "filled_qty": float(filled),
        "levels_used": float(levels_used),
    }


def stop_limit_fill_probability_sell(
    limit_price: float,
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    order_size: float,
) -> float:
    """
    Fraction of order that can be filled immediately by a stop-limit SELL.

    SELL stop-limit can hit bids >= limit_price.
    """
    px = np.asarray(bid_prices, dtype="float64")
    sz = np.maximum(np.asarray(bid_sizes, dtype="float64"), 0.0)
    q = max(float(order_size), 0.0)

    eligible = sz[px >= limit_price].sum()
    if q <= 0:
        return 0.0
    return float(np.clip(eligible / q, 0.0, 1.0))


def compute_quote_fading_features(
    df_lob: pd.DataFrame,
    depth_levels: int = 3,
    spread_z_window: int = 200,
) -> pd.DataFrame:
    """
    Compute quote-fading / liquidity-vacuum diagnostics.

    Features:
    - spread, spread_z
    - top_bid_depth, top_ask_depth
    - depth_drop ratios
    - vacuum scores and binary flags
    """
    cols = detect_lob_cols(df_lob)
    k = max(1, min(depth_levels, len(cols.bid_px)))

    out = df_lob.copy()

    best_bid = out[cols.bid_px[0]].to_numpy(dtype="float64")
    best_ask = out[cols.ask_px[0]].to_numpy(dtype="float64")
    spread = best_ask - best_bid

    bid_depth = out[cols.bid_sz[:k]].sum(axis=1).to_numpy(dtype="float64")
    ask_depth = out[cols.ask_sz[:k]].sum(axis=1).to_numpy(dtype="float64")

    bid_depth_ma = pd.Series(bid_depth).rolling(spread_z_window).mean().bfill().to_numpy(dtype="float64")
    ask_depth_ma = pd.Series(ask_depth).rolling(spread_z_window).mean().bfill().to_numpy(dtype="float64")

    depth_drop_bid = 1.0 - (bid_depth / (bid_depth_ma + _EPS))
    depth_drop_ask = 1.0 - (ask_depth / (ask_depth_ma + _EPS))

    spread_s = pd.Series(spread)
    spread_mu = spread_s.rolling(spread_z_window).mean().bfill()
    spread_sd = spread_s.rolling(spread_z_window).std().replace(0, np.nan).bfill().fillna(1.0)
    spread_z = ((spread_s - spread_mu) / spread_sd).to_numpy(dtype="float64")

    # Vacuum score: widened spread + vanished top-of-book depth
    vacuum_down_score = np.maximum(spread_z, 0.0) * np.maximum(depth_drop_bid, 0.0)
    vacuum_up_score = np.maximum(spread_z, 0.0) * np.maximum(depth_drop_ask, 0.0)

    thr_dn = np.quantile(vacuum_down_score, 0.85)
    thr_up = np.quantile(vacuum_up_score, 0.85)

    out["spread"] = spread
    out["spread_z"] = spread_z
    out["top_bid_depth"] = bid_depth
    out["top_ask_depth"] = ask_depth
    out["depth_drop_bid"] = depth_drop_bid
    out["depth_drop_ask"] = depth_drop_ask
    out["vacuum_down_score"] = vacuum_down_score
    out["vacuum_up_score"] = vacuum_up_score
    out["vacuum_down_flag"] = (vacuum_down_score >= thr_dn).astype("int8")
    out["vacuum_up_flag"] = (vacuum_up_score >= thr_up).astype("int8")

    return out


def convex_impact_proxy(
    order_size: pd.Series | np.ndarray,
    liquidity: pd.Series | np.ndarray,
    beta: float = 1.4,
    scale: float = 1.0,
) -> pd.Series:
    """
    Convex impact proxy:
        impact ~ scale * (order_size / liquidity)^beta
    """
    q = pd.to_numeric(order_size, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    liq = pd.to_numeric(liquidity, errors="coerce").fillna(0.0).to_numpy(dtype="float64")

    ratio = q / (liq + _EPS)
    impact = scale * np.power(np.maximum(ratio, 0.0), beta)
    return pd.Series(impact, name="convex_impact", dtype="float64")


def hawkes_cascade_intensity(
    trigger_events: pd.Series | np.ndarray,
    mu0: float = 0.05,
    alpha: float = 0.7,
    beta_decay: float = 0.6,
    dt: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    """
    Hawkes-like intensity for stop-trigger contagion.

    lambda_t = mu0 + exp(-beta_decay*dt_t)*(lambda_{t-1}-mu0) + alpha*trigger_{t-1}
    """
    trig = pd.to_numeric(trigger_events, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    n = len(trig)

    if dt is None:
        dt_arr = np.ones(n, dtype="float64")
    else:
        if isinstance(dt, np.ndarray):
            dt_arr = pd.to_numeric(pd.Series(dt), errors="coerce").fillna(1.0).to_numpy(dtype="float64")
        else:
            dt_arr = pd.to_numeric(dt, errors="coerce").fillna(1.0).to_numpy(dtype="float64")
        dt_arr = np.array(dt_arr, dtype="float64", copy=True)
        dt_arr[dt_arr <= 0] = 1.0

    out = np.empty(n, dtype="float64")
    lam = mu0

    for i in range(n):
        decay = np.exp(-beta_decay * dt_arr[i])
        lam = mu0 + decay * (lam - mu0)
        if i > 0 and trig[i - 1] > 0:
            lam += alpha * trig[i - 1]
        out[i] = lam

    return pd.Series(out, name="cascade_lambda", dtype="float64")


def robust_stop_plan(
    trigger_price: float,
    current_mid: float,
    sigma_inst: float,
    side: str = "sell",
    tau_microseconds: int = 250,
    alpha_limit: float = 2.0,
    parent_qty: float = 1.0,
    child_qty: float = 0.2,
    spread_z: float = 0.0,
    max_spread_z_hold: float = 2.0,
) -> dict[str, float | str | int]:
    """
    Build a robust stop execution plan (institutional style).

    Logic:
    1) Delay/gate if spread regime indicates transient vacuum.
    2) Replace stop-market with dynamic stop-limit band.
    3) Slice parent into child orders.
    """
    s = side.lower().strip()
    if s not in {"sell", "buy"}:
        raise ValueError("side must be 'sell' or 'buy'")

    hold = int(spread_z >= max_spread_z_hold)

    if s == "sell":
        # dynamic lower band
        limit_price = current_mid - alpha_limit * sigma_inst * current_mid
        trigger_hit = int(current_mid <= trigger_price)
    else:
        # dynamic upper band
        limit_price = current_mid + alpha_limit * sigma_inst * current_mid
        trigger_hit = int(current_mid >= trigger_price)

    n_child = int(np.ceil(max(parent_qty, 0.0) / max(child_qty, _EPS)))

    mode = "hold" if hold and trigger_hit else "route"
    order_type = "stop_limit_sliced" if trigger_hit else "idle"

    return {
        "mode": mode,
        "trigger_hit": trigger_hit,
        "order_type": order_type,
        "tau_us": int(tau_microseconds),
        "dynamic_limit_price": float(limit_price),
        "parent_qty": float(parent_qty),
        "child_qty": float(child_qty),
        "n_child": int(n_child),
    }


def compute_stop_cascade_features(
    df_lob: pd.DataFrame,
    stop_trigger_col: str = "stop_trigger_event",
    est_stop_order_size_col: str = "est_stop_size",
    depth_levels: int = 3,
) -> pd.DataFrame:
    """
    End-to-end feature pipeline for stop-cascade risk.

    Output:
    - quote fading + vacuum features
    - cascade intensity
    - convex impact proxy
    - cascade risk score and regime flag
    """
    out = compute_quote_fading_features(df_lob=df_lob, depth_levels=depth_levels)

    if stop_trigger_col in out.columns:
        trig = pd.to_numeric(out[stop_trigger_col], errors="coerce").fillna(0.0)
    else:
        trig = out["vacuum_down_flag"].astype("float64")

    dt = _infer_dt(out)
    out["cascade_lambda"] = hawkes_cascade_intensity(trig, dt=dt)

    if est_stop_order_size_col in out.columns:
        size = pd.to_numeric(out[est_stop_order_size_col], errors="coerce").fillna(0.0)
    else:
        size = pd.Series(np.ones(len(out), dtype="float64"), index=out.index)

    out["convex_impact"] = convex_impact_proxy(
        order_size=size,
        liquidity=out["top_bid_depth"],
        beta=1.4,
        scale=1.0,
    )

    out["cascade_risk_score"] = (
        0.45 * out["vacuum_down_score"]
        + 0.35 * out["cascade_lambda"]
        + 0.20 * out["convex_impact"]
    )

    thr = out["cascade_risk_score"].quantile(0.85)
    out["cascade_risk_flag"] = (out["cascade_risk_score"] >= thr).astype("int8")

    return out
