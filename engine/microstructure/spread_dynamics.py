"""
engine/microstructure/spread_dynamics.py

Spread and microstructure dynamics utilities for backtest and execution analytics.

Implements reusable calculations for:
- quoted spread, relative spread (bps), spread-to-tick ratio
- midprice and microprice (queue-weighted)
- effective spread, realized spread, adverse-selection component
- depth-of-market cumulative and slope proxies
- tick-constrained regime classification
- queue fill probability proxy under FIFO dynamics
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
    found: list[tuple[int, str]] = []
    for c in columns:
        m = pat.match(c)
        if m:
            found.append((int(m.group(1)), c))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found]


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


def quoted_spread(best_bid: pd.Series, best_ask: pd.Series) -> pd.Series:
    """Quoted spread S_q = ask - bid."""
    b = pd.to_numeric(best_bid, errors="coerce").ffill().bfill()
    a = pd.to_numeric(best_ask, errors="coerce").ffill().bfill()
    return (a - b).astype("float64").rename("spread_quoted")


def mid_price(best_bid: pd.Series, best_ask: pd.Series) -> pd.Series:
    """Mid price m = (bid+ask)/2."""
    b = pd.to_numeric(best_bid, errors="coerce").ffill().bfill()
    a = pd.to_numeric(best_ask, errors="coerce").ffill().bfill()
    return (0.5 * (a + b)).astype("float64").rename("mid")


def relative_spread_bps(spread: pd.Series, mid: pd.Series) -> pd.Series:
    """Relative quoted spread in bps: spread / mid * 1e4."""
    s = pd.to_numeric(spread, errors="coerce").fillna(0.0)
    m = pd.to_numeric(mid, errors="coerce").ffill().bfill()
    return ((s / (m + _EPS)) * 1e4).astype("float64").rename("spread_rel_bps")


def microprice(best_bid: pd.Series, best_ask: pd.Series, bid_sz_1: pd.Series, ask_sz_1: pd.Series) -> pd.Series:
    """
    Queue-weighted microprice:
      micro = (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)

    This shifts fair value toward the side with stronger opposite queue pressure.
    """
    b = pd.to_numeric(best_bid, errors="coerce").ffill().bfill()
    a = pd.to_numeric(best_ask, errors="coerce").ffill().bfill()
    qb = pd.to_numeric(bid_sz_1, errors="coerce").fillna(0.0)
    qa = pd.to_numeric(ask_sz_1, errors="coerce").fillna(0.0)

    mp = (b * qa + a * qb) / (qb + qa + _EPS)
    return mp.astype("float64").rename("microprice")


def spread_to_tick_ratio(spread: pd.Series, tick_size: float) -> pd.Series:
    """Spread-to-tick ratio, key regime indicator."""
    t = max(float(tick_size), _EPS)
    s = pd.to_numeric(spread, errors="coerce").fillna(0.0)
    return (s / t).astype("float64").rename("spread_tick_ratio")


def tick_regime_label(spread_tick_ratio: pd.Series, near_locked_threshold: float = 1.2) -> pd.Series:
    """
    Regime label:
    - tick_constrained if spread/tick <= threshold
    - price_competitive otherwise
    """
    r = pd.to_numeric(spread_tick_ratio, errors="coerce").fillna(np.inf)
    out = np.where(r <= near_locked_threshold, "tick_constrained", "price_competitive")
    return pd.Series(out, index=spread_tick_ratio.index, name="tick_regime")


def effective_spread_bps(trade_price: pd.Series, mid_at_trade: pd.Series, direction: pd.Series) -> pd.Series:
    """
    Effective spread (bps):
      S_e = 2 * D * (P_k - M_t) / M_t * 1e4
    where D=+1 for buy-initiated, -1 for sell-initiated.
    """
    p = pd.to_numeric(trade_price, errors="coerce").ffill().bfill()
    m = pd.to_numeric(mid_at_trade, errors="coerce").ffill().bfill()
    d = pd.to_numeric(direction, errors="coerce").fillna(0.0).clip(-1, 1)

    s = 2.0 * d * (p - m) / (m + _EPS) * 1e4
    return s.astype("float64").rename("effective_spread_bps")


def realized_spread_bps(trade_price: pd.Series, future_mid: pd.Series, direction: pd.Series) -> pd.Series:
    """
    Realized spread (bps):
      S_r = 2 * D * (P_k - M_{t+tau}) / M_{t+tau} * 1e4
    """
    p = pd.to_numeric(trade_price, errors="coerce").ffill().bfill()
    mf = pd.to_numeric(future_mid, errors="coerce").ffill().bfill()
    d = pd.to_numeric(direction, errors="coerce").fillna(0.0).clip(-1, 1)

    s = 2.0 * d * (p - mf) / (mf + _EPS) * 1e4
    return s.astype("float64").rename("realized_spread_bps")


def adverse_selection_bps(effective_bps: pd.Series, realized_bps: pd.Series) -> pd.Series:
    """
    Adverse selection component from spread decomposition:
      effective = realized + adverse_selection
      adverse_selection = effective - realized
    """
    e = pd.to_numeric(effective_bps, errors="coerce").fillna(0.0)
    r = pd.to_numeric(realized_bps, errors="coerce").fillna(0.0)
    return (e - r).astype("float64").rename("adverse_selection_bps")


def dom_depth_features(df_lob: pd.DataFrame, depth_levels: int = 5, tick_size: float = 0.01) -> pd.DataFrame:
    """
    Depth-of-market features (3D surface proxies):
    - cumulative bid/ask depth at top-K
    - local depth slope proxy per price increment
    - imbalance from cumulative depth
    """
    cols = detect_lob_cols(df_lob)
    k = max(1, min(depth_levels, len(cols.bid_px)))

    out = df_lob.copy()

    bid_depth = out[cols.bid_sz[:k]].sum(axis=1).to_numpy(dtype="float64")
    ask_depth = out[cols.ask_sz[:k]].sum(axis=1).to_numpy(dtype="float64")

    px_bid_1 = out[cols.bid_px[0]].to_numpy(dtype="float64")
    px_bid_k = out[cols.bid_px[k - 1]].to_numpy(dtype="float64")
    px_ask_1 = out[cols.ask_px[0]].to_numpy(dtype="float64")
    px_ask_k = out[cols.ask_px[k - 1]].to_numpy(dtype="float64")

    bid_range_ticks = np.maximum((px_bid_1 - px_bid_k) / max(tick_size, _EPS), 1.0)
    ask_range_ticks = np.maximum((px_ask_k - px_ask_1) / max(tick_size, _EPS), 1.0)

    slope_bid = bid_depth / bid_range_ticks
    slope_ask = ask_depth / ask_range_ticks

    out["dom_bid_depth_k"] = bid_depth
    out["dom_ask_depth_k"] = ask_depth
    out["dom_slope_bid"] = slope_bid
    out["dom_slope_ask"] = slope_ask
    out["dom_depth_imbalance"] = (bid_depth - ask_depth) / (bid_depth + ask_depth + _EPS)

    return out


def queue_fill_probability(lambda_arrival: pd.Series, q_pos: pd.Series, cancel_rate: pd.Series, horizon_sec: float = 1.0) -> pd.Series:
    """
    FIFO fill-probability proxy using queue dynamics.

    Intuition:
    - higher arrival intensity and cancellation rate ahead increase fill chance
    - deeper queue position lowers fill chance

    Proxy formula:
      p_fill = 1 - exp( - ((lambda + theta) * horizon) / (1 + q_pos) )
    """
    lam = pd.to_numeric(lambda_arrival, errors="coerce").fillna(0.0).clip(lower=0.0)
    q = pd.to_numeric(q_pos, errors="coerce").fillna(0.0).clip(lower=0.0)
    th = pd.to_numeric(cancel_rate, errors="coerce").fillna(0.0).clip(lower=0.0)

    x = ((lam + th) * max(float(horizon_sec), _EPS)) / (1.0 + q)
    p = 1.0 - np.exp(-x)
    return pd.Series(np.clip(p, 0.0, 1.0), index=lam.index, name="p_fill_proxy")


def compute_spread_dynamics_features(
    df_lob: pd.DataFrame,
    tick_size: float = 0.01,
    depth_levels: int = 5,
) -> pd.DataFrame:
    """
    End-to-end spread dynamics feature pipeline from LOB snapshots.

    Adds:
    - spread_quoted, mid, spread_rel_bps
    - microprice, microprice_delta_bps
    - spread_tick_ratio, tick_regime
    - DOM depth/slope/imbalance features
    """
    cols = detect_lob_cols(df_lob)
    out = df_lob.copy()

    b1 = out[cols.bid_px[0]]
    a1 = out[cols.ask_px[0]]

    out["spread_quoted"] = quoted_spread(b1, a1)
    out["mid"] = mid_price(b1, a1)
    out["spread_rel_bps"] = relative_spread_bps(out["spread_quoted"], out["mid"])

    out["microprice"] = microprice(
        best_bid=b1,
        best_ask=a1,
        bid_sz_1=out[cols.bid_sz[0]],
        ask_sz_1=out[cols.ask_sz[0]],
    )
    out["microprice_delta_bps"] = ((out["microprice"] - out["mid"]) / (out["mid"] + _EPS) * 1e4).astype("float64")

    out["spread_tick_ratio"] = spread_to_tick_ratio(out["spread_quoted"], tick_size=tick_size)
    out["tick_regime"] = tick_regime_label(out["spread_tick_ratio"])

    out = dom_depth_features(out, depth_levels=depth_levels, tick_size=tick_size)

    return out


def compute_trade_spread_tca(
    trade_price: pd.Series,
    direction: pd.Series,
    mid_at_trade: pd.Series,
    future_mid_tau: pd.Series,
) -> pd.DataFrame:
    """
    Compute trade-level TCA spread decomposition.

    Output columns:
    - effective_spread_bps
    - realized_spread_bps
    - adverse_selection_bps
    """
    eff = effective_spread_bps(trade_price, mid_at_trade, direction)
    rea = realized_spread_bps(trade_price, future_mid_tau, direction)
    adv = adverse_selection_bps(eff, rea)

    return pd.DataFrame(
        {
            "effective_spread_bps": eff,
            "realized_spread_bps": rea,
            "adverse_selection_bps": adv,
        }
    )
