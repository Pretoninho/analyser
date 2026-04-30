"""
engine/orderbook_imbalance.py

Order-book microstructure features for backtest and live research.

This module implements reusable, vectorized features from LOB snapshots and
trade prints:
- normalized Order Book Imbalance (OBI)
- OBI velocity / acceleration
- naive midprice and microprice
- microprice divergence
- cumulative liquidity profiles (single snapshot helper)
- liquidity void scores / flags
- trade imbalance from prints

Expected LOB dataframe schema (level-based):
    bid_px_1, bid_sz_1, ask_px_1, ask_sz_1,
    bid_px_2, bid_sz_2, ask_px_2, ask_sz_2, ...
Optional:
    ts (epoch seconds) or timestamp (datetime-like)

Expected trades dataframe schema (prints):
    side in {"buy","sell"} (or custom mapping), qty, optional ts/timestamp
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass(frozen=True)
class LOBColumns:
    bid_px: list[str]
    bid_sz: list[str]
    ask_px: list[str]
    ask_sz: list[str]


def _sorted_level_columns(columns: Iterable[str], prefix: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    found: list[tuple[int, str]] = []
    for c in columns:
        m = pat.match(c)
        if m:
            found.append((int(m.group(1)), c))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found]


def detect_lob_columns(df: pd.DataFrame) -> LOBColumns:
    """Detect level-based LOB columns from a dataframe."""
    bid_px = _sorted_level_columns(df.columns, "bid_px_")
    bid_sz = _sorted_level_columns(df.columns, "bid_sz_")
    ask_px = _sorted_level_columns(df.columns, "ask_px_")
    ask_sz = _sorted_level_columns(df.columns, "ask_sz_")

    n = min(len(bid_px), len(bid_sz), len(ask_px), len(ask_sz))
    if n == 0:
        raise ValueError("No LOB columns detected. Expected bid_px_i/bid_sz_i/ask_px_i/ask_sz_i.")

    return LOBColumns(
        bid_px=bid_px[:n],
        bid_sz=bid_sz[:n],
        ask_px=ask_px[:n],
        ask_sz=ask_sz[:n],
    )


def _infer_dt_seconds(df: pd.DataFrame) -> np.ndarray:
    if "ts" in df.columns:
        t = pd.to_numeric(df["ts"], errors="coerce").values.astype("float64")
    elif "timestamp" in df.columns:
        ts_ns = pd.to_datetime(df["timestamp"], utc=True).astype("int64")
        t = ts_ns.to_numpy(dtype="float64") / 1e9
    else:
        return np.ones(len(df), dtype="float64")

    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1.0
    return dt


def _distance_weights(
    bid_px: np.ndarray,
    ask_px: np.ndarray,
    mid: np.ndarray,
    tick_size: float,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    if decay <= 0:
        ones_b = np.ones_like(bid_px, dtype="float64")
        ones_a = np.ones_like(ask_px, dtype="float64")
        return ones_b, ones_a

    bid_dist_ticks = np.abs(mid[:, None] - bid_px) / max(tick_size, _EPS)
    ask_dist_ticks = np.abs(ask_px - mid[:, None]) / max(tick_size, _EPS)

    w_bid = np.exp(-decay * bid_dist_ticks)
    w_ask = np.exp(-decay * ask_dist_ticks)
    return w_bid, w_ask


def cumulative_profile_snapshot(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid: float | None = None,
    decay: float = 0.0,
    tick_size: float = 1.0,
) -> dict[str, np.ndarray | float]:
    """
    Build cumulative liquidity profile for one snapshot.

    Returns cumulative volumes from the inside outward for both sides and
    normalized OBI at each depth level.
    """
    bid_px = np.asarray(bid_px, dtype="float64")
    bid_sz = np.asarray(bid_sz, dtype="float64")
    ask_px = np.asarray(ask_px, dtype="float64")
    ask_sz = np.asarray(ask_sz, dtype="float64")

    n = min(len(bid_px), len(bid_sz), len(ask_px), len(ask_sz))
    bid_px, bid_sz, ask_px, ask_sz = bid_px[:n], bid_sz[:n], ask_px[:n], ask_sz[:n]

    if mid is None:
        mid = 0.5 * (bid_px[0] + ask_px[0])

    if decay > 0:
        w_bid = np.exp(-decay * np.abs(mid - bid_px) / max(tick_size, _EPS))
        w_ask = np.exp(-decay * np.abs(ask_px - mid) / max(tick_size, _EPS))
        wb = bid_sz * w_bid
        wa = ask_sz * w_ask
    else:
        wb, wa = bid_sz, ask_sz

    cum_bid = np.cumsum(wb)
    cum_ask = np.cumsum(wa)
    obi_lvl = (cum_bid - cum_ask) / (cum_bid + cum_ask + _EPS)

    return {
        "mid": float(mid),
        "cum_bid": cum_bid,
        "cum_ask": cum_ask,
        "obi_by_level": obi_lvl,
    }


def compute_obi_features_from_lob(
    df_lob: pd.DataFrame,
    depth: int | None = None,
    decay: float = 0.0,
    tick_size: float = 1.0,
    obi_threshold: float = 0.2,
    void_top_levels: int = 3,
    void_pct_threshold: float = 0.15,
) -> pd.DataFrame:
    """
    Compute vectorized OBI/microprice/liquidity-void features from LOB snapshots.

    Formulas:
        bid_cum = sum_k w_k * bid_sz_k
        ask_cum = sum_k w_k * ask_sz_k
        obi_raw = bid_cum - ask_cum
        obi_norm = obi_raw / (bid_cum + ask_cum)

        mid = (bid_px_1 + ask_px_1) / 2
        microprice = (bid_px_1 * ask_sz_1 + ask_px_1 * bid_sz_1) / (bid_sz_1 + ask_sz_1)

        obi_vel = d(obi_norm) / dt
        obi_acc = d(obi_vel) / dt

    Distance weighting (optional):
        w_k = exp(-decay * distance_in_ticks_from_mid_k)

    Liquidity-void score (up/down):
        void_up_score   = 1 / sum_{ask top K} weighted_size
        void_down_score = 1 / sum_{bid top K} weighted_size
    """
    cols = detect_lob_columns(df_lob)

    n_levels = len(cols.bid_px)
    use_depth = n_levels if depth is None else max(1, min(depth, n_levels))

    bid_px = df_lob[cols.bid_px[:use_depth]].to_numpy(dtype="float64")
    bid_sz = df_lob[cols.bid_sz[:use_depth]].to_numpy(dtype="float64")
    ask_px = df_lob[cols.ask_px[:use_depth]].to_numpy(dtype="float64")
    ask_sz = df_lob[cols.ask_sz[:use_depth]].to_numpy(dtype="float64")

    best_bid = bid_px[:, 0]
    best_ask = ask_px[:, 0]

    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid

    w_bid, w_ask = _distance_weights(
        bid_px=bid_px,
        ask_px=ask_px,
        mid=mid,
        tick_size=tick_size,
        decay=decay,
    )

    wb = bid_sz * w_bid
    wa = ask_sz * w_ask

    bid_cum = wb.sum(axis=1)
    ask_cum = wa.sum(axis=1)

    obi_raw = bid_cum - ask_cum
    obi_norm = obi_raw / (bid_cum + ask_cum + _EPS)

    # OBI dynamics
    dt = _infer_dt_seconds(df_lob)
    obi_vel = np.diff(obi_norm, prepend=obi_norm[0]) / dt
    obi_acc = np.diff(obi_vel, prepend=obi_vel[0]) / dt

    # Microprice (top-of-book weighted midpoint)
    bid_sz1 = bid_sz[:, 0]
    ask_sz1 = ask_sz[:, 0]
    microprice = (best_bid * ask_sz1 + best_ask * bid_sz1) / (bid_sz1 + ask_sz1 + _EPS)
    microprice_delta = microprice - mid

    # Simple cumulative-profile slope proxy: local density at top K levels
    k = max(1, min(void_top_levels, use_depth))
    top_bid_liq = wb[:, :k].sum(axis=1)
    top_ask_liq = wa[:, :k].sum(axis=1)

    void_up_score = 1.0 / (top_ask_liq + _EPS)     # fragility for upward sweep
    void_down_score = 1.0 / (top_bid_liq + _EPS)   # fragility for downward sweep

    up_thr = np.quantile(void_up_score, 1.0 - void_pct_threshold)
    dn_thr = np.quantile(void_down_score, 1.0 - void_pct_threshold)

    void_up_flag = (void_up_score >= up_thr).astype("int8")
    void_down_flag = (void_down_score >= dn_thr).astype("int8")

    # Directional trigger from OBI + velocity
    obi_buy_pressure = ((obi_norm >= obi_threshold) & (obi_vel > 0)).astype("int8")
    obi_sell_pressure = ((obi_norm <= -obi_threshold) & (obi_vel < 0)).astype("int8")

    out = df_lob.copy()
    out["mid"] = mid.astype("float64")
    out["spread"] = spread.astype("float64")

    out["bid_cum_w"] = bid_cum.astype("float64")
    out["ask_cum_w"] = ask_cum.astype("float64")
    out["obi_raw"] = obi_raw.astype("float64")
    out["obi_norm"] = np.clip(obi_norm, -1.0, 1.0).astype("float64")

    out["obi_vel"] = obi_vel.astype("float64")
    out["obi_acc"] = obi_acc.astype("float64")

    out["microprice"] = microprice.astype("float64")
    out["microprice_delta"] = microprice_delta.astype("float64")
    out["microprice_divergence_bps"] = ((microprice_delta / (mid + _EPS)) * 1e4).astype("float64")

    out["void_up_score"] = void_up_score.astype("float64")
    out["void_down_score"] = void_down_score.astype("float64")
    out["void_up_flag"] = void_up_flag
    out["void_down_flag"] = void_down_flag

    out["obi_buy_pressure"] = obi_buy_pressure
    out["obi_sell_pressure"] = obi_sell_pressure

    return out


def compute_trade_imbalance(
    df_trades: pd.DataFrame,
    qty_col: str = "qty",
    side_col: str = "side",
    buy_values: tuple[str, ...] = ("buy", "B", "1", "true"),
) -> pd.DataFrame:
    """
    Compute trade-flow imbalance from prints.

    Normalized formula:
        tfi_norm = (buy_qty - sell_qty) / (buy_qty + sell_qty)

    The function works row-by-row when each row is one print.
    """
    out = df_trades.copy()

    side = out[side_col].astype(str)
    buy_mask = side.str.lower().isin({s.lower() for s in buy_values})

    qty = pd.to_numeric(out[qty_col], errors="coerce").fillna(0.0)
    buy_qty = np.where(buy_mask, qty, 0.0)
    sell_qty = np.where(~buy_mask, qty, 0.0)

    out["buy_qty"] = buy_qty.astype("float64")
    out["sell_qty"] = sell_qty.astype("float64")

    out["tfi_raw"] = (out["buy_qty"] - out["sell_qty"]).astype("float64")
    out["tfi_norm"] = (
        out["tfi_raw"] / (out["buy_qty"] + out["sell_qty"] + _EPS)
    ).astype("float64")

    return out


def merge_lob_trade_features_asof(
    lob_features: pd.DataFrame,
    trade_features: pd.DataFrame,
    ts_col_lob: str = "timestamp",
    ts_col_trade: str = "timestamp",
    tolerance: str = "1s",
) -> pd.DataFrame:
    """
    As-of merge trade imbalance onto LOB features for event-time backtests.
    """
    left = lob_features.copy()
    right = trade_features.copy()

    left[ts_col_lob] = pd.to_datetime(left[ts_col_lob], utc=True)
    right[ts_col_trade] = pd.to_datetime(right[ts_col_trade], utc=True)

    left = left.sort_values(ts_col_lob)
    right = right.sort_values(ts_col_trade)

    merged = pd.merge_asof(
        left,
        right[[ts_col_trade, "tfi_raw", "tfi_norm"]],
        left_on=ts_col_lob,
        right_on=ts_col_trade,
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    return merged
