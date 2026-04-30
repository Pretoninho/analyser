"""
engine/microstructure/aggressive_execution.py

Aggressive execution and market-order impact utilities.

Implements reusable functions for:
- limit-order-book traversal under a continuous double auction
- market-order sweep simulation and VWAP execution price
- cumulative depth and clearing-price lookup
- implementation shortfall and slippage diagnostics
- square-root impact and Almgren-Chriss style execution-cost decomposition
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spread_dynamics import detect_lob_cols, mid_price


_EPS = 1e-12


def _safe_numeric(x: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    return pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype="float64")


def _validate_side(side: str) -> str:
    s = side.lower().strip()
    if s not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    return s


def cumulative_depth_curve(
    prices: pd.Series | np.ndarray,
    sizes: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Cumulative depth profile across sequential price levels.
    """
    px = _safe_numeric(prices)
    sz = np.maximum(_safe_numeric(sizes), 0.0)
    cum = np.cumsum(sz)
    return pd.DataFrame({"price": px, "size": sz, "cum_size": cum})


def clearing_price_for_quantity(
    prices: pd.Series | np.ndarray,
    sizes: pd.Series | np.ndarray,
    quantity: float,
) -> float:
    """
    Smallest price level whose cumulative depth fills the requested quantity.
    """
    q = max(float(quantity), 0.0)
    curve = cumulative_depth_curve(prices, sizes)
    if q <= 0 or curve.empty:
        return float("nan")

    hit = curve.loc[curve["cum_size"] >= q, "price"]
    if hit.empty:
        return float(curve["price"].iloc[-1])
    return float(hit.iloc[0])


def simulate_market_sweep(
    side: str,
    quantity: float,
    bid_prices: pd.Series | np.ndarray,
    bid_sizes: pd.Series | np.ndarray,
    ask_prices: pd.Series | np.ndarray,
    ask_sizes: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Simulate a deterministic market-order sweep through the book.

    For a buy market order, consume asks from best to worst.
    For a sell market order, consume bids from best to worst.
    """
    s = _validate_side(side)
    q = max(float(quantity), 0.0)

    if s == "buy":
        px = _safe_numeric(ask_prices)
        sz = np.maximum(_safe_numeric(ask_sizes), 0.0)
    else:
        px = _safe_numeric(bid_prices)
        sz = np.maximum(_safe_numeric(bid_sizes), 0.0)

    if q <= 0 or len(px) == 0:
        return {
            "filled_qty": 0.0,
            "vwap_exec": float("nan"),
            "levels_used": 0.0,
            "clearing_price": float("nan"),
            "avg_slippage_abs": 0.0,
        }

    rem = q
    notional = 0.0
    filled = 0.0
    levels_used = 0
    best_px = float(px[0])

    for i in range(len(px)):
        take = min(rem, sz[i])
        if take > 0:
            notional += take * px[i]
            filled += take
            rem -= take
            levels_used = i + 1
        if rem <= _EPS:
            break

    vwap = notional / (filled + _EPS) if filled > 0 else float("nan")
    clearing = float(px[min(max(levels_used - 1, 0), len(px) - 1)]) if filled > 0 else float("nan")

    if filled > 0 and np.isfinite(vwap):
        if s == "buy":
            slip = vwap - best_px
        else:
            slip = best_px - vwap
    else:
        slip = 0.0

    return {
        "filled_qty": float(filled),
        "vwap_exec": float(vwap),
        "levels_used": float(levels_used),
        "clearing_price": float(clearing),
        "avg_slippage_abs": float(max(slip, 0.0)),
    }


def implementation_shortfall_bps(
    exec_price: pd.Series | np.ndarray,
    decision_price: pd.Series | np.ndarray,
    side: str | pd.Series,
) -> pd.Series:
    """
    Implementation shortfall in basis points relative to decision price.
    Positive values are worse costs for the aggressor.
    """
    p_exec = _safe_numeric(exec_price)
    p_dec = _safe_numeric(decision_price)

    if isinstance(side, pd.Series):
        side_num = np.where(side.astype(str).str.lower().eq("buy"), 1.0, -1.0)
    else:
        side_num = np.full(len(p_exec), 1.0 if _validate_side(side) == "buy" else -1.0, dtype="float64")

    shortfall = side_num * (p_exec - p_dec) / (p_dec + _EPS) * 1e4
    return pd.Series(shortfall, name="implementation_shortfall_bps", dtype="float64")


def square_root_impact_bps(
    quantity: pd.Series | np.ndarray,
    daily_volume: pd.Series | np.ndarray,
    daily_volatility: pd.Series | np.ndarray,
    impact_scale: float = 1.0,
) -> pd.Series:
    """
    Square-root market impact law in bps:
        impact = Y * sigma * sqrt(Q / V) * 1e4
    """
    q = np.maximum(_safe_numeric(quantity), 0.0)
    v = np.maximum(_safe_numeric(daily_volume), _EPS)
    sigma = np.maximum(_safe_numeric(daily_volatility), 0.0)
    impact = float(impact_scale) * sigma * np.sqrt(q / v) * 1e4
    return pd.Series(impact, name="impact_sqrt_bps", dtype="float64")


def almgren_chriss_cost_bps(
    quantity: pd.Series | np.ndarray,
    horizon_seconds: pd.Series | np.ndarray,
    volatility: pd.Series | np.ndarray,
    liquidity: pd.Series | np.ndarray,
    eta_temp: float = 0.10,
    gamma_perm: float = 0.02,
) -> pd.DataFrame:
    """
    Simplified Almgren-Chriss style impact decomposition in bps.

    temporary impact grows with trading speed and inverse liquidity.
    permanent impact scales linearly with participation against liquidity.
    risk term proxies variance exposure from slower execution.
    """
    q = np.maximum(_safe_numeric(quantity), 0.0)
    h = np.maximum(_safe_numeric(horizon_seconds), _EPS)
    sigma = np.maximum(_safe_numeric(volatility), 0.0)
    liq = np.maximum(_safe_numeric(liquidity), _EPS)

    speed = q / h
    temporary = float(eta_temp) * np.power(speed / liq, 0.60) * 1e4
    permanent = float(gamma_perm) * (q / liq) * 1e4
    risk = sigma * np.sqrt(h) * 1e4
    total = temporary + permanent + risk

    return pd.DataFrame(
        {
            "impact_temp_bps": temporary,
            "impact_perm_bps": permanent,
            "impact_risk_bps": risk,
            "impact_total_bps": total,
        }
    )


def optimal_slicing_schedule(
    parent_qty: float,
    n_child: int,
    curve: str = "linear",
) -> np.ndarray:
    """
    Simple normalized execution schedule for child-order slicing.
    """
    n = max(int(n_child), 1)
    q = max(float(parent_qty), 0.0)

    if curve == "front_loaded":
        w = np.linspace(n, 1, n, dtype="float64")
    elif curve == "back_loaded":
        w = np.linspace(1, n, n, dtype="float64")
    else:
        w = np.ones(n, dtype="float64")

    w = w / (np.sum(w) + _EPS)
    return q * w


def compute_aggressive_execution_features(
    df_lob: pd.DataFrame,
    order_qty_col: str = "aggr_order_qty",
    side_col: str = "aggr_side",
    adv_col: str = "adv",
    sigma_daily_col: str = "sigma_daily",
    tick_size: float = 0.01,
) -> pd.DataFrame:
    """
    End-to-end aggressive execution feature pipeline from LOB snapshots.
    """
    out = df_lob.copy()
    cols = detect_lob_cols(out)

    if order_qty_col in out.columns:
        qty = np.maximum(_safe_numeric(out[order_qty_col]), 0.0)
    else:
        qty = np.ones(len(out), dtype="float64")

    if side_col in out.columns:
        side_series = out[side_col].astype(str).str.lower().fillna("buy")
    else:
        side_series = pd.Series(np.repeat("buy", len(out)), index=out.index)

    b1 = out[cols.bid_px[0]]
    a1 = out[cols.ask_px[0]]
    mid = mid_price(b1, a1)
    out["mid"] = mid

    vwap_exec = np.empty(len(out), dtype="float64")
    levels_used = np.empty(len(out), dtype="float64")
    clearing_px = np.empty(len(out), dtype="float64")
    slippage_abs = np.empty(len(out), dtype="float64")

    for i in range(len(out)):
        bid_px = out.loc[out.index[i], cols.bid_px].to_numpy(dtype="float64")
        bid_sz = out.loc[out.index[i], cols.bid_sz].to_numpy(dtype="float64")
        ask_px = out.loc[out.index[i], cols.ask_px].to_numpy(dtype="float64")
        ask_sz = out.loc[out.index[i], cols.ask_sz].to_numpy(dtype="float64")
        sim = simulate_market_sweep(
            side=side_series.iloc[i],
            quantity=float(qty[i]),
            bid_prices=bid_px,
            bid_sizes=bid_sz,
            ask_prices=ask_px,
            ask_sizes=ask_sz,
        )
        vwap_exec[i] = sim["vwap_exec"]
        levels_used[i] = sim["levels_used"]
        clearing_px[i] = sim["clearing_price"]
        slippage_abs[i] = sim["avg_slippage_abs"]

    out["exec_vwap"] = vwap_exec
    out["exec_levels_used"] = levels_used
    out["exec_clearing_price"] = clearing_px
    out["exec_slippage_abs"] = slippage_abs
    out["exec_slippage_bps"] = (slippage_abs / (out["mid"] + _EPS) * 1e4).astype("float64")

    out["implementation_shortfall_bps"] = implementation_shortfall_bps(
        exec_price=out["exec_vwap"],
        decision_price=out["mid"],
        side=side_series,
    )

    if adv_col in out.columns:
        adv = np.maximum(_safe_numeric(out[adv_col]), _EPS)
    else:
        depth_proxy = out[cols.bid_sz].sum(axis=1).to_numpy(dtype="float64") + out[cols.ask_sz].sum(axis=1).to_numpy(dtype="float64")
        adv = np.maximum(depth_proxy * 500.0, _EPS)

    if sigma_daily_col in out.columns:
        sigma_daily = np.maximum(_safe_numeric(out[sigma_daily_col]), 0.0)
    else:
        sigma_daily = np.full(len(out), 0.02, dtype="float64")

    out["impact_sqrt_bps"] = square_root_impact_bps(qty, adv, sigma_daily)

    liquidity = out[cols.bid_sz].sum(axis=1).to_numpy(dtype="float64") + out[cols.ask_sz].sum(axis=1).to_numpy(dtype="float64")
    ac = almgren_chriss_cost_bps(
        quantity=qty,
        horizon_seconds=np.full(len(out), 1.0, dtype="float64"),
        volatility=sigma_daily,
        liquidity=liquidity,
    )
    out = pd.concat([out, ac], axis=1)

    out["impact_gap_bps"] = out["implementation_shortfall_bps"] - out["impact_sqrt_bps"]
    out["taker_cost_flag"] = (
        (out["implementation_shortfall_bps"] > out["impact_sqrt_bps"]) | (out["impact_total_bps"] > out["impact_sqrt_bps"])
    ).astype("int8")

    return out