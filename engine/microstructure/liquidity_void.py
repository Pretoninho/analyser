"""
engine/microstructure/liquidity_void.py

Liquidity-void and extreme microvolatility features.

This module provides reusable diagnostics for:
- liquidity-void severity and persistence
- cancellation-to-arrival stress and replenishment speed
- sweep impact and phantom-book style depletion
- extreme-tail risk proxies via POT/Hill estimators
- protective quoting signals during spread blowout regimes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spread_dynamics import detect_lob_cols, mid_price, quoted_spread


_EPS = 1e-12


def _safe_numeric(x: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    return pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype="float64")


def _infer_dt_seconds(df: pd.DataFrame) -> np.ndarray:
    if "ts" in df.columns:
        t = pd.to_numeric(df["ts"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype="float64")
    elif "timestamp" in df.columns:
        ts_ns = pd.to_datetime(df["timestamp"], utc=True).astype("int64")
        t = ts_ns.to_numpy(dtype="float64") / 1e9
    else:
        return np.ones(len(df), dtype="float64")

    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1.0
    return dt


def cancellation_arrival_ratio(
    cancel_volume: pd.Series | np.ndarray,
    new_limit_volume: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Stress ratio of disappearing liquidity versus replenishing liquidity.
    """
    cancel = np.maximum(_safe_numeric(cancel_volume), 0.0)
    new = np.maximum(_safe_numeric(new_limit_volume), 0.0)
    ratio = cancel / (new + _EPS)
    return pd.Series(ratio, name="cancel_arrival_ratio", dtype="float64")


def replenishment_rate(
    new_limit_volume: pd.Series | np.ndarray,
    dt_seconds: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Local replenishment intensity of new passive liquidity.
    """
    new = np.maximum(_safe_numeric(new_limit_volume), 0.0)
    dt = np.maximum(_safe_numeric(dt_seconds), _EPS)
    rate = new / dt
    return pd.Series(rate, name="replenishment_rate", dtype="float64")


def market_impact_coefficient(
    swept_volume: pd.Series | np.ndarray,
    price_displacement_ticks: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Instantaneous impact coefficient: price displacement per swept volume.
    """
    vol = np.maximum(_safe_numeric(swept_volume), 0.0)
    disp = np.maximum(_safe_numeric(price_displacement_ticks), 0.0)
    coeff = disp / (vol + _EPS)
    return pd.Series(coeff, name="impact_coeff", dtype="float64")


def top_level_depletion_ratio(
    current_depth: pd.Series | np.ndarray,
    baseline_depth: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Fractional depletion of local displayed depth.
    """
    cur = np.maximum(_safe_numeric(current_depth), 0.0)
    base = np.maximum(_safe_numeric(baseline_depth), 0.0)
    dep = 1.0 - cur / (base + _EPS)
    return pd.Series(np.clip(dep, 0.0, 1.0), name="depth_depletion_ratio", dtype="float64")


def void_duration_clock(
    void_flag: pd.Series | np.ndarray,
    dt_seconds: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Running duration spent inside a detected liquidity void.
    """
    flag = (_safe_numeric(void_flag) > 0).astype("int8")
    dt = np.maximum(_safe_numeric(dt_seconds), _EPS)
    out = np.zeros(len(flag), dtype="float64")
    acc = 0.0

    for i in range(len(flag)):
        if flag[i] == 1:
            acc += dt[i]
        else:
            acc = 0.0
        out[i] = acc

    return pd.Series(out, name="void_duration_sec", dtype="float64")


def liquidity_void_score(
    spread_z: pd.Series | np.ndarray,
    depth_depletion: pd.Series | np.ndarray,
    cancel_arrival: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Composite void severity: wider spread + depleted depth + cancellation stress.
    """
    sz = np.maximum(_safe_numeric(spread_z), 0.0)
    dep = np.clip(_safe_numeric(depth_depletion), 0.0, 1.0)
    car = np.maximum(_safe_numeric(cancel_arrival), 0.0)
    score = sz * dep * np.log1p(car)
    return pd.Series(score, name="liquidity_void_score", dtype="float64")


def rolling_hill_tail_index(
    losses: pd.Series | np.ndarray,
    window: int = 250,
    tail_fraction: float = 0.10,
) -> pd.Series:
    """
    Hill estimator proxy for right-tail heaviness.

    Larger values imply heavier tails.
    """
    x = np.maximum(_safe_numeric(losses), 0.0)
    n = len(x)
    out = np.full(n, np.nan, dtype="float64")
    frac = min(max(float(tail_fraction), 0.01), 0.50)

    for i in range(window - 1, n):
        w = x[i - window + 1 : i + 1]
        w = w[np.isfinite(w)]
        if len(w) < 10:
            continue
        w = np.sort(w)
        k = max(int(len(w) * frac), 1)
        tail = w[-k:]
        xk = max(w[-k], _EPS)
        logs = np.log(np.maximum(tail, xk) / xk)
        hill = np.mean(logs)
        out[i] = max(hill, 0.0)

    ser = pd.Series(out).bfill().fillna(0.0)
    ser.name = "tail_index_hill"
    return ser.astype("float64")


def pot_exceedance_features(
    losses: pd.Series | np.ndarray,
    window: int = 250,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """
    Peaks-over-threshold features from rolling loss distribution.

    Uses empirical exceedances rather than full MLE-GPD fitting for robustness.
    """
    x = np.maximum(_safe_numeric(losses), 0.0)
    s = pd.Series(x, dtype="float64")
    u = s.rolling(window, min_periods=max(20, window // 5)).quantile(quantile).bfill()
    exceed = np.maximum(x - u.to_numpy(dtype="float64"), 0.0)
    p_exceed = (pd.Series((exceed > 0).astype("float64")).rolling(window, min_periods=max(20, window // 5)).mean().bfill())

    exceed_series = pd.Series(np.where(exceed > 0, exceed, np.nan), dtype="float64")
    mean_exc = exceed_series.rolling(window, min_periods=max(20, window // 5)).mean().bfill().fillna(0.0)
    es = s.where(s >= u, np.nan).rolling(window, min_periods=max(20, window // 5)).mean().bfill()

    return pd.DataFrame(
        {
            "pot_threshold": u.astype("float64"),
            "pot_exceedance": exceed.astype("float64"),
            "pot_exceedance_prob": p_exceed.astype("float64"),
            "pot_mean_excess": mean_exc.astype("float64"),
            "pot_expected_shortfall": es.fillna(0.0).astype("float64"),
        }
    )


def protective_quote_signal(
    void_score: pd.Series | np.ndarray,
    tail_index: pd.Series | np.ndarray,
    impact_coeff: pd.Series | np.ndarray,
    widen_threshold: float = 0.75,
    pause_threshold: float = 1.25,
) -> pd.Series:
    """
    Three-state quoting regime:
    - normal
    - widen
    - pause
    """
    vs = np.maximum(_safe_numeric(void_score), 0.0)
    ti = np.maximum(_safe_numeric(tail_index), 0.0)
    ic = np.maximum(_safe_numeric(impact_coeff), 0.0)
    risk = vs + 0.5 * ti + 2.0 * ic

    label = np.where(risk >= pause_threshold, "pause", np.where(risk >= widen_threshold, "widen", "normal"))
    return pd.Series(label, name="protective_quote_regime")


def compute_liquidity_void_features(
    df_lob: pd.DataFrame,
    tick_size: float = 0.01,
    depth_levels: int = 5,
    spread_z_window: int = 200,
    evt_window: int = 250,
    evt_quantile: float = 0.95,
) -> pd.DataFrame:
    """
    End-to-end liquidity-void feature pipeline.

    Expected optional event-flow columns:
    - cancel_vol
    - new_limit_vol
    - sweep_vol
    """
    out = df_lob.copy()
    cols = detect_lob_cols(out)
    k = max(1, min(depth_levels, len(cols.bid_px)))

    dt = _infer_dt_seconds(out)
    out["dt_seconds"] = dt

    best_bid = out[cols.bid_px[0]]
    best_ask = out[cols.ask_px[0]]
    out["mid"] = mid_price(best_bid, best_ask)
    out["spread"] = quoted_spread(best_bid, best_ask)

    spread_s = pd.Series(out["spread"], dtype="float64")
    spread_mu = spread_s.rolling(spread_z_window, min_periods=max(20, spread_z_window // 5)).mean().bfill()
    spread_sd = spread_s.rolling(spread_z_window, min_periods=max(20, spread_z_window // 5)).std().replace(0.0, np.nan).bfill().fillna(1.0)
    out["spread_z"] = ((spread_s - spread_mu) / spread_sd).to_numpy(dtype="float64")

    top_bid_depth = out[cols.bid_sz[:k]].sum(axis=1).to_numpy(dtype="float64")
    top_ask_depth = out[cols.ask_sz[:k]].sum(axis=1).to_numpy(dtype="float64")
    out["top_bid_depth"] = top_bid_depth
    out["top_ask_depth"] = top_ask_depth
    out["top_total_depth"] = top_bid_depth + top_ask_depth

    baseline_depth = pd.Series(out["top_total_depth"], dtype="float64").rolling(spread_z_window, min_periods=max(20, spread_z_window // 5)).mean().bfill()
    out["depth_depletion_ratio"] = top_level_depletion_ratio(out["top_total_depth"], baseline_depth)

    cancel_vol = out["cancel_vol"] if "cancel_vol" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    new_limit_vol = out["new_limit_vol"] if "new_limit_vol" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
    sweep_vol = out["sweep_vol"] if "sweep_vol" in out.columns else pd.Series(np.zeros(len(out)), index=out.index)

    out["cancel_arrival_ratio"] = cancellation_arrival_ratio(cancel_vol, new_limit_vol)
    out["replenishment_rate"] = replenishment_rate(new_limit_vol, dt)

    price_jump_ticks = np.abs(pd.Series(out["mid"], dtype="float64").diff().fillna(0.0).to_numpy(dtype="float64")) / max(float(tick_size), _EPS)
    out["impact_coeff"] = market_impact_coefficient(sweep_vol, price_jump_ticks)

    out["liquidity_void_score"] = liquidity_void_score(
        out["spread_z"],
        out["depth_depletion_ratio"],
        out["cancel_arrival_ratio"],
    )

    score_arr = pd.to_numeric(out["liquidity_void_score"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    positive_scores = score_arr[score_arr > 0.0]
    thr = float(np.quantile(positive_scores, 0.85)) if len(positive_scores) else np.inf
    out["void_flag"] = ((score_arr > 0.0) & (score_arr >= thr)).astype("int8")
    out["void_duration_sec"] = void_duration_clock(out["void_flag"], dt)

    micro_ret_bps = np.abs(pd.Series(out["mid"], dtype="float64").pct_change().fillna(0.0).to_numpy(dtype="float64")) * 1e4
    out["micro_return_abs_bps"] = micro_ret_bps
    out["tail_index_hill"] = rolling_hill_tail_index(micro_ret_bps, window=evt_window, tail_fraction=0.10)
    pot = pot_exceedance_features(micro_ret_bps, window=evt_window, quantile=evt_quantile)
    out = pd.concat([out, pot], axis=1)

    out["protective_quote_regime"] = protective_quote_signal(
        out["liquidity_void_score"],
        out["tail_index_hill"],
        out["impact_coeff"],
    )
    out["quote_widen_flag"] = (out["protective_quote_regime"] == "widen").astype("int8")
    out["quote_pause_flag"] = (out["protective_quote_regime"] == "pause").astype("int8")

    return out