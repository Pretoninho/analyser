"""
engine/microstructure/queue_survival.py

Queue survival and passive-order value features.

This module turns queue-depth intuition into measurable features using:
- Survival analysis (hazard and survival functions)
- Cox-style proportional hazard adjustment
- Kaplan-Meier empirical survival curve
- Compound-Poisson fill probability (Gaussian approximation)
- Markov-modulated survival probability for volatile regime switches
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def normal_cdf(z: pd.Series | np.ndarray) -> np.ndarray:
    """
    Smooth approximation of standard normal CDF without scipy.
    """
    x = _safe_numeric(z)
    t = np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))
    return 0.5 * (1.0 + np.tanh(t))


def hazard_from_events(
    n_events: pd.Series | np.ndarray,
    n_at_risk: pd.Series | np.ndarray,
    dt_seconds: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Estimate baseline hazard from event counts.

    lambda(t) = events / (at_risk * dt)
    """
    e = np.maximum(_safe_numeric(n_events), 0.0)
    r = np.maximum(_safe_numeric(n_at_risk), 0.0)
    dt = np.maximum(_safe_numeric(dt_seconds), _EPS)
    lam = e / (r * dt + _EPS)
    return pd.Series(np.maximum(lam, 0.0), name="hazard_baseline", dtype="float64")


def cox_hazard(
    baseline_hazard: pd.Series | np.ndarray,
    covariates: pd.DataFrame,
    beta: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Cox proportional hazard:
        lambda(t|x) = lambda0(t) * exp(beta^T x)
    """
    lam0 = np.maximum(_safe_numeric(baseline_hazard), 0.0)
    x = covariates.astype("float64").to_numpy()
    b = _safe_numeric(beta)

    if x.shape[1] != b.shape[0]:
        raise ValueError("beta length must match number of covariate columns")

    lin = x @ b
    lam = lam0 * np.exp(np.clip(lin, -40.0, 40.0))
    return pd.Series(np.maximum(lam, 0.0), index=covariates.index, name="hazard_cox", dtype="float64")


def survival_from_hazard(
    hazard: pd.Series | np.ndarray,
    dt_seconds: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Convert hazard trajectory to survival trajectory:
        S(t) = exp(- integral lambda(u) du)
    """
    lam = np.maximum(_safe_numeric(hazard), 0.0)
    dt = np.maximum(_safe_numeric(dt_seconds), _EPS)
    cum_h = np.cumsum(lam * dt)
    s = np.exp(-cum_h)
    return pd.Series(np.clip(s, 0.0, 1.0), name="survival", dtype="float64")


def kaplan_meier_curve(
    durations: pd.Series | np.ndarray,
    event_observed: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Kaplan-Meier product-limit estimator.

    event_observed: 1 if terminal event observed, 0 if right-censored.
    """
    t = np.maximum(_safe_numeric(durations), 0.0)
    e = (_safe_numeric(event_observed) > 0).astype("int64")

    order = np.argsort(t)
    t = t[order]
    e = e[order]

    unique_t = np.unique(t)
    n_total = len(t)
    n_at_risk = n_total
    surv = 1.0

    rows: list[tuple[float, float, float, float]] = []
    idx_start = 0

    for ti in unique_t:
        idx_end = idx_start
        while idx_end < n_total and t[idx_end] == ti:
            idx_end += 1

        chunk_e = e[idx_start:idx_end]
        d_i = float(np.sum(chunk_e == 1))
        c_i = float(np.sum(chunk_e == 0))

        if n_at_risk > 0:
            surv *= 1.0 - d_i / n_at_risk

        rows.append((ti, float(n_at_risk), d_i, float(np.clip(surv, 0.0, 1.0))))
        n_at_risk -= int(d_i + c_i)
        idx_start = idx_end

    return pd.DataFrame(rows, columns=["time", "n_at_risk", "n_events", "survival_km"])


def compound_poisson_fill_probability_gaussian(
    volume_ahead: pd.Series | np.ndarray,
    order_size: float,
    horizon_seconds: float,
    arrival_intensity: pd.Series | np.ndarray,
    mean_order_size: pd.Series | np.ndarray,
    var_order_size: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Fill probability with Gaussian approximation of compound-Poisson flow.
    """
    v_a = np.maximum(_safe_numeric(volume_ahead), 0.0)
    lam = np.maximum(_safe_numeric(arrival_intensity), 0.0)
    mz = np.maximum(_safe_numeric(mean_order_size), 0.0)
    vz = np.maximum(_safe_numeric(var_order_size), 0.0)

    h = max(float(horizon_seconds), _EPS)
    threshold = v_a + max(float(order_size), 0.0)

    mu = lam * h * mz
    var = lam * h * (vz + mz * mz)
    std = np.sqrt(np.maximum(var, _EPS))

    z = (threshold - mu) / std
    p = 1.0 - normal_cdf(z)
    return pd.Series(np.clip(p, 0.0, 1.0), name="p_fill_cp_gauss", dtype="float64")


def ewma_compound_poisson_params(
    aggressive_volume: pd.Series | np.ndarray,
    dt_seconds: pd.Series | np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Recursive parameter tracking for compound-Poisson approximation.
    """
    vol = np.maximum(_safe_numeric(aggressive_volume), 0.0)
    dt = np.maximum(_safe_numeric(dt_seconds), _EPS)

    event = (vol > 0.0).astype("float64")
    inst_intensity = event / dt
    size_obs = np.where(event > 0.0, vol, np.nan)

    lam = pd.Series(inst_intensity).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype="float64")
    mean_size = pd.Series(size_obs).ffill().fillna(0.0).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype="float64")
    sq_dev = np.power(np.where(np.isnan(size_obs), mean_size, size_obs) - mean_size, 2.0)
    var_size = pd.Series(sq_dev).ewm(alpha=alpha, adjust=False).mean().fillna(0.0).to_numpy(dtype="float64")

    return pd.DataFrame(
        {
            "lambda_arrival": np.maximum(lam, 0.0),
            "mean_order_size": np.maximum(mean_size, 0.0),
            "var_order_size": np.maximum(var_size, 0.0),
        }
    )


def matrix_exponential_eig(a: np.ndarray, t: float) -> np.ndarray:
    """
    Matrix exponential using eigendecomposition.
    Suitable for small state spaces in execution models.
    """
    vals, vecs = np.linalg.eig(a)
    inv_vecs = np.linalg.inv(vecs)
    exp_diag = np.diag(np.exp(vals * float(t)))
    out = vecs @ exp_diag @ inv_vecs
    return np.real_if_close(out).astype("float64")


def markov_modulated_survival(
    horizon_seconds: float,
    generator_q: np.ndarray,
    hazard_by_state: pd.Series | np.ndarray,
    pi0: pd.Series | np.ndarray,
) -> float:
    """
    Survival probability under hidden Markov regime:
        S(t) = pi0^T exp((Q - diag(lambda)) t) 1
    """
    q = np.asarray(generator_q, dtype="float64")
    hz = np.maximum(_safe_numeric(hazard_by_state), 0.0)
    p0 = np.maximum(_safe_numeric(pi0), 0.0)

    if q.shape[0] != q.shape[1] or q.shape[0] != hz.shape[0] or p0.shape[0] != hz.shape[0]:
        raise ValueError("dimension mismatch between generator, hazard vector, and pi0")

    p0 = p0 / (np.sum(p0) + _EPS)
    a = q - np.diag(hz)
    expm_a = matrix_exponential_eig(a, float(horizon_seconds))
    s = float(p0 @ expm_a @ np.ones(hz.shape[0], dtype="float64"))
    return float(np.clip(s, 0.0, 1.0))


def expected_passive_value_bps(
    p_fill: pd.Series | np.ndarray,
    half_spread_bps: float,
    expected_impact_bps: pd.Series | np.ndarray,
    maker_rebate_bps: float = 0.1,
) -> pd.Series:
    """
    Passive EV decomposition:
        EV = p_fill * (half_spread + rebate - E[impact | fill])
    """
    pf = np.clip(_safe_numeric(p_fill), 0.0, 1.0)
    impact = np.maximum(_safe_numeric(expected_impact_bps), 0.0)
    edge = float(half_spread_bps) + float(maker_rebate_bps)
    ev = pf * (edge - impact)
    return pd.Series(ev, name="ev_passive_bps", dtype="float64")


def compute_queue_survival_features(
    df: pd.DataFrame,
    queue_ahead_col: str = "queue_ahead_vol",
    imbalance_col: str = "dom_depth_imbalance",
    sigma_col: str = "sigma_ewma",
    toxicity_col: str = "ofi_norm",
    aggressive_vol_col: str = "consume_ahead_vol",
    order_size: float = 1.0,
    horizon_seconds: float = 1.0,
    half_spread_bps: float = 0.5,
    maker_rebate_bps: float = 0.1,
    beta: tuple[float, float, float, float] = (-0.70, 0.90, 0.60, 0.70),
    cancel_ev_threshold_bps: float = 0.0,
    cancel_toxicity_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    End-to-end queue survival and passive EV feature pipeline.
    """
    out = df.copy()
    n = len(out)

    ahead = np.maximum(_safe_numeric(out[queue_ahead_col]), 0.0)
    imbalance = _safe_numeric(out[imbalance_col]) if imbalance_col in out.columns else np.zeros(n, dtype="float64")
    sigma = np.maximum(_safe_numeric(out[sigma_col]), 0.0) if sigma_col in out.columns else np.zeros(n, dtype="float64")
    toxicity = np.abs(_safe_numeric(out[toxicity_col])) if toxicity_col in out.columns else np.zeros(n, dtype="float64")
    aggr = np.maximum(_safe_numeric(out[aggressive_vol_col]), 0.0) if aggressive_vol_col in out.columns else np.zeros(n, dtype="float64")

    dt = _infer_dt_seconds(out)
    out["dt_seconds"] = dt

    # Baseline event intensity from observed aggressive flow.
    cp = ewma_compound_poisson_params(aggr, dt, alpha=0.05)
    out = pd.concat([out, cp], axis=1)

    n_events = (aggr > 0.0).astype("float64")
    n_at_risk = 1.0 + ahead
    out["hazard_baseline"] = hazard_from_events(n_events, n_at_risk, dt)

    # Cox hazard over microstructure covariates.
    cov = pd.DataFrame(
        {
            "x_ahead": -np.log1p(ahead),
            "x_imbalance": np.abs(imbalance),
            "x_sigma": sigma,
            "x_toxicity": toxicity,
        },
        index=out.index,
    )
    out["hazard_cox"] = cox_hazard(out["hazard_baseline"], cov, np.array(beta, dtype="float64"))

    # One-step survival and fill proxy from hazard.
    out["survival_exec_tau"] = np.exp(-np.maximum(_safe_numeric(out["hazard_cox"]), 0.0) * max(float(horizon_seconds), _EPS))
    out["p_fill_hazard"] = 1.0 - out["survival_exec_tau"]

    # Fill probability from compound-Poisson Gaussian approximation.
    out["p_fill_cp_gauss"] = compound_poisson_fill_probability_gaussian(
        volume_ahead=ahead,
        order_size=order_size,
        horizon_seconds=horizon_seconds,
        arrival_intensity=out["lambda_arrival"],
        mean_order_size=out["mean_order_size"],
        var_order_size=out["var_order_size"],
    )

    # Two-state Markov-modulated survival (quiescent vs toxic).
    q_gen = np.array([[-0.60, 0.60], [0.85, -0.85]], dtype="float64")
    depth_norm = ahead / (np.nanmedian(ahead) + 1.0)
    hz_q = 0.04 + 0.05 * np.abs(imbalance) + 0.03 * depth_norm
    hz_v = 0.15 + 0.35 * sigma + 0.25 * toxicity + 0.06 * depth_norm

    mm_surv = np.empty(n, dtype="float64")
    for i in range(n):
        mm_surv[i] = markov_modulated_survival(
            horizon_seconds=horizon_seconds,
            generator_q=q_gen,
            hazard_by_state=np.array([hz_q[i], hz_v[i]], dtype="float64"),
            pi0=np.array([0.75, 0.25], dtype="float64"),
        )
    out["survival_markov_tau"] = np.clip(mm_surv, 0.0, 1.0)

    # Passive EV and cancellation logic.
    expected_impact = 0.30 + 1.8 * toxicity + 2.5 * sigma + 0.8 * np.maximum(-imbalance, 0.0)
    out["expected_impact_bps"] = expected_impact
    out["ev_passive_bps"] = expected_passive_value_bps(
        p_fill=out["p_fill_cp_gauss"],
        half_spread_bps=half_spread_bps,
        expected_impact_bps=out["expected_impact_bps"],
        maker_rebate_bps=maker_rebate_bps,
    )

    out["cancel_by_ev"] = (out["ev_passive_bps"] < float(cancel_ev_threshold_bps)).astype("int8")
    out["cancel_by_toxicity"] = ((toxicity > float(cancel_toxicity_threshold)) & (out["p_fill_hazard"] > 0.10)).astype("int8")
    out["cancel_signal"] = ((out["cancel_by_ev"] == 1) | (out["cancel_by_toxicity"] == 1)).astype("int8")

    return out