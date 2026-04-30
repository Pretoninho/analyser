"""
engine/microstructure/queue_dynamics.py

Queue-position and adverse-selection features for passive execution models.

Implements reusable, backtest-oriented functions from queue dynamics theory:
- FIFO queue rank metrics (ahead/behind/ratio)
- Exponential fill probability vs queue depth
- Dynamic kappa estimation (decay slope)
- Bayesian survival update under toxic-flow evidence
- Adverse-selection expected value and cancel triggers
- Event-time queue progression simulation

Expected event dataframe (minimum for full pipeline):
    queue_ahead_vol         : volume ahead of our order at level (>=0)
    queue_behind_vol        : volume behind our order (>=0, optional)
    consume_ahead_vol       : market orders consuming queue ahead (>=0)
    cancel_ahead_vol        : cancellations/mods ahead of us (>=0)

Optional features used to adapt kappa and toxic posterior:
    ofi_norm                : order flow imbalance proxy [-1, +1]
    sigma_ewma              : local volatility proxy
    microprice_divergence_bps : microprice divergence signal
    ts / timestamp          : for dt-aware derivatives
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-12

# Default parameters (conservative baselines for calibration)
_KAPPA_BASE = 0.02
_KAPPA_VOL_SCALE = 30.0
_KAPPA_OFI_SCALE = 0.5

_TOXIC_PRIOR = 0.30

_SPREAD_HALF_BPS = 0.5
_MAKER_REBATE_BPS = 0.1
_TOXIC_MARKOUT_BPS = -2.0

_SURVIVAL_CANCEL_THRESHOLD = 0.20
_EV_CANCEL_THRESHOLD_BPS = 0.0


def _safe_numeric(x: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        return arr
    return pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype="float64")


def _infer_dt_seconds(df: pd.DataFrame) -> np.ndarray:
    if "ts" in df.columns:
        t = pd.to_numeric(df["ts"], errors="coerce").to_numpy(dtype="float64")
    elif "timestamp" in df.columns:
        ts_ns = pd.to_datetime(df["timestamp"], utc=True).astype("int64")
        t = ts_ns.to_numpy(dtype="float64") / 1e9
    else:
        return np.ones(len(df), dtype="float64")

    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = 1.0
    return dt


def compute_queue_state_metrics(
    queue_ahead_vol: pd.Series | np.ndarray,
    queue_behind_vol: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute core queue-state metrics.

    Metrics:
        q_ahead_ratio = ahead / (ahead + behind)
        q_support_ratio = behind / (ahead + behind)
        q_front_proximity = 1 / (1 + ahead)

    Returns:
        DataFrame with queue-state columns.
    """
    ahead = np.maximum(_safe_numeric(queue_ahead_vol), 0.0)

    if queue_behind_vol is None:
        behind = np.zeros_like(ahead)
    else:
        behind = np.maximum(_safe_numeric(queue_behind_vol), 0.0)

    total = ahead + behind

    out = pd.DataFrame(
        {
            "queue_ahead": ahead,
            "queue_behind": behind,
            "queue_total": total,
            "q_ahead_ratio": ahead / (total + _EPS),
            "q_support_ratio": behind / (total + _EPS),
            "q_front_proximity": 1.0 / (1.0 + ahead),
        }
    )
    return out.astype("float64")


def dynamic_kappa(
    df: pd.DataFrame,
    base_kappa: float = _KAPPA_BASE,
    vol_scale: float = _KAPPA_VOL_SCALE,
    ofi_scale: float = _KAPPA_OFI_SCALE,
    sigma_col: str = "sigma_ewma",
    ofi_col: str = "ofi_norm",
) -> pd.Series:
    """
    Estimate dynamic decay slope for fill probability.

    Model intuition:
    - higher volatility -> flatter decay -> lower kappa
    - stronger absolute OFI -> faster queue transitions -> lower kappa

    kappa_t = base_kappa / (1 + vol_scale*sigma + ofi_scale*|ofi|)
    """
    sigma = np.zeros(len(df), dtype="float64")
    if sigma_col in df.columns:
        sigma = np.maximum(_safe_numeric(df[sigma_col]), 0.0)

    ofi = np.zeros(len(df), dtype="float64")
    if ofi_col in df.columns:
        ofi = np.abs(_safe_numeric(df[ofi_col]))

    denom = 1.0 + vol_scale * sigma + ofi_scale * ofi
    kappa = base_kappa / np.maximum(denom, _EPS)
    return pd.Series(kappa, index=df.index, name="kappa_dyn", dtype="float64")


def fill_probability_exponential(
    queue_ahead: pd.Series | np.ndarray,
    kappa: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Exponential fill probability from queue depth.

    Formula:
        p_fill = exp(-kappa * queue_ahead)
    """
    d = np.maximum(_safe_numeric(queue_ahead), 0.0)
    k = np.maximum(_safe_numeric(kappa), 0.0)
    p = np.exp(-k * d)
    return pd.Series(np.clip(p, 0.0, 1.0), name="p_fill_exp", dtype="float64")


def bayesian_survival_update(
    prior_survival: pd.Series | np.ndarray,
    evidence_likelihood_if_survive: pd.Series | np.ndarray,
    evidence_likelihood_if_toxic: pd.Series | np.ndarray,
) -> pd.Series:
    """
    Bayesian posterior update for survival probability.

    posterior = P(S|E) = P(S)*P(E|S) / [P(S)*P(E|S) + (1-P(S))*P(E|T)]
    where T is toxic regime.
    """
    p_s = np.clip(_safe_numeric(prior_survival), 0.0, 1.0)
    l_s = np.clip(_safe_numeric(evidence_likelihood_if_survive), _EPS, 1.0)
    l_t = np.clip(_safe_numeric(evidence_likelihood_if_toxic), _EPS, 1.0)

    num = p_s * l_s
    den = num + (1.0 - p_s) * l_t
    post = num / (den + _EPS)

    return pd.Series(np.clip(post, 0.0, 1.0), name="p_survival_post", dtype="float64")


def toxic_evidence_likelihoods(
    df: pd.DataFrame,
    ofi_col: str = "ofi_norm",
    micro_div_col: str = "microprice_divergence_bps",
    consume_col: str = "consume_ahead_vol",
) -> pd.DataFrame:
    """
    Build simple likelihood proxies P(E|survive) and P(E|toxic) from signals.

    Heuristic mapping:
    - large negative OFI, adverse microprice divergence, and high consume flow
      increase toxic likelihood.
    """
    n = len(df)

    ofi = np.zeros(n, dtype="float64")
    if ofi_col in df.columns:
        ofi = _safe_numeric(df[ofi_col])

    micro = np.zeros(n, dtype="float64")
    if micro_div_col in df.columns:
        micro = _safe_numeric(df[micro_div_col])

    consume = np.zeros(n, dtype="float64")
    if consume_col in df.columns:
        consume = np.maximum(_safe_numeric(df[consume_col]), 0.0)

    # Toxic pressure score (direction-agnostic absolute pressure + downside proxy)
    score = (
        1.2 * np.maximum(-ofi, 0.0)
        + 0.8 * np.maximum(-micro / 10.0, 0.0)
        + 0.5 * (consume / (np.nanmedian(consume) + 1.0))
    )

    # Logistic transform to [0,1]
    p_toxic = 1.0 / (1.0 + np.exp(-(score - 1.0)))
    p_survive = 1.0 - p_toxic

    return pd.DataFrame(
        {
            "likelihood_e_given_survive": np.clip(p_survive, _EPS, 1.0),
            "likelihood_e_given_toxic": np.clip(p_toxic, _EPS, 1.0),
        },
        index=df.index,
    )


def expected_passive_ev_bps(
    p_fill: pd.Series | np.ndarray,
    p_survival: pd.Series | np.ndarray,
    spread_half_bps: float = _SPREAD_HALF_BPS,
    maker_rebate_bps: float = _MAKER_REBATE_BPS,
    toxic_markout_bps: float = _TOXIC_MARKOUT_BPS,
) -> pd.Series:
    """
    Expected value (bps) of resting passively.

    If filled and survives toxicity:
        gain = spread_half + maker_rebate
    If filled and toxic:
        markout loss = toxic_markout_bps (typically negative)

    EV = p_fill * [ p_survival*(spread_half+rebate) + (1-p_survival)*toxic_markout ]
    """
    pf = np.clip(_safe_numeric(p_fill), 0.0, 1.0)
    ps = np.clip(_safe_numeric(p_survival), 0.0, 1.0)

    edge_good = spread_half_bps + maker_rebate_bps
    edge_bad = toxic_markout_bps

    ev = pf * (ps * edge_good + (1.0 - ps) * edge_bad)
    return pd.Series(ev, name="ev_passive_bps", dtype="float64")


def simulate_queue_ahead_progress(
    df: pd.DataFrame,
    initial_ahead: float,
    consume_col: str = "consume_ahead_vol",
    cancel_col: str = "cancel_ahead_vol",
) -> pd.Series:
    """
    Simulate queue-ahead trajectory from event-time consume/cancel flows.

    Recurrence:
        ahead_t = max(ahead_{t-1} - consume_t - cancel_t, 0)
    """
    consume = np.maximum(_safe_numeric(df[consume_col]), 0.0) if consume_col in df.columns else np.zeros(len(df))
    cancel = np.maximum(_safe_numeric(df[cancel_col]), 0.0) if cancel_col in df.columns else np.zeros(len(df))

    out = np.empty(len(df), dtype="float64")
    ahead = max(float(initial_ahead), 0.0)

    for i in range(len(df)):
        ahead = max(ahead - consume[i] - cancel[i], 0.0)
        out[i] = ahead

    return pd.Series(out, index=df.index, name="queue_ahead_sim", dtype="float64")


def compute_queue_execution_features(
    df_events: pd.DataFrame,
    queue_ahead_col: str = "queue_ahead_vol",
    queue_behind_col: str = "queue_behind_vol",
    consume_col: str = "consume_ahead_vol",
    cancel_col: str = "cancel_ahead_vol",
    survival_cancel_threshold: float = _SURVIVAL_CANCEL_THRESHOLD,
    ev_cancel_threshold_bps: float = _EV_CANCEL_THRESHOLD_BPS,
) -> pd.DataFrame:
    """
    End-to-end queue/execution feature pipeline.

    Output columns:
        queue_ahead, queue_behind, queue_total,
        q_ahead_ratio, q_support_ratio, q_front_proximity,
        kappa_dyn, p_fill_exp,
        likelihood_e_given_survive, likelihood_e_given_toxic,
        p_survival_post,
        ev_passive_bps,
        cancel_by_survival, cancel_by_ev, cancel_signal
    """
    out = df_events.copy()

    # Base queue metrics
    qs = compute_queue_state_metrics(out[queue_ahead_col], out.get(queue_behind_col))
    out = pd.concat([out, qs], axis=1)

    # Dynamic kappa and fill probability
    out["kappa_dyn"] = dynamic_kappa(out)
    out["p_fill_exp"] = fill_probability_exponential(out["queue_ahead"], out["kappa_dyn"])

    # Bayesian toxic/survival update
    like = toxic_evidence_likelihoods(out, consume_col=consume_col)
    out = pd.concat([out, like], axis=1)

    prior_survival = 1.0 - _TOXIC_PRIOR
    prior_vec = np.full(len(out), prior_survival, dtype="float64")
    out["p_survival_post"] = bayesian_survival_update(
        prior_survival=prior_vec,
        evidence_likelihood_if_survive=out["likelihood_e_given_survive"],
        evidence_likelihood_if_toxic=out["likelihood_e_given_toxic"],
    )

    # EV and cancellation decisions
    out["ev_passive_bps"] = expected_passive_ev_bps(
        p_fill=out["p_fill_exp"],
        p_survival=out["p_survival_post"],
    )

    out["cancel_by_survival"] = (out["p_survival_post"] < survival_cancel_threshold).astype("int8")
    out["cancel_by_ev"] = (out["ev_passive_bps"] < ev_cancel_threshold_bps).astype("int8")
    out["cancel_signal"] = ((out["cancel_by_survival"] == 1) | (out["cancel_by_ev"] == 1)).astype("int8")

    # Optional queue simulation if consume/cancel flows available
    if consume_col in out.columns or cancel_col in out.columns:
        init_ahead = float(np.maximum(pd.to_numeric(out[queue_ahead_col], errors="coerce").fillna(0.0).iloc[0], 0.0))
        out["queue_ahead_sim"] = simulate_queue_ahead_progress(
            out,
            initial_ahead=init_ahead,
            consume_col=consume_col,
            cancel_col=cancel_col,
        )

    return out
