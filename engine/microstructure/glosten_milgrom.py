"""
engine/microstructure/glosten_milgrom.py

Glosten-Milgrom inspired microstructure utilities for adverse-selection modeling.

Core ideas implemented:
- Bayesian posterior updates from order direction (buy/sell)
- Information-driven bid/ask quotes from conditional expectation
- Dynamic toxicity parameter mu from rolling order-flow imbalance
- Spread decomposition: information + processing + inventory components
- Sequential quote evolution from an order-sign stream

Notation:
- V_high, V_low: terminal fundamental states
- delta: prior probability P(V=V_high)
- mu: informed trader share in [0, 1]
- noise traders buy/sell with probability 0.5

Under standard GM assumptions:
- Informed buy only in high state
- Informed sell only in low state
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-12


def expected_fundamental(delta: float, v_high: float, v_low: float) -> float:
    """Unconditional expected value E[V]."""
    d = float(np.clip(delta, 0.0, 1.0))
    return d * v_high + (1.0 - d) * v_low


def order_likelihoods(mu: float) -> dict[str, float]:
    """
    Return conditional order probabilities under GM assumptions.

    P(B|H) = mu + (1-mu)/2
    P(B|L) = (1-mu)/2
    P(S|H) = (1-mu)/2
    P(S|L) = mu + (1-mu)/2
    """
    m = float(np.clip(mu, 0.0, 1.0))
    p_noise = (1.0 - m) * 0.5

    return {
        "p_b_given_h": m + p_noise,
        "p_b_given_l": p_noise,
        "p_s_given_h": p_noise,
        "p_s_given_l": m + p_noise,
    }


def posterior_high_given_order(delta: float, mu: float, order_sign: int) -> float:
    """
    Bayesian posterior P(H|order).

    order_sign:
    - +1 for buy
    - -1 for sell
    """
    d = float(np.clip(delta, 0.0, 1.0))
    lk = order_likelihoods(mu)

    if order_sign >= 0:
        num = d * lk["p_b_given_h"]
        den = num + (1.0 - d) * lk["p_b_given_l"]
    else:
        num = d * lk["p_s_given_h"]
        den = num + (1.0 - d) * lk["p_s_given_l"]

    return float(num / (den + _EPS))


def gm_quotes_from_prior(
    delta: float,
    mu: float,
    v_high: float,
    v_low: float,
) -> dict[str, float]:
    """
    Compute GM-consistent ask/bid and spread from current prior.

    ask = E[V | Buy]
    bid = E[V | Sell]
    spread = ask - bid
    """
    post_h_buy = posterior_high_given_order(delta=delta, mu=mu, order_sign=+1)
    post_h_sell = posterior_high_given_order(delta=delta, mu=mu, order_sign=-1)

    ask = post_h_buy * v_high + (1.0 - post_h_buy) * v_low
    bid = post_h_sell * v_high + (1.0 - post_h_sell) * v_low

    mid_uncond = expected_fundamental(delta=delta, v_high=v_high, v_low=v_low)

    return {
        "ask_info": float(ask),
        "bid_info": float(bid),
        "mid_uncond": float(mid_uncond),
        "spread_info": float(ask - bid),
        "adverse_premium_ask": float(ask - mid_uncond),
        "adverse_premium_bid": float(mid_uncond - bid),
    }


def spread_decomposition(
    spread_info: pd.Series | np.ndarray,
    processing_bps: float = 0.02,
    inventory_bps: float = 0.03,
) -> pd.DataFrame:
    """
    Decompose total spread into three components.

    total_spread = info + processing + inventory

    processing and inventory are modeled as fixed bps add-ons.
    """
    info = pd.to_numeric(spread_info, errors="coerce").fillna(0.0)

    out = pd.DataFrame(index=info.index if isinstance(info, pd.Series) else None)
    out["spread_info"] = info.astype("float64")
    out["spread_processing"] = float(processing_bps)
    out["spread_inventory"] = float(inventory_bps)
    out["spread_total"] = (
        out["spread_info"] + out["spread_processing"] + out["spread_inventory"]
    )
    return out


def estimate_mu_from_order_flow(
    order_sign: pd.Series,
    window: int = 100,
    mu_min: float = 0.01,
    mu_max: float = 0.45,
    sensitivity: float = 4.0,
) -> pd.Series:
    """
    Estimate informed-flow share mu from rolling order-sign imbalance.

    order_sign expected in {-1, +1}.

    Steps:
    - rolling imbalance I in [-1, 1]
    - map |I| through a logistic-like transform to [mu_min, mu_max]

    This is a pragmatic proxy for live execution engines, not an MLE estimator.
    """
    s = pd.to_numeric(order_sign, errors="coerce").fillna(0.0).clip(-1, 1)

    imbalance = s.rolling(window).mean().fillna(0.0).abs()
    z = 1.0 / (1.0 + np.exp(-sensitivity * (imbalance - 0.25)))
    mu = mu_min + (mu_max - mu_min) * z

    return mu.astype("float64").rename("mu_hat")


def sequential_gm_quotes(
    order_sign: pd.Series,
    v_high: float,
    v_low: float,
    delta0: float = 0.5,
    mu_series: pd.Series | None = None,
    processing_bps: float = 0.02,
    inventory_bps: float = 0.03,
) -> pd.DataFrame:
    """
    Sequential GM quote update over an order flow stream.

    For each event t:
    1) use current prior delta_t and mu_t to compute quotes
    2) observe order_sign_t and update prior delta_{t+1}

    Returns a dataframe with per-event quotes and posterior dynamics.
    """
    s = pd.to_numeric(order_sign, errors="coerce").fillna(0.0)
    s = s.where(s >= 0, -1).where(s <= 0, 1)  # map non-zero to +/-1

    if mu_series is None:
        mu = estimate_mu_from_order_flow(s)
    else:
        mu = pd.to_numeric(mu_series, errors="coerce").fillna(0.1).clip(0.0, 1.0)

    n = len(s)
    delta_prior = np.empty(n, dtype="float64")
    delta_post = np.empty(n, dtype="float64")

    ask_info = np.empty(n, dtype="float64")
    bid_info = np.empty(n, dtype="float64")
    spread_info = np.empty(n, dtype="float64")
    mid_uncond = np.empty(n, dtype="float64")

    d = float(np.clip(delta0, 0.0, 1.0))

    s_vals = s.to_numpy(dtype="float64")
    mu_vals = mu.to_numpy(dtype="float64")

    for i in range(n):
        delta_prior[i] = d

        q = gm_quotes_from_prior(delta=d, mu=mu_vals[i], v_high=v_high, v_low=v_low)
        ask_info[i] = q["ask_info"]
        bid_info[i] = q["bid_info"]
        spread_info[i] = q["spread_info"]
        mid_uncond[i] = q["mid_uncond"]

        d = posterior_high_given_order(delta=d, mu=mu_vals[i], order_sign=int(np.sign(s_vals[i]) or 1))
        delta_post[i] = d

    out = pd.DataFrame(
        {
            "order_sign": s_vals.astype("int8"),
            "mu_hat": mu_vals,
            "delta_prior": delta_prior,
            "delta_post": delta_post,
            "mid_uncond": mid_uncond,
            "bid_info": bid_info,
            "ask_info": ask_info,
            "spread_info": spread_info,
        },
        index=order_sign.index,
    )

    decomp = spread_decomposition(
        out["spread_info"],
        processing_bps=processing_bps,
        inventory_bps=inventory_bps,
    )
    out = pd.concat([out, decomp[["spread_processing", "spread_inventory", "spread_total"]]], axis=1)

    # Optional binary toxicity regime marker
    out["toxic_regime"] = (out["mu_hat"] >= np.quantile(out["mu_hat"], 0.75)).astype("int8")

    return out


def classify_order_sign(
    price: pd.Series,
    bid: pd.Series,
    ask: pd.Series,
) -> pd.Series:
    """
    Lee-Ready style coarse order-sign classification from trade price and quotes.

    +1 if trade >= mid, else -1.
    """
    p = pd.to_numeric(price, errors="coerce").fillna(method="ffill").fillna(0.0)
    b = pd.to_numeric(bid, errors="coerce").fillna(method="ffill").fillna(0.0)
    a = pd.to_numeric(ask, errors="coerce").fillna(method="ffill").fillna(0.0)
    mid = 0.5 * (b + a)

    sign = np.where(p >= mid, 1, -1).astype("int8")
    return pd.Series(sign, index=price.index, name="order_sign")
