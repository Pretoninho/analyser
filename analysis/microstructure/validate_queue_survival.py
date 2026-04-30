"""
Quick validation for queue survival feature pipeline.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.microstructure.queue_survival import compute_queue_survival_features, kaplan_meier_curve


def make_synthetic_events(n: int = 5000, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ts = np.cumsum(np.maximum(rng.normal(1.0, 0.10, size=n), 0.2))
    queue_ahead = np.maximum(1.0, rng.gamma(shape=2.0, scale=12.0, size=n))
    imbalance = np.clip(rng.normal(0.0, 0.30, size=n), -1.0, 1.0)
    sigma = np.maximum(0.0, rng.lognormal(mean=-2.7, sigma=0.35, size=n))
    ofi = np.clip(rng.normal(0.0, 0.55, size=n), -1.0, 1.0)

    # Aggressive volume process with clustered bursts.
    burst = rng.binomial(1, 0.06, size=n)
    base = rng.poisson(lam=0.4, size=n).astype("float64")
    jump = burst * rng.poisson(lam=6.0, size=n)
    consume = base + jump

    return pd.DataFrame(
        {
            "ts": ts,
            "queue_ahead_vol": queue_ahead,
            "dom_depth_imbalance": imbalance,
            "sigma_ewma": sigma,
            "ofi_norm": ofi,
            "consume_ahead_vol": consume,
        }
    )


def validate_pipeline() -> None:
    df = make_synthetic_events()
    feat = compute_queue_survival_features(df)

    required_cols = [
        "hazard_cox",
        "survival_exec_tau",
        "p_fill_cp_gauss",
        "survival_markov_tau",
        "ev_passive_bps",
        "cancel_signal",
    ]
    for c in required_cols:
        assert c in feat.columns, f"missing column: {c}"

    durations = np.random.default_rng(42).exponential(scale=2.0, size=4000)
    observed = np.random.default_rng(43).binomial(1, 0.68, size=4000)
    km = kaplan_meier_curve(durations, observed)

    assert len(km) > 0, "Kaplan-Meier output is empty"
    assert km["survival_km"].iloc[-1] <= 1.0 + 1e-9
    assert km["survival_km"].iloc[-1] >= -1e-9

    print("QUEUE SURVIVAL FEATURES SHAPE:", feat.shape)
    print("KM CURVE SHAPE:", km.shape)
    print("Fill probability (CP) mean:", round(float(feat["p_fill_cp_gauss"].mean()), 6))
    print("Hazard (Cox) mean:", round(float(feat["hazard_cox"].mean()), 6))
    print("EV passive bps mean:", round(float(feat["ev_passive_bps"].mean()), 6))
    print("Cancel ratio:", round(float(feat["cancel_signal"].mean()), 6))
    print("OK", feat.shape, km.shape)


if __name__ == "__main__":
    validate_pipeline()