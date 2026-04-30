from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.microstructure.queue_dynamics import compute_queue_execution_features


def build_synthetic_queue_events(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ts = pd.date_range("2026-01-01", periods=n, freq="ms", tz="UTC")

    # Base queue state
    queue_ahead = np.maximum(rng.gamma(shape=2.0, scale=25.0, size=n), 1.0)
    queue_behind = np.maximum(rng.gamma(shape=1.8, scale=18.0, size=n), 0.0)

    # Event flows ahead
    consume = np.maximum(rng.exponential(scale=3.0, size=n), 0.0)
    cancel = np.maximum(rng.exponential(scale=1.2, size=n), 0.0)

    # Inject toxic episodes
    toxic_idx = (np.arange(n) > 1700) & (np.arange(n) < 2200)
    consume[toxic_idx] *= 3.0
    cancel[toxic_idx] *= 2.5

    # Signals
    ofi = rng.normal(0.0, 0.25, size=n)
    ofi[toxic_idx] -= 0.8

    sigma = np.abs(rng.normal(0.0005, 0.0002, size=n))
    sigma[toxic_idx] *= 2.0

    micro_div = rng.normal(0.0, 2.0, size=n)
    micro_div[toxic_idx] -= 7.0

    return pd.DataFrame(
        {
            "timestamp": ts,
            "queue_ahead_vol": queue_ahead,
            "queue_behind_vol": queue_behind,
            "consume_ahead_vol": consume,
            "cancel_ahead_vol": cancel,
            "ofi_norm": ofi,
            "sigma_ewma": sigma,
            "microprice_divergence_bps": micro_div,
        }
    )


def main() -> None:
    df = build_synthetic_queue_events()
    out = compute_queue_execution_features(df)

    cols = [
        "queue_ahead",
        "q_ahead_ratio",
        "kappa_dyn",
        "p_fill_exp",
        "p_survival_post",
        "ev_passive_bps",
        "cancel_signal",
    ]

    print(out[cols].describe().round(4).to_string())
    print("\nCancel counts:")
    print(out["cancel_signal"].value_counts().sort_index())

    print("\nHead sample:")
    print(out[cols].head(5).to_string(index=False))

    print("\nToxic window cancel rate (1701..2199):")
    toxic_slice = out.iloc[1701:2200]
    print(float(toxic_slice["cancel_signal"].mean()))

    print("\nOK", out.shape)


if __name__ == "__main__":
    main()
