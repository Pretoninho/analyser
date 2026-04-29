from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "display" / "analysis"
DB_DIR = ROOT / "db"

SHORTLIST_AGGR_2 = OUT_DIR / "htf_combo_ranking_2_agressif.csv"
SHORTLIST_AGGR_3 = OUT_DIR / "htf_combo_ranking_3_agressif.csv"
SHORTLIST_RELAX_2 = OUT_DIR / "htf_combo_ranking_2_equilibre_assoupli.csv"
SHORTLIST_RELAX_3 = OUT_DIR / "htf_combo_ranking_3_equilibre_assoupli.csv"


def _load_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _q_vector(row: pd.Series) -> np.ndarray:
    q_flat = 0.0
    q_long = float(row.get("long_avg_fwd_ret_pct", 0.0)) / 100.0
    q_short = float(row.get("short_avg_fwd_ret_pct", 0.0)) / 100.0
    return np.array([q_flat, q_long, q_short], dtype=np.float64)


def _policy_from_q(q: np.ndarray) -> str:
    idx = int(np.argmax(q))
    return ["FLAT", "LONG", "SHORT"][idx]


def _seed_from_frames(df2: pd.DataFrame, df3: pd.DataFrame, profile_name: str) -> dict:
    frames = []
    if not df2.empty:
        tmp2 = df2.copy()
        tmp2["combo_size"] = 2
        frames.append(tmp2)
    if not df3.empty:
        tmp3 = df3.copy()
        tmp3["combo_size"] = 3
        frames.append(tmp3)

    if not frames:
        return {
            "version": "htf_seed_v1",
            "profile": profile_name,
            "n_states": 0,
            "states": [],
            "actions": ["FLAT", "LONG", "SHORT"],
            "q_table": np.zeros((0, 3), dtype=np.float64),
        }

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.sort_values(by=["combo_size", "rank"], ascending=[True, True]).reset_index(drop=True)

    states = []
    q_rows = []
    for row in all_rows.itertuples(index=False):
        s = row._asdict()
        q = _q_vector(pd.Series(s))
        policy = _policy_from_q(q)
        states.append(
            {
                "state_id": f"HTF_{profile_name}_{len(states):03d}",
                "combo_id": s.get("combo_id", ""),
                "state_codes": s.get("state_codes", ""),
                "state_names": s.get("state_names", ""),
                "timeframes": s.get("timeframes", ""),
                "combo_size": int(s.get("combo_size", 0)),
                "q_flat": float(q[0]),
                "q_long": float(q[1]),
                "q_short": float(q[2]),
                "policy": policy,
                "qtable_status": s.get("qtable_status", ""),
                "win_rate_pct": float(s.get("win_rate_pct", 0.0)),
                "wilson_lb_pct": float(s.get("wilson_lb_pct", 0.0)),
                "n_trades": int(s.get("n_trades", 0)),
                "avg_trade_pct": float(s.get("avg_trade_pct", 0.0)),
                "total_return_pct": float(s.get("total_return_pct", 0.0)),
                "max_drawdown_pct": float(s.get("max_drawdown_pct", 0.0)),
            }
        )
        q_rows.append(q)

    q_table = np.vstack(q_rows) if q_rows else np.zeros((0, 3), dtype=np.float64)
    return {
        "version": "htf_seed_v1",
        "profile": profile_name,
        "n_states": len(states),
        "states": states,
        "actions": ["FLAT", "LONG", "SHORT"],
        "q_table": q_table,
    }


def _save_seed(seed: dict, profile_name: str) -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pkl_path = DB_DIR / f"stats_agent_htf_seed_{profile_name}.pkl"
    csv_path = OUT_DIR / f"htf_qtable_seed_{profile_name}.csv"

    with pkl_path.open("wb") as f:
        pickle.dump(seed, f)

    pd.DataFrame(seed["states"]).to_csv(csv_path, index=False)

    print(f"seed_profile={profile_name}")
    print(f"states={seed['n_states']}")
    print(f"pickle={pkl_path}")
    print(f"csv={csv_path}")


def main() -> None:
    aggr2 = _load_if_exists(SHORTLIST_AGGR_2)
    aggr3 = _load_if_exists(SHORTLIST_AGGR_3)
    relaxed2 = _load_if_exists(SHORTLIST_RELAX_2)
    relaxed3 = _load_if_exists(SHORTLIST_RELAX_3)

    seed_aggressive = _seed_from_frames(aggr2, aggr3, "agressif")
    seed_relaxed = _seed_from_frames(relaxed2, relaxed3, "equilibre_assoupli")

    _save_seed(seed_aggressive, "agressif")
    _save_seed(seed_relaxed, "equilibre_assoupli")


if __name__ == "__main__":
    main()
