# ── Ensemble Voting v2 — Majority Consensus avec Pool Qualifié ──
#
# Approche moins restrictive :
#   - Pool large : tous configs avec n_OOS >= 5 et wr_OOS >= 55%
#   - Vote : "majority consensus" (>50% des matching configs votent direction)
#   - Plus flexibles sur feature matching

import pandas as pd
import numpy as np
from pathlib import Path


class EnsembleVoterV2:
    """Voteur ensemble plus flexible basé sur pool qualifié."""

    def __init__(self, results_dir: Path, min_n_oos: int = 5, min_wr_oos: float = 0.55):
        """
        Charge tous configs stables OOS, not just top-3.

        Args:
            results_dir: chemin du répertoire results
            min_n_oos: minimum trades OOS
            min_wr_oos: minimum WR OOS
        """
        self.results_dir = Path(results_dir)
        self.min_n_oos = min_n_oos
        self.min_wr_oos = min_wr_oos
        self.qualified_configs = self._load_qualified_configs()
        self.stats = self._compute_stats()

    def _load_qualified_configs(self) -> list:
        """Charge tous configs stables OOS (pool large)."""
        path = self.results_dir / "sweep_IS_vs_OOS.csv"
        if not path.exists():
            return []

        df = pd.read_csv(path)
        df["wr_drop"] = df["wr_OOS"] - df["wr_IS"]

        # Filter qualifiés : n_OOS >= min, wr_OOS >= min
        filt = df[
            (df["n_OOS"] >= self.min_n_oos) &
            (df["wr_OOS"] >= self.min_wr_oos)
        ].copy()

        configs = []
        for _, row in filt.iterrows():
            configs.append({
                "params": row["params"],
                "direction": row["direction"],
                "regime": row["regime"],
                "ema_state": int(row["ema_state"]),
                "ema_slope": int(row["ema_slope"]),
                "swing": int(row["swing"]),
                "rsi_state": str(row["rsi_state"]),
                "stoch_state": str(row["stoch_state"]),
                "atr_state": str(row["atr_state"]),
                "vwap_state": int(row["vwap_state"]),
                "wr_OOS": row["wr_OOS"],
                "exp_R_OOS": row["exp_R_OOS"],
                "n_OOS": int(row["n_OOS"]),
            })

        return configs

    def _compute_stats(self) -> dict:
        """Compute stats pool."""
        if not self.qualified_configs:
            return {
                "total_configs": 0,
                "by_regime": {},
                "by_direction": {},
            }

        df = pd.DataFrame(self.qualified_configs)
        return {
            "total_configs": len(df),
            "by_regime": df["regime"].value_counts().to_dict(),
            "by_direction": df["direction"].value_counts().to_dict(),
        }

    def vote(self, regime: str, features: dict, direction_candidate: str) -> tuple:
        """
        Vote sur le signal candidat (majority consensus).

        Args:
            regime: 'bull' ou 'bear'
            features: dict avec ema_state, ema_slope, swing, rsi_state, etc.
            direction_candidate: 'LONG' ou 'SHORT' du trigger

        Returns:
            (total_voters, favorable_votes, consensus_direction, avg_confidence)
              total_voters : nombre de configs matching les features
              favorable_votes : nombre votant pour direction_candidate
              consensus_direction : direction si vote_favorable > 50%, None sinon
              avg_confidence : moyenne exp_R des favorable voters
        """
        # Filter configs du même régime
        candidates = [c for c in self.qualified_configs if c["regime"] == regime]

        if not candidates:
            return 0, 0, None, 0.0

        # Match feature + direction
        favorable = []
        total_matching = 0

        # Match feature + direction (cast int pour éviter int vs float mismatch depuis CSV)
        f_ema_state  = int(features["ema_state"])
        f_ema_slope  = int(features["ema_slope"])
        f_swing      = int(features["swing"])
        f_vwap       = int(features.get("vwap_state", 0))
        f_rsi        = str(features["rsi_state"])
        f_stoch      = str(features["stoch_state"])
        f_atr        = str(features["atr_state"])

        for config in candidates:
            # Feature match (exact)
            if (f_ema_state != int(config["ema_state"]) or
                f_ema_slope != int(config["ema_slope"]) or
                f_swing     != int(config["swing"]) or
                f_rsi       != str(config["rsi_state"]) or
                f_stoch     != str(config["stoch_state"]) or
                f_atr       != str(config["atr_state"])):
                continue

            total_matching += 1

            # Direction match
            if direction_candidate == config["direction"]:
                favorable.append(config["exp_R_OOS"])

        # Consensus
        if total_matching == 0:
            return 0, 0, None, 0.0

        favorable_count = len(favorable)
        favorable_pct = favorable_count / total_matching

        if favorable_pct > 0.5:
            consensus_direction = direction_candidate
            avg_confidence = np.mean(favorable) if favorable else 0.0
        else:
            consensus_direction = None
            avg_confidence = 0.0

        return total_matching, favorable_count, consensus_direction, avg_confidence

    def get_qualified_summary(self) -> pd.DataFrame:
        """Summary des configs qualifiés."""
        return pd.DataFrame(self.qualified_configs)

    def get_pool_stats(self) -> dict:
        """Pool stats."""
        return self.stats
