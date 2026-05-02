# ── Ensemble Voting — Vote 2/3 pour consensus ──
#
# Au lieu de picker 1 config stable, on vote entre 3 configs top OOS.
# Signal valide seulement si 2/3 configs votent le même direction+regime.

import pandas as pd
import numpy as np
from pathlib import Path


class EnsembleVoter:
    """Gestionnaire ensemble voting pour TA strategy."""

    def __init__(self, results_dir: Path):
        """
        Charge les top-3 configs stables OOS par régime.
        """
        self.results_dir = Path(results_dir)
        self.top_configs = self._load_top_configs()

    def _load_top_configs(self) -> dict:
        """
        Retourne dict :
          regime -> list of 3 dicts {params, direction, features}
        """
        path = self.results_dir / "sweep_IS_vs_OOS.csv"
        if not path.exists():
            return {}

        df = pd.read_csv(path)
        df["wr_drop"] = df["wr_OOS"] - df["wr_IS"]

        top_by_regime = {}
        for regime in ["bull", "bear"]:
            filt = df[
                (df["regime"] == regime) &
                (df["wr_drop"] >= -0.05) &
                (df["n_OOS"] >= 5)
            ].nlargest(3, "exp_R_OOS")

            top_by_regime[regime] = []
            for _, row in filt.iterrows():
                top_by_regime[regime].append({
                    "params": row["params"],
                    "direction": row["direction"],
                    "regime": regime,
                    "ema_state": int(row["ema_state"]),
                    "ema_slope": int(row["ema_slope"]),
                    "swing": int(row["swing"]),
                    "rsi_state": str(row["rsi_state"]),
                    "stoch_state": str(row["stoch_state"]),
                    "atr_state": str(row["atr_state"]),
                    "vwap_state": int(row["vwap_state"]),
                    "wr_OOS": row["wr_OOS"],
                    "exp_R_OOS": row["exp_R_OOS"],
                })

        return top_by_regime

    def vote(self, regime: str, features: dict, direction_candidate: str) -> tuple:
        """
        Vote sur le signal candidat.

        Args:
            regime: 'bull' ou 'bear'
            features: dict avec {ema_state, ema_slope, swing, rsi_state, stoch_state, atr_state, vwap_state}
            direction_candidate: 'LONG' ou 'SHORT' du trigger

        Returns:
            (vote_count, consensus_direction, confidence)
              vote_count : 0-3 votes pour la direction candidate
              consensus_direction : direction votée si vote_count >= 2, None sinon
              confidence : moyenne exp_R_OOS des voters (0 si no consensus)
        """
        if regime not in self.top_configs:
            return 0, None, 0.0

        configs = self.top_configs[regime]
        votes = []

        for config in configs:
            # Check feature match
            if features["ema_state"] != config["ema_state"]:
                continue
            if features["ema_slope"] != config["ema_slope"]:
                continue
            if features["swing"] != config["swing"]:
                continue
            if features["rsi_state"] != config["rsi_state"]:
                continue
            if features["stoch_state"] != config["stoch_state"]:
                continue
            if features["atr_state"] != config["atr_state"]:
                continue

            # Config match — check si direction cohérente
            if direction_candidate == config["direction"]:
                votes.append(config["exp_R_OOS"])

        # Result
        vote_count = len(votes)
        if vote_count >= 2:
            consensus_direction = direction_candidate
            confidence = np.mean(votes)
        else:
            consensus_direction = None
            confidence = 0.0

        return vote_count, consensus_direction, confidence

    def get_top_configs_summary(self) -> pd.DataFrame:
        """Retourne summary des configs top OOS pour debug/display."""
        rows = []
        for regime, configs in self.top_configs.items():
            for i, cfg in enumerate(configs):
                rows.append({
                    "regime": regime,
                    "rank": i + 1,
                    "params": cfg["params"],
                    "direction": cfg["direction"],
                    "wr_OOS": cfg["wr_OOS"],
                    "exp_R_OOS": cfg["exp_R_OOS"],
                })
        return pd.DataFrame(rows)
