"""
engine/markov.py — Chaîne de Markov sur les états de marché Pi*.

La chaîne stocke :
    - counts[s, s']   : nombre de fois que l'état s a été suivi de s'
    - prob[s, s']     : probabilité P(s'|s) = counts / sum_row
    - confidence[s]   : nombre total d'observations depuis s

Elle se renforce automatiquement à chaque épisode d'entraînement
et lors des backtests.

Persistance : db/markov.npy (counts bruts — les probabilités sont
recalculées à la volée pour rester cohérentes après chaque update).
"""

import numpy as np
import pickle
from pathlib import Path

# Nombre d'états marché (SESSION×VOL×STRUCTURE, sans encodage position)
N_MARKET_STATES = 100

# Seuil minimum d'observations pour considérer une probabilité fiable
MIN_CONFIDENCE = 5


class MarkovChain:

    def __init__(self):
        self.counts = np.zeros((N_MARKET_STATES, N_MARKET_STATES), dtype=np.float64)

    # ── Construction ───────────────────────────────────────────

    def fit(self, sessions: list) -> "MarkovChain":
        """Construit la chaîne depuis une liste de sessions (DataFrames)."""
        self.counts = np.zeros((N_MARKET_STATES, N_MARKET_STATES), dtype=np.float64)
        for ep_df in sessions:
            states = ep_df.reset_index(drop=True)["state_id"].astype(int).tolist()
            for s, s_next in zip(states[:-1], states[1:]):
                self.counts[s, s_next] += 1
        return self

    def update(self, from_state: int, to_state: int, weight: float = 1.0):
        """Mise à jour incrémentale après un step d'entraînement ou backtest."""
        self.counts[from_state, to_state] += weight

    # ── Probabilités ───────────────────────────────────────────

    @property
    def prob(self) -> np.ndarray:
        """Matrice de probabilité P(s'|s) normalisée ligne par ligne."""
        row_sums = self.counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return self.counts / row_sums

    @property
    def confidence(self) -> np.ndarray:
        """Nombre total d'observations depuis chaque état."""
        return self.counts.sum(axis=1)

    def is_reliable(self, state: int) -> bool:
        return float(self.confidence[state]) >= MIN_CONFIDENCE

    # ── Prédiction ─────────────────────────────────────────────

    def predict(self, state: int, top_k: int = 3, exclude_self: bool = True) -> list[dict]:
        """
        Retourne les top_k états suivants les plus probables depuis state.

        Returns:
            list de dict {state, prob, confidence, reliable}
            triée par probabilité décroissante
        """
        p = self.prob[state].copy()
        if exclude_self:
            p[state] = 0.0

        total_excl = p.sum()
        if total_excl > 0:
            p = p / total_excl  # renormaliser après exclusion self

        indices = np.argsort(p)[::-1][:top_k]
        results = []
        for idx in indices:
            if p[idx] > 0:
                results.append({
                    "state":      int(idx),
                    "prob":       float(p[idx]),
                    "count":      int(self.counts[state, idx]),
                    "reliable":   self.is_reliable(state),
                })
        return results

    def predict_structure(self, state: int, top_k: int = 3) -> list[dict]:
        """
        Prédit la structure de prix la plus probable au prochain état,
        en agrégeant les probabilités par PRICE_STRUCTURE (0-4).
        Utile pour l'anticipation de pattern.
        """
        p = self.prob[state].copy()
        p[state] = 0.0
        total = p.sum()
        if total == 0:
            return []
        p = p / total

        # Agréger par structure : state % 5
        struct_prob = np.zeros(5)
        for s in range(self.counts.shape[0]):
            struct_prob[s % 5] += p[s]

        indices = np.argsort(struct_prob)[::-1][:top_k]
        PS_NAMES = {0: "RANGE", 1: "UPTREND", 2: "DOWNTREND", 3: "BREAKOUT", 4: "REJECTION"}
        return [
            {"structure": PS_NAMES[int(i)], "prob": float(struct_prob[i])}
            for i in indices if struct_prob[i] > 0
        ]

    # ── Analyse ────────────────────────────────────────────────

    def stationary_distribution(self) -> np.ndarray:
        """
        Distribution stationnaire π telle que π = π × P.
        Calculée par itération de puissance.
        Retourne un vecteur (N_STATES,) ou zeros si non convergé.
        """
        visited = self.confidence > 0
        if visited.sum() < 2:
            return np.zeros(self.counts.shape[0])

        P = self.prob.copy()
        n  = self.counts.shape[0]
        pi = np.ones(n) / n
        for _ in range(1000):
            pi_new = pi @ P
            if np.max(np.abs(pi_new - pi)) < 1e-8:
                break
            pi = pi_new
        return pi_new

    def most_visited_states(self, top_k: int = 10) -> list[dict]:
        """États les plus fréquemment visités (selon les counts)."""
        conf = self.confidence
        indices = np.argsort(conf)[::-1][:top_k]
        return [
            {"state": int(i), "visits": int(conf[i])}
            for i in indices if conf[i] > 0
        ]

    def transition_entropy(self, state: int) -> float:
        """
        Entropie de Shannon de la distribution de transition depuis state.
        Haute entropie = transitions imprévisibles.
        Basse entropie = transitions déterministes (prévisibles).
        """
        p = self.prob[state]
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def summary(self) -> dict:
        """Résumé global de la chaîne."""
        conf = self.confidence
        visited = (conf > 0).sum()
        reliable = (conf >= MIN_CONFIDENCE).sum()
        total_transitions = int(self.counts.sum())
        return {
            "etats_visites":   int(visited),
            "etats_fiables":   int(reliable),
            "transitions_tot": total_transitions,
            "transitions_uniq": int((self.counts > 0).sum()),
            "entropie_moy":    float(np.mean([
                self.transition_entropy(s) for s in range(self.counts.shape[0]) if conf[s] > 0
            ])),
        }

    # ── Validation (backtest) ──────────────────────────────────

    def backtest_accuracy(self, sessions: list) -> dict:
        """
        Mesure la précision de prédiction top-1 sur les sessions de test.
        Pour chaque transition, vérifie si l'état prédit (argmax) = état réel.
        """
        correct_top1 = 0
        correct_top3 = 0
        total = 0

        for ep_df in sessions:
            states = ep_df.reset_index(drop=True)["state_id"].astype(int).tolist()
            for s, s_next in zip(states[:-1], states[1:]):
                if not self.is_reliable(s):
                    continue
                p = self.prob[s].copy()
                p[s] = 0.0
                if p.sum() == 0:
                    continue
                top3 = np.argsort(p)[::-1][:3]
                if top3[0] == s_next:
                    correct_top1 += 1
                if s_next in top3:
                    correct_top3 += 1
                total += 1

        if total == 0:
            return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "n_evaluated": 0}
        return {
            "top1_accuracy": round(correct_top1 / total, 4),
            "top3_accuracy": round(correct_top3 / total, 4),
            "n_evaluated":   total,
        }

    # ── Persistance ────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"counts": self.counts}, f)
        print(f"[markov] Sauvegardé : {path} ({int(self.counts.sum())} transitions)")

    @classmethod
    def load(cls, path: str) -> "MarkovChain":
        with open(path, "rb") as f:
            data = pickle.load(f)
        mc = cls()
        mc.counts = data["counts"]
        print(f"[markov] Chargé : {path} ({int(mc.counts.sum())} transitions)")
        return mc
