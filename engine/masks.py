"""
engine/masks.py — Génération automatique des masques d'actions.

Deux sources :
    1. Transitions réelles  : quels états se suivent dans les données historiques
    2. Q-table              : quelles (état, action) ont une Q-valeur négative
"""

import numpy as np
import pandas as pd
from engine.rl_env import N_STATES, N_ACTIONS

EV_THRESHOLD = -0.01   # Q-valeur sous laquelle une action est masquée


def compute_transition_stats(sessions: list) -> pd.DataFrame:
    """
    Parcourt toutes les sessions, compte les transitions état→état réelles.
    Retourne un DataFrame trié par fréquence décroissante.
    """
    counts: dict = {}
    for ep_df in sessions:
        states = ep_df.reset_index(drop=True)["state_id"].astype(int).tolist()
        for s_from, s_to in zip(states[:-1], states[1:]):
            key = (s_from, s_to)
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        return pd.DataFrame(columns=["from_state", "to_state", "count", "freq_pct"])

    rows = [{"from_state": k[0], "to_state": k[1], "count": v}
            for k, v in counts.items()]
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    df["freq_pct"] = (df["count"] / df["count"].sum() * 100).round(3)
    return df


def build_action_mask(q_table: np.ndarray,
                      ev_threshold: float = EV_THRESHOLD) -> np.ndarray:
    """
    Construit un masque booléen (N_STATES × N_ACTIONS).
    True  = action autorisée
    False = action masquée

    Règles :
        - État non visité → toutes actions autorisées (pas assez de données)
        - Q[s][a] < ev_threshold → action masquée
        - FLAT (action 0) reste toujours disponible comme fallback
    """
    visited = q_table.sum(axis=1) != 0
    mask = np.ones((N_STATES, N_ACTIONS), dtype=bool)

    for s in np.where(visited)[0]:
        for a in range(N_ACTIONS):
            if q_table[s, a] < ev_threshold:
                mask[s, a] = False
        # Fallback : FLAT toujours autorisé
        mask[s, 0] = True

    return mask


def mask_summary(mask: np.ndarray) -> dict:
    return {
        "états_avec_masque":    int((~mask).any(axis=1).sum()),
        "actions_masquées":     int((~mask).sum()),
        "pct_masquées":         round((~mask).sum() / mask.size * 100, 1),
        "FLAT_masqué":          int((~mask[:, 0]).sum()),
        "LONG_masqué":          int((~mask[:, 1]).sum()),
        "SHORT_masqué":         int((~mask[:, 2]).sum()),
    }
