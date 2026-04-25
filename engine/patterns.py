"""
engine/patterns.py — Détection de patterns Price Action sur séquences d'états.

Un pattern est défini par la transition (structure_précédente → structure_courante).
Il contraint les actions autorisées, filtrant les signaux Q-table incompatibles
avec le contexte de marché observé.

Encodage structure : RANGE_MID=0, UPTREND=1, DOWNTREND=2, BREAKOUT=3, REJECTION=4,
                     RANGE_TOP=5, RANGE_BOTTOM=6
"""

import numpy as np

N_ACTIONS = 3  # FLAT=0, LONG=1, SHORT=2

# ── Définition des patterns ────────────────────────────────────
# Clé : (structure_prev, structure_curr)
# Valeur : (nom, description courte)

PATTERN_NAMES = {
    (3, 4): ("FAUX_BREAKOUT",    "Breakout échoué — mèche de rejet"),
    (1, 3): ("CONTINUATION_H",   "Uptrend accéléré — breakout haussier"),
    (2, 3): ("CONTINUATION_B",   "Downtrend accéléré — breakout baissier"),
    (4, 1): ("REVERSAL_H",       "Rejet suivi d'uptrend — retournement haussier"),
    (4, 2): ("REVERSAL_B",       "Rejet suivi de downtrend — retournement baissier"),
    (4, 4): ("DOUBLE_REJECTION", "Double rejet — signal de retournement fort"),
    (0, 3): ("RANGE_BREAKOUT",   "Sortie de range avec volume"),
    (1, 4): ("TOP",              "Uptrend + rejet — sommet potentiel"),
    (2, 4): ("BOTTOM",           "Downtrend + rejet — bas potentiel"),
    (1, 1): ("UPTREND_CONT",     "Continuation de tendance haussière"),
    (2, 2): ("DOWNTREND_CONT",   "Continuation de tendance baissière"),
    # Patterns RANGE_TOP (5) et RANGE_BOTTOM (6)
    (5, 4): ("TOP",              "Range top + rejet — retournement SHORT"),
    (6, 4): ("BOTTOM",           "Range bottom + rejet — retournement LONG"),
    (5, 3): ("RANGE_BREAKOUT",   "Sortie de range top avec volume"),
    (6, 3): ("RANGE_BREAKOUT",   "Sortie de range bottom avec volume"),
    (5, 5): ("RANGE_TOP_CONT",   "Persistance en haut de range — SHORT privilégié"),
    (6, 6): ("RANGE_BOT_CONT",   "Persistance en bas de range — LONG privilégié"),
    (5, 2): ("REVERSAL_B",       "Range top → downtrend — retournement baissier"),
    (6, 1): ("REVERSAL_H",       "Range bottom → uptrend — retournement haussier"),
}

# ── Actions autorisées par pattern ────────────────────────────
# 0=FLAT, 1=LONG, 2=SHORT
# FLAT est toujours inclus (fallback)

PATTERN_ALLOWED = {
    "FAUX_BREAKOUT":    [0, 2],      # SHORT privilégié
    "CONTINUATION_H":   [0, 1],      # LONG privilégié
    "CONTINUATION_B":   [0, 2],      # SHORT privilégié
    "REVERSAL_H":       [0, 1],      # LONG privilégié
    "REVERSAL_B":       [0, 2],      # SHORT privilégié
    "DOUBLE_REJECTION": [0, 1, 2],   # direction inconnue sans contexte vol
    "RANGE_BREAKOUT":   [0, 1, 2],   # direction inconnue
    "TOP":              [0, 2],      # SHORT privilégié
    "BOTTOM":           [0, 1],      # LONG privilégié
    "UPTREND_CONT":     [0, 1],      # LONG privilégié
    "DOWNTREND_CONT":   [0, 2],      # SHORT privilégié
    "RANGE_TOP_CONT":   [0, 2],      # SHORT mean-reversion
    "RANGE_BOT_CONT":   [0, 1],      # LONG mean-reversion
}


def detect_pattern(prev_structure: int, curr_structure: int) -> str | None:
    """
    Retourne le nom du pattern détecté ou None si aucun pattern connu.

    Args:
        prev_structure : PriceStructure de la bougie précédente (0-4)
        curr_structure : PriceStructure de la bougie courante (0-4)
    """
    entry = PATTERN_NAMES.get((prev_structure, curr_structure))
    return entry[0] if entry is not None else None


def pattern_description(pattern: str | None) -> str:
    """Retourne la description textuelle d'un pattern."""
    if pattern is None:
        return "Aucun pattern"
    for (_, desc) in PATTERN_NAMES.values():
        pass
    for key, (name, desc) in PATTERN_NAMES.items():
        if name == pattern:
            return desc
    return pattern


def pattern_mask(pattern: str | None) -> np.ndarray:
    """
    Retourne un masque booléen (N_ACTIONS,) basé sur le pattern.
    True = action autorisée par le pattern.
    Si pattern=None, toutes les actions sont autorisées.
    """
    mask = np.ones(N_ACTIONS, dtype=bool)
    if pattern is None:
        return mask
    allowed = PATTERN_ALLOWED.get(pattern, [0, 1, 2])
    for a in range(N_ACTIONS):
        mask[a] = a in allowed
    return mask


def apply_pattern_to_state_mask(
    state_mask: np.ndarray,
    pattern: str | None,
    state_id: int,
) -> np.ndarray:
    """
    Combine le masque global (N_STATES × N_ACTIONS) avec le masque pattern
    pour un état donné. Retourne un masque (N_ACTIONS,) final.

    FLAT (action 0) est toujours garanti True.
    """
    pm = pattern_mask(pattern)

    if state_mask is not None:
        sm = state_mask[state_id].copy()
        combined = sm & pm
    else:
        combined = pm.copy()

    combined[0] = True  # FLAT toujours disponible
    return combined
