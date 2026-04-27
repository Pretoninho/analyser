"""
pi_config.py -- Configuration centrale de la strategie Pi*.

Transfert SHADOW -> LIVE :
  1. Retirer mac_idx de SHADOW_MACROS
  2. Ajouter mac_idx a LIVE_MACROS

Transfert LIVE -> SHADOW :
  1. Retirer mac_idx de LIVE_MACROS
  2. Ajouter mac_idx a SHADOW_MACROS
"""

# ── Macros : source de verite unique ─────────────────────────────
#
#   mac_idx  Heure ET   Statut
#       1    08:50      SHADOW  (pre-NYSE, signal OOS negatif)
#       2    09:50      LIVE    (macro active STAR)
#       3    10:50      SHADOW  (pas de signal consistant OOS)
#       4    11:50      SHADOW  (actif en backtest, pas encore valide live)
#       5    12:50      SHADOW  (pas de signal consistant OOS)
#       6    13:50      SHADOW  (pas de signal consistant OOS)
#       7    14:50      SHADOW  (Power Hour = phenomene equity, sans effet BTC)

LIVE_MACROS   = frozenset({2})              # tradees en live  (live_signal.py)
SHADOW_MACROS = frozenset({1, 3, 5, 6, 7}) # paper trading EOD (shadow_signal.py)

# ── Parametres de trading ─────────────────────────────────────────
SL_PCT        = 0.006
RR            = 2.5
EXIT_HM       = 960    # 16:00 ET en minutes depuis minuit
Q_THRESHOLD   = 0.0
ALIGNED_ONLY  = True
SKIP_DAYS     = frozenset({0})  # 0 = lundi

# ── Regles macro ──────────────────────────────────────────────────
# (mac_idx, lc, pc) -> frozenset des sc autorises
# frozenset() vide = scenario bloque
MACRO_RULES = {
    (2, 1, 1): frozenset({1}),  # 09:50 + RAID_H + BSL_swept -> SWEEP_H seulement
    (2, 0, 1): frozenset(),     # 09:50 + NO_RAID + BSL_swept -> bloque
}

# ── Simulation ────────────────────────────────────────────────────
FEE  = 0.0005
SLIP = 0.0002
