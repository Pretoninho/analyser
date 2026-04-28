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

# ── Conditions statistiques validees OOS ──────────────────────────
# (mac_idx, lc, sc, pc) -> (direction: +1 LONG / -1 SHORT, label)
# Ces regles court-circuitent la Q-table quand la condition est active.
#
#   C1 — NO_RAID x SWEEP_L x SSL_SWEPT -> SHORT
#        test N=26  WR=50%  avg=+0.300%  total=+7.79%  (3 macros)
#   C2 — RAID_H  x SWEEP_H x NEUTRAL   -> LONG
#        test N=17  WR=53%  avg=+0.289%  total=+4.92%  (2 macros)
#   C3 — RAID_H  x SWEEP_H x BSL_SWEPT -> LONG
#        test N=4   WR=100% avg=+1.202%  total=+4.81%  (2 macros, N faible)
CONDITION_RULES = {
    # C1 — macros 08:50, 10:50, 14:50
    (1, 0, 2, 2): (-1, "C1"),
    (3, 0, 2, 2): (-1, "C1"),
    (7, 0, 2, 2): (-1, "C1"),
    # C2 — macros 10:50, 12:50
    (3, 1, 1, 0): (+1, "C2"),
    (5, 1, 1, 0): (+1, "C2"),
    # C3 — macros 08:50, 10:50
    (1, 1, 1, 1): (+1, "C3"),
    (3, 1, 1, 1): (+1, "C3"),
}

# ── Simulation ────────────────────────────────────────────────────
FEE  = 0.0005
SLIP = 0.0002
