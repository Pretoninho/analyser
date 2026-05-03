# ── TA Strategy — configuration centrale ────────────────────────────────────
# Isolé de Pi*. Aucun import depuis engine/ ou pi_config.py.

import os
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data_binance" / "raw"
RESULTS_DIR = ROOT / "strategies" / "ta" / "results"
SYMBOL      = os.getenv("BINANCE_SYMBOL", "BTCUSDT").upper()

# ── Espace paramétrique — sweep exhaustif ────────────────────────────────────
# EMA appliqué sur les bougies 4H (biais directionnel)
EMA_LENGTHS   = [21, 50, 100, 200]

# RSI appliqué sur les bougies 15m
RSI_LENGTHS   = [7, 14, 21]

# Stochastique (k_period, smooth_k, d_period) appliqué sur 15m
STOCH_PARAMS  = [
    (5,  3, 3),
    (14, 3, 3),
    (21, 5, 5),
]

# ATR appliqué sur 15m — utilisé comme feature (régime volatilité)
# La taille de window de normalisation est toujours 20 périodes
ATR_LENGTHS   = [7, 14, 21]

# ── TP / SL (standardisé, indépendant du sweep) ──────────────────────────────
# ATR_14 sur 15m est utilisé pour dimensionner TP et SL.
# Cela permet de calculer les outcomes UNE seule fois pour tous les combos.
TP_MULT      = 2.0   # TP = entry ± TP_MULT × ATR_14
SL_MULT      = 1.0   # SL = entry ∓ SL_MULT × ATR_14
TP_SL_ATR    = 14    # longueur ATR fixe pour le sizing
MAX_BARS     = 48    # timeout = 48 bougies 15m = 12h

# ── Sessions de trading (UTC) ─────────────────────────────────────────────────
# Seules les bougies dans ces fenêtres génèrent des signaux
# (start_hour inclus, end_hour exclus)
SESSIONS_UTC = [(7, 11), (13, 17)]   # London + NY open

# ── Filtres de validation OOS ────────────────────────────────────────────────
MIN_TRADES   = 20     # nombre minimum de trades pour un état valide
MIN_WR       = 0.55   # win rate minimum (55%)
MIN_EXP      = 0.10   # expectancy minimum en R (> 0.10R)

# ── Discrétisation des features ──────────────────────────────────────────────
# RSI  : 4 buckets — [0,30[ oversold | [30,50[ weak | [50,70[ strong | [70,100] overbought
RSI_BINS    = [0, 30, 50, 70, 100]
RSI_LABELS  = ["oversold", "weak", "strong", "overbought"]

# Stoch : 4 buckets — identique RSI
STOCH_BINS   = [0, 20, 50, 80, 100]
STOCH_LABELS = ["oversold", "weak", "strong", "overbought"]

# ATR ratio (atr / rolling_mean_atr_20) : 3 buckets
ATR_BINS    = [0.0, 0.8, 1.3, float("inf")]
ATR_LABELS  = ["compression", "neutral", "expansion"]

# ── Regime macro (EMA200 daily) ───────────────────────────────────────────────
# Longueur EMA et période de slope pour la détection bull/bear/range
REGIME_EMA_LEN   = 200   # EMA200 daily
REGIME_SLOPE_DAYS = 5    # slope = diff(ema, 5 jours) > 0 → montant

# ── Rolling sweep ─────────────────────────────────────────────────────────────
# Fenêtre glissante de refit : on re-sweepke sur les ROLLING_WINDOW_MONTHS
# derniers mois de données à chaque run
ROLLING_WINDOW_MONTHS = 18   # 18 mois de données pour le refit
