"""
config.py — Configuration centrale du moteur d'analyse.

Pour ajouter un actif : ajouter une entrée dans ASSETS.
Tout le reste s'adapte automatiquement.
"""

from pathlib import Path

# ── Répertoires ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "db"
DATA_DIR.mkdir(exist_ok=True)

PRICES_DB_PATH  = DATA_DIR / "prices.db"
RESULTS_DB_PATH = DATA_DIR / "results.db"

# ── Actifs suivis ──────────────────────────────────────────────
# Pour ajouter ETH : { "kraken": "XETHZUSD", "deribit": "ETH" }
ASSETS = {
    "BTC": {
        "kraken": "XBTUSD",       # symbole Kraken
        "deribit": "BTC",         # currency Deribit
        "deribit_index": "btc_usd",
    },
    # "ETH": {
    #     "kraken": "XETHZUSD",
    #     "deribit": "ETH",
    #     "deribit_index": "eth_usd",
    # },
}

# ── Timeframes disponibles (en minutes) ───────────────────────
TIMEFRAMES = {
    "1m":  1,
    "15m": 15,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
}

# ── Paramètres de collecte ─────────────────────────────────────
FETCH = {
    "ohlcv_limit":     720,   # nombre de bougies à récupérer
    "refresh_seconds": 900,   # intervalle de rafraîchissement (15min)
}

# ── Paramètres du moteur de calcul ────────────────────────────
ENGINE = {
    # Volatilité réalisée
    "vol_window_short":  20,    # jours
    "vol_window_long":   90,    # jours

    # Régimes (percentiles sur vol_window_long)
    "regime_low_pct":   33,
    "regime_high_pct":  66,

    # ATR
    "atr_period":       14,

    # Z-score
    "zscore_window":    30,     # jours de référence
}

# ── Sources ────────────────────────────────────────────────────
SOURCES = {
    "deribit_base_url": "https://www.deribit.com/api/v2/public",
}