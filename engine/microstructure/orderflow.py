"""
engine/orderflow.py — Order Flow features pour backtest Pi*.

Implémente les variables issues du framework OFI (Order Flow Imbalance)
calibrées pour données Binance 1m (taker_buy_vol disponible).

Variables produites :
  ofi_norm       : déséquilibre net acheteurs/vendeurs  [-1, +1]
  beta_proxy     : sensibilité prix / OFI (market thinness)
  hawkes_lambda  : intensité Hawkes (clustering de volatilité)
  depth_proxy    : profondeur de marché implicite
  sigma_ewma     : variance EWMA des returns (microvolatilité)
  lri            : Liquidity Replenishment Indicator (multi-fenêtre)

Toutes les fonctions acceptent un DataFrame 1m Binance standard
(colonnes : open, high, low, close, volume, taker_buy_vol).

Usage :
    from engine.orderflow import compute_orderflow_features
    df = compute_orderflow_features(df_1m)
    # -> colonnes ofi_norm, beta_proxy, hawkes_lambda, depth_proxy,
    #    sigma_ewma, lri ajoutées in-place (copy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Constantes par défaut ───────────────────────────────────────────────────

# OFI
_OFI_EPS = 1e-9           # évite division par zéro

# Beta proxy (rolling OLS)
_BETA_WINDOW = 20         # nombre de bougies pour la rolling regression

# Hawkes
_HAWKES_MU    = 0.1       # taux baseline (proportion de bougies actives)
_HAWKES_ALPHA = 0.5       # amplitude du spike sur choc
_HAWKES_BETA  = 0.3       # vitesse de décroissance (par bougie)
_HAWKES_SHOCK_THRESHOLD = 0.5   # |ofi_norm| > seuil -> choc

# Depth proxy
_DEPTH_EPS  = 1e-9
_DEPTH_CLIP = 1e6         # cap pour éviter les infinis sur bougies plates

# Variance EWMA (style RiskMetrics)
_EWMA_LAMBDA = 0.94

# LRI (Liquidity Replenishment Indicator)
_LRI_SHORT = 3            # fenêtre courte (bougies)
_LRI_LONG  = 20           # fenêtre longue (bougies)


# ── 1. OFI normalisé ───────────────────────────────────────────────────────

def compute_ofi_norm(df: pd.DataFrame) -> pd.Series:
    """
    OFI agrégé à partir du taker flow Binance.

    Formule :
        ofi_norm = (taker_buy_vol - taker_sell_vol) / (volume + eps)
                 = (2 * taker_buy_vol - volume) / (volume + eps)

    Résultat dans [-1, +1] :
        +1  = 100% market buy orders (pression acheteuse maximale)
        -1  = 100% market sell orders (pression vendeuse maximale)
         0  = neutre

    Args:
        df : DataFrame 1m avec colonnes 'taker_buy_vol' et 'volume'.

    Returns:
        Series float32 'ofi_norm'.
    """
    buy  = df["taker_buy_vol"]
    vol  = df["volume"]
    sell = vol - buy

    ofi = (buy - sell) / (vol + _OFI_EPS)
    return ofi.clip(-1.0, 1.0).astype("float32").rename("ofi_norm")


# ── 2. Beta proxy (price impact coefficient) ──────────────────────────────

def compute_beta_proxy(
    df: pd.DataFrame,
    ofi_norm: pd.Series | None = None,
    window: int = _BETA_WINDOW,
) -> pd.Series:
    """
    Estime le coefficient d'impact prix = Cov(delta_close, ofi_norm) / Var(ofi_norm)
    sur une fenêtre glissante.

    Interprétation :
        beta élevé  -> book mince, volatilité imminente
        beta faible -> liquidité profonde, absorption forte

    Args:
        df       : DataFrame 1m avec colonne 'close'.
        ofi_norm : Series ofi_norm précalculée (si None, calculée ici).
        window   : taille de la fenêtre rolling (défaut 20 bougies).

    Returns:
        Series float32 'beta_proxy'.
    """
    if ofi_norm is None:
        ofi_norm = compute_ofi_norm(df)

    delta_close = df["close"].pct_change().fillna(0)

    # rolling covariance / rolling variance
    roll_cov = delta_close.rolling(window).cov(ofi_norm)
    roll_var = ofi_norm.rolling(window).var()

    beta = (roll_cov / (roll_var + _OFI_EPS)).fillna(0)
    return beta.astype("float32").rename("beta_proxy")


# ── 3. Hawkes intensity (clustering de volatilité) ────────────────────────

def compute_hawkes_lambda(
    ofi_norm: pd.Series,
    mu: float = _HAWKES_MU,
    alpha: float = _HAWKES_ALPHA,
    beta_decay: float = _HAWKES_BETA,
    shock_threshold: float = _HAWKES_SHOCK_THRESHOLD,
) -> pd.Series:
    """
    Intensité Hawkes discrétisée — update récursif O(n).

    Modélise la self-excitation : chaque choc OFI élève la probabilité
    du prochain choc, décroissant exponentiellement vers le baseline mu.

    Formule récursive :
        lambda(t) = mu + exp(-beta_decay) * (lambda(t-1) - mu)
                    + alpha * 1[|ofi_norm(t-1)| > shock_threshold]

    Args:
        ofi_norm        : Series ofi_norm dans [-1, +1].
        mu              : taux baseline (proportion attendue de chocs).
        alpha           : amplitude du spike après un choc.
        beta_decay      : vitesse d'oubli (par bougie).
        shock_threshold : seuil |ofi_norm| pour déclarer un choc.

    Returns:
        Series float32 'hawkes_lambda' (>= mu).
    """
    n = len(ofi_norm)
    values = ofi_norm.values
    result = np.empty(n, dtype="float32")

    decay = np.exp(-beta_decay)
    lam   = mu  # état initial

    for i in range(n):
        # decay de l'état précédent
        lam = mu + decay * (lam - mu)
        # choc si bougie précédente dépasse le seuil
        if i > 0 and abs(values[i - 1]) > shock_threshold:
            lam += alpha
        result[i] = lam

    return pd.Series(result, index=ofi_norm.index, name="hawkes_lambda")


# ── 4. Depth proxy (profondeur implicite) ────────────────────────────────

def compute_depth_proxy(
    df: pd.DataFrame,
    window: int = _BETA_WINDOW,
) -> pd.Series:
    """
    Profondeur implicite = volume / |delta_close|, en rolling moyen.

    Interprétation :
        élevé  -> marché profond, ordres absorbés sans mouvement
        faible -> marché mince, chaque order déplace le prix

    Args:
        df     : DataFrame 1m avec colonnes 'close' et 'volume'.
        window : fenêtre de lissage (défaut 20 bougies).

    Returns:
        Series float32 'depth_proxy'.
    """
    delta_close = df["close"].diff().abs().replace(0, _DEPTH_EPS)
    raw_depth   = (df["volume"] / delta_close).clip(upper=_DEPTH_CLIP)

    # lissage rolling pour réduire le bruit des bougies plates
    depth = raw_depth.rolling(window).mean().fillna(0)
    return depth.astype("float32").rename("depth_proxy")


# ── 5. Sigma EWMA (microvolatilité) ──────────────────────────────────────

def compute_sigma_ewma(
    df: pd.DataFrame,
    lam: float = _EWMA_LAMBDA,
) -> pd.Series:
    """
    Variance EWMA des log-returns (style RiskMetrics).

    Formule :
        r(t)    = log(close(t) / close(t-1))
        var(t)  = (1 - lam) * r(t)^2 + lam * var(t-1)
        sigma(t) = sqrt(var(t))

    Args:
        df  : DataFrame 1m avec colonne 'close'.
        lam : facteur de lissage EWMA (défaut 0.94 — standard RiskMetrics).

    Returns:
        Series float32 'sigma_ewma' (écart-type en log-return).
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0).values
    n       = len(log_ret)
    var     = np.empty(n, dtype="float64")

    var[0] = log_ret[0] ** 2
    for i in range(1, n):
        var[i] = (1 - lam) * log_ret[i] ** 2 + lam * var[i - 1]

    sigma = np.sqrt(var).astype("float32")
    return pd.Series(sigma, index=df.index, name="sigma_ewma")


# ── 6. LRI — Liquidity Replenishment Indicator ───────────────────────────

def compute_lri(
    ofi_norm: pd.Series,
    short_window: int = _LRI_SHORT,
    long_window: int  = _LRI_LONG,
) -> pd.Series:
    """
    Divergence entre OFI court terme et OFI long terme.

    Interprétation :
        LRI >> 0  : pression court terme sans confirmation long terme
                    -> risque de faux break, signal peu fiable
        LRI ~  0  : pression cohérente multi-fenêtre -> signal fiable
        LRI << 0  : momentum long terme non confirmé court terme
                    -> potentiel retournement

    Formule :
        lri = rolling_mean(ofi_norm, short) - rolling_mean(ofi_norm, long)

    Args:
        ofi_norm     : Series ofi_norm dans [-1, +1].
        short_window : fenêtre courte (défaut 3 bougies).
        long_window  : fenêtre longue (défaut 20 bougies).

    Returns:
        Series float32 'lri'.
    """
    ofi_short = ofi_norm.rolling(short_window).mean()
    ofi_long  = ofi_norm.rolling(long_window).mean()
    lri       = (ofi_short - ofi_long).fillna(0)
    return lri.astype("float32").rename("lri")


# ── Pipeline complet ──────────────────────────────────────────────────────

def compute_orderflow_features(
    df: pd.DataFrame,
    beta_window:     int   = _BETA_WINDOW,
    hawkes_mu:       float = _HAWKES_MU,
    hawkes_alpha:    float = _HAWKES_ALPHA,
    hawkes_beta:     float = _HAWKES_BETA,
    hawkes_shock:    float = _HAWKES_SHOCK_THRESHOLD,
    ewma_lambda:     float = _EWMA_LAMBDA,
    lri_short:       int   = _LRI_SHORT,
    lri_long:        int   = _LRI_LONG,
) -> pd.DataFrame:
    """
    Calcule toutes les features order flow et les ajoute au DataFrame.

    Colonnes ajoutées :
        ofi_norm      [-1, +1]   déséquilibre taker buy/sell
        beta_proxy    float      sensibilité prix / OFI (rolling)
        hawkes_lambda float      intensité Hawkes (>= mu)
        depth_proxy   float      profondeur implicite (rolling)
        sigma_ewma    float      volatilité EWMA (log-return std)
        lri           float      Liquidity Replenishment Indicator

    Requis : colonnes 'open', 'close', 'volume', 'taker_buy_vol'.

    Args:
        df           : DataFrame 1m Binance standard.
        beta_window  : fenêtre rolling pour beta et depth (bougies).
        hawkes_mu    : baseline Hawkes.
        hawkes_alpha : amplitude spike Hawkes.
        hawkes_beta  : décroissance Hawkes.
        hawkes_shock : seuil |ofi_norm| pour déclarer un choc Hawkes.
        ewma_lambda  : facteur lissage variance EWMA.
        lri_short    : fenêtre courte LRI.
        lri_long     : fenêtre longue LRI.

    Returns:
        Nouveau DataFrame avec les 6 colonnes order flow ajoutées.

    Example:
        from data.binance import load_binance_1m
        from engine.orderflow import compute_orderflow_features

        df = load_binance_1m()
        df = compute_orderflow_features(df)
        print(df[["close", "ofi_norm", "beta_proxy", "hawkes_lambda",
                   "depth_proxy", "sigma_ewma", "lri"]].tail(10))
    """
    out = df.copy()

    ofi = compute_ofi_norm(out)
    out["ofi_norm"]      = ofi
    out["beta_proxy"]    = compute_beta_proxy(out, ofi, window=beta_window)
    out["hawkes_lambda"] = compute_hawkes_lambda(
        ofi,
        mu=hawkes_mu,
        alpha=hawkes_alpha,
        beta_decay=hawkes_beta,
        shock_threshold=hawkes_shock,
    )
    out["depth_proxy"]   = compute_depth_proxy(out, window=beta_window)
    out["sigma_ewma"]    = compute_sigma_ewma(out, lam=ewma_lambda)
    out["lri"]           = compute_lri(ofi, short_window=lri_short, long_window=lri_long)

    return out


# ── Utilitaires backtest ──────────────────────────────────────────────────

def ofi_regime(
    df: pd.DataFrame,
    ofi_col: str = "ofi_norm",
    sigma_col: str = "sigma_ewma",
    hawkes_col: str = "hawkes_lambda",
    ofi_threshold: float = 0.3,
    sigma_high_pct: float = 0.75,
    hawkes_high_pct: float = 0.75,
) -> pd.Series:
    """
    Encode le régime order flow en 5 états discrets pour intégration Q-table.

    États :
        0 = NEUTRAL        (OFI faible, volatilité normale)
        1 = BUYING_QUIET   (OFI positif fort, hawkes faible)
        2 = BUYING_BURST   (OFI positif fort, hawkes élevé)
        3 = SELLING_QUIET  (OFI négatif fort, hawkes faible)
        4 = SELLING_BURST  (OFI négatif fort, hawkes élevé)

    Args:
        df               : DataFrame avec colonnes ofi_norm, sigma_ewma, hawkes_lambda.
        ofi_col          : nom colonne OFI normalisé.
        sigma_col        : nom colonne sigma EWMA.
        hawkes_col       : nom colonne intensité Hawkes.
        ofi_threshold    : seuil |ofi_norm| pour considérer un signal directionnel.
        sigma_high_pct   : percentile pour qualifier sigma comme "élevé".
        hawkes_high_pct  : percentile pour qualifier hawkes comme "élevé".

    Returns:
        Series int8 'ofi_regime' (valeurs 0-4).
    """
    ofi    = df[ofi_col]
    hawkes = df[hawkes_col]

    hawkes_threshold = hawkes.quantile(hawkes_high_pct)

    hawkes_high = hawkes > hawkes_threshold
    ofi_buy     = ofi >  ofi_threshold
    ofi_sell    = ofi < -ofi_threshold

    regime = pd.Series(0, index=df.index, dtype="int8", name="ofi_regime")
    regime[ofi_buy  & ~hawkes_high] = 1
    regime[ofi_buy  &  hawkes_high] = 2
    regime[ofi_sell & ~hawkes_high] = 3
    regime[ofi_sell &  hawkes_high] = 4

    return regime
