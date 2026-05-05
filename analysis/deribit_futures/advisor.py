"""
analysis/deribit_futures/advisor.py — Options Advisor multi-asset.

Produit une recommandation options actionnable basée sur :
  - IVP (IV Percentile 52 semaines via DVOL Deribit)
  - DVOL state + régime
  - Signal directionnel Deribit futures
  - Vol premium (IV implicite vs vol réalisée)

Actifs supportés nativement (DVOL) : BTC, ETH
Autres actifs : IVP None, fallback vol_premium
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

VolRegime    = Literal["SELL_VOL", "BUY_VOL", "NEUTRAL"]
DirectBias   = Literal["BULLISH", "BEARISH", "NEUTRAL"]
StrategyName = Literal[
    "SHORT_STRANGLE", "SHORT_PUT", "SHORT_CALL",
    "IRON_CONDOR", "LONG_STRADDLE", "LONG_STRANGLE",
    "BULL_CALL_SPREAD", "BEAR_PUT_SPREAD", "WAIT",
]

# Actifs avec DVOL natif Deribit
DVOL_ASSETS = {"BTC", "ETH"}

# ── Normale inverse (Acklam) — sans scipy ─────────────────────────────────────

def _norm_ppf(p: float) -> float:
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plo, phi = 0.02425, 1.0 - 0.02425
    if p < plo:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p <= phi:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


# ── IVP ──────────────────────────────────────────────────────────────────────

def compute_ivp(asset: str, days: int = 365) -> float | None:
    """
    IV Percentile : rang de l'IV courante vs les `days` derniers jours de DVOL daily.
    Retourne [0,1] — 0=IV historiquement basse, 1=haute. None si DVOL indisponible.
    """
    if asset.upper() not in DVOL_ASSETS:
        return None
    try:
        from data.deribit import fetch_dvol_history
        df = fetch_dvol_history(asset=asset.upper(), timeframe="1d", days=days)
        if df.empty or len(df) < 30:
            return None
        close = df["dvol_close"].astype(float)
        current = float(close.iloc[-1])
        ivp = float((close < current).sum()) / len(close)
        return round(ivp, 4)
    except Exception as e:
        print(f"[advisor] IVP {asset} erreur: {e}")
        return None


# ── Régimes ───────────────────────────────────────────────────────────────────

def _vol_regime(ivp: float | None, vol_premium_bias: str = "NEUTRAL") -> VolRegime:
    if ivp is not None:
        if ivp > 0.70:
            return "SELL_VOL"
        if ivp < 0.30:
            return "BUY_VOL"
        return "NEUTRAL"
    # Fallback: vol_premium quand DVOL indisponible
    if vol_premium_bias == "SELL_VOL":
        return "SELL_VOL"
    if vol_premium_bias == "BUY_VOL":
        return "BUY_VOL"
    return "NEUTRAL"


def _directional_bias(signal_action: str, dvol_state: str) -> DirectBias:
    if signal_action == "LONG":
        return "BULLISH"
    if signal_action == "SHORT":
        return "BEARISH"
    # WATCH/FLAT : utiliser DVOL comme biais secondaire
    if dvol_state == "VOL_CRUSH_DOWN":
        return "BULLISH"
    if dvol_state == "VOL_SHOCK_UP":
        return "BEARISH"
    return "NEUTRAL"


# ── Matrice stratégies ────────────────────────────────────────────────────────

_STRATEGY_MATRIX: dict[tuple[str, str], StrategyName] = {
    ("SELL_VOL", "BULLISH"): "SHORT_PUT",
    ("SELL_VOL", "BEARISH"): "SHORT_CALL",
    ("SELL_VOL", "NEUTRAL"): "SHORT_STRANGLE",
    ("BUY_VOL",  "BULLISH"): "BULL_CALL_SPREAD",
    ("BUY_VOL",  "BEARISH"): "BEAR_PUT_SPREAD",
    ("BUY_VOL",  "NEUTRAL"): "LONG_STRADDLE",
    ("NEUTRAL",  "BULLISH"): "BULL_CALL_SPREAD",
    ("NEUTRAL",  "BEARISH"): "BEAR_PUT_SPREAD",
    ("NEUTRAL",  "NEUTRAL"): "IRON_CONDOR",
}

_STRATEGY_META: dict[str, dict] = {
    "SHORT_PUT":        {"fr": "Short Put OTM",       "risk": "LIMITÉ",    "color": "emerald", "verb": "Vendre vol, biais haussier"},
    "SHORT_CALL":       {"fr": "Short Call OTM",      "risk": "LIMITÉ",    "color": "emerald", "verb": "Vendre vol, biais baissier"},
    "SHORT_STRANGLE":   {"fr": "Short Strangle",      "risk": "ILLIMITÉ",  "color": "amber",   "verb": "Vendre vol des deux côtés"},
    "BULL_CALL_SPREAD": {"fr": "Bull Call Spread",    "risk": "LIMITÉ",    "color": "cyan",    "verb": "Acheter directionalité haussière"},
    "BEAR_PUT_SPREAD":  {"fr": "Bear Put Spread",     "risk": "LIMITÉ",    "color": "cyan",    "verb": "Acheter directionalité baissière"},
    "LONG_STRADDLE":    {"fr": "Long Straddle",       "risk": "LIMITÉ",    "color": "violet",  "verb": "Acheter vol, breakout attendu"},
    "LONG_STRANGLE":    {"fr": "Long Strangle",       "risk": "LIMITÉ",    "color": "violet",  "verb": "Acheter vol bon marché"},
    "IRON_CONDOR":      {"fr": "Iron Condor",         "risk": "LIMITÉ",    "color": "slate",   "verb": "Range play, vol neutre"},
    "WAIT":             {"fr": "Attente",              "risk": "AUCUN",     "color": "slate",   "verb": "Signal ambigu, pas de position"},
}


# ── Calcul des strikes cibles ─────────────────────────────────────────────────

def _strike_at_delta(spot: float, iv_dec: float, dte_days: float, target_delta: float) -> float:
    """Strike BS approché pour un delta cible. Pas de scipy requis."""
    T = dte_days / 365.0
    if T <= 0 or iv_dec <= 0 or spot <= 0:
        return spot
    is_put = target_delta < 0
    abs_d = abs(target_delta)
    d1_target = _norm_ppf(1.0 - abs_d) if is_put else _norm_ppf(abs_d)
    sigma_sqrtT = iv_dec * math.sqrt(T)
    log_SK = d1_target * sigma_sqrtT - 0.5 * iv_dec ** 2 * T
    K = spot * math.exp(-log_SK)
    # Arrondir à la centaine la plus proche (pour BTC) ou 10 (ETH/SOL)
    step = 100 if spot > 5000 else (10 if spot > 100 else 1)
    return round(K / step) * step


def _build_legs(strategy: str, spot: float, iv_pct: float, dte_days: int) -> list[dict]:
    iv = iv_pct / 100.0
    atm  = _strike_at_delta(spot, iv, dte_days, 0.50)
    c25  = _strike_at_delta(spot, iv, dte_days, 0.25)
    p25  = _strike_at_delta(spot, iv, dte_days, -0.25)
    c10  = _strike_at_delta(spot, iv, dte_days, 0.10)
    p10  = _strike_at_delta(spot, iv, dte_days, -0.10)

    legs_map: dict[str, list[dict]] = {
        "SHORT_PUT":        [{"action": "SELL", "type": "PUT",  "strike": p25,  "dte": dte_days, "delta": "−0.25"}],
        "SHORT_CALL":       [{"action": "SELL", "type": "CALL", "strike": c25,  "dte": dte_days, "delta": "+0.25"}],
        "SHORT_STRANGLE":   [
            {"action": "SELL", "type": "PUT",  "strike": p25,  "dte": dte_days, "delta": "−0.25"},
            {"action": "SELL", "type": "CALL", "strike": c25,  "dte": dte_days, "delta": "+0.25"},
        ],
        "LONG_STRADDLE":    [
            {"action": "BUY",  "type": "CALL", "strike": atm,  "dte": dte_days, "delta": "~+0.50"},
            {"action": "BUY",  "type": "PUT",  "strike": atm,  "dte": dte_days, "delta": "~−0.50"},
        ],
        "LONG_STRANGLE":    [
            {"action": "BUY",  "type": "PUT",  "strike": p25,  "dte": dte_days, "delta": "−0.25"},
            {"action": "BUY",  "type": "CALL", "strike": c25,  "dte": dte_days, "delta": "+0.25"},
        ],
        "BULL_CALL_SPREAD":  [
            {"action": "BUY",  "type": "CALL", "strike": atm,  "dte": dte_days, "delta": "~+0.50"},
            {"action": "SELL", "type": "CALL", "strike": c25,  "dte": dte_days, "delta": "+0.25"},
        ],
        "BEAR_PUT_SPREAD":   [
            {"action": "BUY",  "type": "PUT",  "strike": atm,  "dte": dte_days, "delta": "~−0.50"},
            {"action": "SELL", "type": "PUT",  "strike": p25,  "dte": dte_days, "delta": "−0.25"},
        ],
        "IRON_CONDOR":      [
            {"action": "SELL", "type": "PUT",  "strike": p25,  "dte": dte_days, "delta": "−0.25"},
            {"action": "BUY",  "type": "PUT",  "strike": p10,  "dte": dte_days, "delta": "−0.10"},
            {"action": "SELL", "type": "CALL", "strike": c25,  "dte": dte_days, "delta": "+0.25"},
            {"action": "BUY",  "type": "CALL", "strike": c10,  "dte": dte_days, "delta": "+0.10"},
        ],
        "WAIT": [],
    }
    return legs_map.get(strategy, [])


def _build_rationale(
    strategy: str,
    ivp: float | None,
    vol_regime: str,
    directional_bias: str,
    dvol_state: str,
    skew: float | None,
    signal_action: str,
) -> str:
    parts: list[str] = []

    if ivp is not None:
        parts.append(f"IVP {ivp*100:.0f}% → vol {'chère' if vol_regime == 'SELL_VOL' else 'bon marché' if vol_regime == 'BUY_VOL' else 'neutre'}")
    else:
        parts.append("DVOL indisponible → fallback vol premium")

    if signal_action not in ("WATCH", "FLAT"):
        parts.append(f"signal {signal_action}")

    if dvol_state != "NEUTRAL":
        parts.append(f"DVOL {dvol_state.replace('_', ' ').lower()}")

    if skew is not None and abs(skew) > 2.0:
        side = "puts chers" if skew > 0 else "calls chers"
        parts.append(f"skew {skew:+.1f}% ({side})")

    meta = _STRATEGY_META.get(strategy, {})
    return f"{meta.get('verb', strategy)}. " + " · ".join(parts) + "."


# ── Point d'entrée principal ──────────────────────────────────────────────────

def compute_advisor(asset: str = "BTC", timeframe: str = "1h", days: int = 60) -> dict:
    """
    Recommandation options actionnable pour `asset`.
    BTC/ETH : IVP natif via DVOL Deribit.
    Autres actifs : IVP None, fallback vol_premium.
    """
    from .dvol import DvolDetectorConfig, detect_dvol_variation
    from .signal import SignalConfig, build_deribit_signal

    asset = asset.upper()

    # 1. IVP
    ivp = compute_ivp(asset, days=365)

    # 2. DVOL
    dvol_state  = "NEUTRAL"
    dvol_close  = None
    dvol_z      = None
    try:
        dvol = detect_dvol_variation(DvolDetectorConfig(asset=asset, timeframe=timeframe, days=days))
        dvol.pop("frame", None)
        dvol_state = dvol.get("dvol_state", "NEUTRAL")
        dvol_close = dvol.get("dvol_close")
        dvol_z     = dvol.get("dvol_z")
    except Exception as e:
        print(f"[advisor] DVOL {asset}: {e}")

    # 3. Signal directionnel + snapshot options
    signal_action   = "WATCH"
    spot            = None
    iv_atm          = None
    skew            = None
    realized_vol    = None
    vol_premium_bias = "NEUTRAL"
    try:
        sig = build_deribit_signal(SignalConfig(asset=asset, timeframe=timeframe, days=90))
        signal_action = sig.get("signal", {}).get("action", "WATCH")
        spot          = sig.get("close")
        opts          = sig.get("options") or {}
        iv_atm        = opts.get("iv_atm")
        skew          = opts.get("iv_skew_25d")
        realized_vol  = sig.get("realized_vol_annual")
    except Exception as e:
        print(f"[advisor] Signal {asset}: {e}")

    # 4. Vol premium bias (fallback si IVP indisponible)
    if iv_atm and realized_vol:
        try:
            premium = float(iv_atm) / 100.0 - float(realized_vol)
            if premium > 0.05:
                vol_premium_bias = "SELL_VOL"
            elif premium < -0.05:
                vol_premium_bias = "BUY_VOL"
        except Exception:
            pass

    # 5. Régimes
    vol_regime       = _vol_regime(ivp, vol_premium_bias)
    directional_bias = _directional_bias(signal_action, dvol_state)

    # 6. Stratégie
    if not spot or not iv_atm:
        strategy = "WAIT"
    else:
        strategy = _STRATEGY_MATRIX.get((vol_regime, directional_bias), "IRON_CONDOR")

    # 7. DTE cible : 21j pour vente, 35j pour achat
    dte_days = 21 if vol_regime in ("SELL_VOL", "NEUTRAL") else 35

    legs = _build_legs(strategy, float(spot or 0), float(iv_atm or 0), dte_days) if strategy != "WAIT" else []

    meta      = _STRATEGY_META.get(strategy, {})
    rationale = _build_rationale(strategy, ivp, vol_regime, directional_bias, dvol_state, skew, signal_action)

    return {
        "asset":              asset,
        "timestamp":          datetime.utcnow().isoformat(),
        "dvol_asset_supported": asset in DVOL_ASSETS,
        "ivp":                ivp,
        "ivp_pct":            round(ivp * 100, 1) if ivp is not None else None,
        "vol_regime":         vol_regime,
        "directional_bias":   directional_bias,
        "signal_action":      signal_action,
        "dvol_state":         dvol_state,
        "dvol_close":         dvol_close,
        "dvol_z":             dvol_z,
        "spot":               spot,
        "iv_atm":             iv_atm,
        "skew_25d":           skew,
        "strategy":           strategy,
        "strategy_label":     meta.get("fr", strategy),
        "risk_profile":       meta.get("risk", "LIMITÉ"),
        "color":              meta.get("color", "slate"),
        "dte_days":           dte_days,
        "legs":               legs,
        "rationale":          rationale,
    }
