"""
engine/entry_stats.py — Statistiques comparatives des techniques d'entree ICT.

Pour chaque technique, mesure dans le contexte des macros ICT :
  - Frequence (nb de setups detectes)
  - Win rate (le prix bouge dans la direction attendue > seuil)
  - Avg P&L si on entre sur ce signal
  - Comparaison avec le baseline (entree aleatoire en macro)

Techniques analysees :
  FVG      — Fair Value Gap (imbalance 3 bougies)
  OTE      — Optimal Trade Entry (retracement 62-79% Fibonacci)
  NWOG     — New Week Opening Gap
  Breaker  — Failed swing reteste
"""

import numpy as np
import pandas as pd
import pytz
from engine.stats_state import ET_TZ, MACROS

WIN_THRESHOLD = 0.0010   # 0.10% — mouvement minimum pour compter comme "win"


# ── Detecteurs de setup ────────────────────────────────────────

def detect_fvg(df: pd.DataFrame) -> int:
    """
    Detecte le FVG le plus recent dans df.
    Bullish FVG : low[i] > high[i-2]  (gap vers le haut)
    Bearish FVG : high[i] < low[i-2]  (gap vers le bas)
    Returns : 1=bullish, -1=bearish, 0=aucun
    """
    if len(df) < 3:
        return 0
    highs = df["high"].values.astype(float)
    lows  = df["low"].values.astype(float)
    for i in range(len(df) - 1, 1, -1):
        if lows[i]  > highs[i - 2]:
            return 1    # bullish FVG
        if highs[i] < lows[i - 2]:
            return -1   # bearish FVG
    return 0


def detect_ote(df: pd.DataFrame) -> int:
    """
    Detecte si le prix courant est dans la zone OTE (62-79% retracement).
    Identifie le swing dominant sur df puis verifie le retracement.
    Returns : 1=OTE bullish, -1=OTE bearish, 0=hors zone
    """
    if len(df) < 8:
        return 0

    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)

    swing_h  = float(highs.max())
    swing_l  = float(lows.min())
    rng      = swing_h - swing_l

    if rng < 0.0005 * swing_l:
        return 0

    idx_h = int(highs.argmax())
    idx_l = int(lows.argmin())
    price = closes[-1]

    # Uptrend : low d'abord, puis high → OTE = retracement 62-79% depuis le high
    if idx_l < idx_h:
        fib_lo = swing_h - 0.786 * rng
        fib_hi = swing_h - 0.618 * rng
        if fib_lo <= price <= fib_hi:
            return 1

    # Downtrend : high d'abord, puis low → OTE = retracement 62-79% depuis le low
    if idx_h < idx_l:
        fib_lo = swing_l + 0.618 * rng
        fib_hi = swing_l + 0.786 * rng
        if fib_lo <= price <= fib_hi:
            return -1

    return 0


def detect_nwog(sun_open: float, fri_close: float, price: float) -> int:
    """
    Detecte la position du prix par rapport au NWOG.
    Returns : 1=prix sous le gap (aimant vers le haut), -1=prix au-dessus (aimant vers le bas), 0=dans le gap / gap trop petit
    """
    if sun_open is None or fri_close is None:
        return 0
    top    = max(sun_open, fri_close)
    bottom = min(sun_open, fri_close)
    if (top - bottom) < 0.0001 * price:   # gap < 0.01%, negligeable
        return 0
    if price < bottom:
        return 1    # price sous le gap → fill vers le haut
    if price > top:
        return -1   # price au-dessus → fill vers le bas
    return 0


def detect_breaker(df: pd.DataFrame) -> int:
    """
    Detecte un breaker block : swing high/low pris puis reteste.
    Returns : 1=bullish breaker, -1=bearish breaker, 0=aucun
    """
    if len(df) < 12:
        return 0

    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n      = len(df)
    price  = closes[-1]

    for i in range(2, n - 3):
        # Swing low local → candidat bullish breaker
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            # Ce swing low a-t-il ete swept ?
            if lows[i + 1:].min() < lows[i]:
                # Le prix est-il dans la range du candle i (zone breaker) ?
                if lows[i] <= price <= highs[i]:
                    return 1

        # Swing high local → candidat bearish breaker
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            if highs[i + 1:].max() > highs[i]:
                if lows[i] <= price <= highs[i]:
                    return -1

    return 0


# ── Calcul NWOG hebdomadaire ───────────────────────────────────

# Approximation CME BTC futures : ferme ven 17:00 ET, rouvre dim 19:00 ET
_CME_CLOSE_HM = 17 * 60   # 1020 min = 17:00 ET (16:00 CT)
_CME_OPEN_HM  = 19 * 60   # 1140 min = 19:00 ET (18:00 CT)
_CME_TOL      = 5          # tolerance ±5 min si bougie exacte absente

def _build_nwog_map(df: pd.DataFrame) -> dict:
    """
    Construit un dict date -> (sunday_open, friday_close) pour toutes les semaines.
    Approxime le NWOG CME : close vendredi 17:00 ET -> open dimanche 19:00 ET.
    Sur Binance (24/7), le vrai gap CME est importe via la correlation ES/BTC.
    """
    et_tz = pytz.timezone("America/New_York")
    df = df.copy()
    df["ts_et"]    = df["timestamp"].dt.tz_convert(et_tz)
    df["date_et"]  = df["ts_et"].dt.date
    df["dow"]      = df["ts_et"].dt.dayofweek   # 0=Lun 4=Ven 6=Dim
    df["hm_et"]    = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["iso_year"] = df["ts_et"].dt.isocalendar().year
    df["iso_week"] = df["ts_et"].dt.isocalendar().week

    nwog = {}

    for (yr, wk), grp in df.groupby(["iso_year", "iso_week"]):
        fri = grp[grp["dow"] == 4]
        sun = grp[grp["dow"] == 6]
        if fri.empty or sun.empty:
            continue

        # Vendredi : bougie la plus proche de 17:00 ET (fermeture CME)
        fri_near = fri[abs(fri["hm_et"] - _CME_CLOSE_HM) <= _CME_TOL]
        if fri_near.empty:
            continue
        fri_close = float(
            fri_near.iloc[fri_near["hm_et"].sub(_CME_CLOSE_HM).abs().argmin()]["close"]
        )

        # Dimanche : bougie la plus proche de 19:00 ET (reouverture CME)
        sun_near = sun[abs(sun["hm_et"] - _CME_OPEN_HM) <= _CME_TOL]
        if sun_near.empty:
            continue
        sun_open = float(
            sun_near.iloc[sun_near["hm_et"].sub(_CME_OPEN_HM).abs().argmin()]["open"]
        )

        for d in grp["date_et"].unique():
            nwog[d] = (sun_open, fri_close)

    return nwog


# ── Analyse principale ─────────────────────────────────────────

def compute_entry_stats(df_1m: pd.DataFrame) -> dict:
    """
    Analyse les 4 techniques d'entree sur toutes les macros ICT disponibles.

    Returns dict avec, pour chaque technique :
        n        : nombre de setups detectes
        win_rate : % de trades gagnants (mouvement > WIN_THRESHOLD dans la bonne direction)
        avg_pnl  : P&L moyen (%)
        avg_win  : gain moyen sur les winners (%)
        avg_loss : perte moyenne sur les losers (%)
        pf       : profit factor
    """
    df = df_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    et_tz = pytz.timezone("America/New_York")
    df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
    df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"] = df["ts_et"].dt.date

    nwog_map = _build_nwog_map(df)

    buckets = {k: [] for k in ("baseline", "fvg", "ote", "nwog", "breaker")}

    for date, day_grp in df.groupby("date_et"):
        day_grp = day_grp.sort_values("ts_et").reset_index(drop=True)
        nwog_info = nwog_map.get(date)

        for mac_idx, (mac_start, mac_end) in MACROS.items():
            pre_start = mac_start - 20
            wide_start = mac_start - 50   # fenetre elargie pour breaker

            pre_mask  = (day_grp["hm_et"] >= pre_start)  & (day_grp["hm_et"] < mac_start)
            wide_mask = (day_grp["hm_et"] >= wide_start) & (day_grp["hm_et"] < mac_start)
            mac_mask  = (day_grp["hm_et"] >= mac_start)  & (day_grp["hm_et"] < mac_end)

            pre_df  = day_grp[pre_mask]
            wide_df = day_grp[wide_mask]
            mac_df  = day_grp[mac_mask]

            if len(pre_df) < 5 or len(mac_df) < 5:
                continue

            entry_px = float(mac_df.iloc[0]["open"])
            exit_px  = float(mac_df.iloc[-1]["close"])
            move     = (exit_px - entry_px) / entry_px   # P&L long brut

            # Baseline : toutes les entrees long/short en debut de macro
            buckets["baseline"].append(move)

            # FVG
            fvg_dir = detect_fvg(pre_df)
            if fvg_dir != 0:
                buckets["fvg"].append(fvg_dir * move)

            # OTE
            ote_dir = detect_ote(pre_df)
            if ote_dir != 0:
                buckets["ote"].append(ote_dir * move)

            # NWOG
            if nwog_info is not None:
                nwog_dir = detect_nwog(nwog_info[0], nwog_info[1], entry_px)
                if nwog_dir != 0:
                    buckets["nwog"].append(nwog_dir * move)

            # Breaker (fenetre elargie)
            brk_dir = detect_breaker(wide_df)
            if brk_dir != 0:
                buckets["breaker"].append(brk_dir * move)

    results = {}
    for name, pnls in buckets.items():
        arr = np.array(pnls)
        n   = len(arr)
        if n == 0:
            results[name] = {"n": 0, "win_rate": 0, "avg_pnl": 0,
                             "avg_win": 0, "avg_loss": 0, "pf": 0}
            continue

        wins   = arr[arr >  WIN_THRESHOLD]
        losses = arr[arr < -WIN_THRESHOLD]
        neutral= arr[np.abs(arr) <= WIN_THRESHOLD]

        pf = abs(float(wins.sum()) / float(losses.sum())) if losses.sum() != 0 else float("inf")

        results[name] = {
            "n":        n,
            "win_rate": round(len(wins) / n * 100, 1),
            "avg_pnl":  round(float(arr.mean())    * 100, 4),
            "avg_win":  round(float(wins.mean())   * 100, 4) if len(wins)   > 0 else 0.0,
            "avg_loss": round(float(losses.mean()) * 100, 4) if len(losses) > 0 else 0.0,
            "pf":       round(pf, 3),
            "n_neutral":len(neutral),
        }

    return results


def print_entry_stats(results: dict):
    print("\n" + "=" * 68)
    print(f"  {'Technique':<12} {'N':>6} {'Win%':>7} {'AvgP&L%':>9} {'AvgWin%':>9} {'AvgLoss%':>9} {'PF':>6}")
    print("  " + "-" * 66)
    for name in ("baseline", "fvg", "ote", "nwog", "breaker"):
        r = results.get(name, {})
        if r.get("n", 0) == 0:
            print(f"  {name:<12} {'—':>6}")
            continue
        print(
            f"  {name:<12} {r['n']:>6} {r['win_rate']:>6.1f}% "
            f"{r['avg_pnl']:>+8.4f}% {r['avg_win']:>+8.4f}% "
            f"{r['avg_loss']:>+8.4f}% {r['pf']:>6.3f}"
        )
    print("=" * 68)
    print(f"  Seuil win : mouvement > {WIN_THRESHOLD*100:.2f}% dans la direction du signal")
    print("=" * 68)
