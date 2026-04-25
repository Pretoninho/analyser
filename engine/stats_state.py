"""
engine/stats_state.py — Encodeur d'etat statistique ICT pour Pi*.

Dimensions :
    month_ctx  (3) : 0=WEAK  1=NEUTRAL  2=STRONG
    day_ctx    (3) : 0=WEAK  1=NEUTRAL  2=STRONG
    london_ctx (3) : 0=NO_RAID  1=RAID_HIGH  2=RAID_LOW
    macro_ctx  (8) : 0=NONE  1=08:50  2=09:50*  3=10:50  4=11:50  5=12:50  6=13:50  7=14:50
    sweep_ctx  (3) : 0=NO_SWEEP  1=SWEEP_HIGH  2=SWEEP_LOW
    pool_ctx   (3) : 0=NEUTRAL  1=BSL_SWEPT  2=SSL_SWEPT

Total : 3 * 3 * 3 * 8 * 3 * 3 = 1944 etats

Encodage : state = mc*648 + dc*216 + lc*72 + mac*9 + sc*3 + pc

Liquidite :
    BSL (Buy-Side Liquidity)  : au-dessus de London High ou PWH
    SSL (Sell-Side Liquidity) : en-dessous de London Low ou PWL
    Sweep BSL -> reversal baissier attendu (pool_ctx=1)
    Sweep SSL -> reversal haussier attendu (pool_ctx=2)
"""

import pytz
import pandas as pd

N_STATES  = 1944   # 3*3*3*8*3*3
N_ACTIONS = 3      # 0=FLAT  1=LONG  2=SHORT

ET_TZ = pytz.timezone('America/New_York')

# ICT Macros en minutes ET (heure*60 + minute)
MACROS = {
    1: (530, 550),   # 08:50-09:10
    2: (590, 610),   # 09:50-10:10  STAR
    3: (650, 670),   # 10:50-11:10
    4: (710, 730),   # 11:50-12:10
    5: (770, 790),   # 12:50-13:10
    6: (830, 850),   # 13:50-14:10
    7: (890, 910),   # 14:50-15:10
}

# Pre-macro : 20 min avant le debut de chaque macro
PRE_MACRO_WINDOWS = {
    idx: (start - 20, start)
    for idx, (start, _) in MACROS.items()
}

STRONG_MONTHS = {10, 7, 3, 1}
WEAK_MONTHS   = {2, 6, 9}
STRONG_DAYS   = {0, 2}   # Lundi, Mercredi
WEAK_DAYS     = set()     # Jeudi retire — laissons les donnees decider


def month_ctx(month: int) -> int:
    if month in STRONG_MONTHS: return 2
    if month in WEAK_MONTHS:   return 0
    return 1


def day_ctx(dow: int) -> int:
    if dow in STRONG_DAYS: return 2
    if dow in WEAK_DAYS:   return 0
    return 1


def macro_ctx(hm_et: int) -> int:
    for idx, (start, end) in MACROS.items():
        if start <= hm_et < end:
            return idx
    return 0


def encode(mc: int, dc: int, lc: int, mac: int, sc: int, pc: int = 0) -> int:
    return mc * 648 + dc * 216 + lc * 72 + mac * 9 + sc * 3 + pc


def decode(state: int) -> dict:
    pc  = state % 3;  state //= 3
    sc  = state % 3;  state //= 3
    mac = state % 8;  state //= 8
    lc  = state % 3;  state //= 3
    dc  = state % 3;  state //= 3
    mc  = state % 3
    return {
        "month_ctx": mc, "day_ctx": dc, "london_ctx": lc,
        "macro_ctx": mac, "sweep_ctx": sc, "pool_ctx": pc,
    }


def compute_pool_ctx(pre_high: float, pre_low: float,
                     london_h, london_l,
                     pwh, pwl) -> int:
    """
    Detecte si un pool de liquidite significatif a ete sweepé avant la macro.

    BSL swept (pool_ctx=1) : pre_high depasse London High ou PWH
                              -> reversal baissier attendu
    SSL swept (pool_ctx=2) : pre_low passe sous London Low ou PWL
                              -> reversal haussier attendu
    """
    bsl_levels = [l for l in (london_h, pwh) if l is not None]
    ssl_levels = [l for l in (london_l, pwl) if l is not None]

    bsl_swept = any(pre_high > lvl for lvl in bsl_levels)
    ssl_swept = any(pre_low  < lvl for lvl in ssl_levels)

    if bsl_swept and not ssl_swept:
        return 1   # BSL cleared
    if ssl_swept and not bsl_swept:
        return 2   # SSL cleared
    return 0


def build_weekly_levels(df_1m: pd.DataFrame) -> dict:
    """
    Calcule le PWH/PWL (Previous Week High/Low) pour chaque date.
    Utilise les semaines ISO (lundi-dimanche).
    Returns : {date -> (pwh, pwl)}
    """
    df = df_1m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    et_tz = pytz.timezone("America/New_York")
    df["ts_et"]    = df["timestamp"].dt.tz_convert(et_tz)
    df["date_et"]  = df["ts_et"].dt.date
    df["iso_year"] = df["ts_et"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["ts_et"].dt.isocalendar().week.astype(int)

    # High/Low par semaine
    weekly = (
        df.groupby(["iso_year", "iso_week"])
        .agg(wh=("high", "max"), wl=("low", "min"))
        .reset_index()
    )
    weekly_dict = {
        (int(r.iso_year), int(r.iso_week)): (float(r.wh), float(r.wl))
        for r in weekly.itertuples()
    }

    # Associer chaque date a la semaine precedente
    date_week = (
        df[["date_et", "iso_year", "iso_week"]]
        .drop_duplicates()
    )

    result = {}
    for row in date_week.itertuples():
        yr, wk = int(row.iso_year), int(row.iso_week)
        prev_wk, prev_yr = wk - 1, yr
        if prev_wk == 0:
            prev_yr -= 1
            prev_wk = 52   # simplification — semaine 53 traitee comme 52
        pw = weekly_dict.get((prev_yr, prev_wk))
        result[row.date_et] = pw if pw else (None, None)

    return result


def compute_daily_context(df_day: pd.DataFrame, pwh=None, pwl=None) -> dict:
    """
    Calcule le contexte fixe du jour (Asia range + London raid + PWH/PWL +
    session_high/low pour detection BSL/SSL).

    session_high/low = max/min de toute la periode pre-macro (00:00-08:30 ET).
    Capture le London High etabli jusqu'a 08:30 ET, plus large que la fenetre
    London (03:00-06:00 ET) utilisee uniquement pour le raid Asia.
    """
    ts     = df_day["timestamp"]
    h      = df_day["high"]
    lo     = df_day["low"]
    hm_utc = ts.dt.hour * 60 + ts.dt.minute

    # Asia range : 01:00-05:00 UTC
    asia_mask = (hm_utc >= 60) & (hm_utc < 300)
    asia_h = h[asia_mask].max()  if asia_mask.any() else None
    asia_l = lo[asia_mask].min() if asia_mask.any() else None

    # London KZ : 07:00-10:00 UTC (03:00-06:00 ET) — pour london_ctx uniquement
    ldn_mask = (hm_utc >= 420) & (hm_utc < 600)
    ldn_h = h[ldn_mask].max()  if ldn_mask.any() else None
    ldn_l = lo[ldn_mask].min() if ldn_mask.any() else None

    lc = 0  # NO_RAID par defaut
    if asia_h is not None and ldn_h is not None:
        raid_h = ldn_h > asia_h
        raid_l = ldn_l < asia_l
        if raid_h and not raid_l:
            lc = 1   # RAID_HIGH
        elif raid_l and not raid_h:
            lc = 2   # RAID_LOW

    # Session pre-macro : tout avant 08:30 ET = reference BSL/SSL
    ts_et_series = ts.dt.tz_convert(ET_TZ)
    hm_et_series = ts_et_series.dt.hour * 60 + ts_et_series.dt.minute
    pre_mask = hm_et_series < 510   # avant 08:30 ET
    sess_h = float(h[pre_mask].max())  if pre_mask.any() else ldn_h
    sess_l = float(lo[pre_mask].min()) if pre_mask.any() else ldn_l

    # Contexte mois/jour depuis la premiere bougie
    ts_et = ts.iloc[0].tz_convert(ET_TZ)
    mc    = month_ctx(ts_et.month)
    dc    = day_ctx(ts_et.dayofweek)

    return {
        "asia_high":    asia_h,
        "asia_low":     asia_l,
        "london_high":  ldn_h,
        "london_low":   ldn_l,
        "london_ctx":   lc,
        "month_ctx":    mc,
        "day_ctx":      dc,
        "pwh":          pwh,
        "pwl":          pwl,
        "session_high": sess_h,
        "session_low":  sess_l,
    }
