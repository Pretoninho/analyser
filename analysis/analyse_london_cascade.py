"""
analyse_london_cascade.py — Protocole London: LOR -> Silver Bullet -> London Macro.

Spec validee:
- Asia range ET: 20:00-00:00 (veille)
- Cascade: LOR 02:00-02:30 -> SB 03:00-04:00 -> Macro 04:10-04:30
- 1 trade max / jour (strict)
- SB active seulement si LOR trigger detecte mais pas d'entree validee
- Entree comparee: MSS+FVG/retest vs heure fixe
- Comparatif MSS 1m vs 5m
- Comparatif SB direction: contrarian (Judas) vs momentum
- Exit max: 08:30 ET
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytz

from data.binance import load_binance_1m
from engine.stats_state import build_weekly_levels
from main import _sim_trade_rr

sys.stdout.reconfigure(encoding="utf-8")

ET_TZ = pytz.timezone("America/New_York")

SL_PCT = 0.006
RR = 2.5
FEE_RATE = 0.0005
SLIPPAGE = 0.0002
EXIT_HM = 510  # 08:30 ET

SKIP_DAYS = {0}  # lundi
TEST_RATIO = 0.2


@dataclass
class StageWindow:
    name: str
    sweep_start: int
    sweep_end: int
    confirm_end: int
    fixed_entry: int


LOR = StageWindow("LOR", 120, 150, 180, 150)   # 02:30
SB = StageWindow("SB", 180, 240, 250, 240)      # 04:00
LM = StageWindow("LM", 250, 270, EXIT_HM, 270)  # 04:30


@dataclass
class SetupResult:
    has_trigger: bool
    has_entry: bool
    direction: int | None = None
    entry_hm: int | None = None
    entry_px: float | None = None
    method: str = ""
    sweep_side: str | None = None
    score: int = 0
    notes: str = ""


def _resample_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    d = df_1m.set_index("ts_et").sort_index()
    r = (
        d.resample("5min", label="right", closed="right")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return r


def _first_asia_sweep(win_df: pd.DataFrame, asia_high: float, asia_low: float) -> tuple[str, pd.Timestamp] | tuple[None, None]:
    for _, r in win_df.iterrows():
        h = float(r["high"])
        l = float(r["low"])
        up = h > asia_high
        dn = l < asia_low
        if up and dn:
            continue
        if up:
            return "HIGH", r["ts_et"]
        if dn:
            return "LOW", r["ts_et"]
    return None, None


def _find_mss(candles: pd.DataFrame, bias: int, start_ts: pd.Timestamp, lookback: int) -> tuple[int, float] | tuple[None, None]:
    c = candles.reset_index(drop=True)
    for i in range(lookback, len(c)):
        if c.loc[i, "ts_et"] < start_ts:
            continue
        lo_ref = float(c.loc[i - lookback:i - 1, "low"].min())
        hi_ref = float(c.loc[i - lookback:i - 1, "high"].max())
        cls = float(c.loc[i, "close"])
        if bias == -1 and cls < lo_ref:
            return i, lo_ref
        if bias == +1 and cls > hi_ref:
            return i, hi_ref
    return None, None


def _find_fvg(candles: pd.DataFrame, bias: int, start_idx: int) -> tuple[pd.Timestamp, float, float] | tuple[None, None, None]:
    c = candles.reset_index(drop=True)
    for i in range(max(2, start_idx + 1), len(c)):
        h0 = float(c.loc[i - 2, "high"])
        l0 = float(c.loc[i - 2, "low"])
        h2 = float(c.loc[i, "high"])
        l2 = float(c.loc[i, "low"])
        if bias == -1 and h2 < l0:
            return c.loc[i, "ts_et"], h2, l0
        if bias == +1 and l2 > h0:
            return c.loc[i, "ts_et"], h0, l2
    return None, None, None


def _find_entry_after_signal(day_df: pd.DataFrame, bias: int, fvg_ts: pd.Timestamp | None,
                             z_bot: float | None, z_top: float | None,
                             mss_ts: pd.Timestamp, mss_lvl: float,
                             confirm_end: int) -> tuple[int, float, str] | tuple[None, None, None]:
    minute = day_df[(day_df["ts_et"] > mss_ts) & (day_df["hm_et"] < confirm_end)].copy()
    if len(minute) == 0:
        return None, None, None

    if fvg_ts is not None and z_bot is not None and z_top is not None:
        mid = (z_bot + z_top) / 2.0
        fvg_scan = minute[minute["ts_et"] > fvg_ts]
        for _, r in fvg_scan.iterrows():
            h = float(r["high"])
            l = float(r["low"])
            if h >= z_bot and l <= z_top:
                return int(r["hm_et"]), float(mid), "FVG"

    for _, r in minute.iterrows():
        h = float(r["high"])
        l = float(r["low"])
        if bias == -1 and h >= mss_lvl:
            return int(r["hm_et"]), float(mss_lvl), "RETEST"
        if bias == +1 and l <= mss_lvl:
            return int(r["hm_et"]), float(mss_lvl), "RETEST"

    return None, None, None


def _nearest_target_dist(entry_px: float, direction: int, levels: list[float]) -> float | None:
    if direction == -1:
        tgt = [lv for lv in levels if lv < entry_px]
        if not tgt:
            return None
        return (entry_px - max(tgt)) / entry_px
    tgt = [lv for lv in levels if lv > entry_px]
    if not tgt:
        return None
    return (min(tgt) - entry_px) / entry_px


def _context_score(direction: int, entry_px: float,
                   pdh: float | None, pdl: float | None,
                   pwh: float | None, pwl: float | None,
                   sweep_side: str) -> int:
    levels = [lv for lv in (pdh, pdl, pwh, pwl) if lv is not None]
    score = 0
    dist = _nearest_target_dist(entry_px, direction, levels)
    if dist is not None:
        score += 1
    if dist is not None and dist > SL_PCT:
        score += 1
    if (direction == -1 and sweep_side == "HIGH") or (direction == +1 and sweep_side == "LOW"):
        score += 1
    return score


def _detect_stage_setup(day_df: pd.DataFrame, asia_high: float, asia_low: float, stage: StageWindow,
                        timeframe: str, direction_mode: str, entry_model: str,
                        pdh: float | None, pdl: float | None,
                        pwh: float | None, pwl: float | None) -> SetupResult:
    win = day_df[(day_df["hm_et"] >= stage.sweep_start) & (day_df["hm_et"] < stage.sweep_end)]
    if len(win) == 0:
        return SetupResult(False, False, notes="empty window")

    sweep_side, sweep_ts = _first_asia_sweep(win, asia_high, asia_low)
    if sweep_side is None:
        return SetupResult(False, False, notes="no sweep")

    base_bias = -1 if sweep_side == "HIGH" else +1
    if direction_mode == "momentum":
        base_bias *= -1

    if entry_model == "fixed_time":
        if sweep_ts is not None and int(sweep_ts.hour * 60 + sweep_ts.minute) >= stage.fixed_entry:
            return SetupResult(True, False, sweep_side=sweep_side, notes="sweep after fixed entry")
        ebar = day_df[day_df["hm_et"] >= stage.fixed_entry]
        if len(ebar) == 0:
            return SetupResult(True, False, sweep_side=sweep_side, notes="no fixed entry bar")
        entry = ebar.iloc[0]
        entry_hm = int(entry["hm_et"])
        entry_px = float(entry["open"])
        score = _context_score(base_bias, entry_px, pdh, pdl, pwh, pwl, sweep_side)
        return SetupResult(True, True, base_bias, entry_hm, entry_px, "FIXED_TIME", sweep_side, score,
                           f"{stage.name}:{timeframe}:{direction_mode}:fixed")

    scan = day_df[(day_df["hm_et"] >= stage.sweep_start) & (day_df["hm_et"] < stage.confirm_end)].copy()
    if len(scan) < 8:
        return SetupResult(True, False, sweep_side=sweep_side, notes="not enough bars")

    if timeframe == "5m":
        cand = _resample_5m(scan)
        lookback = 3
    else:
        cand = scan[["ts_et", "open", "high", "low", "close"]].copy().reset_index(drop=True)
        lookback = 5

    if len(cand) < lookback + 3:
        return SetupResult(True, False, sweep_side=sweep_side, notes="not enough tf bars")

    mss_idx, mss_lvl = _find_mss(cand, base_bias, sweep_ts, lookback)
    if mss_idx is None:
        return SetupResult(True, False, sweep_side=sweep_side, notes="no MSS")

    mss_ts = cand.loc[mss_idx, "ts_et"]
    fvg_ts, z_bot, z_top = _find_fvg(cand, base_bias, mss_idx)
    entry_hm, entry_px, method = _find_entry_after_signal(
        day_df, base_bias, fvg_ts, z_bot, z_top, mss_ts, mss_lvl, stage.confirm_end
    )
    if entry_hm is None:
        return SetupResult(True, False, sweep_side=sweep_side, notes="MSS no entry")

    score = _context_score(base_bias, entry_px, pdh, pdl, pwh, pwl, sweep_side)
    return SetupResult(True, True, base_bias, entry_hm, entry_px, method, sweep_side, score,
                       f"{stage.name}:{timeframe}:{direction_mode}:mss")


def _simulate_trade(day_df: pd.DataFrame, setup: SetupResult) -> dict | None:
    if not setup.has_entry or setup.direction is None or setup.entry_hm is None or setup.entry_px is None:
        return None

    exit_df = day_df[(day_df["hm_et"] >= setup.entry_hm) & (day_df["hm_et"] < EXIT_HM)]
    if len(exit_df) < 5:
        return None

    epx = setup.entry_px * (1 + SLIPPAGE) if setup.direction == 1 else setup.entry_px * (1 - SLIPPAGE)
    pnl, reason, _, _, _, nc = _sim_trade_rr(
        exit_df, epx, setup.direction, SL_PCT, SL_PCT * RR,
        fee=FEE_RATE, slip=SLIPPAGE, verbose=True
    )
    return {
        "entry_hm": setup.entry_hm,
        "direction": "LONG" if setup.direction == 1 else "SHORT",
        "pnl": float(pnl),
        "reason": reason,
        "n_candles": int(nc),
        "method": setup.method,
        "score": int(setup.score),
        "sweep_side": setup.sweep_side,
        "notes": setup.notes,
    }


def _stats(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {"n": 0, "wr": 0.0, "avg": 0.0, "pf": 0.0, "total": 0.0}
    arr = sub["pnl"].values
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    pf = float("inf") if losses.sum() == 0 else abs(wins.sum() / losses.sum())
    return {
        "n": int(len(sub)),
        "wr": float((arr > 0).mean() * 100.0),
        "avg": float(arr.mean() * 100.0),
        "pf": float(pf),
        "total": float(arr.sum() * 100.0),
    }


def _run_variant(days: list, by_date: dict, weekly: dict,
                 timeframe: str, sb_dir_mode: str, macro_role: str,
                 entry_model: str) -> pd.DataFrame:
    rows = []
    split_cut = int(len(days) * (1 - TEST_RATIO))

    for i, d in enumerate(days):
        if d.weekday() in SKIP_DAYS:
            continue

        day_df = by_date.get(d)
        prev_df = by_date.get(d - timedelta(days=1))
        if day_df is None or prev_df is None:
            continue

        asia = prev_df[prev_df["hm_et"] >= 1200]
        if len(asia) < 10:
            continue

        asia_high = float(asia["high"].max())
        asia_low = float(asia["low"].min())

        pwh, pwl = weekly.get(d, (None, None))
        pdh = float(prev_df["high"].max()) if len(prev_df) else None
        pdl = float(prev_df["low"].min()) if len(prev_df) else None

        split = "TRAIN" if i < split_cut else "TEST"

        if macro_role == "standalone":
            lm = _detect_stage_setup(day_df, asia_high, asia_low, LM,
                                     timeframe, "contrarian", entry_model,
                                     pdh, pdl, pwh, pwl)
            tr = _simulate_trade(day_df, lm)
            if tr:
                rows.append({
                    "date": d, "split": split, "stage": "LM",
                    "timeframe": timeframe, "sb_dir_mode": sb_dir_mode,
                    "macro_role": macro_role, "entry_model": entry_model,
                    **tr,
                })
            continue

        lor = _detect_stage_setup(day_df, asia_high, asia_low, LOR,
                                  timeframe, "contrarian", entry_model,
                                  pdh, pdl, pwh, pwl)

        chosen = None
        stage = ""

        if lor.has_entry:
            chosen = lor
            stage = "LOR"
        else:
            if lor.has_trigger and not lor.has_entry:
                sb = _detect_stage_setup(day_df, asia_high, asia_low, SB,
                                         timeframe, sb_dir_mode, entry_model,
                                         pdh, pdl, pwh, pwl)
            else:
                sb = SetupResult(False, False, notes="SB skipped")

            if sb.has_entry:
                if macro_role == "sb_filter":
                    lm_filter = _detect_stage_setup(day_df, asia_high, asia_low, LM,
                                                    timeframe, "contrarian", entry_model,
                                                    pdh, pdl, pwh, pwl)
                    if lm_filter.has_trigger:
                        chosen = sb
                        stage = "SB"
                else:
                    chosen = sb
                    stage = "SB"

            if chosen is None and macro_role == "fallback":
                lm = _detect_stage_setup(day_df, asia_high, asia_low, LM,
                                         timeframe, "contrarian", entry_model,
                                         pdh, pdl, pwh, pwl)
                if lm.has_entry:
                    chosen = lm
                    stage = "LM"

        if chosen is None:
            continue

        tr = _simulate_trade(day_df, chosen)
        if tr is None:
            continue

        rows.append({
            "date": d, "split": split, "stage": stage,
            "timeframe": timeframe, "sb_dir_mode": sb_dir_mode,
            "macro_role": macro_role, "entry_model": entry_model,
            **tr,
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("[cascade] Chargement BTCUSDT 1m...")
    df = load_binance_1m()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ts_et"] = df["timestamp"].dt.tz_convert(ET_TZ)
    df["hm_et"] = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date_et"] = df["ts_et"].dt.date
    df = df.sort_values("ts_et").reset_index(drop=True)

    print(f"[cascade] {len(df):,} bougies | {df['date_et'].min()} -> {df['date_et'].max()}")

    by_date = {d: g.sort_values("ts_et").reset_index(drop=True) for d, g in df.groupby("date_et")}
    days = sorted(by_date.keys())
    weekly = build_weekly_levels(df)

    variants = [
        ("1m", "contrarian", "fallback", "mss_fvg"),
        ("1m", "contrarian", "fallback", "fixed_time"),
        ("1m", "momentum", "fallback", "fixed_time"),
        ("5m", "contrarian", "fallback", "mss_fvg"),
        ("5m", "contrarian", "fallback", "fixed_time"),
        ("5m", "momentum", "fallback", "fixed_time"),
        ("1m", "contrarian", "standalone", "fixed_time"),
        ("5m", "contrarian", "standalone", "fixed_time"),
    ]

    all_rows = []
    for tf, sb_mode, role, entry_model in variants:
        res = _run_variant(days, by_date, weekly, tf, sb_mode, role, entry_model)
        if len(res):
            all_rows.append(res)

    if not all_rows:
        print("[cascade] Aucun trade genere par les variantes.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out_dir = Path("display/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "london_cascade_results.csv"
    out.to_csv(out_csv, index=False)

    print("\n" + "=" * 84)
    print("  LONDON CASCADE — COMPARATIF VARIANTES")
    print("=" * 84)
    print(f"  Export CSV: {out_csv}")

    keys = ["timeframe", "sb_dir_mode", "macro_role", "entry_model"]
    for k, sub in out.groupby(keys):
        train = sub[sub["split"] == "TRAIN"]
        test = sub[sub["split"] == "TEST"]
        st_tr = _stats(train)
        st_te = _stats(test)
        label = f"tf={k[0]} | sb={k[1]} | role={k[2]} | entry={k[3]}"
        print("\n" + "-" * 84)
        print(f"  {label}")
        print(
            f"    TRAIN N={st_tr['n']:>4} WR={st_tr['wr']:>5.1f}% Avg={st_tr['avg']:>+7.3f}% "
            f"PF={st_tr['pf']:.3f} Total={st_tr['total']:+.2f}%"
        )
        print(
            f"    TEST  N={st_te['n']:>4} WR={st_te['wr']:>5.1f}% Avg={st_te['avg']:>+7.3f}% "
            f"PF={st_te['pf']:.3f} Total={st_te['total']:+.2f}%"
        )
        stage_dist = sub.groupby("stage").size().to_dict()
        print(f"    Stage mix: {stage_dist}")

    print("\n" + "=" * 84)
    print("  Top variantes (TEST avg desc, N>=15)")
    print("=" * 84)
    rank_rows = []
    for k, sub in out.groupby(keys):
        t = sub[sub["split"] == "TEST"]
        s = _stats(t)
        rank_rows.append({
            "timeframe": k[0], "sb_dir_mode": k[1], "macro_role": k[2], "entry_model": k[3],
            "n": s["n"], "wr": s["wr"], "avg": s["avg"], "pf": s["pf"], "total": s["total"],
        })

    rank = pd.DataFrame(rank_rows)
    rank = rank[rank["n"] >= 15].sort_values(["avg", "pf", "wr"], ascending=False)
    if len(rank) == 0:
        print("  Aucun variant avec N>=15 en TEST.")
    else:
        for _, r in rank.head(10).iterrows():
            print(
                f"  tf={r['timeframe']:<2} sb={r['sb_dir_mode']:<10} role={r['macro_role']:<10} "
                f"entry={r['entry_model']:<10} N={int(r['n']):>4} WR={r['wr']:>5.1f}% "
                f"Avg={r['avg']:>+7.3f}% PF={r['pf']:.3f} Total={r['total']:+.2f}%"
            )


if __name__ == "__main__":
    main()
