# ── TA Strategy — sweep glissant (rolling refit) ─────────────────────────────
#
# Lance un sweep exhaustif sur une fenetre glissante de ROLLING_WINDOW_MONTHS mois.
# Cela permet de re-valider les configurations tous les 3 mois et de detecter
# les configs qui ne survivent plus au regime actuel.
#
# Utilisation typique (a planifier en cron tous les 3 mois) :
#   python strategies/ta/sweep_rolling.py
#   python strategies/ta/sweep_rolling.py --window 12   # 12 derniers mois
#
# Output dans results/rolling/ :
#   sweep_rolling_YYYY-MM.csv       <- configs valides sur la fenetre courante
#   sweep_rolling_history.csv       <- historique de toutes les executions
#   regime_drift.csv                <- configs disparues / apparues vs run precedent

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strategies.ta.config import (
    EMA_LENGTHS, RSI_LENGTHS, STOCH_PARAMS, ATR_LENGTHS,
    RESULTS_DIR, MIN_TRADES, MIN_WR, MIN_EXP,
    ROLLING_WINDOW_MONTHS,
)
from strategies.ta.features import load_15m, compute_features, STATE_COLS
from strategies.ta.backtest import build_trades
from strategies.ta.sweep import run_sweep, _aggregate_states


ROLLING_DIR = RESULTS_DIR / "rolling"


def _parse_args():
    p = argparse.ArgumentParser(description="Sweep glissant TA Strategy")
    p.add_argument("--window", type=int, default=ROLLING_WINDOW_MONTHS,
                   help=f"Fenetre glissante en mois (defaut: {ROLLING_WINDOW_MONTHS})")
    p.add_argument("--ref-date", default=None,
                   help="Date de reference ISO (ex: 2026-05-01). Defaut: aujourd'hui.")
    return p.parse_args()


def _state_key(row: pd.Series) -> str:
    """Cle unique pour identifier un etat (params + state_combo + direction)."""
    parts = [str(row["params"]), str(row["direction"])]
    for c in STATE_COLS:
        if c in row.index:
            parts.append(str(row[c]))
    return "|".join(parts)


def main():
    args = _parse_args()
    ROLLING_DIR.mkdir(parents=True, exist_ok=True)

    ref_date = (
        datetime.fromisoformat(args.ref_date).replace(tzinfo=timezone.utc)
        if args.ref_date
        else datetime.now(timezone.utc)
    )
    window_start = ref_date - relativedelta(months=args.window)

    date_from = window_start.strftime("%Y-%m")
    date_to   = ref_date.strftime("%Y-%m")
    run_label = ref_date.strftime("%Y-%m")

    print(f"[rolling] Fenetre : {date_from} -> {date_to} ({args.window} mois)", flush=True)

    # ── Chargement et filtrage ────────────────────────────────────────────────
    df15_all = load_15m()
    mask = (df15_all.index >= pd.Timestamp(date_from, tz="UTC")) & \
           (df15_all.index <= pd.Timestamp(date_to, tz="UTC") + pd.offsets.MonthEnd(1))
    df15 = df15_all[mask]

    print(f"[rolling] {len(df15):,} bougies 15m dans la fenetre", flush=True)

    # ── Trades et sweep ───────────────────────────────────────────────────────
    trades = build_trades(df15)
    print(f"[rolling] {len(trades):,} trades identifies", flush=True)

    results = run_sweep(df15, trades, label_prefix=f"ROLL-{run_label}")

    # Filtrage
    valid = results[
        (results["n"]     >= MIN_TRADES) &
        (results["wr"]    >= MIN_WR)     &
        (results["exp_R"] >= MIN_EXP)
    ].copy()
    valid["run_date"] = run_label
    valid["key"]      = valid.apply(_state_key, axis=1)

    out_run = ROLLING_DIR / f"sweep_rolling_{run_label}.csv"
    valid.to_csv(out_run, index=False)
    print(f"[rolling] {len(valid)} etats valides -> {out_run}", flush=True)

    # ── Historique cumulatif ──────────────────────────────────────────────────
    hist_path = ROLLING_DIR / "sweep_rolling_history.csv"
    if hist_path.exists():
        history = pd.read_csv(hist_path)
        history = pd.concat([history, valid], ignore_index=True)
    else:
        history = valid.copy()
    history.to_csv(hist_path, index=False)
    print(f"[rolling] Historique mis a jour -> {hist_path}", flush=True)

    # ── Drift detection : configs disparues / apparues ────────────────────────
    prev_runs = sorted(ROLLING_DIR.glob("sweep_rolling_????-??.csv"))
    # Exclure le run actuel
    prev_runs = [f for f in prev_runs if f.name != out_run.name]

    if prev_runs:
        prev = pd.read_csv(prev_runs[-1])
        prev["key"] = prev.apply(_state_key, axis=1)

        prev_keys    = set(prev["key"])
        current_keys = set(valid["key"])

        appeared  = current_keys - prev_keys
        disappeared = prev_keys - current_keys
        stable    = current_keys & prev_keys

        print(f"\n[rolling] === DRIFT vs run precedent ({prev_runs[-1].stem}) ===", flush=True)
        print(f"  Stables    : {len(stable)}", flush=True)
        print(f"  Apparus    : {len(appeared)}", flush=True)
        print(f"  Disparus   : {len(disappeared)}", flush=True)

        drift = pd.DataFrame([
            {"key": k, "status": "appeared",   "run_date": run_label}
            for k in appeared
        ] + [
            {"key": k, "status": "disappeared", "run_date": run_label}
            for k in disappeared
        ])
        if not drift.empty:
            drift_path = ROLLING_DIR / "regime_drift.csv"
            if drift_path.exists():
                old_drift = pd.read_csv(drift_path)
                drift = pd.concat([old_drift, drift], ignore_index=True)
            drift.to_csv(drift_path, index=False)
            print(f"[rolling] Drift enregistre -> {drift_path}", flush=True)

        # Alerte : configs du top-10 precedent qui ont disparu
        top_prev = prev.sort_values("exp_R", ascending=False).head(10)
        lost_top = top_prev[top_prev["key"].isin(disappeared)]
        if not lost_top.empty:
            print(f"\n[rolling] ⚠  {len(lost_top)} configs du TOP-10 precedent ont disparu !", flush=True)
            cols = ["params","direction","regime","n","wr","exp_R"] if "regime" in lost_top.columns \
                   else ["params","direction","n","wr","exp_R"]
            print(lost_top[cols].to_string(index=False), flush=True)

    # ── Top 20 du run courant ─────────────────────────────────────────────────
    print(f"\n[rolling] === TOP 20 configs valides ({run_label}) ===", flush=True)
    disp_cols = ["params","direction","regime","ema_state","swing","rsi_state",
                 "stoch_state","atr_state","n","wr","exp_R"]
    disp_cols = [c for c in disp_cols if c in valid.columns]
    print(valid.sort_values("exp_R", ascending=False).head(20)[disp_cols].to_string(index=False),
          flush=True)

    print("\n[rolling] Termine.", flush=True)


if __name__ == "__main__":
    main()
