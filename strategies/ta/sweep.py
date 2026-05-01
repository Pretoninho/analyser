# ── TA Strategy — sweep exhaustif des combinaisons paramétriques ─────────────
#
# Pour chaque combinaison de paramètres (EMA × RSI × Stoch × ATR = 108 combos) :
#   1. Calcule les features sur tous les bars 15m
#   2. Extrait les features aux positions des trades (calculés une seule fois)
#   3. Agrège les stats par (state_combo, direction)
#   4. Filtre : n >= MIN_TRADES, WR >= MIN_WR, Exp >= MIN_EXP
#   5. Sauvegarde le résultat complet + le résumé filtré dans results/
#
# Usage :
#   python strategies/ta/sweep.py [--from YYYY-MM] [--to YYYY-MM]

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strategies.ta.config import (
    EMA_LENGTHS, RSI_LENGTHS, STOCH_PARAMS, ATR_LENGTHS,
    RESULTS_DIR, MIN_TRADES, MIN_WR, MIN_EXP,
)
from strategies.ta.features import load_15m, compute_features, STATE_COLS
from strategies.ta.backtest import build_trades


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Sweep exhaustif TA Strategy")
    p.add_argument("--from", dest="date_from", default=None,
                   help="Date de début IS (ex: 2020-01). Défaut: toutes les données.")
    p.add_argument("--to",   dest="date_to",   default=None,
                   help="Date de fin IS (ex: 2024-12). Défaut: toutes les données.")
    p.add_argument("--oos-from", dest="oos_from", default=None,
                   help="Début OOS (ex: 2025-01). Si absent, pas de validation OOS.")
    return p.parse_args()


def _filter_df(df: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    if date_from:
        df = df[df.index >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        end = pd.Timestamp(date_to, tz="UTC") + pd.offsets.MonthEnd(1)
        df = df[df.index <= end]
    return df


def _aggregate_states(trades_with_feat: pd.DataFrame,
                      param_label: str) -> pd.DataFrame:
    """
    Pour chaque (state_combo, direction), calcule n, wins, WR, expectancy.
    RR fixe = TP_MULT / SL_MULT = 2.0
    Expectancy (en R) = WR * TP_MULT - (1-WR) * SL_MULT
    """
    from strategies.ta.config import TP_MULT, SL_MULT

    group_cols = STATE_COLS + ["direction"]
    rows = []

    for key, grp in trades_with_feat.groupby(group_cols, observed=True):
        n    = len(grp)
        wins = (grp["outcome"] == "win").sum()
        wr   = wins / n
        exp  = wr * TP_MULT - (1 - wr) * SL_MULT

        row = dict(zip(group_cols, key))
        row.update({"params": param_label, "n": n, "wins": wins,
                    "wr": round(wr, 4), "exp_R": round(exp, 4)})
        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(df15: pd.DataFrame, trades_base: pd.DataFrame,
              label_prefix: str = "IS") -> pd.DataFrame:
    """
    Itère sur les 108 combos paramétriques.
    Pour chaque combo, extrait les features au moment des trades
    et agrège les stats.
    Retourne le DataFrame complet (toutes combinaisons).
    """
    all_results = []
    total = len(EMA_LENGTHS) * len(RSI_LENGTHS) * len(STOCH_PARAMS) * len(ATR_LENGTHS)
    done  = 0

    for ema_len in EMA_LENGTHS:
        for rsi_len in RSI_LENGTHS:
            for (sk, ss, sd) in STOCH_PARAMS:
                for atr_len in ATR_LENGTHS:
                    done += 1
                    param_label = f"EMA{ema_len}_RSI{rsi_len}_SK{sk}SS{ss}SD{sd}_ATR{atr_len}"
                    print(f"[{label_prefix}] {done}/{total}  {param_label}", flush=True)

                    # Calcul features sur tout le df15
                    feat = compute_features(
                        df15,
                        ema_len=ema_len,
                        rsi_len=rsi_len,
                        stoch_k_period=sk,
                        stoch_smooth_k=ss,
                        stoch_d_period=sd,
                        atr_len=atr_len,
                    )

                    # Extraction features aux indices des trades
                    idx = trades_base["entry_idx"].values
                    # Garde uniquement les trades dans le range de feat
                    valid = (idx >= 0) & (idx < len(feat))
                    trades_sub = trades_base[valid].copy()
                    idx_valid  = idx[valid]

                    for col in STATE_COLS:
                        trades_sub[col] = feat.iloc[idx_valid][col].values

                    # Drop les lignes avec NaN dans les states
                    trades_sub = trades_sub.dropna(subset=STATE_COLS)

                    if trades_sub.empty:
                        continue

                    agg = _aggregate_states(trades_sub, param_label)
                    all_results.append(agg)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def main():
    args = _parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[sweep] Chargement des données 15m...", flush=True)
    df15_all = load_15m()
    print(f"[sweep] Données : {df15_all.index[0]} → {df15_all.index[-1]} "
          f"({len(df15_all):,} bougies 15m)", flush=True)

    # ── In-Sample ────────────────────────────────────────────────────────────
    df15_is = _filter_df(df15_all, args.date_from, args.date_to)
    print(f"[sweep] IS : {len(df15_is):,} bougies", flush=True)

    print("[sweep] Calcul des trades IS (triggers + outcomes)...", flush=True)
    trades_is = build_trades(df15_is)
    print(f"[sweep] {len(trades_is):,} trades IS identifiés", flush=True)

    results_is = run_sweep(df15_is, trades_is, label_prefix="IS")

    # Sauvegarde complète IS
    out_full = RESULTS_DIR / "sweep_full_IS.csv"
    results_is.to_csv(out_full, index=False)
    print(f"[sweep] Resultats complets IS -> {out_full}", flush=True)

    # Résumé filtré IS
    filt = results_is[
        (results_is["n"]     >= MIN_TRADES) &
        (results_is["wr"]    >= MIN_WR)     &
        (results_is["exp_R"] >= MIN_EXP)
    ].sort_values("exp_R", ascending=False)

    out_filt = RESULTS_DIR / "sweep_filtered_IS.csv"
    filt.to_csv(out_filt, index=False)
    print(f"[sweep] {len(filt)} etats valides IS -> {out_filt}", flush=True)

    # ── Out-of-Sample (optionnel) ─────────────────────────────────────────────
    if args.oos_from:
        df15_oos = _filter_df(df15_all, args.oos_from, None)
        print(f"[sweep] OOS : {len(df15_oos):,} bougies", flush=True)

        trades_oos = build_trades(df15_oos)
        print(f"[sweep] {len(trades_oos):,} trades OOS identifiés", flush=True)

        # Uniquement les combos validés en IS
        valid_params = set(filt["params"].unique())

        results_oos_parts = []
        for ema_len in EMA_LENGTHS:
            for rsi_len in RSI_LENGTHS:
                for (sk, ss, sd) in STOCH_PARAMS:
                    for atr_len in ATR_LENGTHS:
                        param_label = f"EMA{ema_len}_RSI{rsi_len}_SK{sk}SS{ss}SD{sd}_ATR{atr_len}"
                        if param_label not in valid_params:
                            continue

                        feat = compute_features(df15_oos, ema_len, rsi_len, sk, ss, sd, atr_len)
                        idx  = trades_oos["entry_idx"].values
                        valid = (idx >= 0) & (idx < len(feat))
                        trades_sub = trades_oos[valid].copy()
                        idx_valid  = idx[valid]
                        for col in STATE_COLS:
                            trades_sub[col] = feat.iloc[idx_valid][col].values
                        trades_sub = trades_sub.dropna(subset=STATE_COLS)
                        if trades_sub.empty:
                            continue
                        agg = _aggregate_states(trades_sub, param_label)
                        results_oos_parts.append(agg)

        if results_oos_parts:
            results_oos = pd.concat(results_oos_parts, ignore_index=True)
            out_oos = RESULTS_DIR / "sweep_filtered_OOS.csv"
            results_oos.to_csv(out_oos, index=False)
            print(f"[sweep] Resultats OOS -> {out_oos}", flush=True)

            # Comparaison IS vs OOS
            merged = filt.merge(
                results_oos[["params"] + STATE_COLS + ["direction", "n", "wr", "exp_R"]],
                on=["params"] + STATE_COLS + ["direction"],
                suffixes=("_IS", "_OOS"),
                how="left",
            )
            out_cmp = RESULTS_DIR / "sweep_IS_vs_OOS.csv"
            merged.to_csv(out_cmp, index=False)
            print(f"[sweep] Comparaison IS/OOS -> {out_cmp}", flush=True)

    print("[sweep] Terminé.", flush=True)


if __name__ == "__main__":
    main()
