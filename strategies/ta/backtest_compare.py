# ── Comparative Backtest — Old vs Enhanced Trigger + Ensemble Voting ──

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.ta.features import load_15m, compute_features
from strategies.ta.backtest import build_trades
from strategies.ta.trigger_enhanced import build_trades_enhanced
from strategies.ta.ensemble_voting import EnsembleVoter
from strategies.ta.config import (
    EMA_LENGTHS, RSI_LENGTHS, STOCH_PARAMS, ATR_LENGTHS,
    RESULTS_DIR, MIN_TRADES, MIN_WR, MIN_EXP,
)


def attach_features_to_trades(trades_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Join trades avec features au moment de l'entry."""
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    trades = trades.set_index("entry_time")
    trades = trades.join(features_df, how="left")
    trades = trades.reset_index()
    return trades


def sweep_trades_by_features(trades_df: pd.DataFrame, paramset: tuple) -> pd.DataFrame:
    """
    Ajoute colonne state et paramset pour chaque trade.
    Paramset = (ema_len, rsi_len, stoch_params, atr_len)
    """
    ema_len, rsi_len, stoch_params, atr_len = paramset

    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()

    # Match features à chaque trade
    for idx, row in trades.iterrows():
        features = {
            "ema_state": int(row.get("ema_state", 0)),
            "ema_slope": int(row.get("ema_slope", 0)),
            "swing": int(row.get("swing", 0)),
            "rsi_state": str(row.get("rsi_state", "weak")),
            "stoch_state": str(row.get("stoch_state", "weak")),
            "atr_state": str(row.get("atr_state", "neutral")),
            "vwap_state": int(row.get("vwap_state", 0)),
        }
        trades.at[idx, "features"] = features

    trades["params"] = str(paramset)
    return trades


def evaluate_trades(trades_df: pd.DataFrame, name: str = "") -> dict:
    """Compute stats on trades."""
    if trades_df.empty:
        return {
            "method": name,
            "n_trades": 0,
            "n_wins": 0,
            "wr": 0.0,
            "exp_R": 0.0,
            "avg_R": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }

    n_total = len(trades_df)
    n_wins = (trades_df["outcome"] == "win").sum()
    wr = n_wins / n_total if n_total > 0 else 0.0

    # Expectancy en R
    wins_avg = trades_df[trades_df["outcome"] == "win"]["atr_at_entry"].mean()
    loss_avg = trades_df[trades_df["outcome"] == "loss"]["atr_at_entry"].mean()
    exp_R = wr * (2.0 * wins_avg) - (1 - wr) * (1.0 * loss_avg)  # TP=2*ATR, SL=1*ATR
    avg_R = (wr * 2.0) - ((1 - wr) * 1.0)

    return {
        "method": name,
        "n_trades": n_total,
        "n_wins": n_wins,
        "wr": wr,
        "exp_R": exp_R,
        "avg_R": avg_R,
        "avg_win": wins_avg if n_wins > 0 else 0.0,
        "avg_loss": loss_avg if n_total - n_wins > 0 else 0.0,
    }


def main():
    print("[backtest_compare] Chargement données...\n")

    df15 = load_15m()
    print(f"  Loaded {len(df15)} candles 15m")

    # Split IS (2020-2024) vs OOS (2025-2026)
    split_date = pd.Timestamp("2025-01-01", tz="UTC")
    df_is = df15[df15.index < split_date]
    df_oos = df15[df15.index >= split_date]

    print(f"  IS period: {len(df_is)} candles ({df_is.index[0].date()} - {df_is.index[-1].date()})")
    print(f"  OOS period: {len(df_oos)} candles ({df_oos.index[0].date()} - {df_oos.index[-1].date()})\n")

    # Features
    print("[backtest_compare] Calcul features...\n")
    features_is = compute_features(
        df_is, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )
    features_oos = compute_features(
        df_oos, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )

    # ── Trigger Comparison ──────────────────────────────────────────────────
    print("[backtest_compare] Trigger comparison:\n")

    print("  OLD trigger (2-bar reversal simple):")
    trades_old_is = build_trades(df_is)
    trades_old_oos = build_trades(df_oos)
    print(f"    IS: {len(trades_old_is)} trades")
    print(f"    OOS: {len(trades_old_oos)} trades\n")

    print("  ENHANCED trigger (multi-structure + ATR filter):")
    trades_enh_is = build_trades_enhanced(df_is)
    trades_enh_oos = build_trades_enhanced(df_oos)
    print(f"    IS: {len(trades_enh_is)} trades")
    print(f"    OOS: {len(trades_enh_oos)} trades\n")

    # Attach features
    trades_old_is = attach_features_to_trades(trades_old_is, features_is)
    trades_old_oos = attach_features_to_trades(trades_old_oos, features_oos)
    trades_enh_is = attach_features_to_trades(trades_enh_is, features_is)
    trades_enh_oos = attach_features_to_trades(trades_enh_oos, features_oos)

    # Stats
    print("=== STATS WITHOUT ENSEMBLE VOTING ===\n")
    stats_old_is = evaluate_trades(trades_old_is, "OLD IS")
    stats_old_oos = evaluate_trades(trades_old_oos, "OLD OOS")
    stats_enh_is = evaluate_trades(trades_enh_is, "ENH IS")
    stats_enh_oos = evaluate_trades(trades_enh_oos, "ENH OOS")

    stats_df = pd.DataFrame([stats_old_is, stats_old_oos, stats_enh_is, stats_enh_oos])
    print(stats_df.to_string(index=False))
    print()

    # ── Ensemble Voting ────────────────────────────────────────────────────
    print("\n=== ENSEMBLE VOTING (2/3 consensus) ===\n")

    voter = EnsembleVoter(RESULTS_DIR)
    print("Top-3 configs OOS stables loaded:\n")
    print(voter.get_top_configs_summary().to_string(index=False))
    print()

    # Apply voting filter
    def apply_ensemble_filter(trades_df: pd.DataFrame, voter: EnsembleVoter) -> pd.DataFrame:
        """Garde seulement les trades votés 2/3."""
        filtered = []
        for idx, trade in trades_df.iterrows():
            regime = trade.get("regime", "bull")
            features = {
                "ema_state": int(trade.get("ema_state", 0)),
                "ema_slope": int(trade.get("ema_slope", 0)),
                "swing": int(trade.get("swing", 0)),
                "rsi_state": str(trade.get("rsi_state", "weak")),
                "stoch_state": str(trade.get("stoch_state", "weak")),
                "atr_state": str(trade.get("atr_state", "neutral")),
                "vwap_state": int(trade.get("vwap_state", 0)),
            }
            vote_count, consensus_dir, confidence = voter.vote(regime, features, trade["direction"])
            if vote_count >= 2:
                trade = trade.copy()
                trade["vote_count"] = vote_count
                trade["confidence"] = confidence
                filtered.append(trade)

        return pd.DataFrame(filtered)

    trades_old_is_voted = apply_ensemble_filter(trades_old_is, voter)
    trades_old_oos_voted = apply_ensemble_filter(trades_old_oos, voter)
    trades_enh_is_voted = apply_ensemble_filter(trades_enh_is, voter)
    trades_enh_oos_voted = apply_ensemble_filter(trades_enh_oos, voter)

    print("After ensemble voting (2/3 consensus):\n")
    print(f"  OLD IS: {len(trades_old_is)} -> {len(trades_old_is_voted)} trades "
          f"({len(trades_old_is_voted)/len(trades_old_is)*100:.1f}% kept)")
    print(f"  OLD OOS: {len(trades_old_oos)} -> {len(trades_old_oos_voted)} trades "
          f"({len(trades_old_oos_voted)/len(trades_old_oos)*100:.1f}% kept)")
    print(f"  ENH IS: {len(trades_enh_is)} -> {len(trades_enh_is_voted)} trades "
          f"({len(trades_enh_is_voted)/len(trades_enh_is)*100:.1f}% kept)")
    print(f"  ENH OOS: {len(trades_enh_oos)} -> {len(trades_enh_oos_voted)} trades "
          f"({len(trades_enh_oos_voted)/len(trades_enh_oos)*100:.1f}% kept)\n")

    print("=== STATS WITH ENSEMBLE VOTING ===\n")
    stats_voted = [
        evaluate_trades(trades_old_is_voted, "OLD+VOTE IS"),
        evaluate_trades(trades_old_oos_voted, "OLD+VOTE OOS"),
        evaluate_trades(trades_enh_is_voted, "ENH+VOTE IS"),
        evaluate_trades(trades_enh_oos_voted, "ENH+VOTE OOS"),
    ]
    stats_voted_df = pd.DataFrame(stats_voted)
    print(stats_voted_df.to_string(index=False))
    print()

    # ── Summary Comparison ──────────────────────────────────────────────────
    print("=== SUMMARY METRICS (IS -> OOS degradation) ===\n")
    for old_is, old_oos, enh_is, enh_oos, label in [
        (stats_old_is, stats_old_oos, stats_enh_is, stats_enh_oos, "Without Voting"),
        (stats_voted[0], stats_voted[1], stats_voted[2], stats_voted[3], "With Ensemble Voting"),
    ]:
        print(f"{label}:")
        print(f"  OLD:  n={old_is['n_trades']:3d} -> {old_oos['n_trades']:3d}  "
              f"WR {old_is['wr']:.3f} -> {old_oos['wr']:.3f} "
              f"(drop {old_oos['wr']-old_is['wr']:+.3f})")
        print(f"  ENH:  n={enh_is['n_trades']:3d} -> {enh_oos['n_trades']:3d}  "
              f"WR {enh_is['wr']:.3f} -> {enh_oos['wr']:.3f} "
              f"(drop {enh_oos['wr']-enh_is['wr']:+.3f})")
        print()


if __name__ == "__main__":
    main()
