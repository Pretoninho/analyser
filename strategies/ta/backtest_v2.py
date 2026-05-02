# ── Backtest v2 — Simple 2-bar + Ensemble Voting v2 ──

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from strategies.ta.features import load_15m, compute_features
from strategies.ta.backtest import build_trades
from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2
from strategies.ta.config import RESULTS_DIR


def attach_features_to_trades(trades_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Join trades avec features au moment de l'entry."""
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    trades = trades.set_index("entry_time")
    trades = trades.join(features_df, how="left")
    trades = trades.reset_index()
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
        }

    n_total = len(trades_df)
    n_wins = (trades_df["outcome"] == "win").sum()
    wr = n_wins / n_total if n_total > 0 else 0.0

    # Expectancy en R (TP=2*ATR, SL=1*ATR)
    wins_avg = trades_df[trades_df["outcome"] == "win"]["atr_at_entry"].mean() if n_wins > 0 else 0.0
    loss_avg = trades_df[trades_df["outcome"] == "loss"]["atr_at_entry"].mean() if n_total - n_wins > 0 else 0.0

    if n_total > 0:
        exp_R = wr * (2.0 * wins_avg if wins_avg > 0 else 1.0) - (1 - wr) * (1.0 * loss_avg if loss_avg > 0 else 1.0)
    else:
        exp_R = 0.0

    return {
        "method": name,
        "n_trades": n_total,
        "n_wins": n_wins,
        "wr": wr,
        "exp_R": exp_R,
    }


def apply_voting_filter(trades_df: pd.DataFrame, voter: EnsembleVoterV2) -> pd.DataFrame:
    """Filtre les trades par ensemble voting."""
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

        total_voters, favorable_votes, consensus_dir, confidence = voter.vote(
            regime, features, trade["direction"]
        )

        if consensus_dir is not None:
            trade_copy = trade.copy()
            trade_copy["vote_total"] = total_voters
            trade_copy["vote_favorable"] = favorable_votes
            trade_copy["vote_confidence"] = confidence
            filtered.append(trade_copy)

    return pd.DataFrame(filtered)


def main():
    print("[backtest_v2] Chargement données...\n")

    df15 = load_15m()
    print(f"  Loaded {len(df15)} candles 15m\n")

    # Split IS (2020-2024) vs OOS (2025-2026)
    split_date = pd.Timestamp("2025-01-01", tz="UTC")
    df_is = df15[df15.index < split_date]
    df_oos = df15[df15.index >= split_date]

    print(f"  IS period: {len(df_is)} candles ({df_is.index[0].date()} - {df_is.index[-1].date()})")
    print(f"  OOS period: {len(df_oos)} candles ({df_oos.index[0].date()} - {df_oos.index[-1].date()})\n")

    # Features
    print("[backtest_v2] Calcul features...\n")
    features_is = compute_features(
        df_is, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )
    features_oos = compute_features(
        df_oos, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )

    # Build trades (simple 2-bar reversal)
    print("[backtest_v2] Building trades (2-bar reversal)...\n")
    trades_is = attach_features_to_trades(build_trades(df_is), features_is)
    trades_oos = attach_features_to_trades(build_trades(df_oos), features_oos)

    print(f"  IS: {len(trades_is)} trades")
    print(f"  OOS: {len(trades_oos)} trades\n")

    # Baseline stats
    print("=== BASELINE (NO VOTING) ===\n")
    stats_is = evaluate_trades(trades_is, "IS")
    stats_oos = evaluate_trades(trades_oos, "OOS")

    baseline = pd.DataFrame([stats_is, stats_oos])
    print(baseline[["method", "n_trades", "wr", "exp_R"]].to_string(index=False))
    print()

    # Test different voting thresholds
    print("\n=== ENSEMBLE VOTING v2 — Different Qualification Thresholds ===\n")

    thresholds = [
        (5, 0.50, "liberal (n>=5, WR>=50%)"),
        (5, 0.55, "standard (n>=5, WR>=55%)"),
        (5, 0.60, "strict (n>=5, WR>=60%)"),
        (10, 0.55, "conservative (n>=10, WR>=55%)"),
    ]

    results_by_threshold = []

    for min_n, min_wr, label in thresholds:
        print(f"Testing {label}...")

        voter = EnsembleVoterV2(RESULTS_DIR, min_n_oos=min_n, min_wr_oos=min_wr)
        pool_stats = voter.get_pool_stats()
        print(f"  Pool: {pool_stats['total_configs']} qualified configs\n")

        # Apply voting
        trades_is_voted = apply_voting_filter(trades_is, voter)
        trades_oos_voted = apply_voting_filter(trades_oos, voter)

        print(f"  After voting: IS {len(trades_is)} -> {len(trades_is_voted)} "
              f"({len(trades_is_voted)/len(trades_is)*100:.1f}%), "
              f"OOS {len(trades_oos)} -> {len(trades_oos_voted)} "
              f"({len(trades_oos_voted)/len(trades_oos)*100:.1f}%)\n")

        # Evaluate
        stats_is_v = evaluate_trades(trades_is_voted, f"IS_{label}")
        stats_oos_v = evaluate_trades(trades_oos_voted, f"OOS_{label}")

        results_by_threshold.append({
            "threshold": label,
            "pool_size": pool_stats['total_configs'],
            "n_IS": len(trades_is_voted),
            "wr_IS": stats_is_v["wr"],
            "exp_R_IS": stats_is_v["exp_R"],
            "n_OOS": len(trades_oos_voted),
            "wr_OOS": stats_oos_v["wr"],
            "exp_R_OOS": stats_oos_v["exp_R"],
            "wr_drop": stats_oos_v["wr"] - stats_is_v["wr"],
        })

    results_df = pd.DataFrame(results_by_threshold)
    print("\n=== RESULTS SUMMARY ===\n")
    print(results_df.to_string(index=False))

    # Best threshold
    print("\n=== RECOMMENDATION ===\n")
    best_idx = results_df["exp_R_OOS"].idxmax()
    best = results_df.iloc[best_idx]
    print(f"Best: {best['threshold']}")
    print(f"  OOS Performance: n={int(best['n_OOS'])} trades, WR={best['wr_OOS']:.1%}, Exp={best['exp_R_OOS']:.3f}R")
    print(f"  IS->OOS WR drop: {best['wr_drop']:+.1%}")
    print()


if __name__ == "__main__":
    main()
