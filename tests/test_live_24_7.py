# ── Test Live 24/7 — Signaux par jour de semaine ──

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strategies.ta.features import load_15m, compute_features
from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2
from strategies.ta.config import RESULTS_DIR
from strategies.ta.live_runner_v2 import scan_signals
import pandas as pd

print("[test_live_24_7] Loading data...\n")

df15 = load_15m()
df15 = compute_features(
    df15, ema_len=50, rsi_len=7,
    stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
)

print("[test_live_24_7] Loading voter...\n")
voter = EnsembleVoterV2(RESULTS_DIR, min_n_oos=5, min_wr_oos=0.60)

print("[test_live_24_7] Scanning last 10 days with all signals...\n")

# Last 10 days
df_last_10 = df15.tail(10 * 96)  # 96 * 15m = 24h
signals = scan_signals(df_last_10, voter)

if not signals:
    print("  No signals detected")
else:
    # Groupe par jour de semaine
    signals_df = pd.DataFrame(signals)
    signals_df["day_name"] = signals_df["timestamp"].dt.day_name()
    signals_df["date"] = signals_df["timestamp"].dt.date

    print(f"  Total signals: {len(signals)}\n")

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in days_order:
        sub = signals_df[signals_df["day_name"] == day]
        if sub.empty:
            continue

        print(f"  {day:10s} ({sub['date'].unique()[0]}):")
        for _, sig in sub.iterrows():
            ts = sig["timestamp"].strftime("%H:%M UTC")
            vote_fav = sig["vote_favorable"]
            vote_tot = sig["vote_total"]
            conf = sig["confidence"]
            direction = sig["direction"]
            regime = sig["regime"]
            print(f"    {ts} | {direction:5s} ({regime:5s}) | vote {vote_fav:2d}/{vote_tot:2d} | conf={conf:.2f}")
        print()
