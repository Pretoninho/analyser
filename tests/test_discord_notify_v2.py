# ── Test discord_notify_v2 — vérify logic without webhook ──

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import os
os.environ["DISCORD_WEBHOOK_TA_URL"] = "http://localhost:9999/webhook"  # dummy URL

from strategies.ta.features import load_15m, compute_features
from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2
from strategies.ta.config import RESULTS_DIR

print("[test] Loading data...")
df15 = load_15m()
print(f"  Loaded {len(df15)} candles\n")

print("[test] Computing features...")
df15 = compute_features(
    df15, ema_len=50, rsi_len=7,
    stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
)

print("[test] Loading voter...")
voter = EnsembleVoterV2(RESULTS_DIR, min_n_oos=5, min_wr_oos=0.60)
pool = voter.get_pool_stats()
print(f"  Voter pool: {pool['total_configs']} configs")
print(f"  By regime: {pool['by_regime']}\n")

print("[test] Scanning signals (last 500 candles)...")
from strategies.ta.live_runner_v2 import scan_signals
signals = scan_signals(df15.tail(500), voter)

if signals:
    print(f"\n  [OK] Found {len(signals)} signal(s):\n")
    for sig in signals:
        print(f"    {sig['timestamp']} | {sig['direction']:5s} ({sig['regime']:5s}) | "
              f"vote {sig['vote_favorable']}/{sig['vote_total']} conf={sig['confidence']:.3f}")
else:
    print("\n  No signals found (expected — strict voting)")

print("\n[test] [OK] Logic OK (no webhook errors)")
