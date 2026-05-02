# ── discord_notify_v2.py — Signaux TA avec Ensemble Voting ──
#
# Nouvelle version avec voting intégré.
# Appelée par scheduler toutes les 15min en session.

import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
from datetime import datetime

from strategies.ta.features import load_15m, compute_features
from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2
from strategies.ta.signal_logger import log_signal
from strategies.ta.config import RESULTS_DIR

WEBHOOK_ENV = "DISCORD_WEBHOOK_TA_URL"
_DIR_EMOJI = {"LONG": "🟢", "SHORT": "🔴"}
_REGIME_EMOJI = {"bull": "🐂", "bear": "🐻", "range": "↔️"}


def _format_message_v2(signals: list) -> str:
    """Formate message Discord pour signaux avec voting."""
    if not signals:
        return None

    sig = signals[0]  # premier signal (récent)

    dir_emoji = _DIR_EMOJI.get(sig["direction"], "")
    regime_emoji = _REGIME_EMOJI.get(sig["regime"], "")

    entry_px = sig.get("entry_price", 0)
    vote_fav = sig.get("vote_favorable", 0)
    vote_tot = sig.get("vote_total", 0)
    confidence = sig.get("confidence", 0.0)
    ts = sig.get("timestamp", datetime.utcnow()).isoformat()[:16].replace("T", " ")

    lines = [
        f"## {dir_emoji} TA Signal v2 — **{sig['direction']}** BTC/USDT 15m",
        f"`{ts} UTC` — Entry: **${entry_px:,.2f}**",
        f"Régime: {regime_emoji} **{sig['regime'].upper()}**",
        "",
        f"**Ensemble Voting:** {vote_fav}/{vote_tot} configs votent → **{confidence:.2f} confidence**",
        "",
        f"Features: EMA={sig['ema_state']:+d} slope={sig['ema_slope']:+d} | "
        f"Swing={sig['swing']:+d} | RSI={sig['rsi_state']} | "
        f"Stoch={sig['stoch_state']} | ATR={sig['atr_state']}",
    ]

    return "\n".join(lines)


def scan_and_notify_v2(symbol: str = "BTCUSDT") -> bool:
    """
    Lance le scan TA v2 avec ensemble voting et envoie Discord si signaux.
    """
    webhook_url = os.environ.get(WEBHOOK_ENV, "")
    if not webhook_url:
        print(f"[ta_notify_v2] Skipped — {WEBHOOK_ENV} not set", flush=True)
        return False

    try:
        print("[ta_notify_v2] Loading data...", flush=True)

        # Load features
        df15 = load_15m()
        if len(df15) < 100:
            print("[ta_notify_v2] Insufficient data", flush=True)
            return False

        # Compute features (fixed params)
        df15 = compute_features(
            df15, ema_len=50, rsi_len=7,
            stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
        )

        # Load voter
        print("[ta_notify_v2] Loading ensemble voter...", flush=True)
        voter = EnsembleVoterV2(RESULTS_DIR, min_n_oos=5, min_wr_oos=0.60)

        # Scan signals (from live_runner_v2)
        from strategies.ta.live_runner_v2 import scan_signals
        signals = scan_signals(df15, voter)

        if not signals:
            print("[ta_notify_v2] No signals with consensus", flush=True)
            return False

        print(f"[ta_notify_v2] Found {len(signals)} signal(s) with consensus", flush=True)

        # Format et envoyer
        msg = _format_message_v2(signals)
        if not msg:
            return False

        print(f"[ta_notify_v2] Sending to Discord: {len(msg)} chars", flush=True)
        resp = requests.post(webhook_url, json={"content": msg}, timeout=10)
        resp.raise_for_status()

        # Log signals
        for sig in signals:
            log_signal({
                "timestamp": sig["timestamp"],
                "direction": sig["direction"],
                "regime": sig["regime"],
                "entry_price": sig["entry_price"],
                "confidence": sig["confidence"],
                "vote_favorable": sig["vote_favorable"],
                "vote_total": sig["vote_total"],
            })

        print(f"[ta_notify_v2] Signal sent and logged", flush=True)
        return True

    except Exception as e:
        print(f"[ta_notify_v2] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    scan_and_notify_v2()
