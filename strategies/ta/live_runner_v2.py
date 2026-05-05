# ── Live Runner v2 — Avec Ensemble Voting ──
#
# Moteur de détection TA temps réel avec ensemble voting v2 intégré.
# Signal valide seulement si majority des configs qualifiées votent.

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

from strategies.ta.config import (
    EMA_LENGTHS, RSI_LENGTHS, STOCH_PARAMS, ATR_LENGTHS,
    SESSIONS_UTC, RESULTS_DIR, SYMBOL,
    RSI_BINS, RSI_LABELS, STOCH_BINS, STOCH_LABELS,
    ATR_BINS, ATR_LABELS,
    REGIME_EMA_LEN, REGIME_SLOPE_DAYS,
)
from strategies.ta.features import (
    _ema, _rsi, _atr, _stoch_k, _vwap_daily, _swing_state_4h, _regime_daily
)
from strategies.ta.ensemble_voting_v2 import EnsembleVoterV2

# ── Constantes ────────────────────────────────────────────────────────────────
BINANCE_URL    = "https://api.binance.com/api/v3/klines"
REF_PARAMS     = "EMA50_RSI14_SK14SS3SD3_ATR14"

# Seuil minimal de corps pour éviter les doji (en % de la bougie)
DOJI_THRESHOLD = 0.1  # corps < 10% du range total = doji, ignoré

# Singleton du voter : chargé une seule fois au niveau module
_voter_singleton: "EnsembleVoterV2 | None" = None


def get_voter(min_n_oos: int = VOTING_MIN_N_OOS,
              min_wr_oos: float = VOTING_MIN_WR_OOS) -> "EnsembleVoterV2":
    """Retourne un voter singleton (relit le CSV seulement au 1er appel)."""
    global _voter_singleton
    if _voter_singleton is None:
        _voter_singleton = EnsembleVoterV2(
            RESULTS_DIR, min_n_oos=min_n_oos, min_wr_oos=min_wr_oos
        )
    return _voter_singleton
# ─────────────────────────────────────────────────────────────────────────────
# Fetch Binance
# ─────────────────────────────────────────────────────────────────────────────

def _parse_klines(data: list) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_ts", "qv", "trades", "tbbv", "tbqv", "ignore",
    ])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype("float64")
    return df.set_index("ts")[["open", "high", "low", "close", "volume"]]


def fetch_klines(interval: str, limit: int, symbol: str = SYMBOL) -> pd.DataFrame:
    resp = requests.get(
        BINANCE_URL,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return _parse_klines(resp.json())


# ─────────────────────────────────────────────────────────────────────────────
# Feature Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_2bar_reversal(df15: pd.DataFrame, idx: int) -> tuple:
    """
    Détecte 2-bar reversal à index idx.

    Returns:
        (direction, is_reversal) — direction='LONG'|'SHORT', is_reversal=bool
    """
    if idx < 2:
        return None, False

    rng = (df15["high"].values - df15["low"].values)  # range de chaque bougie
    body_abs = (df15["close"].values - df15["open"].values)
    # Ignorer les doji : corps < DOJI_THRESHOLD * range
    is_doji = np.where(
        rng > 0, np.abs(body_abs) / rng < DOJI_THRESHOLD, True
    )
    body = np.where(is_doji, 0, np.sign(body_abs))
    b0, b1, b2 = body[idx], body[idx - 1], body[idx - 2]

    long_trigger  = (b0 > 0) and (b1 < 0) and (b2 < 0)
    short_trigger = (b0 < 0) and (b1 > 0) and (b2 > 0)

    if long_trigger:
        return "LONG", True
    elif short_trigger:
        return "SHORT", True
    else:
        return None, False


def get_regime_at_idx(df15: pd.DataFrame, idx: int) -> str:
    """Retourne le régime (bull/bear/range) au moment de idx."""
    try:
        regime = df15.iloc[idx]["regime"]
        return str(regime) if regime in ["bull", "bear", "range"] else "bull"
    except Exception as e:
        logging.warning("[get_regime_at_idx] idx=%d err=%s", idx, e)
        return "bull"


def get_features_at_idx(df15: pd.DataFrame, idx: int) -> dict:
    """Extrait features au moment de idx."""
    try:
        row = df15.iloc[idx]
        return {
            "ema_state": int(row.get("ema_state", 0) or 0),
            "ema_slope": int(row.get("ema_slope", 0) or 0),
            "swing": int(row.get("swing", 0) or 0),
            "rsi_state": str(row.get("rsi_state", "weak")),
            "stoch_state": str(row.get("stoch_state", "weak")),
            "atr_state": str(row.get("atr_state", "neutral")),
            "vwap_state": int(row.get("vwap_state", 0) or 0),
        }
    except Exception as e:
        logging.warning("[get_features_at_idx] idx=%d err=%s", idx, e)
        return {
            "ema_state": 0, "ema_slope": 0, "swing": 0,
            "rsi_state": "weak", "stoch_state": "weak",
            "atr_state": "neutral", "vwap_state": 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Detection
# ─────────────────────────────────────────────────────────────────────────────

def scan_signals(df15: pd.DataFrame, voter: EnsembleVoterV2) -> list:
    """
    Scan pour tous les signaux 2-bar avec ensemble voting.

    Returns:
        list of dicts {timestamp, direction, regime, features, vote_count, confidence}
    """
    signals = []

    # Check session mask
    h = df15.index.hour
    session_mask = np.zeros(len(df15), dtype=bool)
    for start, end in SESSIONS_UTC:
        session_mask |= (h >= start) & (h < end)

    # Scan chaque bar
    for i in range(2, len(df15)):
        if not session_mask[i]:
            continue

        # Detect reversal
        direction, is_reversal = detect_2bar_reversal(df15, i)
        if not is_reversal:
            continue

        # Get regime et features
        regime = get_regime_at_idx(df15, i)
        features = get_features_at_idx(df15, i)

        # Ensemble voting
        total_voters, favorable_votes, consensus_dir, confidence = voter.vote(
            regime, features, direction
        )

        if consensus_dir is None:
            continue

        signals.append({
            "timestamp": df15.index[i],
            "direction": direction,
            "regime": regime,
            "ema_state": features["ema_state"],
            "ema_slope": features["ema_slope"],
            "swing": features["swing"],
            "rsi_state": features["rsi_state"],
            "stoch_state": features["stoch_state"],
            "atr_state": features["atr_state"],
            "vwap_state": features["vwap_state"],
            "entry_price": float(df15.iloc[i + 1]["open"]) if i + 1 < len(df15) else float(df15.iloc[i]["close"]),
            "vote_total": total_voters,
            "vote_favorable": favorable_votes,
            "confidence": confidence,
        })

    return signals


def main():
    print(f"[live_runner_v2] {datetime.utcnow().isoformat()} UTC\n")

    # Load ensemble voter (singleton)
    print("[live_runner_v2] Loading ensemble voter...\n")
    voter = get_voter()
    pool_stats = voter.get_pool_stats()
    print(f"  Voter pool: {pool_stats['total_configs']} qualified configs")
    print(f"    By regime: {pool_stats['by_regime']}")
    print(f"    By direction: {pool_stats['by_direction']}\n")

    # Fetch klines
    print("[live_runner_v2] Fetching Binance data...\n")
    df15 = fetch_klines("15m", 500)  # ~5 days
    print(f"  Loaded {len(df15)} candles 15m")
    print(f"  Period: {df15.index[0].date()} to {df15.index[-1].date()}\n")

    # Compute features (fixed params for ref)
    print("[live_runner_v2] Computing features...\n")
    from strategies.ta.features import compute_features
    df15 = compute_features(
        df15, ema_len=50, rsi_len=7,
        stoch_k_period=5, stoch_smooth_k=3, stoch_d_period=3, atr_len=7
    )

    # Scan signals
    print("[live_runner_v2] Scanning signals with ensemble voting...\n")
    signals = scan_signals(df15, voter)

    if signals:
        print(f"  Found {len(signals)} signals:\n")
        for sig in signals:
            print(f"  {sig['timestamp'].isoformat():20s} | "
                  f"{sig['direction']:5s} ({sig['regime']:5s}) | "
                  f"vote={sig['vote_favorable']}/{sig['vote_total']} "
                  f"conf={sig['confidence']:.3f}")
    else:
        print("  No signals with consensus.\n")

    return signals


if __name__ == "__main__":
    signals = main()
