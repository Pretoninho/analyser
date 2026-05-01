"""
discord_notify.py — Envoi des signaux TA sur Discord.

Appelé par le scheduler api/app.py toutes les 15 minutes en session.
Variable d'environnement requise : DISCORD_WEBHOOK_TA_URL
"""

import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests
import pandas as pd

from strategies.ta.live_runner import scan

WEBHOOK_ENV = "DISCORD_WEBHOOK_TA_URL"

# Emojis direction
_DIR_EMOJI = {"LONG": "🟢", "SHORT": "🔴"}
_REGIME_EMOJI = {"bull": "🐂", "bear": "🐻", "range": "↔️"}


def _format_message(result: dict) -> str:
    matches    = result["matches"]
    trigger    = result["trigger"]
    last_price = result["last_price"]
    last_bar   = result["last_bar_time"][:16].replace("T", " ")
    cs         = result["current_state"]
    regime     = cs.get("regime", "?")

    dir_emoji    = _DIR_EMOJI.get(trigger, "")
    regime_emoji = _REGIME_EMOJI.get(regime, "")

    lines = [
        f"## {dir_emoji} TA Signal — **{trigger}** BTC/USDT 15m",
        f"`{last_bar} UTC` — Prix : **${last_price:,.2f}**",
        f"Régime : {regime_emoji} **{regime.upper()}** | "
        f"EMA: {cs.get('ema_state','')} | Swing: {cs.get('swing','')} | "
        f"RSI: {cs.get('rsi_state','')} | Stoch: {cs.get('stoch_state','')} | "
        f"ATR: {cs.get('atr_state','')}",
        "",
        f"**{len(matches)} config(s) validée(s) IS+OOS :**",
    ]

    for i, m in enumerate(matches[:8], 1):
        lines.append(
            f"`{i}.` `{m['params']}` "
            f"WR OOS **{m['wr_OOS']:.1%}** | "
            f"Exp **{m['exp_R_OOS']:.2f}R** | "
            f"n={m['n_OOS']}"
        )

    if len(matches) > 8:
        lines.append(f"_...et {len(matches) - 8} autres configs_")

    return "\n".join(lines)


def run_and_notify(symbol: str = "BTCUSDT") -> bool:
    """
    Lance le scan TA et envoie un message Discord si signal actif.
    Retourne True si un signal a été envoyé.
    """
    webhook_url = os.environ.get(WEBHOOK_ENV, "")
    if not webhook_url:
        print(f"[ta_notify] Skipped — {WEBHOOK_ENV} not set", flush=True)
        return False

    try:
        result = scan(symbol)
    except Exception as e:
        print(f"[ta_notify] Scan failed: {e}", flush=True)
        return False

    trigger  = result.get("trigger")
    matches  = result.get("matches", [])
    n        = result.get("n_matches", 0)

    if not trigger or n == 0:
        regime = result.get("current_state", {}).get("regime", "?")
        print(
            f"[ta_notify] No signal "
            f"(trigger={result.get('trigger_raw')}, in_session={result.get('in_session')}, "
            f"matches={n}, regime={regime})",
            flush=True,
        )
        return False

    msg = _format_message(result)
    try:
        resp = requests.post(
            webhook_url,
            json={"content": msg},
            timeout=10,
        )
        resp.raise_for_status()
        print(
            f"[ta_notify] Signal sent: {trigger} | {n} matches | "
            f"regime={result['current_state'].get('regime')} | "
            f"top_exp={matches[0]['exp_R_OOS']:.2f}R",
            flush=True,
        )
        return True
    except Exception as e:
        print(f"[ta_notify] Discord send failed: {e}", flush=True)
        return False


if __name__ == "__main__":
    sent = run_and_notify()
    print("Signal envoyé." if sent else "Aucun signal.")
