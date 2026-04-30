from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.deribit_futures.signal import SignalConfig, build_deribit_signal, format_discord_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Deribit futures actionable signal.")
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--notify", action="store_true", help="Send signal to Discord webhook")
    args = parser.parse_args()

    cfg = SignalConfig(asset=args.asset, timeframe=args.timeframe, days=args.days)
    signal = build_deribit_signal(cfg)

    msg = format_discord_signal(signal)
    print(msg)

    if args.notify:
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
        if not webhook:
            raise RuntimeError("DISCORD_WEBHOOK_URL is not set.")
        resp = requests.post(webhook, json={"content": msg}, timeout=10)
        resp.raise_for_status()
        print("\nDiscord notification sent.")


if __name__ == "__main__":
    main()
