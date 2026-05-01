from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.deribit_futures.dvol import (
    DvolDetectorConfig,
    detect_dvol_variation,
    format_dvol_signal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect DVOL variation from Deribit volatility index data.")
    parser.add_argument("--asset", type=str, default="BTC")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--notify", action="store_true", help="Send detector output to Discord")
    args = parser.parse_args()

    cfg = DvolDetectorConfig(asset=args.asset.upper(), timeframe=args.timeframe, days=args.days)
    payload = detect_dvol_variation(cfg)
    msg = format_dvol_signal(payload)
    print(msg)

    if args.notify:
        webhook = (
            os.environ.get("DISCORD_WEBHOOK_DVOL_URL", "")
            or os.environ.get("DISCORD_WEBHOOK_DERIBIT_URL", "")
            or os.environ.get("DISCORD_WEBHOOK_URL", "")
        )
        if not webhook:
            raise RuntimeError(
                "DISCORD_WEBHOOK_DVOL_URL or DISCORD_WEBHOOK_DERIBIT_URL or DISCORD_WEBHOOK_URL is not set."
            )
        resp = requests.post(webhook, json={"content": msg}, timeout=10)
        resp.raise_for_status()
        print("\nDiscord notification sent.")


if __name__ == "__main__":
    main()
