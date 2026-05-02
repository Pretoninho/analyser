"""
Fractal Signal Orchestrator
Manages all three setups (STRICT, MODÉRÉ, FRÉQUENT) and sends Discord notifications
"""
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
import json
from typing import List, Dict, Optional

from detector_strict import FractalDetectorStrict
from detector_modere import FractalDetectorModere
from detector_frequent import FractalDetectorFrequent


class FractalOrchestrator:
    def __init__(self, discord_webhook_url: str = None):
        self.strict_detector = FractalDetectorStrict()
        self.modere_detector = FractalDetectorModere()
        self.frequent_detector = FractalDetectorFrequent()

        self.discord_webhook = discord_webhook_url
        self.signals_log = []

    async def send_discord_notification(self, signal: Dict, setup_tag: str):
        """Envoie une notification Discord pour un signal détecté"""
        if not self.discord_webhook:
            return

        embed = {
            "title": f"🎯 Fractal Signal [{setup_tag}]",
            "description": f"**{signal['pattern']}** - {signal['kz']}",
            "color": 16711680 if signal['pattern'] == 'DOWN->UP' else 65280,  # Red or Green
            "fields": [
                {
                    "name": "Setup",
                    "value": signal['setup'],
                    "inline": True
                },
                {
                    "name": "Kill Zone",
                    "value": signal['kz'],
                    "inline": True
                },
                {
                    "name": "Pattern",
                    "value": signal['pattern'],
                    "inline": True
                },
                {
                    "name": "Confidence",
                    "value": f"{signal['confidence']*100:.1f}%",
                    "inline": True
                },
                {
                    "name": "Entry Price",
                    "value": f"{signal['entry_price']:.2f}",
                    "inline": True
                },
                {
                    "name": "Day Date",
                    "value": str(signal['day_date']),
                    "inline": True
                },
                {
                    "name": "Levels",
                    "value": self._format_levels(signal['levels']),
                    "inline": False
                }
            ],
            "timestamp": signal['timestamp'].isoformat()
        }

        payload = {"embeds": [embed]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json=payload) as resp:
                    if resp.status != 204:
                        print(f"Discord notification failed: {resp.status}")
        except Exception as e:
            print(f"Error sending Discord notification: {e}")

    def _format_levels(self, levels: Dict) -> str:
        """Formate les niveaux de prix pour l'affichage Discord"""
        formatted = []
        for key, value in levels.items():
            readable_key = key.replace('_', ' ').title()
            formatted.append(f"{readable_key}: {value:.2f}")
        return "\n".join(formatted)

    async def detect_and_notify(self, df_m15: pd.DataFrame, daily: pd.DataFrame,
                                 weekly: pd.DataFrame, active_setups: List[str] = None):
        """
        Détecte les signaux pour tous les setups et envoie les notifications
        active_setups: ['STRICT', 'MODÉRÉ', 'FRÉQUENT'] ou None pour tous
        """
        if active_setups is None:
            active_setups = ['STRICT', 'MODÉRÉ', 'FRÉQUENT']

        all_signals = []

        # Détection STRICT (W+D+KZ+BR)
        if 'STRICT' in active_setups:
            strict_signals = self.strict_detector.detect(df_m15, daily, weekly)
            for signal in strict_signals:
                await self.send_discord_notification(signal, '[STRICT]')
                all_signals.append(signal)
                self.signals_log.append({
                    **signal,
                    'detected_at': datetime.utcnow(),
                    'setup_tag': '[STRICT]'
                })

        # Détection MODÉRÉ (D+KZ+BR)
        if 'MODÉRÉ' in active_setups:
            modere_signals = self.modere_detector.detect(df_m15, daily)
            for signal in modere_signals:
                # Éviter les doublons avec STRICT
                is_duplicate = any(
                    s['day_date'] == signal['day_date'] and
                    s['kz'] == signal['kz'] and
                    s['pattern'] == signal['pattern'] and
                    s['setup'] == 'STRICT'
                    for s in all_signals
                )
                if not is_duplicate:
                    await self.send_discord_notification(signal, '[MODÉRÉ]')
                    all_signals.append(signal)
                    self.signals_log.append({
                        **signal,
                        'detected_at': datetime.utcnow(),
                        'setup_tag': '[MODÉRÉ]'
                    })

        # Détection FRÉQUENT (KZ+BR)
        if 'FRÉQUENT' in active_setups:
            frequent_signals = self.frequent_detector.detect(df_m15)
            for signal in frequent_signals:
                # Éviter les doublons avec STRICT et MODÉRÉ
                is_duplicate = any(
                    s['day_date'] == signal['day_date'] and
                    s['kz'] == signal['kz'] and
                    s['pattern'] == signal['pattern'] and
                    s['setup'] in ['STRICT', 'MODÉRÉ']
                    for s in all_signals
                )
                if not is_duplicate:
                    await self.send_discord_notification(signal, '[FRÉQUENT]')
                    all_signals.append(signal)
                    self.signals_log.append({
                        **signal,
                        'detected_at': datetime.utcnow(),
                        'setup_tag': '[FRÉQUENT]'
                    })

        return all_signals

    def get_signals_summary(self) -> Dict:
        """Retourne un résumé des signaux détectés"""
        if not self.signals_log:
            return {"total": 0, "by_setup": {}}

        summary = {
            "total": len(self.signals_log),
            "by_setup": {},
            "by_pattern": {}
        }

        for signal in self.signals_log:
            setup = signal['setup']
            pattern = signal['pattern']

            summary["by_setup"][setup] = summary["by_setup"].get(setup, 0) + 1
            summary["by_pattern"][pattern] = summary["by_pattern"].get(pattern, 0) + 1

        return summary
