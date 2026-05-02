"""
Fractal Detectors API
Endpoints: /api/fractal/strict, /api/fractal/modere, /api/fractal/frequent
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime, timedelta
import json

from orchestrator import FractalOrchestrator

app = FastAPI(title="Fractal Detectors")

# Global orchestrator instance
orchestrator = None


class SignalResponse(BaseModel):
    timestamp: str
    setup: str
    day_date: str
    kz: str
    pattern: str
    entry_price: float
    confidence: float
    levels: Dict


class DetectionRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    active_setups: Optional[List[str]] = None  # ['STRICT', 'MODÉRÉ', 'FRÉQUENT']


class StrategyConfig(BaseModel):
    enabled_setups: List[str] = ["STRICT", "MODÉRÉ", "FRÉQUENT"]
    discord_webhook: Optional[str] = None


strategy_config = StrategyConfig()


@app.on_event("startup")
async def startup():
    global orchestrator
    orchestrator = FractalOrchestrator(
        discord_webhook_url=strategy_config.discord_webhook
    )


@app.post("/api/fractal/config")
async def update_config(config: StrategyConfig):
    """Met à jour la configuration des stratégies activées"""
    global strategy_config, orchestrator
    strategy_config = config
    if orchestrator:
        orchestrator.discord_webhook = config.discord_webhook
    return {"status": "success", "config": config}


@app.post("/api/fractal/detect")
async def detect_signals(request: DetectionRequest, background_tasks: BackgroundTasks):
    """Détecte les signaux fractal"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    try:
        # Charger les données (à adapter selon votre source de données)
        # Pour l'instant, retourner une structure de test
        signals = await orchestrator.detect_and_notify(
            df_m15=pd.DataFrame(),  # À charger depuis Binance
            daily=pd.DataFrame(),   # À charger depuis Binance
            weekly=pd.DataFrame(),  # À charger depuis Binance
            active_setups=request.active_setups or strategy_config.enabled_setups
        )

        return {
            "status": "success",
            "signals_count": len(signals),
            "signals": signals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/fractal/strict")
async def get_strict_signals():
    """Retourne les signaux STRICT (W+D+KZ+BR)"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    strict_signals = [s for s in orchestrator.signals_log if s['setup'] == 'STRICT']
    return {
        "setup": "STRICT",
        "count": len(strict_signals),
        "confidence": 0.946,
        "signals": strict_signals[-10:]  # Retourner les 10 derniers signaux
    }


@app.get("/api/fractal/modere")
async def get_modere_signals():
    """Retourne les signaux MODÉRÉ (D+KZ+BR)"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    modere_signals = [s for s in orchestrator.signals_log if s['setup'] == 'MODÉRÉ']
    return {
        "setup": "MODÉRÉ",
        "count": len(modere_signals),
        "confidence": 0.91,
        "signals": modere_signals[-10:]  # Retourner les 10 derniers signaux
    }


@app.get("/api/fractal/frequent")
async def get_frequent_signals():
    """Retourne les signaux FRÉQUENT (KZ+BR)"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    frequent_signals = [s for s in orchestrator.signals_log if s['setup'] == 'FRÉQUENT']
    return {
        "setup": "FRÉQUENT",
        "count": len(frequent_signals),
        "confidence": 0.875,
        "signals": frequent_signals[-10:]  # Retourner les 10 derniers signaux
    }


@app.get("/api/fractal/stats")
async def get_statistics():
    """Retourne les statistiques globales"""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")

    summary = orchestrator.get_signals_summary()
    return {
        "total_signals": summary.get("total", 0),
        "by_setup": summary.get("by_setup", {}),
        "by_pattern": summary.get("by_pattern", {}),
        "uptime": datetime.utcnow().isoformat()
    }


@app.get("/api/fractal/health")
async def health_check():
    """Vérification de santé de l'API"""
    return {
        "status": "healthy",
        "orchestrator": "active" if orchestrator else "inactive",
        "config": strategy_config.dict()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
