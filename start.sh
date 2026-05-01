#!/bin/bash
# Script de démarrage conditionnel.
# Si APP_TYPE=dashboard → Streamlit TA
# Sinon → FastAPI Pi* (comportement par défaut)
if [ "$APP_TYPE" = "dashboard" ]; then
    streamlit run strategies/ta/live_dashboard.py \
        --server.port "$PORT" \
        --server.headless true \
        --server.enableCORS false
else
    python -m uvicorn api.app:app --host 0.0.0.0 --port "$PORT"
fi
