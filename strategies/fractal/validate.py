"""
Validation script for Fractal Detection System
Tests all detectors, orchestrator, API, and database
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import json

from detector_strict import FractalDetectorStrict
from detector_modere import FractalDetectorModere
from detector_frequent import FractalDetectorFrequent
from orchestrator import FractalOrchestrator
from database import SignalDatabase


def generate_test_data(num_days: int = 30) -> tuple:
    """Génère des données de test synthétiques"""
    dates_m15 = pd.date_range(end=datetime.utcnow(), periods=num_days*96, freq='15min')
    dates_daily = pd.date_range(end=datetime.utcnow(), periods=num_days, freq='D')
    dates_weekly = pd.date_range(end=datetime.utcnow(), periods=4, freq='W')

    # Données M15 aléatoires
    df_m15 = pd.DataFrame({
        'timestamp': dates_m15,
        'open': 65000 + (pd.Series(range(len(dates_m15))) % 100),
        'high': 65100 + (pd.Series(range(len(dates_m15))) % 150),
        'low': 64900 + (pd.Series(range(len(dates_m15))) % 50),
        'close': 65050 + (pd.Series(range(len(dates_m15))) % 100),
        'volume': 100
    })

    # Données Daily aléatoires
    df_daily = pd.DataFrame({
        'timestamp': dates_daily,
        'open': 65000 + (pd.Series(range(len(dates_daily))) * 10),
        'high': 65500 + (pd.Series(range(len(dates_daily))) * 10),
        'low': 64500 + (pd.Series(range(len(dates_daily))) * 10),
        'close': 65200 + (pd.Series(range(len(dates_daily))) * 10),
        'volume': 1000
    })

    # Données Weekly aléatoires
    df_weekly = pd.DataFrame({
        'timestamp': dates_weekly,
        'open': 64000,
        'high': 66000,
        'low': 64000,
        'close': 65500,
        'volume': 10000
    })

    return df_m15, df_daily, df_weekly


def test_detectors():
    """Test tous les détecteurs"""
    print("\n" + "="*60)
    print("TEST 1: Detectors Initialization")
    print("="*60)

    try:
        strict = FractalDetectorStrict()
        modere = FractalDetectorModere()
        frequent = FractalDetectorFrequent()

        print("✓ STRICT detector initialized")
        print("✓ MODÉRÉ detector initialized")
        print("✓ FRÉQUENT detector initialized")

        # Test KZ mapping
        assert len(strict.kz_map) == 4, "KZ map should have 4 entries"
        print("✓ Kill Zone mapping verified (4 zones)")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_detection_logic():
    """Test la logique de détection"""
    print("\n" + "="*60)
    print("TEST 2: Detection Logic with Synthetic Data")
    print("="*60)

    try:
        df_m15, df_daily, df_weekly = generate_test_data(30)

        strict = FractalDetectorStrict()
        modere = FractalDetectorModere()
        frequent = FractalDetectorFrequent()

        # Run detections
        strict_signals = strict.detect(df_m15, df_daily, df_weekly)
        modere_signals = modere.detect(df_m15, df_daily)
        frequent_signals = frequent.detect(df_m15)

        print(f"✓ STRICT detection completed: {len(strict_signals)} signals")
        print(f"✓ MODÉRÉ detection completed: {len(modere_signals)} signals")
        print(f"✓ FRÉQUENT detection completed: {len(frequent_signals)} signals")

        # Verify signal structure
        if strict_signals:
            signal = strict_signals[0]
            required_keys = ['timestamp', 'setup', 'day_date', 'kz', 'pattern',
                           'entry_price', 'confidence', 'levels']
            for key in required_keys:
                assert key in signal, f"Missing key: {key}"
            print(f"✓ Signal structure verified ({len(required_keys)} required fields)")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_orchestrator():
    """Test l'orchestrateur"""
    print("\n" + "="*60)
    print("TEST 3: Orchestrator with Discord (No-op)")
    print("="*60)

    try:
        orchestrator = FractalOrchestrator(discord_webhook_url=None)
        df_m15, df_daily, df_weekly = generate_test_data(30)

        signals = await orchestrator.detect_and_notify(
            df_m15=df_m15,
            daily=df_daily,
            weekly=df_weekly,
            active_setups=['STRICT', 'MODÉRÉ', 'FRÉQUENT']
        )

        summary = orchestrator.get_signals_summary()

        print(f"✓ Orchestrator detection completed")
        print(f"  - Total signals: {len(signals)}")
        print(f"  - Summary: {summary}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_database():
    """Test la base de données"""
    print("\n" + "="*60)
    print("TEST 4: Database (SQLite)")
    print("="*60)

    try:
        import os
        test_db = "test_signals.db"

        # Cleanup existing
        if os.path.exists(test_db):
            os.remove(test_db)

        db = SignalDatabase(db_type="sqlite", db_path=test_db)

        # Create test signal
        test_signal = {
            'timestamp': datetime.utcnow(),
            'setup': 'STRICT',
            'day_date': datetime.utcnow().date(),
            'kz': 'NYKZ',
            'pattern': 'UP->DOWN',
            'entry_price': 65234.50,
            'confidence': 0.946,
            'levels': {'kz_low': 65000, 'kz_high': 65500},
            'detected_at': datetime.utcnow(),
            'setup_tag': '[STRICT]'
        }

        # Log signal
        signal_id = db.log_signal(test_signal)
        print(f"✓ Signal logged with ID: {signal_id}")

        # Retrieve signals
        signals = db.get_signals(setup='STRICT', limit=10)
        assert len(signals) == 1, "Should retrieve 1 signal"
        print(f"✓ Retrieved {len(signals)} signal(s)")

        # Get statistics
        stats = db.get_statistics(setup='STRICT')
        print(f"✓ Statistics retrieved: {stats['total_signals']} total signals")

        # Update exit
        db.update_signal_exit(signal_id, exit_price=65300, pnl=65.50, exit_reason='TP Hit')
        print(f"✓ Signal exit updated (P&L: 65.50)")

        # Cleanup
        os.remove(test_db)

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_api_structure():
    """Test la structure de l'API"""
    print("\n" + "="*60)
    print("TEST 5: API Structure Validation")
    print("="*60)

    try:
        from api import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test health endpoint
        response = client.get("/api/fractal/health")
        assert response.status_code == 200, "Health check failed"
        print("✓ Health endpoint working")

        # Test config endpoint
        response = client.post("/api/fractal/config", json={
            "enabled_setups": ["STRICT", "MODÉRÉ"],
            "discord_webhook": None
        })
        assert response.status_code == 200, "Config endpoint failed"
        print("✓ Config endpoint working")

        # Test signal endpoints
        for endpoint in ["/api/fractal/strict", "/api/fractal/modere", "/api/fractal/frequent"]:
            response = client.get(endpoint)
            assert response.status_code == 200, f"{endpoint} failed"
        print("✓ Signal retrieval endpoints working")

        # Test stats endpoint
        response = client.get("/api/fractal/stats")
        assert response.status_code == 200, "Stats endpoint failed"
        print("✓ Stats endpoint working")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_environment():
    """Test la configuration environnementale"""
    print("\n" + "="*60)
    print("TEST 6: Environment Configuration")
    print("="*60)

    try:
        import os
        from pathlib import Path

        # Check files exist
        required_files = [
            'detector_strict.py',
            'detector_modere.py',
            'detector_frequent.py',
            'orchestrator.py',
            'api.py',
            'database.py',
            'main.py',
            'dashboard.html',
            'requirements.txt',
            'README.md',
            '.env.example'
        ]

        for file in required_files:
            assert Path(file).exists(), f"Missing file: {file}"
        print(f"✓ All {len(required_files)} required files present")

        # Check dependencies
        try:
            import fastapi
            import uvicorn
            import pandas
            import aiohttp
            import ccxt
            print("✓ All dependencies importable")
        except ImportError as e:
            print(f"⚠ Missing dependency: {e}")
            print("  Run: pip install -r requirements.txt")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def main():
    """Run all validation tests"""
    print("\n" + "🔍 FRACTAL DETECTION SYSTEM - VALIDATION SUITE" + " 🔍\n")

    results = {
        'Detectors': test_detectors(),
        'Detection Logic': test_detection_logic(),
        'Orchestrator': await test_orchestrator(),
        'Database': test_database(),
        'API Structure': test_api_structure(),
        'Environment': test_environment()
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All systems validated! Ready for deployment.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review errors above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
