# Fractal Detection System - Deployment Checklist

## ✅ Pre-Deployment Verification

Before pushing to GitHub and deploying to Railway, verify:

### Code Quality
- [ ] All 3 detectors created (`detector_strict.py`, `detector_modere.py`, `detector_frequent.py`)
- [ ] Orchestrator handles Discord notifications (`orchestrator.py`)
- [ ] API endpoints expose all 3 setups (`api.py`)
- [ ] Database module supports SQLite/PostgreSQL (`database.py`)
- [ ] Main runner loads Binance data (`main.py`)
- [ ] Dashboard UI allows setup selection (`dashboard.html`)
- [ ] Validation tests pass (`python validate.py`)

### Configuration Files
- [ ] Requirements.txt includes all dependencies
- [ ] .env.example has all required variables
- [ ] Procfile defines web and detection services
- [ ] README.md has complete documentation
- [ ] DEPLOYMENT.md has Railway instructions

### Local Testing
```bash
cd strategies/fractal
python validate.py
```

Should output:
```
TEST 1: Detectors Initialization ✓
TEST 2: Detection Logic ✓
TEST 3: Orchestrator ✓
TEST 4: Database ✓
TEST 5: API Structure ✓
TEST 6: Environment ✓

All systems validated! Ready for deployment.
```

## 🚀 Deployment Steps

### Step 1: Prepare Git Repository

```bash
# Ensure you're in the repo root
cd /c/Users/PC/analyser

# Add all fractal system files
git add strategies/fractal/

# Verify what will be committed
git status

# Commit with descriptive message
git commit -m "feat: Complete Fractal Detection System with 3 setups

- STRICT (W+D+KZ+BR): 94.6% WR @ 313 signals/year
- MODÉRÉ (D+KZ+BR): 91% WR @ 225 signals/year  
- FRÉQUENT (KZ+BR): 87.5% WR @ 650 signals/year

Includes:
- Multi-timeframe consolidation detection
- ICT Kill Zone filtering (LKZ, NYKZ, LnCl, AKZ)
- Discord notifications with setup tags
- REST API for signal retrieval
- SQLite/PostgreSQL signal logging
- FastAPI + Uvicorn web server
- HTML5 dashboard for setup selection
- Full Railway deployment ready"

# Push to GitHub
git push origin main
```

### Step 2: Railway Setup (5 minutes)

1. **Go to railway.app and create account/login**

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub"
   - Authorize Railway
   - Select your repository
   - Railway should auto-detect `Procfile` in `strategies/fractal/`

3. **Configure Variables**
   
   In Railway dashboard, set:
   ```
   BINANCE_SYMBOL=BTC/USDT
   DISCORD_WEBHOOK=https://discordapp.com/api/webhooks/YOUR_ID/YOUR_TOKEN
   DETECTION_INTERVAL=3600
   ACTIVE_SETUPS=STRICT,MODÉRÉ,FRÉQUENT
   DB_TYPE=sqlite
   API_HOST=0.0.0.0
   API_PORT=$PORT
   ```

4. **Deploy**
   - Railway auto-deploys when you push
   - Check build log for errors
   - Verify both services start:
     - `web`: FastAPI on port $PORT
     - `detection`: Detection loop running

### Step 3: Verify Deployment

```bash
# After Railway deploy completes, test endpoints:

# 1. Health check
curl https://your-app.up.railway.app/api/fractal/health

# 2. Get stats
curl https://your-app.up.railway.app/api/fractal/stats

# 3. Get strict signals
curl https://your-app.up.railway.app/api/fractal/strict

# 4. Check logs
railway logs -f
```

### Step 4: Test Signals

1. **Check Discord**
   - Wait 5-60 minutes for first detection cycle
   - Should see signals like:
     ```
     🎯 Fractal Signal [STRICT]
     DOWN->UP - NYKZ
     Confidence: 94.6%
     ...
     ```

2. **Monitor Dashboard**
   - Open your API URL in browser
   - Or navigate: `https://your-app.up.railway.app/`
   - Watch signals update every 10 seconds

## 📋 File Inventory

### Core Detectors (3 setups)
- `detector_strict.py` - Inside Week+Day+KZ+BR
- `detector_modere.py` - Inside Day+KZ+BR
- `detector_frequent.py` - Inside KZ+BR

### Infrastructure
- `orchestrator.py` - Unified manager + Discord
- `api.py` - FastAPI endpoints
- `database.py` - Signal persistence
- `main.py` - Binance data loader + runner

### Frontend & Config
- `dashboard.html` - Web UI for setup selection
- `requirements.txt` - Python dependencies
- `.env.example` - Configuration template
- `Procfile` - Railway service definitions

### Documentation
- `README.md` - Full technical documentation
- `DEPLOYMENT.md` - Railway deployment guide
- `SYSTEM_OVERVIEW.md` - Architecture and concepts
- `CHECKLIST.md` - This file

### Testing
- `validate.py` - System validation suite

## 🔧 Post-Deployment Configuration

### Adjust Signal Frequency

If too many signals:
```
ACTIVE_SETUPS=STRICT,MODÉRÉ        # Remove FRÉQUENT
DETECTION_INTERVAL=7200            # Check every 2 hours
```

If too few signals:
```
ACTIVE_SETUPS=STRICT,MODÉRÉ,FRÉQUENT    # Add all
DETECTION_INTERVAL=1800                  # Check every 30 min
```

### Enable PostgreSQL for Persistence

1. Add PostgreSQL to Railway:
   - Click "Add" in Services
   - Select "PostgreSQL"
   - Railway auto-creates DATABASE_URL

2. Update variables:
   ```
   DB_TYPE=postgresql
   ```

3. Restart: `railway restart`

## 📊 Monitoring Dashboard

Access via: `https://your-app.up.railway.app/`

Features:
- Toggle setups (STRICT/MODÉRÉ/FRÉQUENT) on/off
- View last 10 signals from each setup
- See total signal count
- Monitor confidence levels
- Check entry prices

## 🚨 Troubleshooting Deployment

### Deploy fails with "Module not found"
- Verify `requirements.txt` in `strategies/fractal/`
- Check Python version (should be 3.9+)
- Rebuild: `railway restart`

### No signals after 1 hour
- Check logs: `railway logs detection`
- Verify BINANCE_SYMBOL is correct (BTC/USDT, not BTCUSDT)
- Verify ACTIVE_SETUPS is set
- Check Binance API rate limits

### Discord notifications not sending
- Verify DISCORD_WEBHOOK URL is valid
- Check it hasn't been revoked
- Verify Discord channel still exists
- Check Railway logs for errors

### API returns 500
- Restart: `railway restart`
- Check logs: `railway logs web`
- Verify database connection (if using PostgreSQL)

## 📈 Success Criteria

After deployment, you should see:

✅ **Within 5-60 minutes**
- First detection cycle completes
- 0-3 signals generated (depending on market)
- Discord receives notification(s)
- API responds with signals

✅ **Within 24 hours**
- 3-10 signals across all setups
- Pattern breakdown: mostly FRÉQUENT, some MODÉRÉ, few STRICT
- Database logging working
- Dashboard updates in real-time

✅ **Within 1 week**
- Consistent signal generation
- Can validate patterns manually on charts
- Performance statistics accumulate
- Ready to integrate with trading execution

## 🎯 Next: Live Trading Integration

Once deployed and validated:

1. **Real Money Preparation**
   - Set position sizing rules
   - Define stop loss levels
   - Set profit targets
   - Plan portfolio risk limits

2. **Signal Validation**
   - Manual chart review first 50 signals
   - Compare entry price vs actual fills
   - Validate pattern recognition

3. **Execution Integration**
   - Connect to broker API (binance, coinbase, etc.)
   - Implement order placement for STRICT signals
   - Track entry/exit prices back to database
   - Calculate P&L

4. **Performance Optimization**
   - Monitor actual win rates vs backtest
   - Adjust setups if needed
   - Scale position size based on equity curve

## 💡 Pro Tips

- Start with STRICT only for first week
- Use small position size during validation
- Keep Discord webhook URL private
- Monitor logs regularly
- Review signal statistics weekly
- Backup database monthly

## ✨ You're Done!

All components are now deployed and running 24/7.

**Congratulations!** Your fractal detection system is live. 🚀

---

**Questions?** Check the documentation:
- General: README.md
- Deployment: DEPLOYMENT.md  
- Concepts: SYSTEM_OVERVIEW.md
- Troubleshooting: DEPLOYMENT.md (end of file)

**Ready?** Run: `git push origin main` and Railway will auto-deploy!
