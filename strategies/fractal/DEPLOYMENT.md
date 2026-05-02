# Railway Deployment Guide

## Quick Deploy in 5 Minutes

### 1. Prepare Your Repository

```bash
cd /path/to/your/repo
git add strategies/fractal/
git commit -m "feat: Add Fractal Detection System (STRICT/MODÉRÉ/FRÉQUENT)"
git push origin main
```

### 2. Create Railway Project

- Go to [railway.app](https://railway.app)
- Click "New Project"
- Select "Deploy from GitHub"
- Authorize Railway with your GitHub account
- Select your repository
- Select root directory: `/` (Railway will auto-detect subdirectories)

### 3. Configure Environment Variables

In Railway dashboard → Variables, add:

```
BINANCE_SYMBOL=BTC/USDT
DISCORD_WEBHOOK=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
DETECTION_INTERVAL=3600
ACTIVE_SETUPS=STRICT,MODÉRÉ,FRÉQUENT
DB_TYPE=sqlite
API_HOST=0.0.0.0
API_PORT=$PORT
```

### 4. Add PostgreSQL (Optional)

For persistent data across redeploys:

- Click "Add" in Railway
- Select "PostgreSQL"
- Railway auto-creates `DATABASE_URL`
- Update in Variables: `DB_TYPE=postgresql`

### 5. Deploy

- Push another commit or click "Deploy" manually
- Railway auto-detects `Procfile` and `requirements.txt`
- Deploys with both services:
  - **web**: FastAPI on `$PORT` (assigned by Railway)
  - **detection**: Continuous detection loop

### Verification

After deploy completes:

```bash
# Check web service
curl https://your-app.up.railway.app/api/fractal/health

# Check logs
railway logs --tail=50
```

## Advanced Configuration

### Custom Domain

1. Go to Railway → Your App → Settings
2. Add custom domain
3. Configure DNS with your registrar

### Webhook Secrets

Protect your Discord webhook:

1. Go to Railway → Variables
2. Mark `DISCORD_WEBHOOK` as "Reference" (private)

### Multiple Environments

Deploy separate instances:

- `main` branch → Production (all setups enabled)
- `staging` branch → Test (STRICT only)

```
# In Railway for staging:
ACTIVE_SETUPS=STRICT
```

### Database Backups

For PostgreSQL:

```bash
# Export signals
railway run pg_dump $DATABASE_URL > signals_backup.sql

# Import signals
railway run psql $DATABASE_URL < signals_backup.sql
```

## Monitoring & Maintenance

### View Metrics

Railway dashboard shows:
- CPU usage
- Memory usage
- Network I/O
- Build time

### Check Live Signals

API endpoints available at: `https://your-app.up.railway.app/api/fractal/`

```bash
curl https://your-app.up.railway.app/api/fractal/stats
curl https://your-app.up.railway.app/api/fractal/strict
curl https://your-app.up.railway.app/api/fractal/modere
curl https://your-app.up.railway.app/api/fractal/frequent
```

### Restart Services

```bash
railway restart
```

### View Logs

```bash
# Last 100 lines
railway logs --tail=100

# Real-time
railway logs -f

# Filter by service
railway logs web
railway logs detection
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'ccxt'"

- Check `requirements.txt` is in root of `strategies/fractal/`
- Check `Procfile` uses correct paths

### No signals detected

- Verify ACTIVE_SETUPS is set
- Check logs: `railway logs detection`
- Ensure BINANCE_SYMBOL format is correct (BTC/USDT, not BTCUSDT)

### API returns 500 error

- Check health: `curl your-app.up.railway.app/api/fractal/health`
- Restart: `railway restart`
- Check logs for exceptions

### Slow detection or timeouts

- Increase `DETECTION_INTERVAL` (default 3600s = 1 hour)
- Check Binance API rate limits in logs
- Reduce data load (fewer candles) if needed

### Database errors

SQLite:
- Rebuilds on each redeploy (data lost)
- Switch to PostgreSQL for persistence

PostgreSQL:
- Verify `DATABASE_URL` is set
- Check PostgreSQL service is running: `railway logs`

## Cost Estimation (Railway)

Monthly cost for continuous monitoring:

| Component | Usage | Cost |
|-----------|-------|------|
| Web dyno | 24/7 running | ~$5 |
| Detection dyno | 24/7 running | ~$5 |
| PostgreSQL (opt) | Basic plan | ~$12 |
| Bandwidth | Minimal (API only) | Free (50GB/month) |
| **Total** | **Production** | **~$10-17** |

*Prices as of 2026-05 (check current Railway pricing)*

## Example: Full Setup with Discord

```bash
# 1. Clone and navigate
git clone your-repo
cd strategies/fractal

# 2. Create .env locally for testing
cp .env.example .env
# Edit .env with your Discord webhook

# 3. Test locally
pip install -r requirements.txt
python validate.py  # Should pass all tests

# 4. Push to GitHub
git add .
git commit -m "feat: Fractal Detection ready for Railway"
git push

# 5. Deploy to Railway
# - Create project on railway.app
# - Set environment variables
# - Trigger deploy

# 6. Monitor
railway logs -f
curl https://your-app.up.railway.app/api/fractal/stats
```

## Alternative: Docker Deploy

If Railway doesn't auto-detect, create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY strategies/fractal .
RUN pip install -r requirements.txt
CMD uvicorn api:app --host 0.0.0.0 --port $PORT & python main.py
```

Then push and Railway will use Dockerfile automatically.

## Next Steps

After deployment:

1. **Verify signals**: Check Discord notifications incoming
2. **Monitor performance**: Watch win rates in dashboard
3. **Adjust if needed**: Change ACTIVE_SETUPS based on signal volume
4. **Integrate trading**: Connect exit detection to your broker API

---

**Need help?** Check Railway docs: https://docs.railway.app
