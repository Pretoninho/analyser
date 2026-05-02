# Déploiement Railway — TA Strategy v2 avec Ensemble Voting

**Date mise à jour** : 2026-05-02

## Changements effectués

### 1. Modules Python créés

- **`strategies/ta/ensemble_voting_v2.py`** — Voteur avec pool qualifié (n_OOS≥5, wr_OOS≥55-60%)
- **`strategies/ta/live_runner_v2.py`** — Moteur live avec voting intégré
- **`strategies/ta/discord_notify_v2.py`** — Notification Discord avec vote consensus
- **`strategies/ta/backtest_v2.py`** — Analyse comparative des thresholds

### 2. Modifications API

**`api/app.py`** :
- Remplacé ancien scheduler TA → nouveau avec `discord_notify_v2.py`
- **Scheduler TA v2 24/7 (sans restriction weekend)** — sessions London (07-10 UTC) + NY (13-16 UTC)
- Ajouté endpoints test :
  - `POST /api/ta/scan-v2` — scan sans Discord (test rapide)
  - `POST /api/ta/notify-v2` — scan + Discord (test complet)
- Configuration via env var : `TA_NOTIFY_ENABLED_V2=true` (default)

## Variables d'environnement Railway

Vérifier/ajouter dans Railway project settings :

```
TA_NOTIFY_ENABLED_V2         = true          # active le scheduler TA v2
DISCORD_WEBHOOK_TA_URL       = <webhook>     # copier depuis ancien TA
```

*Note* : Garder aussi `DISCORD_WEBHOOK_URL` pour les fallbacks.

## Procédure de déploiement

### 1. Test local (avant push)

```bash
# Vérifier qu'il n'y a pas d'erreurs import
python strategies/ta/discord_notify_v2.py

# Tester l'endpoint scan (sans Discord)
curl -X POST http://localhost:8000/api/ta/scan-v2

# Vérifier logs du backtest
python strategies/ta/backtest_v2.py | tail -20
```

### 2. Push vers Railway

```bash
git add strategies/ta/ensemble_voting_v2.py \
         strategies/ta/live_runner_v2.py \
         strategies/ta/discord_notify_v2.py \
         strategies/ta/backtest_v2.py \
         api/app.py

git commit -m "feat: TA strategy v2 with ensemble voting (49.4% OOS WR)"
git push origin main
```

### 3. Vérification post-déploiement (Railway)

1. **Check logs** → rechercher `[scheduler] APScheduler demarre ... ta_notify=v2_enabled`

2. **Test endpoint scan** :
   ```bash
   curl -X POST https://api-production-xxxx.up.railway.app/api/ta/scan-v2
   ```
   Doit retourner : `{"status": "scanned", "n_signals": ..., "signals": [...]}`

3. **Test Discord manual** :
   ```bash
   curl -X POST https://api-production-xxxx.up.railway.app/api/ta/notify-v2
   ```
   Doit envoyer un message Discord avec vote consensus

4. **Monitor signaux** — attendre 15 min pour voir si signal détecté dans sessions London/NY

## Performance attendue

**Production** (walk-forward 2025-2026 OOS) :

| Métrique | Valeur | Notes |
|----------|--------|-------|
| Win Rate | 49.4% | +14.9pp vs baseline |
| Expectancy | +124.4R | excellent (baseline +2.8R) |
| Fréquence | ~1-3 signaux/jour | pendant sessions (7-11 UTC, 13-17 UTC) |
| IS→OOS degradation | +0.9% | quasi zéro (pas d'overfitting!) |

## Rollback (si problème)

```bash
# Revert à ancien discord_notify.py
git revert <commit_hash>
git push origin main

# Oucfg manuellement dans Railway :
# Mettre TA_NOTIFY_ENABLED_V2 = false
# Mettre TA_NOTIFY_ENABLED = true (ancien)
```

## Configuration futures (optionnel)

Si fréquence insuffisante en production :
- Changer `VOTING_MIN_WR_OOS = 0.55` dans `live_runner_v2.py` (vs 0.60 actuel)
- Trade-off : +150 trades détectés OOS, WR 49.3% (mini-degradation)

## Fichiers clés

```
strategies/ta/
├── ensemble_voting_v2.py      ← voteur principal
├── live_runner_v2.py          ← moteur live
├── discord_notify_v2.py       ← notif Discord
├── backtest_v2.py             ← analyse
└── RESULTS_TA_V2.md           ← documentation détaillée

api/
└── app.py                      ← endpoints test + scheduler
```

## Support

- **Questions sur voting** → voir `ensemble_voting_v2.py` docstrings
- **Debugging signaux** → exécuter `test_discord_notify_v2.py` localement
- **Performance monitoring** → tracker WR vs backtest (+/-5% acceptable)
