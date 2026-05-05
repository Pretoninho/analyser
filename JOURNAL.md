# Journal de bord — Analyser

> Fichier de référence principal. Remplace HANDOFF.md, HANDOFF_HTF.md, PROCEDURE_HTF_COMPLETE.md, HTF_STATES_AGENT_HANDOFF.txt.
> Lire ce fichier en entier avant de toucher au code.

---

## Architecture globale

Le projet est composé de **4 stratégies distinctes** déployées sur Railway :

| Stratégie | Description | Fichiers clés |
|-----------|-------------|---------------|
| **Pi\*** | Q-table ICT BTC 1m — macro 09:50 ET | `main.py`, `pi_config.py`, `live_signal.py`, `shadow_signal.py` |
| **TA v2** | Ensemble voting multi-timeframe 15m | `strategies/ta/`, `api/app.py` APScheduler |
| **Fractals** | Détection W/D/KZ ICT 3 tiers | `strategies/fractal/` |
| **HTF** | States hebdo/daily/4H + Deribit options | `analysis/htf/`, `analysis/deribit_futures/` |

**Infrastructure Railway :**
- Service 1 : API Python (FastAPI + uvicorn, `api/app.py`)
- Service 2 : Cron live_signal.py (`51 13 * * 1-5`)
- Service 3 : Cron shadow_signal.py (`5 20 * * 1-5`)
- Service 4 : Frontend Next.js (`frontend/`)

**Variables d'environnement Railway requises :**
- `DISCORD_WEBHOOK_URL` (fallback global)
- `DISCORD_WEBHOOK_DERIBIT_URL`
- `DISCORD_WEBHOOK_DVOL_URL`
- `DISCORD_WEBHOOK_TA_URL` ← **à configurer pour activer notifications TA**
- `DERIBIT_NOTIFY_ENABLED`, `DERIBIT_NOTIFY_MODE`, `DERIBIT_NOTIFY_TIMEFRAME`, `DERIBIT_NOTIFY_DAYS`, `DERIBIT_NOTIFY_MINUTE`
- `TA_NOTIFY_ENABLED` (default true)

---

## Pi* — État de la stratégie

**Résultats backtest (test set, config active 2026-04-25) :**
- Jours tradés : 43 (9%) | Trades : 45
- WR : 60.0% | Return : +13.07% | Sharpe : +1.685 | MaxDD : -2.68%
- PF : 2.242 | AvgWin : +0.833% | AvgLoss : -0.557% | Expectancy : +0.277%
- Par macro : 09:50*=24 trades WR 50.0% +3.87% | 11:50=21 trades WR 71.4% +8.58%

**Config active (`pi_config.py`) :**
```python
SKIP_MACROS = frozenset({1, 3, 5, 6, 7})   # macros silencieuses
SKIP_DAYS   = frozenset({0})                 # lundi retiré
EXIT_HM     = 960                            # sortie 16:00 ET
sl_pct = 0.006, rr = 2.5                    # recommandé post-sweep 2026-04-25
MACRO_RULES = {
    (2, 1, 1): frozenset({1}),   # 09:50 + RAID_H + BSL_swept -> SWEEP_H seulement
    (2, 0, 1): frozenset(),      # 09:50 + NO_RAID + BSL_swept -> bloqué
}
```

**Espace d'états :**
```
N_STATES = 1944   # 3*3*3*8*3*3
Dimensions : month_ctx(3) × day_ctx(3) × london_ctx(3) × macro_ctx(8) × sweep_ctx(3) × pool_ctx(3)
Encodage : state = mc*648 + dc*216 + lc*72 + mac*9 + sc*3 + pc
```

**Macros silencieuses (raisons) :**
| mac_idx | Heure | Raison |
|---------|-------|--------|
| 1 | 08:50 | Pré-ouverture NYSE, toutes directions négatives OOS |
| 3 | 10:50 | Pas de signal consistant OOS |
| 5 | 12:50 | Pas de signal consistant OOS |
| 6 | 13:50 | Pas de signal consistant OOS |
| 7 | 14:50 | Power Hour = phénomène equity (MOC orders), BTC 24/7 sans effet |

---

## TA v2 — État de la stratégie

**Benchmark OOS (sweep_IS_vs_OOS.csv) :** WR 49.4%, Exp_R +124R

**Pipeline :**
`scan_signals()` → `log_signal()` → `db/ta_signals.csv` → `resolve_pending()` → `live_stats()`

**Scheduler APScheduler (`api/app.py`) :**
- Cron : `hour="7-11,13-17"`, `minute="0,15,30,45"` (London 7h-12h UTC, NY 13h-18h UTC)
- Sessions couvertes : 16 bougies London + 16 bougies NY par jour

**Fichiers :**
```
strategies/ta/
├── features.py              # EMA, RSI, Stoch, ATR, VWAP, Swing, Regime
├── ensemble_voting_v2.py    # Majority consensus depuis sweep_IS_vs_OOS.csv
├── live_runner_v2.py        # Fetch Binance 15m + détection 2-bar reversal
├── discord_notify_v2.py     # Scheduler job : scan + log + Discord
├── signal_logger.py         # Log db/ta_signals.csv + résolution TP/SL + live_stats()
└── backtest_v2.py           # IS/OOS backtest avec ensemble voting filter
```

**Paramètres :**
- `DOJI_THRESHOLD = 0.1` (corps < 10% du range ignoré dans trigger 2-bar)
- `VOTING_MIN_N_OOS = 5`, `VOTING_MIN_WR_OOS = 0.60`

---

## Fractals — État de la stratégie

**3 tiers de détection :**
| Tier | Conditions | Fichier |
|------|-----------|---------|
| STRICT | W + D + KZ + BR | `detector_strict.py` |
| MODÉRÉ | D + KZ + BR | `detector_modere.py` |
| FRÉQUENT | KZ + BR | `detector_frequent.py` |

**Orchestrateur (`orchestrator.py`) :**
- Dédup par session : `_sent_keys: set` sur `(day_date, kz, pattern, setup)`
- Discord conditionnel si `DISCORD_WEBHOOK_URL` présent

**Loader (`main.py`) :**
- `load_live_timeframes()` : M15=384 bars, Daily=14 jours, Weekly=8 semaines
- `kz_after` borné à `day_end + pd.Timedelta(days=2)` pour éviter look-ahead infini

---

## HTF — État de la stratégie

**Architecture états :**
```
weekly_state (W1-W5) × daily_state (D1-D5) × h4_state (H1-H4)
Forward return : pret_j1 = return du lendemain J+1
```

**Profil actif : `equilibre_assoupli`**
| Seuil | Valeur |
|-------|--------|
| N_trades min | ≥ 15 |
| WinRate min | ≥ 52% |
| Wilson_LB min | ≥ 40% |
| AvgTrade min | ≥ +0.10% |

**Dry-run (2026-04-29, profil equilibre_assoupli) :**
| Métrique | HTF | Baseline |
|----------|-----|---------|
| n_days | 2284 | 2284 |
| n_trades | 19 | 19 |
| avg_trade_proxy | +0.5095% | -0.3862% |
| total_proxy | +9.58% | -7.54% |

**Shortlist shorts retenues :**
| ID | États | Direction | AvgTrade | N |
|----|-------|-----------|----------|---|
| R2_018 | W4 + D3 | SHORT | +1.049% | 14 |
| R2_003 | W1 + D3 | SHORT | +0.253% | 18 |
| R3_006 | W1 + D3 + H2 | SHORT (satellite) | variable | 15 |

**Problème identifié :** fréquence ~3 trades/an → trop faible pour usage opérationnel.

**Fichiers DB :**
```
db/htf/
├── htf_state_combinations.db
├── stats_agent_htf_seed_agressif.pkl
└── stats_agent_htf_seed_equilibre_assoupli.pkl
```

---

## Deribit Futures — État du module

**7 edges (`analysis/deribit_futures/features.py`) :**
| Edge | Type | Logique |
|------|------|---------|
| `edge_funding_reversion` | directionnel | funding extrême → mean reversion |
| `edge_carry_momentum` | directionnel | funding positif fort → suivi tendance |
| `edge_carry_stress` | directionnel | funding très négatif → squeeze haussier |
| `edge_mark_dislocation` | directionnel | basis mark/index > 1.5σ → convergence |
| `edge_options_vol_premium` | non-directionnel | IV réalisée < IV implicite |
| `edge_skew_panic` | directionnel | put skew extrême → retournement haussier |
| `edge_term_structure_kink` | non-directionnel | anomalie term structure IV |

**Endpoints API :**
- `GET /api/deribit/edges` — snapshot 7 edge scores
- `GET /api/deribit/backtest` — hit ratio par edge (cache 15 min)
- `GET /api/deribit/signal` — signal actionnable LONG/SHORT/FLAT/WATCH
- `POST /api/deribit/futures/notify` — push Discord

---

## Analyses archivées (pas d'edge exploitable sur BTC)

| Analyse | Conclusion |
|---------|-----------|
| `analyse_0850.py` (08:50) | Globalement négatif OOS. 1 contexte noté pour suivi (N trop petit). |
| `analyse_silver_bullet.py` (10:00) | Aucun contexte validé OOS. OR sweep trop rare (18.9% des jours). |
| `analyse_judas_swing.py` | WR=27%, PF=0.25 — pas d'edge. Conservé en observation. |
| `analyse_london_cascade.py` | avg < -0.08% test — pas d'edge. Conservé en observation. |
| `analyse_cbdr.py` | narrow CBDR négatif, medium CBDR prometteur mais N=11. À enrichir. |
| Weekly Profile ICT | Concepts déjà encodés via pool_ctx + sweep_ctx. Pas de gain additionnel. |

**Note :** ces stratégies pourraient être viables sur Forex (EURUSD, NASDAQ) — à réévaluer.

---

## Pistes en attente

### Trailing stop structurel (P3)
- Trailing fixe non retenu (dégradation OOS nette)
- Prototype à tester : sortie sur cassure EMA9 ± k*ATR(14), actif seulement après +0.30-0.40%
- Validation requise : walk-forward 80/20 strict vs baseline

### HTF — Augmenter la fréquence (P1 HTF)
- Options : abaisser seuils Wilson_LB, élargir combinaisons, utiliser states unitaires
- Script à créer : `analysis/diagnose_htf_frequency.py`
- Critère de succès : ≥ 10 trades/an sans WR < 50% ni AvgTrade < 0%

### Dashboard mobile Streamlit (P9)
- `streamlit_app.py` à la racine — standalone, Streamlit Community Cloud
- Contenu : prix BTC live + contexte marché + bouton test Discord

---

## Règles méthodologiques

1. **Ne jamais modifier `stats_agent.pkl` ni `engine/`** sans walk-forward complet avant.
2. **Walk-forward strict** : 80% train / 20% test temporel. Jamais de shuffle.
3. **HTF est séparé de Pi*** : aucune interférence avec la Q-Table production.
4. **UnicodeEncodeError Windows** : ne pas utiliser de caractères spéciaux dans les print().
5. **Trailing stop** : non retenu en production tant que non validé OOS.

---

## Journal des sessions

---

### 2026-05-05

**Frontend — Options page (commit ceed5fc)**
- Fix build Railway : `openRollSimulator`, `rollType`, `rollStrikeFrom`, etc. ajoutés dans le composant `OptionsPage` (étaient référencés mais non définis → erreur TypeScript strict).

**Fractals — Détecteurs et orchestrateur (commit bc57a36)**
- `detector_strict.py` / `detector_modere.py` / `detector_frequent.py` : `kz_after` borné à `day_end + 2 jours` (était illimité → look-ahead infini).
- `orchestrator.py` : dédup par session via `_sent_keys` sur `(day_date, kz, pattern, setup)` (était réémis à chaque cycle).
- `main.py` : `load_live_timeframes()` remplace `load_all_timeframes()` — M15=384 bars, Daily=14j, Weekly=8w (était 365j de données rechargées).
- Ajout header `# -*- coding: utf-8 -*-` sur tous les fichiers fractals.

**TA v2 — Correction de 9 bugs sur 4 fichiers (commit d7a517a)**
- `features.py` : `regime_1d.reindex(..., method="ffill").bfill()` (bfill manquant) ; `_atr_ratio` avec `.fillna(1.0)` pour éviter NaN → crash `pd.cut`.
- `ensemble_voting_v2.py` : cast `int()` des deux côtés du comparateur feature/config pour corriger mismatch float/int CSV (pool toujours vide).
- `backtest_v2.py` : `exp_R = (n_wins * 2.0 - n_losses * 1.0) / n_total` (était ATR absolu) ; `int(trade.get("ema_state", 0) or 0)` pour éviter `int(NaN)`.
- `live_runner_v2.py` : `DOJI_THRESHOLD = 0.1` ; `except Exception` au lieu de bare except silencieux ; `entry_price` fallback sur `close` au lieu de `0.0` ; singleton `get_voter()`.

**TA v2 — NameError scheduler (commit bec2948)**
- `live_runner_v2.py` : `VOTING_MIN_N_OOS` et `VOTING_MIN_WR_OOS` définis avant `get_voter()`.

**Discord pipeline — Fix end-to-end (commit bb7c09a)**
- `discord_notify_v2.py` : scan continue même si `DISCORD_WEBHOOK_TA_URL` absent (warning au lieu de `return False`) ; `log_signal()` appelé **avant** l'envoi Discord.

**Signal logger — Réécriture format v2 (commit 17d10a3)**
- `signal_logger.py` : `log_signal()` réécrit pour format v2 `{timestamp, direction, entry_price, regime, ema_state, ..., vote_total, vote_favorable, confidence}` ; `_load()` cast colonnes outcome/exit en `object` pour éviter TypeError dtype ; import corrigé vers `live_runner_v2`.

**Scheduler — Cron TA (commit 83645f1)**
- `api/app.py` : `hour="7-10,13-16"` → `hour="7-11,13-17"` (bougies 11h00 et 17h00 UTC manquées corrigées).

**Tests end-to-end validés :**
- 2 signaux loggés et résolus (win +2R, loss -1R) — `db/ta_signals.csv` fonctionnel.
- `live_stats()` opérationnel : WR, Exp_R, by_regime, by_direction.

**Action restante :**
- Configurer `DISCORD_WEBHOOK_TA_URL` dans les variables Railway pour activer les notifications Discord TA.
