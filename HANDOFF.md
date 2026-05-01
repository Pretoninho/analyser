# Pi* — Fichier de passation pour agent IA

> Mis à jour le 2026-05-02. À lire en entier avant de toucher au code.

> **Fichier HTF separé : [HANDOFF_HTF.md](HANDOFF_HTF.md)**

---

## 1. Vue d'ensemble du projet

**Pi*** est un agent de trading ICT (Inner Circle Trader) sur BTCUSDT Binance, 1-minute candles.
- Données : Binance historique 2020-01-01 -> aujourd'hui (3,3M bougies), stockées localement dans `data_binance/`
- Stratégie : Q-table empirique (pas de RL loop) — moyenne des P&L réels par état
- Backtest : walk-forward 80% train / 20% test temporel
- Live : déployé sur Railway (API + scheduler + dashboards)

**Résultats actuels (test set, config active) :**
- WR : 84.6% | Return : +6.02% | Profit Factor : 5.99 | Sharpe : +1.699
- N trades (test) : 13 — très sélectif mais propre

---

## 2. Architecture — espace d'états

**Fichier clé : `engine/stats_state.py`**

```
N_STATES = 1944   # 3*3*3*8*3*3
N_ACTIONS = 3     # 0=FLAT  1=LONG  2=SHORT

Dimensions :
  month_ctx  (3) : 0=WEAK  1=NEUTRAL  2=STRONG
  day_ctx    (3) : 0=WEAK  1=NEUTRAL  2=STRONG
  london_ctx (3) : 0=NO_RAID  1=RAID_HIGH  2=RAID_LOW
  macro_ctx  (8) : 0=NONE  1=08:50  2=09:50*  3=10:50  4=11:50  5=12:50  6=13:50  7=14:50
  sweep_ctx  (3) : 0=NO_SWEEP  1=SWEEP_HIGH  2=SWEEP_LOW
  pool_ctx   (3) : 0=NEUTRAL   1=BSL_SWEPT   2=SSL_SWEPT

Encodage : state = mc*648 + dc*216 + lc*72 + mac*9 + sc*3 + pc
```

**ICT Macros (minutes ET depuis minuit) :**
```python
MACROS = {
    1: (530, 550),   # 08:50-09:10
    2: (590, 610),   # 09:50-10:10  ← seule macro active (STAR)
    3: (650, 670),   # 10:50-11:10
    4: (710, 730),   # 11:50-12:10
    5: (770, 790),   # 12:50-13:10
    6: (830, 850),   # 13:50-14:10
    7: (890, 910),   # 14:50-15:10
}
```

**Contextes :**
- `lc` (london_ctx) : London raid de l'Asia range (03:00-06:00 ET vs 01:00-05:00 UTC)
- `sc` (sweep_ctx) : sweep de la fenêtre pré-macro (20 min avant) à l'ouverture de la macro
- `pc` (pool_ctx) : pré-macro high/low vs London session high/low + PWH/PWL (previous week)

---

## 3. Configuration active — `sweep.py`

```python
SKIP_MACROS = frozenset({1, 3, 5, 6, 7})   # macros silencieuses
SKIP_DAYS   = frozenset({0})                 # lundi retiré
EXIT_HM     = 960                            # sortie 16:00 ET

MACRO_RULES = {
    (2, 1, 1): frozenset({1}),   # 09:50 + RAID_H + BSL_swept -> SWEEP_H seulement
    (2, 0, 1): frozenset(),      # 09:50 + NO_RAID + BSL_swept -> bloqué
}
```

**Seule macro active : mac_idx=2 (09:50 ET)**
- `aligned_only=True` : skip si sc==0 (pas de sweep à l'entrée)
- `target_pool=True` : TP dynamique vers le pool opposé (ref_h/ref_l)

---

## 4. Macros silencieuses — pourquoi

| mac_idx | Heure | Raison du silence |
|---------|-------|-------------------|
| 1 | 08:50 | Pré-ouverture NYSE — toutes directions négatives en test |
| 3 | 10:50 | Pas de signal consistant OOS |
| 5 | 12:50 | Pas de signal consistant OOS |
| 6 | 13:50 | Pas de signal consistant OOS |
| 7 | 14:50 | Power Hour = phénomène equity (MOC orders 16:00 ET), BTC 24/7 sans effet |

**Workflow de surveillance** : les macros silencieuses sont loggées avec label ✗ (mode monitoring Railway — pas encore implémenté).

---

## 5. Fichiers principaux

```
analyser/
├── pi_config.py               # SOURCE DE VERITE config strategie (SL, RR, LIVE/SHADOW_MACROS, MACRO_RULES)
├── live_signal.py             # Signal live cron 09:51 ET — ecrit db/live_trades.csv
├── shadow_signal.py           # Paper trading EOD cron 16:05 ET — ecrit db/shadow_trades.csv
├── api/
│   └── app.py                 # FastAPI backend (uvicorn) — endpoints /api/daily /api/trades /api/performance /api/qtable /api/candles
├── frontend/                  # Next.js dashboard (Railway service separe)
│   ├── app/
│   │   ├── page.tsx           # Redirect vers /trades
│   │   ├── trades/page.tsx    # Historique trades live + shadow
│   │   ├── trades/[date]/[mac]/page.tsx  # Detail trade + CandleChart TradingView
│   │   ├── daily/[date]/page.tsx         # Rapport journalier toutes macros
│   │   ├── performance/page.tsx          # Metriques + courbe P&L cumulee
│   │   └── qtable/page.tsx              # Q-table etats actifs
│   ├── components/            # CandleChart, CtxBadges, ExitBadge, Sidebar, StatCard
│   ├── lib/api.ts             # Fetches vers l'API Python
│   └── railway.json           # Config Railway service Next.js
├── railway.toml               # Config Railway service API (uvicorn $PORT)
├── Procfile                   # web: uvicorn api.app:app --host 0.0.0.0 --port $PORT
├── requirements-live.txt      # requests, pandas, numpy, pytz, fastapi, uvicorn
├── db/
│   ├── stats_agent.pkl        # Q-table active (1944 x 3)
│   ├── live_trades.csv        # Log trades live (append quotidien)
│   └── shadow_trades.csv      # Log trades shadow (append quotidien)
├── main.py                    # run_build_qtable() + run_backtest_stats() + _sim_trade_rr()
├── sweep.py                   # grid search SL/TP
├── config.py                  # chemins, parametres globaux
├── engine/
│   ├── stats_state.py         # N_STATES, MACROS, encode, decode, compute_pool_ctx, compute_daily_context
│   └── q_agent.py             # QAgent (save/load pkl, act, update)
├── data/
│   └── binance.py             # load_binance_1m(), download_binance_1m()
├── data_binance/              # CSV Binance BTCUSDT 1min (2019-2026)
├── analysis/                  # Scripts d'analyse standalone
    ├── analyse_0850.py
    ├── analyse_0950.py
    ├── analyse_1150.py
    ├── analyse_1450.py
    ├── analyse_macros_silencieux.py
    ├── analyse_silver_bullet.py
    ├── analyse_judas_swing.py
    ├── analyse_london_cascade.py
    ├── analyse_cbdr.py
    ├── run_deribit_futures.py     # CLI edge frame -> db/deribit_futures_edges.csv
    ├── run_deribit_backtest.py    # CLI backtest -> db/deribit_backtest.csv
    ├── run_deribit_signal.py      # CLI signal + optionnel --notify Discord
    └── deribit_futures/           # Package signal Deribit (voir section 12)
        ├── __init__.py
        ├── features.py            # build_deribit_edge_frame (7 edges)
        ├── backtest.py            # hit ratio par edge (horizons +4h/+24h)
        └── signal.py              # build_deribit_signal + format_discord_signal
```

---

## 6. Fonctions clés — signatures

```python
# main.py
run_build_qtable(
    test_ratio=0.2, min_samples=5, exit_hm=960,
    sl_pct=0.005, rr=2.0, target_pool=True, aligned_only=True,
    skip_macros=frozenset({1,3,5,6,7}), skip_days=frozenset({0}),
    macro_rules={(2,1,1): frozenset({1}), (2,0,1): frozenset()}
)

run_backtest_stats(
    test_ratio=0.2, q_threshold=0.0, exit_hm=960,
    sl_pct=0.005, rr=2.0, target_pool=True, aligned_only=True,
    skip_macros=frozenset({1,3,5,6,7}), skip_days=frozenset({0}),
    entry_mode="baseline",  # ou "ote" ou "fvg"
    macro_rules={(2,1,1): frozenset({1}), (2,0,1): frozenset()}
)

_sim_trade_rr(exit_df, entry_px, direction, sl_pct, tp_pct,
              fee=0.0005, slip=0.0002, verbose=False)
# direction : +1 LONG, -1 SHORT
# verbose=True -> (pnl, exit_reason, exit_px, tp_px, sl_px, n_candles)
```

---

## 7. Résultats d'analyses récentes

### Analyse 08:50 (analyse_0850.py)
- **Globalement négatif** à toutes les heures (08:50 → 09:40)
- **1 contexte validé OOS** : `RAID_H × SWEEP_L × BSL_SWEPT @ 09:30 → SHORT`
  - Train : N=10, WR=70%, avg=+0.421%
  - Test  : N=6, avg=+0.157% ✓ (mais N petit)
- Conclusion : 08:50 restera silencieux, le signal 09:30 est noté pour suivi

### Silver Bullet 10:00 (analyse_silver_bullet.py)
- Setup : OR (09:30-10:00) → sweep OR_HIGH/LOW à 10:00
- **Aucun contexte validé OOS** — tous les contextes prometteurs en train échouent en test
- Distribution : seulement 435/2306 jours ont un sweep OR (18.9%) — trop rare
- Conclusion : concept intéressant théoriquement, pas de signal exploitable sur BTC actuellement

---

## 8. Tâches en attente (par priorité)

### P1 — Sweep paramètres (fait)
Sweep execute le 2026-04-25 sur la config active :
```python
SKIP_MACROS = frozenset({1, 3, 5, 6, 7})
# SL_VALUES = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010]
# RR_VALUES  = [1.5, 2.0, 2.5, 3.0]
```
Resultats: `db/sweep_results.csv`

Top combo retenu (return + sharpe + PF):
- `sl_pct=0.006`, `rr=2.5`
- Trades=48, WR=56.2%, Return=+8.91%, PF=1.844, Sharpe=+1.197, MaxDD=-2.06%

Statut: parametres recommandes pour la config active -> `sl_pct=0.006`, `rr=2.5`.

Dernier run complet valide (2026-04-25, build + backtest stats):
- Build Q-table: 1844 episodes train, 145 etats visites, 42 etats avec N>=5
- Backtest (test=461 jours):
    - Jours trades: 43 (9%), Trades totaux: 45
    - Sorties: TP=12 (27%), SL=13 (29%), TRAIL=0 (0%), EOD=20 (44%)
    - Return total: +13.07%
    - Sharpe: +1.685
    - Max drawdown: -2.68%
    - Win rate: 60.0%
    - Profit factor: 2.242
    - Avg win: +0.833%
    - Avg loss: -0.557%
    - Expectancy/trade: +0.2768%
    - Par macro: 09:50*=24 trades, WR 50.0%, total +3.87% | 11:50=21 trades, WR 71.4%, total +8.58%

### P2 — Railway deployment (EN PRODUCTION)

Architecture déployée sur Railway — 3 services :

**Service 1 : API Python (racine)**
- `railway.toml` → `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
- `requirements-live.txt` : requests, pandas, numpy, pytz, fastapi, uvicorn, apscheduler
- Endpoints Pi* : `/api/daily/{date}`, `/api/trades`, `/api/performance`, `/api/qtable`, `/api/candles/{date}/{mac}`, `/health`
- Endpoints Deribit (voir section 12) :
  - `GET /api/deribit/edges?timeframe=1h&days=14` — snapshot 7 edge scores + mark/OI/options
  - `GET /api/deribit/backtest?timeframe=1h&days=90&threshold=0.05` — hit ratio par edge (cache 15 min)
  - `GET /api/deribit/signal?timeframe=1h&days=90` — signal actionnable LONG/SHORT/FLAT + tenor (cache 15 min)
  - `POST /api/deribit/futures/notify?timeframe=1h&days=90` — envoie signal vers Discord webhook dédié
- Lit `db/live_trades.csv` + `db/shadow_trades.csv`

**Service 2 : Cron live_signal.py**
- Cron `51 13 * * 1-5` (EDT) / `51 14 * * 1-5` (EST nov-mars)
- Macro active : mac_idx=2 (09:50 ET)
- Ecrit dans `db/live_trades.csv`, envoie Discord signal live

**Service 3 : Cron shadow_signal.py**
- Cron `5 20 * * 1-5` (EDT) / `5 21 * * 1-5` (EST nov-mars)
- Macros shadow : {1, 3, 5, 6, 7} (toutes les macros silencieuses)
- Ecrit dans `db/shadow_trades.csv`, envoie resume Discord [SHADOW] EOD

**Service 4 : Frontend Next.js (`frontend/`)**
- `frontend/railway.json` → build Next.js
- Pages : `/trades`, `/performance`, `/qtable`, `/daily/{date}`, `/trades/{date}/{mac}`, `/deribit`
- `/deribit` : signal Deribit live (action/confiance/horizon), tenor/contract, 7 edge scores, drivers, snapshot options, bouton "Notifier Discord"
- Se connecte a l'API Python via `lib/api.ts`

**Scheduler APScheduler (dans api/app.py) :**
- live_signal.py : lun-ven 09:51 ET
- shadow_signal.py : lun-ven 16:05 ET
- Notifications Deribit : configurable via variables d'environnement Railway :
  - `DERIBIT_NOTIFY_ENABLED` (default: "true")
  - `DERIBIT_NOTIFY_MODE` : "every_4h" | "hourly_us" | "hourly" (default: "every_4h")
  - `DERIBIT_NOTIFY_TIMEFRAME` (default: "1h")
  - `DERIBIT_NOTIFY_DAYS` (default: "90")
  - `DERIBIT_NOTIFY_MINUTE` (default: "2")
  - `DISCORD_WEBHOOK_DERIBIT_URL` : webhook Discord dédié futures (fallback sur DISCORD_WEBHOOK_URL)

**Config centralisee : `pi_config.py`**
- Transferer une macro de SHADOW vers LIVE : retirer de SHADOW_MACROS, ajouter a LIVE_MACROS
- FEE=0.0005, SLIP=0.0002 inclus dans simulations shadow

**Variables Railway requises :**
- `DISCORD_WEBHOOK_URL` (fallback global)
- `DISCORD_WEBHOOK_DERIBIT_URL` (webhook dédié futures Deribit)
- `DISCORD_WEBHOOK_DVOL_URL` (webhook dédié DVOL, fallback sur DERIBIT_URL puis WEBHOOK_URL)
- `DISCORD_WEBHOOK_TA_URL` (webhook dédié signaux TA)
- `DERIBIT_NOTIFY_ENABLED`, `DERIBIT_NOTIFY_MODE`, `DERIBIT_NOTIFY_TIMEFRAME`, `DERIBIT_NOTIFY_DAYS`, `DERIBIT_NOTIFY_MINUTE`
- `TA_NOTIFY_ENABLED` (default true)
- `APP_TYPE` (service dashboard Streamlit uniquement: `dashboard`)
- `BINANCE_BASE_URL` (optionnel, defaut https://api.binance.com)

### P3 — Trailing stop (idée en attente)
Remplacer SL+TP fixe par : sortie quand P&L non-réalisé baisse de `trailing_delta` depuis son pic.
Fichier mémoire : `memory/idea_trailing_stop.md`

Statut validation (2026-04-25, walk-forward 80/20, config active):
- Baseline (sans trailing): 32 trades, WR 62.5%, Return +4.61%, PF 1.662, Sharpe +0.912
- Trailing 0.2% (`trailing_delta=0.002`): 19 trades, WR 31.6%, Return -0.05%, PF 0.984, Sharpe -0.023
- Trailing 0.3% (`trailing_delta=0.003`): 21 trades, WR 33.3%, Return -1.06%, PF 0.719, Sharpe -0.499

Conclusion actuelle: trailing stop non retenu en production sur la config active (dégradation nette OOS).

Piste suivante a prototyper (avec agent Claude): trailing structurel base moyenne mobile.
- Principe: remplacer le trailing fixe (%) par une sortie sur cassure de tendance court-terme.
- Armement: trailing actif seulement apres un gain minimal (ex: +0.30% a +0.40%).
- Regle LONG: sortir si close < EMA9 - k*ATR(14) (k a tester: 0.1, 0.2, 0.3).
- Regle SHORT: sortir si close > EMA9 + k*ATR(14).
- Confirmation anti-bruit: exiger 1 cloture confirmee (ou 2 en version conservative).
- Garde-fous: conserver hard SL initial et TP dynamique target_pool.
- Validation requise: walk-forward 80/20 strict vs baseline, memes macros/rules, puis comparer Return, PF, Sharpe, DD, WR, distribution TP/SL/TRAIL/EOD.

### P4 — Session de Londres (cloture BTC — observation autres actifs)

Trois approches testees sur BTCUSDT, toutes negatives OOS :

- `analyse_london_cascade.py` : LOR -> SB -> London Macro cascade, 8 variantes, avg < -0.08% test
- `analyse_judas_swing.py` : Judas Swing pur (midnight-05:00 ET), WR=27%, PF=0.25, SL 61% des trades
- Macros London individuelles (03:00-06:00 ET) : deja testees via macros silencieuses

Conclusion : session London sans edge exploitable sur BTC 24/7 (absence de fixings/flow institutionnel Forex).

Scripts conserves en **observation live** — a re-executer periodiquement au fil des donnees :

- `analyse_judas_swing.py` : enrichissement automatique avec chaque semaine de donnees supplementaires. Signal potentiel sur EURUSD ou NASDAQ.
- `analyse_london_cascade.py` : idem.
- `analyse_cbdr.py` : CBDR (14:00-20:00 ET) comme pre-filtre sur macros NY. OOS actuel : narrow CBDR negatif sur BTC (-0.024% avg, WR 46%), medium CBDR prometteur (+0.699%, WR 72.7%) mais N=11 insuffisant. Hypothese ICT non confirmee sur BTC — a enrichir avec donnees live.

Ne pas modifier, ne pas supprimer.

### P5 — Weekly Profile ICT (classé — pas d'intégration)

Transcriptions analysées : **ICT Month 07 Lesson 2 (profils hebdomadaires) + Lesson 3 (Market Maker Manipulation Templates)**.

**Conclusions :**

- Pi* capture déjà l'essence des templates ICT via `pool_ctx` + `sweep_ctx` + `london_ctx` — les PD arrays de rang 1 (liquidity pools) sont déjà encodés.
- La logique est fractale : ce que Pi* fait en intraday (sweep → expansion vers pool opposé) est exactement ce qu'ICT décrit à l'échelle hebdomadaire (Tuesday extreme → expansion Jeudi/Vendredi).
- **PD Array Matrix** : non utile — explosion N_STATES (1944×3 min = 5832 états) non compensée par le volume de trades actuel (~45 test trades).
- **Filtre jour de semaine** (Tuesday/Wednesday = extrême, Jeudi/Vendredi = exécution) : concept testable en standalone (`analyse_weekly_profile.py`), pas encore codé. Les 44% d'EOD exits s'expliquent en partie par des semaines de consolidation non filtrées.
- **Sur Forex** : les concepts qui échouent sur BTC (CBDR, Judas Swing, London Cascade, Weekly Profiles) deviendraient directement applicables et plus fiables — à réévaluer lors du passage Forex.

Aucune modification de `engine/` justifiée à ce stade.

### P6 — Transcriptions à analyser (classé — ICT en pause)

Décision : pas de développement ICT supplémentaire sur BTC. Pi* a extrait ce qu'ICT peut offrir sur ce marché. ICT Month 10/11/12 non analysés — déprioritisés.

**Roman Paolucci — analyses complètes :**

"How to trade with an Edge" :

- Q-table = espérance conditionnelle non-linéaire (valide, architecture correcte)
- Problème central : N=13-45 trop petit pour distinguer edge réel de chance (sample path)
- Diversification EV : une stratégie sur un actif = fragilité structurelle → voir P7

"Profitable vs. Tradable — Why most strategies fail live" :

- Profitable (backtest) != tradable (live). Ce qui compte : stabilité des distributions P&L OOS
- L'espace d'états de Pi* IS un regime model au sens de Paolucci — architecture validée
- Finding clé : régime volatilité médiane = distribution la plus stable. Pi* ne filtre pas par vol → piste pour plus tard
- Stabilité des features (pool_ctx, sweep_ctx) jamais vérifiée explicitement — implicite dans walk-forward
- Conclusion identique aux deux vidéos : P2 (live) est la condition nécessaire pour qualifier Pi* de "tradable"

**ICT Core Content (déprioritisé) :**

- Month 10, 11, 12 — à analyser uniquement si passage Forex décidé

### P9 — Dashboard mobile (Streamlit Cloud)

Créer `streamlit_app.py` à la racine : dashboard léger sans dépendance à la base SQLite ni à `engine/`.

**Contenu cible :**

- Prix BTC live + contexte marché (london_ctx, sweep_ctx, pool_ctx) via Binance public API (même logique que `live_signal.py`)
- Bouton test Discord (déjà dans `dashboard/app.py` sidebar — à porter ici)
- Déploiement : Streamlit Community Cloud (gratuit, URL fixe, `ton-app.streamlit.app`)

**Dépendances uniquement :** `streamlit`, `requests`, `pandas`, `numpy`, `pytz` → `requirements-streamlit.txt`

**Motivation :** PC éteint le soir → accès mobile impossible avec le dashboard local. Streamlit Cloud reste allumé en permanence.

**Prérequis :** aucun — standalone, à faire indépendamment de P2.

---

### P7 — Deribit Futures Signal (EN PRODUCTION — 2026-04-30)

Module complet déployé sur Railway. Statut : **tests Railway 100% pass**.

**Architecture `analysis/deribit_futures/` :**

7 edges calculés sur OHLCV 1h + funding historique + options analytics Deribit :

| Edge | Type | Logique |
|------|------|---------|
| `edge_funding_reversion` | directionnel | funding extrême → mean reversion |
| `edge_carry_momentum` | directionnel | funding positif fort → suivi tendance |
| `edge_carry_stress` | directionnel | funding très négatif → squeeze haussier |
| `edge_mark_dislocation` | directionnel | basis mark/index > 1.5σ → convergence |
| `edge_options_vol_premium` | non-directionnel | IV réalisée < IV implicite → premium vendeur |
| `edge_skew_panic` | directionnel | put skew extrême → retournement haussier |
| `edge_term_structure_kink` | non-directionnel | anomalie term structure IV |

**Signal (`build_deribit_signal`) :**
- Output : `action` (LONG/SHORT/FLAT/WATCH), `horizon` (4h/4h-24h/24h+), `confidence` (0-1)
- Tenor suggéré : PERP (horizon court), 1W DATED_FUTURE (horizon intermédiaire), 1M (horizon long)
- Drivers : top 4 edges classés par contribution

**Backtest récent (BTC 1h, 90j, threshold=0.05) :**
- `edge_carry_momentum` : 91 signaux, WR 57.1% à +4h ✓
- `edge_skew_panic` : 2156 signaux (quasi-permanent, filtre implicite)
- `edge_mark_dislocation` / `edge_term_structure_kink` : 0 signaux sur 90j → threshold à baisser si besoin

**Signal actuel (2026-04-30) :**
- Action : LONG | Tenor : 1W DATED_FUTURE | Confidence : 0.69 | EdgeTotal : 0.047
- Drivers : skew_panic (0.26), options_vol_premium (0.08), funding_reversion (0.02)

**CLI :**
```bash
# Signal + notification Discord
python analysis/run_deribit_signal.py --asset BTC --timeframe 1h --days 90 --notify

# Edge frame brut
python analysis/run_deribit_futures.py --asset BTC --timeframe 1h --days 90 --output db/deribit_futures_edges.csv

# Backtest
python analysis/run_deribit_backtest.py --asset BTC --timeframe 1h --days 90 --threshold 0.05 --output db/deribit_backtest.csv
```

**Bugfix important :**
- `_clean_json_record` dans `api/app.py` doit être **recursive** — `index_price` peut être `NaN` dans les snapshots Deribit (nœud imbriqué), causait HTTP 500. Corrigé commit `44b1d91`.

---

### P8 — Stratégie SPOT BTC (à développer — remplace ancien P7)

Objectif : second pilier de portefeuille, décorrélé de Pi* (FUTURES intraday), pour diversifier les sources d'EV au sens de Paolucci.

**Concept :**

- Pi* = FUTURES, intraday, micro-structure ICT, horizon minutes
- SPOT = LONG ou FLAT uniquement, horizon jours/semaines, signaux macro

**Base déjà existante (à récupérer) :**

L'utilisateur a déjà construit un outil de signal z-score composite combinant :

- Futures à terme (open interest, funding rate, basis)
- Options (put/call ratio, IV term structure, skew)
- Spot macro : surveillance Whales, on-chain flows (SOPR, MVRV, exchange inflows), Fear & Greed Index

**Architecture cible :**

Appliquer la méthode Pi* à ce signal space :

- Z-score composite → buckets d'états (ex: STRONG_LONG / MILD_LONG / NEUTRAL / FLAT)
- Q-table empirique : E[return | bucket_z] sur historique journalier
- Walk-forward 80/20 strict, même seuil OOS > 0.05%
- Action space réduit : LONG ou FLAT (pas de SHORT en SPOT)

**Avantage N vs Pi* :**

Signal journalier → ~2500 jours d'historique BTC 2019-2026 → N beaucoup plus grand, problème de sample path moins critique.

**Pipeline de données à reconstruire :**

- On-chain : Glassnode ou CryptoQuant (API historique)
- Options : Deribit (historique IV, put/call)
- Futures : Binance (funding rate, OI) — déjà en partie disponible

**Prérequis :**

- P2 (Railway live) stabilisé en premier
- Retrouver/reconstruire le pipeline z-score existant
- Définir la granularité du signal (journalier ou hebdomadaire)

Objectif : extraction de concepts ICT/edge testables, pas d'implémentation immédiate.

### P10 — Stratégie TA (EN PRODUCTION PARTIELLE)

Nouveau module isolé dans `strategies/ta/` (indépendant de `engine/` et `pi_config.py`).

**Composants déployés :**
- `strategies/ta/live_runner.py` : scan live Binance 15m, 108 combinaisons (EMA/RSI/Stoch/ATR)
- `strategies/ta/discord_notify.py` : envoi Discord des signaux TA validés IS+OOS
- `strategies/ta/signal_logger.py` : log CSV `db/ta_signals.csv` + résolution TP/SL + stats live
- `strategies/ta/live_dashboard.py` : dashboard Streamlit live (chart + état + performances)
- `strategies/ta/sweep_rolling.py` : revalidation glissante trimestrielle

**Scheduler API (`api/app.py`) :**
- Scan TA : toutes les 15 min, lun-ven, sessions UTC 07-11 et 13-17
- Résolution trades TA : toutes les heures (`minute=5`)

**Résultats IS/OOS avec feature regime :**
- 2105 états valides IS
- régime `range` fragile (majorité des cassures OOS)
- logique live : exclusion des états `regime=range`

### P11 — Détecteur DVOL (EN PRODUCTION)

Nouveau détecteur de variation de volatilité implicite Deribit :
- `data/deribit.py` : `fetch_dvol_history(...)`
- `analysis/deribit_futures/dvol.py` : `detect_dvol_variation(...)`
- `analysis/run_dvol_detector.py` : CLI + `--notify`

**États DVOL :**
- `VOL_SHOCK_UP`
- `VOL_CRUSH_DOWN`
- `NEUTRAL`

**Sortie clé :**
- `dvol_z`, `dvol_roc_24h`, `intensity`, `risk_regime` (RISK_OFF / RISK_ON / BALANCED)

### P12 — Optimiseur vente d'options (NOUVEAU)

Outil pour sélectionner les meilleures options à vendre selon DTE + strike + liquidité + risque :
- `data/deribit.py` : `fetch_option_chain_snapshot(asset)`
- `analysis/run_option_seller_optimizer.py`

**Principe de scoring :**
- DTE filter (ex: 20-45 jours, cible 30)
- OTM minimum (ex: >= 3%)
- liquidité min (`open_interest`, `volume_usd`)
- probabilité OTM à maturité (approx Black-Scholes via IV)
- score final `sell_score` (yield annualisé + prob_otm + OTM + liquidité + fit DTE)

**Output :**
- top candidates dans le terminal
- CSV complet `db/options_sell_candidates.csv`

### P13 — Tâches à faire (prochaine itération)

1. Ajouter un mode profil de risque (`conservateur`, `équilibré`, `agressif`) dans l'optimiseur options
2. Ajouter un trigger Discord automatique si `sell_score` dépasse un seuil
3. Ajouter endpoint API pour stats TA live (`db/ta_signals.csv`) pour le frontend Next.js
4. Ajouter un endpoint API pour DVOL (`/api/deribit/dvol`) avec cache 15 min
5. Ajouter rétention/rotation de `db/ta_signals.csv` (éviter grossissement infini)

---

## 9. Règles méthodologiques importantes

1. **Ne jamais modifier `engine/stats_state.py` sans valider d'abord avec un script d'analyse standalone.**
   - Modifier N_STATES/encode invalide `stats_agent.pkl` et la Q-table.
   - Workflow : `analyse_XXX.py` standalone → OOS validation (avg > 0.05%) → *alors* modifier le core.

2. **Walk-forward strict** : 80% train, 20% test temporel. Jamais de shuffle.

3. **Seuil de validation** : avg_best > 0.05% ET confirmé en test set.

4. **aligned_only=True** : ne trader que si sc != 0 (sweep à l'entrée de la macro).

5. **target_pool=True** : TP dynamique vers le pool opposé — ne pas utiliser TP fixe.

6. **UnicodeEncodeError Windows** : éviter "→", "★", "≥" dans les print() — Windows cp1252 plante. Utiliser "->", "*", ">=".

7. **Pandas UserWarning** : ne pas faire `day_df._ctx = ctx` — passer les contextes comme liste séparée.

---

## 10. Comment reconstruire la Q-table

```python
from main import run_build_qtable, run_backtest_stats

SKIP_MACROS = frozenset({1, 3, 5, 6, 7})
SKIP_DAYS   = frozenset({0})
MACRO_RULES = {(2, 1, 1): frozenset({1}), (2, 0, 1): frozenset()}

run_build_qtable(
    test_ratio=0.2, min_samples=5, exit_hm=960,
    sl_pct=0.006, rr=2.5, target_pool=True, aligned_only=True,
    skip_macros=SKIP_MACROS, skip_days=SKIP_DAYS, macro_rules=MACRO_RULES
)

run_backtest_stats(
    test_ratio=0.2, q_threshold=0.0, exit_hm=960,
    sl_pct=0.006, rr=2.5, target_pool=True, aligned_only=True,
    skip_macros=SKIP_MACROS, skip_days=SKIP_DAYS,
    entry_mode="baseline", macro_rules=MACRO_RULES
)
```

---

## 12. Module Deribit Futures — référence rapide

### Config classes

```python
# features.py
@dataclass
class EdgeBuildConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 14
    funding_days: int = 30
    zscore_lookback_days: int = 90

# backtest.py
@dataclass
class BacktestConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 90
    threshold: float = 0.05
    min_signals: int = 10
    horizons_hours: list = field(default_factory=lambda: [4, 24])

# signal.py
@dataclass
class SignalConfig:
    asset: str = "BTC"
    timeframe: str = "1h"
    days: int = 90
    min_net_score: float = 0.03
```

### Format du signal retourné

```json
{
  "asset": "BTC", "timeframe": "1h", "days": 90,
  "signal": {
    "action": "LONG",
    "horizon": "4h-24h",
    "confidence": 0.69,
    "net_score": 0.284,
    "contract": {
      "instrument": "DATED_FUTURE",
      "tenor": "1W",
      "why": "Horizon intermediaire, echeance hebdo adaptee a une conviction tactique."
    }
  },
  "edges": { "edge_skew_panic": 0.263, "edge_options_vol_premium": 0.085, ... },
  "drivers": [{"name": "skew_panic", "score": 0.263}, ...],
  "snapshot": { "mark_price": 76331, "index_price": null, "open_interest": 958926150, ... },
  "options": { "iv_atm": 31.4, "iv_skew_25d": 3.16, "put_call_ratio": 0.776, "term_1w": 39.4, ... }
}
```

### Règles de sérialisation JSON

`_clean_json_record(obj)` dans `api/app.py` est **récursive** (dict + list + float NaN → None).
Ne jamais la remplacer par une version plate — `index_price` est souvent NaN dans les snapshots Deribit.

---

## 11. Contact / contexte utilisateur

- L'utilisateur développe Pi* seul, trading quantitatif BTC
- Communication en français
- Préfère les réponses concises sans résumé en fin de réponse
- Working dir : `c:\Users\PC\analyser`
- Platform : Windows 11, bash via Claude Code
