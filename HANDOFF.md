# Pi* — Fichier de passation pour agent IA

> Généré le 2026-04-25. À lire en entier avant de toucher au code.

---

## 1. Vue d'ensemble du projet

**Pi*** est un agent de trading ICT (Inner Circle Trader) sur BTCUSDT Binance, 1-minute candles.
- Données : Binance historique 2020-01-01 → aujourd'hui (3,3M bougies), stockées localement dans `data_binance/`
- Stratégie : Q-table empirique (pas de RL loop) — moyenne des P&L réels par état
- Backtest : walk-forward 80% train / 20% test temporel
- Live : pas encore déployé (Railway prévu)

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
├── main.py                    # run_build_qtable() + run_backtest_stats() + _sim_trade_rr()
├── sweep.py                   # grid search SL/TP — config principale
├── config.py                  # chemins, parametres globaux
├── engine/
│   ├── stats_state.py         # N_STATES, MACROS, encode, decode, compute_pool_ctx, build_weekly_levels
│   └── q_agent.py             # QAgent (save/load pkl, act, update)
├── data/
│   └── binance.py             # load_binance_1m(), download_binance_1m()
├── data_binance/              # CSV Binance BTCUSDT 1min (2019-2026)
├── db/
│   └── stats_agent.pkl        # Q-table active (1944 x 3)
└── analysis/                  # Scripts d'analyse standalone (ne pas modifier engine/ sans eux)
    ├── analyse_0850.py        # Macro 08:50 — globalement negatif, 1 contexte OOS note
    ├── analyse_0950.py        # Macro 09:50 — analyse complete + charts
    ├── analyse_1150.py        # Macro 11:50 — analyse complete + charts
    ├── analyse_1450.py        # Macro 14:50 (Power Hour) — silenciee
    ├── analyse_macros_silencieux.py  # 4 macros silencieuses
    ├── analyse_silver_bullet.py      # Silver Bullet 10:00 ET — pas d'edge OOS
    ├── analyse_judas_swing.py        # Judas Swing London — observation live
    ├── analyse_london_cascade.py     # London Cascade LOR->SB->LM — observation live
    └── analyse_cbdr.py               # CBDR 14:00-20:00 ET — observation live
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

### P2 — Railway deployment (après stabilisation stratégie)
- Discord webhook pour signaux live
- Cron 09:51 ET + 11:51 ET (macros 2 et 4, mais 4 silencée)
- Mode monitoring : log macros silencieuses avec label ✗ sans trader

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

### P7 — Stratégie SPOT BTC (à développer après P2)

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

## 11. Contact / contexte utilisateur

- L'utilisateur développe Pi* seul, trading quantitatif BTC
- Communication en français
- Préfère les réponses concises sans résumé en fin de réponse
- Working dir : `c:\Users\PC\analyser`
- Platform : Windows 11, bash via Claude Code
