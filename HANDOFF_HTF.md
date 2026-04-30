# HTF — Fichier de passation pour agent IA

> Genere le 2026-04-29. A lire en entier avant de toucher au code HTF.
> Projet parent Pi* : voir [HANDOFF.md](HANDOFF.md)
> Historique detaille : voir [PROCEDURE_HTF_COMPLETE.md](PROCEDURE_HTF_COMPLETE.md)
> Specification design : voir [HTF_STATES_AGENT_HANDOFF.txt](HTF_STATES_AGENT_HANDOFF.txt)

---

## 1. Contexte et principes

HTF est une **strategie separee** de Pi*, experimentale, sans interference avec la Q-Table production.

- Q-Table Pi* actuelle : **inchangee, en production** — ne pas toucher
- Approche HTF : probabiliste, regime-first, multi-timeframe (W / D / 4H)
- Etats : probabilites de continuation / mean-reversion / no-trade
- Horizon : signal journalier sur forward return J+1

---

## 2. Architecture des etats HTF

**Etats utilises (source : `db/htf_state_combinations.db`) :**

```
Dimensions :
  weekly_state  (W1-W5)  : regime hebdomadaire options + structure
  daily_state   (D1-D5)  : biais operationnel daily
  h4_state      (H1-H4)  : timing structurel 4H

Combinaisons :
  - combos 2 etats (W x D, W x H4, D x H4)
  - combos 3 etats (W x D x H4)

Forward return :
  - pret_j1 = return du lendemain J+1 via close Binance 1m
```

**10 fiches d'analyse design (dans HTF_STATES_AGENT_HANDOFF.txt) :**

| Fiche | Nom | Timeframe |
|-------|-----|-----------|
| 01 | Ordre High/Low Weekly selon Skew 25d | W |
| 02 | Distance Max Pain et retour avant vendredi | W |
| 03 | Regime GEX et range/trend 4H | 4H |
| 04 | IV 3-jours en hausse et sweep->reversal daily | D |
| 05 | Put/Call ratio extreme et comportement 4H | 4H |
| 06 | GEX faible et break du range daily | D |
| 07 | Term structure inversee et ordre HL/LH weekly | W |
| 08 | Skew tres positif et LOW avant HIGH weekly | W |
| 09 | Skew tres negatif et HIGH avant LOW weekly | W |
| 10 | IV haute + GEX bas : trend vs chop daily | D |

---

## 3. Profils d'admission Q-Table HTF

| Profil | Usage | N_trades | WinRate | Wilson_LB | AvgTrade |
|--------|-------|----------|---------|-----------|----------|
| strict | production prudente | >= 40 | >= 58% | >= 50% | >= +0.25% |
| equilibre | premiere version HTF | >= 20 | >= 54% | >= 45% | >= +0.15% |
| equilibre_assoupli | version assouplie | >= 15 | >= 52% | >= 40% | >= +0.10% |
| agressif | exploration/labo | >= 12 | >= 50% | >= 35% | >= +0.10% |

**Profil recommande actuel : equilibre_assoupli** (base du dry-run execute).

---

## 4. Fichiers cles HTF

```
analyser/
├── HANDOFF_HTF.md                      <- CE FICHIER (passation HTF)
├── PROCEDURE_HTF_COMPLETE.md           <- historique chronologique complet
├── HTF_STATES_AGENT_HANDOFF.txt        <- specification design (10 fiches)
├── db/
│   └── htf/
│       ├── htf_state_combinations.db           <- combinaisons enumrees
│       ├── stats_agent_htf_seed_agressif.pkl   <- seed Q-Table profil agressif (2 etats)
│       └── stats_agent_htf_seed_equilibre_assoupli.pkl  <- seed profil eq. assoupli (4 etats)
├── display/analysis/
│   ├── htf_combo_ranking_summary.txt
│   ├── htf_combo_qtable_shortlist_2.csv
│   ├── htf_combo_qtable_shortlist_3.csv
│   ├── htf_qtable_seed_agressif.csv
│   ├── htf_qtable_seed_equilibre_assoupli.csv
│   ├── htf_dry_run_equilibre_assoupli.csv
│   └── htf_dry_run_equilibre_assoupli_report.txt
└── analysis/
    └── htf/
        ├── generate_htf_state_combinations.py
        ├── build_htf_combo_stats.py
        ├── rank_htf_combo_reliability.py
        ├── build_htf_qtable_seed.py
        └── run_htf_dry_run.py
```

---

## 5. Resultats dry-run (profil equilibre_assoupli, execute 2026-04-29)

| Metrique | HTF | Baseline |
|----------|-----|----------|
| n_days | 2284 | 2284 |
| n_trades | 19 | 19 |
| agreement_pct | 99.47% | — |
| avg_trade_proxy | +0.5095% | -0.3862% |
| total_proxy | +9.58% | -7.54% |

**Shortlist shorts retenues (combinaisons 2 etats) :**

| ID | Etats | Direction | AvgTrade | N |
|----|-------|-----------|----------|---|
| R2_018 | W4 + D3 | SHORT | +1.049% | 14 |
| R2_003 | W1 + D3 | SHORT | +0.253% | 18 |
| R3_006 | W1 + D3 + H2 | SHORT (satellite) | variable | 15 |

---

## 6. Etat actuel et decisions actees (2026-04-29)

**Constat :**
- Frequence effective : ~3 trades/an (~0.83% des jours)
- Duree moyenne des sequences signal : 9.5 jours (seulement 2 sequences sur 2284j)
- Jugement : trop faible pour l'objectif operationnel

**Decisions actees :**

1. **Retravail de la frequence** : assouplir les criteres de selection ou elargir les combinaisons
   pour atteindre ~10-15 trades/an minimum sans degrader excessivement la qualite.

2. **Reorganisation des fiches** : fait (ce fichier cree, HANDOFF.md nettoye).

---

## 7. Prochaine session — actions prioritaires

### P1 — Augmenter la frequence des signaux HTF

Options a explorer :
- Option A : abaisser les seuils d'admission (Wilson_LB, AvgTrade) pour capter plus d'etats
- Option B : elargir a plus de combinaisons (inclure combos rejetees proches du seuil)
- Option C : utiliser des states unitaires (W seul, D seul) en plus des combos
- Option D : desagréger — ne pas exiger co-presence W+D+H4, signal sur D seul suffit
- Option E : allonger la periode de look-back pour augmenter N par etat

Critere de succes : atteindre >= 10 trades/an sur backtest walk-forward sans degrader
WinRate < 50% ni AvgTrade < 0%.

### P2 — Script de diagnostic frequence

Creer `analysis/diagnose_htf_frequency.py` :
- Distribuer les etats par frequence d apparition dans le temps
- Identifier les etats proches du seuil de retention (N entre 8 et 15)
- Simuler differents seuils et afficher l'impact sur la frequence resultante

---

## 8. Regles methodologiques HTF

1. **Ne jamais modifier `stats_agent.pkl` ni `engine/`** — HTF est completement separe.
2. **Walk-forward strict** : 80% train, 20% test temporel. Jamais de shuffle.
3. **Source de marche** : auto (DB locale si disponible, sinon Binance historique CSV).
4. **Mode derives** : proxy (Deribit options non disponibles en historique complet).
5. **UnicodeEncodeError Windows** : eviter les caracteres speciaux dans les print().

---

## 9. Methode de reprise rapide

```python
# Relancer le classement avec un profil assoupli
# analysis/htf/rank_htf_combo_reliability.py
# Modifier les seuils en tete de fichier puis relancer

# Relancer le dry-run
# analysis/htf/run_htf_dry_run.py --profil equilibre_assoupli --source binance --mode proxy
```

**Verifier avant de coder :**
- `db/htf_state_combinations.db` existe (enumeration faite)
- `display/analysis/htf_relevant_combo_stats_2.csv` existe (stats calculees)
- Sinon relancer `analysis/build_htf_combo_stats.py` en premier
