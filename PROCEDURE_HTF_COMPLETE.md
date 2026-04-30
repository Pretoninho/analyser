# Procedure Complete HTF (Historique des actions)

Date de mise a jour: 2026-04-29

Ce document resume toute la procedure realisee depuis le debut de la demande HTF, en ordre chronologique, avec objectifs, decisions, code, fichiers et sorties.

## 1) Point de depart

- Demande initiale: creer une autre Q-Table en parallele sans interférer avec l'existante.
- Contrainte retenue:
  - ne pas toucher la Q-Table production existante
  - isoler toute experimentation HTF

## 2) Pivot strategique valide

- Decision utilisateur:
  - nouvelle strategie completement HTF
  - etats differents de la Q-Table actuelle
  - approche probabiliste, non statique
  - phase design d'abord, puis code apres validation

## 3) Formalisation design HTF

- Fichiers de specification produits:
  - [HTF_ETATS_SPEC.txt](HTF_ETATS_SPEC.txt)
  - [HTF_STATES_AGENT_HANDOFF.txt](HTF_STATES_AGENT_HANDOFF.txt)
- Contenu valide:
  - etats Weekly / Daily / 4H
  - approche regime-first probabiliste
  - 10 fiches d'analyse statistique
  - schema de fusion multi-timeframe

## 4) Integration donnees options Deribit

- Objectif: collecter et stocker IV, skew, PCR, term structure, max pain, GEX.
- Fichiers modifies:
  - [data/deribit.py](data/deribit.py)
  - [data/collector.py](data/collector.py)
  - [data/storage.py](data/storage.py)
  - [data/__init__.py](data/__init__.py)
- Champs ajoutes/validees:
  - iv_atm, iv_skew_25d, iv_skew_10d
  - put_call_ratio
  - term_1w, term_1m, term_3m, term_6m
  - max_pain
  - gex

## 5) Corrections techniques effectuees

- Fix max pain dans Deribit:
  - correction d'une erreur liee a `Index.clip`
- Compatibilite pandas:
  - remplacement de `fillna(method="ffill")` par `ffill()`
- Robustesse execution scripts:
  - gestion des imports racine (`sys.path`)
  - fallback auto vers historique Binance si DB locale insuffisante

## 6) Backtest HTF MVP mis en place

- Script cree:
  - [analysis/backtest_htf_probabilistic.py](analysis/backtest_htf_probabilistic.py)
- Capacites:
  - mode derivatives `strict|hybrid|proxy`
  - source marche `auto|db|binance`
  - probabilites continuation / mean reversion / no-trade
- Sorties:
  - [display/analysis/htf_probabilistic_backtest_trades.csv](display/analysis/htf_probabilistic_backtest_trades.csv)
  - [display/analysis/htf_probabilistic_backtest_report.txt](display/analysis/htf_probabilistic_backtest_report.txt)

## 7) Enumeration des combinaisons d'etats

- Script cree:
  - [analysis/generate_htf_state_combinations.py](analysis/generate_htf_state_combinations.py)
- Base creee:
  - [db/htf_state_combinations.db](db/htf_state_combinations.db)
- Volume:
  - 10 etats de base
  - combinaisons 2 etats
  - combinaisons 3 etats

## 8) Statistiques par combinaison (LONG/SHORT/FLAT)

- Script cree:
  - [analysis/build_htf_combo_stats.py](analysis/build_htf_combo_stats.py)
- Exports produits:
  - [display/analysis/htf_relevant_combo_stats_2.csv](display/analysis/htf_relevant_combo_stats_2.csv)
  - [display/analysis/htf_relevant_combo_stats_3.csv](display/analysis/htf_relevant_combo_stats_3.csv)

## 9) Classement fiabilite et admissions Q-Table

- Script cree puis etendu:
  - [analysis/rank_htf_combo_reliability.py](analysis/rank_htf_combo_reliability.py)
- Profils d'admission implementes:
  - strict
  - equilibre
  - equilibre_assoupli
  - agressif
- Sorties principales:
  - [display/analysis/htf_combo_ranking_summary.txt](display/analysis/htf_combo_ranking_summary.txt)
  - [display/analysis/htf_combo_ranking_2_equilibre_assoupli.csv](display/analysis/htf_combo_ranking_2_equilibre_assoupli.csv)
  - [display/analysis/htf_combo_ranking_3_equilibre_assoupli.csv](display/analysis/htf_combo_ranking_3_equilibre_assoupli.csv)
  - [display/analysis/htf_combo_qtable_shortlist_2.csv](display/analysis/htf_combo_qtable_shortlist_2.csv)
  - [display/analysis/htf_combo_qtable_shortlist_3.csv](display/analysis/htf_combo_qtable_shortlist_3.csv)

## 10) Construction seeds Q-Table HTF

- Script cree:
  - [analysis/build_htf_qtable_seed.py](analysis/build_htf_qtable_seed.py)
- Artefacts generes:
  - [db/stats_agent_htf_seed_agressif.pkl](db/stats_agent_htf_seed_agressif.pkl)
  - [display/analysis/htf_qtable_seed_agressif.csv](display/analysis/htf_qtable_seed_agressif.csv)
  - [db/stats_agent_htf_seed_equilibre_assoupli.pkl](db/stats_agent_htf_seed_equilibre_assoupli.pkl)
  - [display/analysis/htf_qtable_seed_equilibre_assoupli.csv](display/analysis/htf_qtable_seed_equilibre_assoupli.csv)

## 11) Dry-run HTF execute

- Script cree:
  - [analysis/run_htf_dry_run.py](analysis/run_htf_dry_run.py)
- Execution realisee:
  - profil `equilibre_assoupli`
  - source marche `binance`
  - mode derives `proxy`
- Resultats resumes:
  - n_days=2284
  - htf_trades=19
  - baseline_trades=19
  - agreement_pct=99.47%
  - htf_avg_trade_proxy=+0.5095%
  - baseline_avg_trade_proxy=-0.3862%
  - htf_total_proxy=+9.58%
  - baseline_total_proxy=-7.54%
- Fichiers dry-run:
  - [display/analysis/htf_dry_run_equilibre_assoupli.csv](display/analysis/htf_dry_run_equilibre_assoupli.csv)
  - [display/analysis/htf_dry_run_equilibre_assoupli_report.txt](display/analysis/htf_dry_run_equilibre_assoupli_report.txt)

## 12) Etat actuel (fin de procedure)

- Q-Table production existante: non modifiee.
- Pipeline HTF experimental: en place.
- Classement et shortlist: en place.
- Seed Q-Table HTF: en place.
- Dry-run HTF: execute avec rapport.

## 13) Prochaine etape conseillee

- Integrer un loader HTF minimal en mode dry-run dans le flux live/shadow,
  sans remplacer la logique actuelle, pour comparer les signaux en production passive.

## 14) Decision actee (2026-04-29) — Reorganisation et frequence

- Constat valide: une frequence autour de 3 trades/an est trop faible pour l objectif.
- Action decidee: retravailler la selection/filtrage pour augmenter la frequence des signaux
  sans degrader excessivement la qualite.
- Action decidee pour la prochaine session:
  - reorganiser les fiches pour separer clairement ce qui concerne Pi* et ce qui concerne HFT
  - appliquer cette separation dans la documentation et le suivi operatif

## 15) Reorganisation documentaire effectuee (2026-04-30)

- Fichier cree: [HANDOFF_HTF.md](HANDOFF_HTF.md) — passation operationnelle HTF (etat actuel, decisions, prochaines etapes)
- Fichier nettoye: [HANDOFF.md](HANDOFF.md) — ne contient plus que Pi*, avec pointeur vers HANDOFF_HTF.md
- Ce fichier (PROCEDURE_HTF_COMPLETE.md) conserve le journal chronologique complet HTF.
- [HTF_STATES_AGENT_HANDOFF.txt](HTF_STATES_AGENT_HANDOFF.txt) conserve la specification design des 10 fiches.

