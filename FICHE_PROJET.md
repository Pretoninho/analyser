# Pi* — Fiche de présentation

> Agent de détection de signaux de trading ICT sur BTC/USDT Futures (Binance)
> Génère des alertes manuelles à des fenêtres horaires précises — aucune exécution automatique.

---

## 1. Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│                        DONNÉES BRUTES                           │
│         Binance BTCUSDT Futures — bougies 1 minute              │
│              2020-01-01 → aujourd'hui (~3,3M bougies)           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYSE DU CONTEXTE                          │
│  Pour chaque jour : quel contexte de marché est en place ?      │
│  - Où est London par rapport à Asia ?  (RAID_H / RAID_L / rien) │
│  - Quels niveaux de liquidité sont visibles ?  (BSL / SSL)      │
│  - Quelle période de l'année ?  (début / milieu / fin)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FENÊTRES MACROS ICT                           │
│  3 moments précis dans la session NY où un signal peut émerger  │
│           09:50 ET  /  11:50 ET  /  14:50 ET                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Q-TABLE (Pi*)                              │
│  1 944 états × 3 actions  →  EV moyen observé historiquement    │
│  "Dans ce contexte précis, LONG / SHORT / FLAT est-il gagnant?" │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SIGNAL                                    │
│          Alerte manuelle → Discord (futur : Railway)            │
│     "09:50 — SHORT — contexte RAID_H × BSL × SW_H"             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. La journée de trading vue par Pi*

Le marché BTC tourne 24h/24. Pi* découpe chaque journée en trois sessions :

```
  UTC :  00h    03h    06h    09h    12h    15h    18h    21h    00h
         │      │      │      │      │      │      │      │      │
  ───────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
         │      │      │      │      │      │      │      │      │
  ASIA   ████████████████                                         │
  (01-05 UTC)                                                      │
                                                                   │
  LONDON              ████████████                                 │
  (07-10 UTC)                                                      │
                                                                   │
  NY                             ██████████████████               │
  (13-21 UTC)                                                      │

  Heure ET (New York) :
         19h    22h    01h    04h    07h    09h    12h    15h    17h
```

**Ce qui intéresse Pi* :**

| Session | Rôle dans le modèle |
|---------|---------------------|
| **Asia** (01h-05h UTC) | Définit le range de référence. Les extrêmes Asia = niveaux de liquidité naturels (BSL au-dessus, SSL en dessous). |
| **London** (07h-10h UTC) | Révèle l'intention institutionnelle. Est-ce que London a cassé au-dessus de l'Asia High (= RAID_H) ou en dessous de l'Asia Low (= RAID_L) ? |
| **New York** (13h-21h UTC) | Là où Pi* trade. 3 fenêtres précises dans cette session. |

---

## 3. Les concepts ICT utilisés

### 3.1 Les niveaux de liquidité (BSL / SSL)

Les institutionnels déplacent le prix vers des zones où des stops sont agrégés, pour se remplir à moindre coût. Pi* cartographie ces zones chaque jour.

```
        Prix
          │
  ════════╪══════ BSL (Buy-Side Liquidity)  ← stops des vendeurs à découvert
          │           = high de la session précédente
          │
          │    marché évolue ici
          │
  ════════╪══════ SSL (Sell-Side Liquidity) ← stops des acheteurs
          │           = low de la session précédente
          │
```

Quand le prix **passe brièvement au-delà** d'un de ces niveaux puis revient, on appelle ça un **sweep** (balayage de liquidité). C'est le signal de déclenchement.

### 3.2 London RAID

```
                 HIGH ASIA
  ─────────────────────────────────────────
               ↑  London monte au-dessus
               │  = RAID_H
               │  → indique que London a "aspiré" la liquidité haute
               │  → NY devrait corriger vers le bas

  ─────────────────────────────────────────
                 LOW ASIA
               ↓  London descend en-dessous
               │  = RAID_L
               │  → indique que London a "aspiré" la liquidité basse
               │  → NY devrait remonter vers le haut
```

### 3.3 Les Macros ICT

Les "Macros" sont des fenêtres de 10 minutes où les algorithmes institutionnels sont particulièrement actifs. Elles sont connues et documentées par la communauté ICT.

```
  Heure ET   09:30        10:00         12:00        14:00        16:00
             │            │             │            │            │
  ───────────┼────────────┼─────────────┼────────────┼────────────┤ fin NY
             │            │             │            │            │
  Open NY    │ ┌──┐ 08:50 │  ┌──┐10:50 │ ┌──┐12:50 │ ┌──┐14:50 │
             │ └──┘       │  └──┘       │ └──┘       │ └──┘      │
             │    ┌──┐    │             │    ┌──┐    │           │
             │    │09:50  │             │    │13:50  │           │
             │    └──┘    │             │    └──┘    │           │
             │       ┌──┐ │             │       ┌──┐ │           │
             │       │   │             │       │11:50           │
             │       └──┘ │             │       └──┘ │           │

  Macros actives dans Pi* :  09:50 ★  /  11:50  /  14:50
  Macros silencieuses :      08:50  /  10:50  /  12:50  /  13:50
```

À chaque macro, Pi* observe les **20 minutes précédentes** (pre-macro) pour détecter si la première bougie de la macro **balaie** le high ou le low de cette pré-période.

```
  Fenêtre pré-macro          │  Ouverture macro (ex: 09:50)
  09:30 ──────────── 09:50   │  09:50 ──────────────────────────
                             │
  pre_high ══════════════════╪═══ → première bougie casse au-dessus → SW_H → SHORT
                             │
  cours évolue ici           │
                             │
  pre_low  ══════════════════╪═══ → première bougie casse en dessous → SW_L → LONG
```

---

## 4. L'espace d'états — le "vocabulaire" de Pi*

Pi* ne voit pas le prix brut. Il traduit chaque situation en un **état discret** parmi 1 944 possibles.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    ÉTAT = 6 composantes                      │
  │                                                              │
  │  1. month_ctx     3 valeurs   Début / Milieu / Fin de mois  │
  │  2. day_ctx       3 valeurs   Jour fort / neutre / faible    │
  │  3. london_ctx    3 valeurs   NO_RAID / RAID_H / RAID_L      │
  │  4. macro_ctx     8 valeurs   Quelle macro (08:50 à 14:50)   │
  │  5. sweep_ctx     3 valeurs   NO_SW / SW_H / SW_L            │
  │  6. pool_ctx      3 valeurs   NEUTRAL / BSL_swept / SSL_swept│
  │                                                              │
  │  Total : 3 × 3 × 3 × 8 × 3 × 3 = 1 944 états               │
  └──────────────────────────────────────────────────────────────┘
```

### Exemple concret — signal 09:50

```
  ┌─────────────────────────────────────────────────────────────┐
  │  Mardi 15 avril 2025, macro 09:50                           │
  │                                                             │
  │  month_ctx  = 1  (milieu de mois)                          │
  │  day_ctx    = 1  (mardi = neutre)                          │
  │  london_ctx = 1  (RAID_H : London a cassé l'Asia High)     │
  │  macro_ctx  = 2  (macro 09:50)                             │
  │  sweep_ctx  = 1  (SW_H : première bougie casse le pre_high)│
  │  pool_ctx   = 1  (BSL_swept : le pool BSL est au-dessus)   │
  │                                                             │
  │  → État encodé : #1030                                     │
  │  → Q[SHORT] = +0.61%  →  SIGNAL : SHORT                   │
  └─────────────────────────────────────────────────────────────┘
```

---

## 5. Construction de la Q-table

La Q-table est une **table de résultats historiques** — pas un réseau de neurones. Pour chaque (état, action), on calcule la moyenne des P&L observés sur les 6 ans de données.

```
  Pour chaque jour du dataset (2020 → 2025) :
  ┌────────────────────────────────────────────────────────────┐
  │  1. Calculer le contexte du jour (london, pool, month...) │
  │  2. Pour chaque macro : détecter le sweep               │
  │  3. Encoder l'état                                      │
  │  4. Simuler : que se passe-t-il si on trade LONG ?      │
  │               que se passe-t-il si on trade SHORT ?     │
  │  5. Accumuler les P&L par (état, action)                │
  └────────────────────────────────────────────────────────────┘

  Q[état, LONG]  = moyenne des P&L LONG observés dans cet état
  Q[état, SHORT] = moyenne des P&L SHORT observés dans cet état
  Q[état, FLAT]  = 0 (ne rien faire = 0% de gain/perte)

  Seuil : si Q[action] ≤ 0 → on ne trade pas (action = FLAT)
```

### Découpage train / test

```
  2020  ───────────────────────────────────────────  2026
  │                                                    │
  │◄────────────── 80% TRAIN (1845 jours) ────────────►│◄── 20% TEST (461 j) ──►│
  │                                                    │                         │
  │  Q-table construite ici                            │  Backtest ici           │
  │  (jamais vu pendant le test)                       │  (jamais vu en train)   │
```

---

## 6. Les filtres — ce qui distingue un signal du bruit

Pi* n'entre en position que si **plusieurs conditions s'alignent simultanément** :

```
  Condition 1 : --aligned-only
  ─────────────────────────────
  La première bougie de la macro DOIT sweeper le high ou le low pré-macro.
  Si la bougie reste dans la range → FLAT (pas de signal)

  Condition 2 : Q-table positive
  ─────────────────────────────
  Q[action] > 0  (l'action a été gagnante en moyenne historique)
  Sinon → FLAT

  Condition 3 : macro_rules (règles directionnelles)
  ─────────────────────────────
  Règles spécifiques encodées par contexte (voir section suivante)
```

### Les règles directionnelles actives

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Macro 09:50 — Règle 1                                              │
  │                                                                     │
  │  Contexte : London RAID_H  ×  BSL_swept  ×  SW_H                   │
  │                                                                     │
  │  London a monté ──►  pool BSL visible  ──►  macro casse en haut    │
  │  au-dessus Asia         au-dessus             = SW_H               │
  │                                                     │               │
  │                                                     ▼               │
  │                                              → SHORT uniquement     │
  │  Formation : WR=66.7%, PF=2.84 sur 12 occurrences historiques       │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Macro 09:50 — Règle 2                                              │
  │                                                                     │
  │  Contexte : NO_RAID  ×  BSL_swept  (toute direction)               │
  │                                                                     │
  │  → TOUJOURS BLOQUÉ                                                  │
  │  Raison : signal régime-dépendant (2020-2023 perdant, 2024 outlier) │
  │  Sans raid London, le pool BSL sans confluence → faux signal        │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Gestion des sorties — SL / TP / EOD

Pi* utilise un **TP dynamique ICT** (`--target-pool`) plutôt qu'un ratio fixe :

```
  Exemple SHORT après BSL sweep :

  Prix
    │
    │  ═══════════════  BSL (pool balayé à la macro)
    │         ↑
    │     entrée SHORT ici (première bougie)
    │         │
    │         ▼ mouvement attendu
    │
    │  ═══════════════  SSL (pool opposé) ← TP dynamique
    │
    │  (si pas de SSL visible : sortie à 16h00 ET)

  SL : 0.5% au-dessus du sweep
  TP : prix du SSL (pool opposé)
  EOD : sortie 16:00 ET si ni SL ni TP atteints
```

```
  Résultats des sorties (test set actuel) :
  ┌─────────────────────────────────────────┐
  │  TP touché  : 17%  ████                 │
  │  SL touché  : 42%  █████████████        │
  │  EOD sortie : 42%  █████████████        │
  └─────────────────────────────────────────┘
  → Beaucoup de sorties EOD = potentiel d'amélioration avec meilleur TP
```

---

## 8. Résultats sur le set de test (461 jours, 2025-2026)

### Par macro

```
  Macro    │  N trades  │  Win rate  │  Return total
  ─────────┼────────────┼────────────┼───────────────
  09:50    │    22      │   45.5%    │    +0.05%   →
  11:50    │    23      │   56.5%    │    +2.10%   ↑
  14:50    │    12      │   25.0%    │    -2.56%   ↓  ← à investiguer
  ─────────┼────────────┼────────────┼───────────────
  TOTAL    │    57      │   45.6%    │    -0.51%
```

### Fréquence de trading

```
  Sur 461 jours de test :

  Jours avec signal : 54  (12%)  ████
  Jours sans signal : 341 (74%)  ████████████████████████
  Jours filtrés (lundi skip) : 66 (14%)

  → En moyenne : ~1 trade par semaine
  → Philosophie : qualité > fréquence
```

### Analyse du signal 09:50 (★ le plus étudié)

```
  Contexte RAID_H × BSL_swept :
  ┌──────────────────────────────────────────────────────────────┐
  │   Avec SW_H → SHORT        │  Avec SW_L → LONG              │
  │   N=12   WR=66.7%          │  N=11   WR=36.4%               │
  │   Avg = +0.432%            │  Avg = -0.179%                  │
  │   PF = 2.84                │  PF = 0.59                      │
  │                            │                                  │
  │   ██████████████████████   │   ████████████                  │
  │   Signal valide ✓          │   Bruit → bloqué ✗              │
  └──────────────────────────────────────────────────────────────┘
```

---

## 9. Architecture technique

```
  analyser/
  │
  ├── main.py              Point d'entrée principal (CLI)
  │                        --build-qtable, --backtest-stats, --download-binance...
  │
  ├── sweep.py             Recherche automatique des meilleurs paramètres SL×RR
  │                        Contient MACRO_RULES (règles directionnelles)
  │
  ├── config.py            Configuration générale
  │
  ├── data/
  │   └── binance.py       Téléchargement Binance Vision (mensuel + journalier)
  │                        → data_binance/btcusdt_1m.parquet (3,3M bougies)
  │
  ├── engine/
  │   ├── stats_state.py   Encodage des états (1944 états), MACROS, encode/decode
  │   ├── q_agent.py       Q-table (sauvegarde / chargement / act)
  │   └── entry_stats.py   Statistiques d'entrée (FVG, OTE, NWOG)
  │
  └── db/
      ├── stats_agent.pkl  Q-table sauvegardée (48 KB) ← commité sur GitHub
      └── trades_log.csv   Log détaillé des trades du dernier backtest
```

### Commandes clés

```bash
# Télécharger les données Binance (depuis 2020)
python main.py --download-binance --start-year 2020 --start-month 1

# Construire la Q-table
python main.py --build-qtable \
  --aligned-only --skip-macros 1,3,5,6 --skip-days 0 \
  --min-samples 5 --macro-rules "2,1,1:1|2,0,1:"

# Backtest
python main.py --backtest-stats \
  --aligned-only --skip-macros 1,3,5,6 --skip-days 0 \
  --sl-pct 0.005 --rr 2.0 --target-pool \
  --macro-rules "2,1,1:1|2,0,1:"

# Recherche des meilleurs paramètres SL/RR
python sweep.py
```

---

## 10. Flux de signal en production (futur — Railway)

```
  Tous les jours de trading :

  09:49 ET ──► Cron Railway déclenche le script
                     │
                     ▼
              Calcul du contexte
              (london_ctx, pool_ctx...)
                     │
                     ▼
              Sweep détecté à 09:50 ?
              ├─ Non  →  FLAT, rien
              └─ Oui  →  Q-table lookup
                               │
                    Q[action] > 0  ?
                    ├─ Non   →  FLAT, rien
                    └─ Oui   →  Vérif macro_rules
                                      │
                         Règle OK ?
                         ├─ Non  →  FLAT, rien
                         └─ Oui  →  Webhook Discord
                                    "09:50 SHORT ↓
                                     RAID_H × BSL × SW_H
                                     TP : 94 200 | SL : 96 300"

  Idem à 11:49 ET et 14:49 ET.
```

---

## 11. Ce que Pi* N'est PAS

```
  ✗  Pi* n'exécute aucun ordre automatiquement
  ✗  Pi* n'est pas un bot de trading
  ✗  Pi* ne prédit pas le prix
  ✗  Pi* ne gère pas de position ouverte

  ✓  Pi* détecte des contextes statistiquement favorables
  ✓  Pi* alerte le trader qui prend la décision finale
  ✓  Pi* apprend de 6 ans d'historique, non d'une intuition
  ✓  Pi* filtre ~88% des jours où le contexte n'est pas en place
```

---

## 12. Lexique rapide

| Terme | Définition |
|-------|-----------|
| **BSL** | Buy-Side Liquidity — stops des vendeurs agrégés au-dessus d'un high |
| **SSL** | Sell-Side Liquidity — stops des acheteurs agrégés sous un low |
| **Sweep** | Le prix casse brièvement un niveau de liquidité puis revient |
| **RAID** | London casse l'extrême du range Asia (RAID_H = casse le haut) |
| **Macro ICT** | Fenêtre de 10 min où les algos institutionnels sont actifs |
| **SW_H** | Sweep High — première bougie de la macro casse le pré-macro high |
| **SW_L** | Sweep Low — première bougie de la macro casse le pré-macro low |
| **Q-table** | Table de valeurs historiques : EV moyen par (état, action) |
| **EOD** | End of Day — sortie forcée à 16:00 ET (fin de la session NY) |
| **PF** | Profit Factor — somme des gains / somme des pertes (>1 = gagnant) |
| **EV** | Expected Value — gain moyen attendu par trade |
| **ET** | Eastern Time (New York) — fuseau horaire de référence ICT |
