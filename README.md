# Projet 5 - Modèle d'allocation de portefeuille par machine learning

---

# 1. Informations du Projet

- **Titre du projet :** Modèle d'allocation de portefeuille par machine learning
- **Nom du groupe :** Groupe 1
- **Membres du groupe :**
  - Étudiant 1 – Maël Pertuisot
  - Étudiant 2 – Valentin Martel
  - Étudiant 3 – Mathias Garnier

- **Cours :** AI In Finance
- **Encadrants :** Nicolas De Roux & Mohamed El Fakir
- **Date de soumission :** 20/04/2026

---

# 2. Description du Projet

L'allocation de portefeuille consiste à répartir un capital entre plusieurs actifs pour maximiser le rendement ajusté au risque. Les méthodes traditionnelles (Markowitz) reposent sur des estimations de rendements et de covariances souvent instables.

Ce projet explore l'utilisation du Machine Learning pour prédire les rendements hebdomadaires de 50 actions du NASDAQ, puis utilise ces prédictions dans un cadre d'optimisation de portefeuille. On compare trois stratégies (Equal Weight, Minimum Variance, Maximum Sharpe avec signaux ML) sur une période out-of-sample 2023–2025, avec un retraining walk-forward trimestriel.

La **frontière efficiente de Markowitz** est tracée pour visualiser le positionnement de chaque stratégie dans l'espace rendement/risque.

Ce travail intéresse les gérants de fonds, analystes quantitatifs, et toute personne travaillant à l'intersection du data science et de la finance.

**Dataset de référence :** Stock Market Dataset (NASDAQ Universe) — données téléchargées via `yfinance`.

---

# 3. Objectif du Projet

Le projet vise à :

- Sélectionner 50 actions NASDAQ couvrant 8 secteurs avec liste de backup automatique
- Calculer des features : rendements passés (lags 1–5), volatilité, momentum, RSI, volume relatif, corrélation au marché, one-hot sectoriel
- Prédire les rendements hebdomadaires (next-week) avec quatre modèles ML (Régression Linéaire, Ridge, Random Forest, GRU)
- Construire trois portefeuilles optimisés à partir des prédictions et d'une covariance Ledoit-Wolf
- Tracer la frontière efficiente et y positionner les différentes stratégies
- Évaluer la performance : ratio de Sharpe, rendement annualisé, maximum drawdown, turnover

---

# 4. Définition de la Tâche

- **Type de tâche :** Régression (prédiction de rendements continus)
- **Variables d'entrée :** Rendements passés (lags 1–5), ratio de moyenne mobile (5j/20j), volatilité glissante (10j et 20j), RSI, momentum (5j et 20j), volume relatif, corrélation glissante au marché (30j), secteur (one-hot)
- **Variable cible :** Rendement cumulé sur les 5 prochains jours (next-week return)
- **Métriques d'évaluation :**
  - Prédiction : RMSE, MAE, R²
  - Portefeuille : Rendement annualisé, Volatilité annualisée, Ratio de Sharpe, Max Drawdown, Turnover moyen

---

# 5. Description du Dataset

## Vue d'ensemble

- **Nombre d'actions :** 50 tickers NASDAQ (avec liste de backup par secteur pour remplacer les tickers indisponibles)
- **Période :** 2018-01-01 à 2026-01-01 (~2 000 jours de trading)
- **Split temporel :** Train 2018–2022 / Test out-of-sample 2023–2025
- **Nombre total d'observations :** ~100 000 (50 tickers × ~2 000 jours)
- **Nombre de features :** 13 features numériques + 8 colonnes one-hot secteur
- **Source des données :** Yahoo Finance via `yfinance`

## Description des features

| Feature | Description | Type |
|---------|------------|------|
| ret_lag1 à ret_lag5 | Rendements journaliers passés (1 à 5 jours) | Numérique |
| ma_ratio | Ratio moyenne mobile 5j / 20j (signal de tendance) | Numérique |
| vol_10j, vol_20j | Volatilité réalisée glissante (10 et 20 jours) | Numérique |
| rsi | Relative Strength Index (14 jours, oscillateur 0–100) | Numérique |
| momentum_5j, momentum_20j | Variation relative sur 5 et 20 jours | Numérique |
| vol_rel | Volume relatif (volume / moyenne mobile 20j) | Numérique |
| corr_marche | Corrélation glissante 30j avec le marché équipondéré | Numérique |
| sect_* | Secteur de l'action (one-hot encoding, 8 colonnes) | Catégoriel |

## Variable cible

- Nom : `target`
- Signification : rendement cumulé sur les 5 prochains jours (`pct_change(5).shift(-5)`)
- Valeurs continues, centrées autour de 0

## Distribution des données

- Les rendements hebdomadaires sont approximativement centrés autour de 0 avec des queues épaisses
- La volatilité varie fortement dans le temps (COVID-19 en 2020, hausse des taux Fed en 2022)
- Les corrélations inter-sectorielles varient dans le temps
- Pas de déséquilibre de classes (régression)

## Qualité des données

- Les tickers avec plus de 5% de NaN sont exclus et remplacés par un backup du même secteur
- Forward-fill ponctuel sur les NaN isolés (jours fériés non communs) avant `dropna()` global
- Pas de doublons

---

# 6. Prétraitement des données

- **Valeurs manquantes :** Tickers avec >5% NaN exclus et remplacés par backup sectoriel, puis `dropna()` sur les features
- **Feature Engineering :** 13 features techniques créées à partir des prix et volumes bruts
- **Encodage catégoriel :** One-hot encoding des 8 secteurs (sans `drop_first` pour préserver l'interprétabilité)
- **Normalisation :** `StandardScaler` sur les features numériques uniquement (fit sur train uniquement, les one-hot ne sont pas scalées)
- **Split temporel :** Train < 2023, Test ≥ 2023 (pas de mélange)

---

# 7. Approche de modélisation

## Modèles utilisés

1. **Régression Linéaire** — baseline interprétable
2. **Ridge** (α=10) — régularisation L2 pour réduire la variance
3. **Random Forest** (100 arbres, max_depth=6, min_samples_leaf=20) — non-linéaire, robuste aux interactions
4. **GRU** (hidden_dim=32, fenêtre=10 jours) — réseau récurrent, deep learning séquentiel

## Choix du GRU

Le GRU (Gated Recurrent Unit) a été préféré au LSTM car :
- Il utilise 2 gates (update, reset) au lieu de 3, ce qui réduit le nombre de paramètres d'environ 25%
- Il est plus rapide à entraîner avec une précision comparable
- Il est particulièrement adapté aux horizons moyens (daily/weekly) selon la littérature
- Le risque d'overfitting est moindre lorsque les données sont limitées

## Stratégie de modélisation

- Baseline linéaire puis complexité croissante
- Panel cross-sectionnel (date × ticker) : ~100 000 observations, un seul modèle pour les 50 actions
- Split temporel strict (pas de shuffle) et `TimeSeriesSplit` (5 folds) pour la validation croisée
- Random Forest avec profondeur limitée (max_depth=6) pour éviter l'overfitting

## Backtest walk-forward

- Retraining **trimestriel** sur la période de test (expanding window, ~12 folds sur 3 ans)
- À chaque date de retraining, le modèle est réajusté sur tout l'historique disponible
- Les prédictions walk-forward du Random Forest servent de signaux pour les stratégies de portefeuille

## Stratégies de portefeuille

| Stratégie | Principe | Signal ML |
|---|---|---|
| **Equal Weight (1/N)** | Benchmark naïf, poids constant 1/N | Non |
| **Minimum Variance (GMV)** | Minimise la variance du portefeuille | Non (risque pur) |
| **Maximum Sharpe (ML)** | Maximise le ratio de Sharpe avec μ = prédictions ML | Oui |

**Contraintes communes :**
- Long-only : w_i ≥ 0
- Budget : Σw_i = 1
- Rebalancement **hebdomadaire** (toutes les 5 séances)
- Covariance estimée par **shrinkage Ledoit-Wolf** sur fenêtre glissante de 60 jours

## Frontière efficiente

La frontière efficiente de Markowitz est tracée sur la période de test avec :
- Les 50 actions individuelles annotées par ticker
- Le portefeuille GMV (Global Minimum Variance)
- Le portefeuille tangent (Maximum Sharpe)
- La Capital Market Line (CML)

Les 3 stratégies sont positionnées dans cet espace rendement/risque.

## Métriques d'évaluation

- **RMSE / MAE / R²** — qualité prédictive des modèles
- **Rendement annualisé** — performance économique
- **Volatilité annualisée** — risque
- **Ratio de Sharpe** — rendement ajusté au risque
- **Max Drawdown** — perte maximale historique (peak-to-trough)
- **Turnover moyen** — coût de rééquilibrage (convention 0.5 × Σ|Δw|)

---

# 8. Structure du projet

```
project_portfolio/
├── project_portfolio.ipynb    # Notebook principal (code complet)
├── README.md                   # Ce fichier
```

Tout le code est dans un seul notebook auto-suffisant.

---

# 9. Installation

Ouvrez le notebook directement dans votre navigateur :

1. Allez sur [colab.research.google.com](https://colab.research.google.com)
2. **Fichier → Ouvrir un notebook → onglet GitHub**
3. Collez l'URL du dépôt et sélectionnez le notebook `project_portfolio.ipynb`
4. **Exécution → Tout exécuter**
