# Projet Final IA in Finance - Master 2 Finance - Université Paris-Panthéon-Assas
---

# 1. Informations sur le Projet

- **Titre du projet :** Allocation de Portefeuille par Machine Learning
- **Nom du groupe :** Groupe 1
- **Membres du groupe :**  
  - Étudiant 1 – Maël Pertuisot  
  - Étudiant 2 – Valentin Martel  
  - Étudiant 3 – Mathias Garnier

- **Nom du cours :** AI In Finance
- **Instructeurs :** Nicolas De Roux & Mohamed EL FAKIR
- **Date de rendu :** 20/04/2026

---

# 2. Description du Projet

L'allocation traditionnelle de portefeuille repose sur des moyennes historiques et l'optimisation classique de Markowitz (1952), qui souffrent d'erreurs d'estimation et ignorent les relations non-linéaires dans les données financières. Ce projet répond au défi de construire un **modèle d'allocation de portefeuille piloté par le machine learning** sur un univers de 50 actions liquides cotées au NASDAQ et au S&P 500.

Le problème est important car même de petites améliorations dans la prédiction des rendements ou l'estimation du risque se traduisent directement par une meilleure performance ajustée au risque pour les investisseurs. Les gérants d'actifs, les hedge funds et les investisseurs particuliers peuvent tous bénéficier de stratégies d'allocation plus orientées données.

Nous combinons des **modèles ML prédictifs** (Ridge, Lasso, Random Forest, Gradient Boosting, LSTM) avec une **optimisation de portefeuille classique** (moyenne-variance de Markowitz, parité de risque) dans un backtest walk-forward couvrant 2018–2024, en comparant les stratégies sur le ratio de Sharpe, le drawdown maximum et le turnover.

---

# 3. Objectif du Projet

Le projet vise à :

1. **Prédire** les rendements à court terme des actions (log-rendements à 5 jours) à partir de features construites sur les prix historiques
2. **Construire** des portefeuilles optimisés en injectant les prédictions ML dans un optimiseur moyenne-variance
3. **Comparer** 6 stratégies de portefeuille : Équipondéré (benchmark), Variance Minimale, Sharpe Maximum, Parité de Risque, Portefeuille ML, et Long-Short ML

Une solution réussie surpasse le benchmark équipondéré sur une base **ajustée au risque** (ratio de Sharpe) dans un backtest walk-forward hors-échantillon, avec un turnover raisonnable et un drawdown contrôlé.

---

# 4. Définition de la Tâche

- **Type de tâche :** Régression (prédiction de rendements) + Optimisation de Portefeuille

- **Variables d'entrée (features) :**
  - Log-rendements à 1j, 5j, 21j, 63j
  - Volatilité réalisée à 10j, 21j, 63j (annualisée)
  - Ratios prix / moyenne mobile (5j, 20j, 60j)
  - Signaux de momentum Jegadeesh-Titman (3m, 6m, 12m)
  - RSI 14 jours (recentré dans [0, 1])
  - Ratio volatilité court terme / long terme
  - Versions normalisées par rang en coupe transversale

- **Variable cible :** Log-rendement à 5 jours : `log(P_{t+5} / P_t)`

- **Métriques d'évaluation :**
  - *Modèles ML :* R² hors-échantillon, RMSE, IC de Spearman (Information Coefficient)
  - *Portefeuilles :* Ratio de Sharpe, Ratio de Sortino, Drawdown Maximum, TCAC, Ratio de Calmar, Turnover, VaR/CVaR 95%

---

# 5. Description du Jeu de Données

## Vue d'ensemble

- **Nombre d'observations :** ~2 250 jours de bourse × 50 actions = ~112 500 observations action-jour
- **Nombre de features :** 21 features construites par observation action-jour
- **Variable cible :** Log-rendement à 5 jours
- **Source des données :** Yahoo Finance via `yfinance` (prix de clôture ajustés des splits et dividendes). Alternative : [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Période couverte :** 01/01/2015 au 01/01/2024

---

## Description des Features

| Feature | Description | Type |
|---|---|---|
| ret_1d | Log-rendement à 1 jour : log(P_t / P_{t-1}) | Numérique |
| ret_5d | Log-rendement à 5 jours | Numérique |
| ret_21d | Log-rendement à 21 jours (~1 mois) | Numérique |
| ret_63d | Log-rendement à 63 jours (~3 mois) | Numérique |
| vol_10d | Volatilité réalisée sur 10 jours (annualisée) | Numérique |
| vol_21d | Volatilité réalisée sur 21 jours (annualisée) | Numérique |
| vol_63d | Volatilité réalisée sur 63 jours (annualisée) | Numérique |
| ma_ratio_5d | Prix / moyenne mobile 5 jours − 1 | Numérique |
| ma_ratio_20d | Prix / moyenne mobile 20 jours − 1 | Numérique |
| ma_ratio_60d | Prix / moyenne mobile 60 jours − 1 | Numérique |
| mom_3m | Momentum 3 mois de Jegadeesh-Titman (skip 1m) | Numérique |
| mom_6m | Momentum 6 mois (skip 1m) | Numérique |
| mom_12m | Momentum 12 mois (skip 1m) | Numérique |
| rsi_14 | RSI 14 jours normalisé dans [0, 1] | Numérique |
| vol_ratio | Volatilité court terme (10j) / long terme (63j) | Numérique |
| rank_ret_1d | Rang centile en coupe transversale de ret_1d | Numérique [0,1] |
| rank_ret_5d | Rang centile en coupe transversale de ret_5d | Numérique [0,1] |
| rank_ret_21d | Rang centile en coupe transversale de ret_21d | Numérique [0,1] |
| rank_vol_21d | Rang centile en coupe transversale de vol_21d | Numérique [0,1] |
| rank_mom_3m | Rang centile en coupe transversale de mom_3m | Numérique [0,1] |
| rank_mom_6m | Rang centile en coupe transversale de mom_6m | Numérique [0,1] |

---

## Variable Cible

- **Nom de la variable :** `fwd_ret_5d`
- **Signification :** Log-rendement à 5 jours pour chaque action à chaque date : `log(P_{t+5} / P_t)`
- **Plage de valeurs :** Continue, approximativement dans [−0,15 ; +0,15] pour des actions liquides
- **Important :** Cette variable utilise des prix futurs et est **uniquement utilisée comme étiquette y**, jamais comme feature, afin d'éviter tout biais de lookahead

---

## Types de Données

Toutes les variables sont des **séries temporelles numériques** :
- Prix bruts : continus, positifs
- Log-rendements : continus, approximativement normaux avec queues épaisses
- Volatilité : continue, positive
- Ratios de moyenne mobile : continus, centrés autour de 0
- RSI et rangs : bornés dans [0, 1]

Aucune variable catégorielle, textuelle ou ordinale n'est utilisée dans le modèle de base. Les labels sectoriels (10 secteurs GICS) sont utilisés uniquement pour la visualisation.

---

## Distribution des Données

- **Rendements :** Approximativement normaux avec **excès de kurtosis** (queues épaisses) — fait stylisé bien connu en finance. Le rendement annualisé moyen varie de ~5% (utilities) à ~35% (technologie/NVDA)
- **Volatilité :** Distribution asymétrique à droite, de ~10% (utilities peu volatiles) à ~60% (tech très volatile) en annualisé
- **Momentum :** Moyenne proche de zéro, distribution symétrique
- **RSI :** Approximativement uniforme sur [0, 1] en période normale, asymétrique en période de tendance
- **Équilibre des classes :** Non applicable — tâche de régression. Les rendements futurs sont approximativement centrés en coupe transversale

---

## Qualité des Données

- **Valeurs manquantes :** 2 à 8% pour certaines actions en début de période (introductions en bourse après 2015). Les actions avec plus de 10% de valeurs manquantes sont supprimées. Les écarts restants sont comblés par forward-fill sur 5 jours consécutifs maximum.
- **Valeurs aberrantes :** Des rendements journaliers extrêmes (>10%) apparaissent lors des crises (COVID mars 2020, mars 2022). Ils sont conservés car ils représentent des événements de marché réels.
- **Non-stationnarité :** Les prix sont non-stationnaires (marche aléatoire). Toutes les features sont construites à partir de rendements et ratios, qui sont stationnaires.
- **Hétéroscédasticité :** La volatilité des rendements varie dans le temps (effets ARCH). Les features de volatilité capturent explicitement ce phénomène.
- **Doublons :** Aucun doublon détecté pour l'ensemble des actions.

---

# 6. Prétraitement des Données

| Étape | Méthode | Justification |
|---|---|---|
| Téléchargement et ajustement | `yfinance` auto_adjust=True | Supprime les distorsions liées aux splits et dividendes |
| Filtre valeurs manquantes | Suppression des actions avec >10% de NaN | Garantit un historique suffisant pour toutes les features |
| Forward-fill | Maximum 5 jours consécutifs | Gère les jours non ouvrés (jours fériés, suspensions) |
| Log-rendements | `log(P_t / P_{t-1})` | Stationnarité, normalité, additivité temporelle |
| Normalisation des features | StandardScaler à l'intérieur de chaque fold CV | Requis pour Ridge/Lasso ; évite la fuite de données entre folds |
| Rang en coupe transversale | `rank(pct=True)` par date | Supprime le facteur de marché commun, concentre le signal relatif |
| Découpage train/test | Basé sur le calendrier (sans mélange aléatoire) | Évite la fuite de données — le futur ne peut pas informer le passé |
| Absence de lookahead | Toutes les features calculées en t avec des données jusqu'à t | Garantit un backtest réaliste |

---

# 7. Approche de Modélisation

## Modèles Utilisés

| Modèle | Type | Idée Principale |
|---|---|---|
| **Ridge** | Linéaire (L2) | Rétrécit tous les coefficients — baseline robuste, gère la multicolinéarité |
| **Lasso** | Linéaire (L1) | Effectue une sélection de variables (coefficients parcimonieux) |
| **ElasticNet** | Linéaire (L1+L2) | Combine Ridge et Lasso — optimal pour des features corrélées |
| **Random Forest** | Ensemble (Bagging) | Combine 200 arbres décorrélés — capture la non-linéarité, réduit la variance |
| **XGBoost/GBM** | Ensemble (Boosting) | Arbres séquentiels corrigeant les résidus — réduit le biais |
| **Ensemble** | Moyenne RF + GBM + EN | Combine les forces de tous les modèles |
| **LSTM** | Deep Learning (RNN) | Capture les dépendances temporelles dans les séquences de features |

---

## Stratégie de Modélisation

**Baseline :** Portefeuille équipondéré (1/N) et régression Ridge — simple, compétitif, difficile à battre.

**Justification du choix des modèles :**
- Les modèles linéaires (Ridge, Lasso) sont rapides, interprétables et fournissent une baseline régularisée. Ils correspondent à l'équivalent ARIMA dans l'univers du ML.
- Random Forest et GBM capturent les **interactions non-linéaires** entre features — l'avantage clé du ML sur l'économétrie classique pour la prévision (Goulet Coulombe et al., 2022).
- Le LSTM exploite la **structure temporelle** des séquences de features (Cours 6 — Deep Learning).

**Réglage des hyperparamètres :**
- Tous les modèles sont évalués avec une **validation croisée TimeSeriesSplit à 5 folds** (pas de K-fold aléatoire qui causerait une fuite de données sur les séries temporelles — cf. slides Cours 5).
- La force de régularisation (α pour les modèles linéaires, max_depth pour les arbres) est sélectionnée pour minimiser le RMSE de validation.

**Protocole de validation croisée :**
```
Entraînement [2015–2020] → Validation [2021–2022] → Test [2022–2024]
TimeSeriesSplit : le fold 1 s'entraîne sur les mois 1–10, valide sur 11–12, etc.
```

---

## Métriques d'Évaluation

**Modèles ML :**
- **R² hors-échantillon :** Part de variance des rendements expliquée. Des valeurs faibles mais positives (~1–5%) sont typiques et significatives en finance.
- **RMSE :** Magnitude de l'erreur dans les mêmes unités que les rendements.
- **IC de Spearman (Information Coefficient) :** Corrélation de rang entre prédictions et rendements réels. Plus robuste aux valeurs extrêmes que Pearson. Un IC > 0,03 est considéré utile en pratique.

**Stratégies de portefeuille :**
- **Ratio de Sharpe :** `(μ_p − r_f) / σ_p × √252` — la métrique de performance ajustée au risque principale
- **Drawdown Maximum :** Plus grande baisse entre un pic et un creux — mesure du risque extrême
- **Ratio de Calmar :** TCAC / |Drawdown Maximum| — rendement par unité de perte maximale
- **Ratio de Sortino :** Ne pénalise que la volatilité à la baisse
- **Turnover :** Proxy du coût de rééquilibrage — important pour la faisabilité pratique
- **VaR / CVaR 95% :** Value-at-Risk journalière et Expected Shortfall

Ces métriques sont appropriées car le R² brut de prédiction est souvent proche de zéro en finance (faible rapport signal/bruit), mais même de petits avantages prédictifs se traduisent par une performance de portefeuille réelle lorsqu'ils sont correctement agrégés via l'optimisation.

---

# 8. Structure du Projet

```
├── data/
│   └── prices.csv                  # Données de prix en cache (généré automatiquement)
├── docs/
│   └── presentation.pptx           # Slides de la présentation finale
├── notebooks/
│   └── portfolio_analysis.ipynb    # Notebook d'analyse complet (à exécuter ici)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Téléchargement et prétraitement des prix
│   ├── features.py                 # Ingénierie des features (rendements, vol, momentum, RSI)
│   ├── models.py                   # Modèles ML (Ridge, RF, GBM, Ensemble)
│   ├── portfolio.py                # Stratégies de portefeuille et backtest walk-forward
│   ├── evaluation.py               # Métriques de performance (Sharpe, MDD, Calmar…)
│   ├── visualization.py            # Utilitaires de visualisation
│   └── lstm_model.py               # Prédicteur LSTM (PyTorch)
├── tests/
│   └── test_pipeline.py            # 26 tests unitaires et d'intégration (pytest)
├── config.py                       # Configuration centrale (dates, hyperparamètres)
├── main.py                         # Point d'entrée en ligne de commande
├── requirements.txt
└── README.md
```

**Dossiers clés :**
- `src/` — Tous les modules Python réutilisables, importables depuis le notebook
- `notebooks/` — Le notebook d'analyse principal avec tous les résultats et visualisations
- `tests/` — Tests automatisés pour vérifier la correction de chaque module
- `config.py` — Modifier ici tous les paramètres (dates, actions, hyperparamètres)

---

# 9. Installation

### Option 1 — Google Colab (recommandé, aucune installation requise)

Ouvrez le notebook directement dans votre navigateur :

1. Allez sur [colab.research.google.com](https://colab.research.google.com)
2. **Fichier → Ouvrir un notebook → onglet GitHub**
3. Collez l'URL du dépôt et sélectionnez `notebooks/portfolio_analysis.ipynb`
4. Ajoutez cette cellule en haut et exécutez-la :

```python
import os, sys
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

REPO = "Portfolio-allocation"
if not os.path.exists(REPO):
    !git clone https://github.com/maelpertuisot-del/{REPO}.git
os.chdir(REPO)
sys.path.insert(0, ".")
!pip install -r requirements.txt -q
print("✅ Prêt !")
```

5. **Exécution → Tout exécuter**

### Option 2 — Installation locale

```bash
# Cloner le dépôt
git clone https://github.com/maelpertuisot-del/Portfolio-analysis.git
cd ai-finance-portfolio

# Installer les dépendances
pip install -r requirements.txt

# Lancer le pipeline complet
python main.py

# Ou lancer le notebook
jupyter notebook notebooks/portfolio_analysis.ipynb

# Lancer les tests
pytest tests/ -v
```
