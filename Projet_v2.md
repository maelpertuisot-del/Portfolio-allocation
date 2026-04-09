# Projet Final IA in Finance - Master 2 Finance - Université Paris-Panthéon-Assas
---

# 1. Project Information

- **Project Title:** ML-Driven Portfolio Allocation
- **Group Name:** Groupe 1
- **Group Members:**  
  - Student 1 – Maël Pertuisot  
  - Student 2 – Valentin Martel  
  - Student 3 – Mathias Garnier

- **Course Name:** AI In Finance
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** 20/04/2026

---

# 2. Project Description

Traditional portfolio allocation relies on historical averages and classical optimisation (Markowitz, 1952), which suffer from estimation errors and ignore nonlinear patterns in financial data. This project addresses the challenge of building a **machine-learning-driven portfolio allocation model** for a universe of 50 liquid US equities listed on the NASDAQ and S&P 500.

The problem is important because even small improvements in return prediction or risk estimation translate directly into higher risk-adjusted performance for investors. Asset managers, hedge funds, and retail investors can all benefit from more data-driven allocation strategies.

We combine **predictive ML models** (Ridge, Lasso, Random Forest, Gradient Boosting, LSTM) with **classical portfolio optimisation** (Markowitz mean-variance, Risk Parity) in a walk-forward backtest spanning 2018–2024, and compare strategies on Sharpe ratio, maximum drawdown, and turnover.

---

# 3. Project Goal

The project aims to:

1. **Predict** short-term stock returns (5-day forward log-returns) using engineered features from historical prices
2. **Construct** optimised portfolios by feeding ML predictions into a mean-variance optimiser
3. **Compare** 6 portfolio strategies: Equal Weight (benchmark), Minimum Variance, Maximum Sharpe, Risk Parity, ML Portfolio, and Long-Short ML

A successful solution outperforms the equal-weight benchmark on a **risk-adjusted basis** (Sharpe ratio) in an out-of-sample walk-forward backtest, with reasonable turnover and controlled drawdown.

---

# 4. Task Definition

- **Task Type:** Regression (return prediction) + Portfolio Optimisation

- **Input Variables:**
  - Log-returns at 1d, 5d, 21d, 63d horizons
  - Realised volatility at 10d, 21d, 63d windows (annualised)
  - Price-over-moving-average ratios (5d, 20d, 60d)
  - Jegadeesh-Titman momentum signals (3m, 6m, 12m)
  - 14-day Relative Strength Index (RSI)
  - Short/long volatility ratio
  - Cross-sectional rank-normalised versions of the above

- **Target Variable:** 5-day forward log-return: `log(P_{t+5} / P_t)`

- **Evaluation Metric(s):**
  - *ML models:* R² (out-of-sample), RMSE, Spearman IC (Information Coefficient)
  - *Portfolios:* Sharpe ratio, Sortino ratio, Max Drawdown, CAGR, Calmar ratio, Turnover, VaR/CVaR 95%

---

# 5. Dataset Description

## Dataset Overview

- **Number of samples:** ~2,250 trading days × 50 stocks = ~112,500 stock-day observations
- **Number of features:** 21 engineered features per stock-day
- **Target variable:** 5-day forward log-return
- **Data source:** Yahoo Finance via `yfinance` (adjusted close prices, auto-adjusted for splits and dividends). Alternative: [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Period:** 2015-01-01 to 2024-01-01

---

## Feature Description

| Feature | Description | Type |
|---|---|---|
| ret_1d | 1-day log-return: log(P_t / P_{t-1}) | Numerical |
| ret_5d | 5-day log-return | Numerical |
| ret_21d | 21-day log-return (~1 month) | Numerical |
| ret_63d | 63-day log-return (~3 months) | Numerical |
| vol_10d | 10-day realised volatility (annualised) | Numerical |
| vol_21d | 21-day realised volatility (annualised) | Numerical |
| vol_63d | 63-day realised volatility (annualised) | Numerical |
| ma_ratio_5d | Price / 5-day moving average − 1 | Numerical |
| ma_ratio_20d | Price / 20-day moving average − 1 | Numerical |
| ma_ratio_60d | Price / 60-day moving average − 1 | Numerical |
| mom_3m | Jegadeesh-Titman 3-month momentum (skip 1m) | Numerical |
| mom_6m | 6-month momentum (skip 1m) | Numerical |
| mom_12m | 12-month momentum (skip 1m) | Numerical |
| rsi_14 | 14-day RSI rescaled to [0, 1] | Numerical |
| vol_ratio | Short-term vol (10d) / Long-term vol (63d) | Numerical |
| rank_ret_1d | Cross-sectional percentile rank of ret_1d | Numerical [0,1] |
| rank_ret_5d | Cross-sectional percentile rank of ret_5d | Numerical [0,1] |
| rank_ret_21d | Cross-sectional percentile rank of ret_21d | Numerical [0,1] |
| rank_vol_21d | Cross-sectional percentile rank of vol_21d | Numerical [0,1] |
| rank_mom_3m | Cross-sectional percentile rank of mom_3m | Numerical [0,1] |
| rank_mom_6m | Cross-sectional percentile rank of mom_6m | Numerical [0,1] |

---

## Target Variable

- **Variable name:** `fwd_ret_5d`
- **Meaning:** 5-day forward log-return for each stock at each date: `log(P_{t+5} / P_t)`
- **Range:** Continuous, approximately in [−0.15, +0.15] for daily stocks
- **Note:** This variable uses future prices and is **only used as the label y**, never as a feature, to avoid lookahead bias

---

## Data Types

All variables are **numerical time-series**:
- Raw prices: continuous, positive
- Log-returns: continuous, approximately normal with fat tails
- Volatility: continuous, positive
- Moving average ratios: continuous, centred around 0
- RSI and ranks: bounded in [0, 1] or [0, 100]

No categorical, text, or ordinal variables are used in the base model. Sector labels (10 GICS sectors) are used only for visualisation.

---

## Data Distribution

- **Returns:** Approximately normal with **excess kurtosis** (fat tails) — a well-known stylised fact in finance. Average annualised return varies from ~5% (utilities) to ~35% (technology/NVDA)
- **Volatility:** Right-skewed, ranges from ~10% (low-vol utilities) to ~60% (high-vol tech) annualised
- **Momentum:** Near-zero mean, symmetric distribution
- **RSI:** Roughly uniform over [0, 1] in normal periods, skewed during trends
- **Class balance:** N/A — regression task. Forward returns are approximately zero-mean across the cross-section

---

## Data Quality

- **Missing values:** ~2–8% for some tickers at the start of the sample (IPOs after 2015). Tickers with >10% missing values are dropped. Remaining gaps are forward-filled up to 5 consecutive days.
- **Outliers:** Extreme daily returns (>10%) occur during crisis periods (COVID March 2020, March 2022). These are retained as they represent genuine market events.
- **Non-stationarity:** Prices are non-stationary (random walk). All features are derived from returns and ratios, which are stationary.
- **Heteroskedasticity:** Return volatility is time-varying (ARCH effects). Volatility features explicitly capture this.
- **No duplicates:** Verified for all tickers.

---

# 6. Data Preprocessing

| Step | Method | Why |
|---|---|---|
| Download & adjust | `yfinance` auto_adjust=True | Removes distortions from splits and dividends |
| Missing value filter | Drop tickers with >10% NaN | Ensures sufficient history for all features |
| Forward-fill | Up to 5 consecutive days | Handles non-trading days (holidays, suspensions) |
| Log-returns | `log(P_t / P_{t-1})` | Stationarity, normality, time-additivity |
| Feature scaling | StandardScaler inside each CV fold | Required for Ridge/Lasso; prevents data leakage across folds |
| Cross-sectional ranking | `rank(pct=True)` per date | Removes common market factor, focuses on relative signal |
| Train/test split | Calendar-based (no shuffling) | Prevents data leakage — future cannot inform past |
| No lookahead | All features computed at time t using data up to t | Ensures realistic backtest |

---

# 7. Modeling Approach

## Chosen Models

| Model | Type | Key Idea |
|---|---|---|
| **Ridge** | Linear (L2) | Shrinks all coefficients — robust baseline, handles multicollinearity |
| **Lasso** | Linear (L1) | Performs variable selection (sparse coefficients) |
| **ElasticNet** | Linear (L1+L2) | Combines Ridge and Lasso — best for correlated features |
| **Random Forest** | Ensemble (Bagging) | Combines 200 decorrelated trees — captures nonlinearity, reduces variance |
| **XGBoost/GBM** | Ensemble (Boosting) | Sequential trees correcting residuals — reduces bias |
| **Ensemble** | Average of RF + GBM + EN | Combines strengths of all models |
| **LSTM** | Deep Learning (RNN) | Captures temporal dependencies in sequential features |

---

## Modeling Strategy

**Baseline:** Equal-weight portfolio (1/N) and Ridge regression — simple, competitive, hard to beat.

**Selection rationale:**
- Linear models (Ridge, Lasso) are fast, interpretable, and provide a regularised baseline. They correspond to the ARIMA-equivalent in the ML world.
- Random Forest and GBM capture **nonlinear interactions** between features — the key advantage of ML over classical econometrics for forecasting (Goulet Coulombe et al., 2022).
- LSTM exploits **temporal structure** in the feature sequences (Cours 6 — Deep Learning).

**Hyperparameter tuning:**
- All models are evaluated with **5-fold TimeSeriesSplit** cross-validation (no random K-fold which would cause data leakage on time series — cf. Cours 5 slides).
- Regularisation strength (α for linear models, max_depth for trees) is selected to minimise validation RMSE.

**Cross-validation protocol:**
```
Train [2015–2020] → Val [2021–2022] → Test [2022–2024]
TimeSeriesSplit: fold 1 trains on months 1–10, validates on 11–12, etc.
```

---

## Evaluation Metrics

**ML models:**
- **R² (out-of-sample):** Fraction of return variance explained. Low but positive values (~1–5%) are typical and meaningful in finance.
- **RMSE:** Error magnitude in the same units as returns.
- **Spearman IC (Information Coefficient):** Rank correlation between predictions and actual returns. More robust to outliers than Pearson. IC > 0.03 is considered useful in practice.

**Portfolio strategies:**
- **Sharpe Ratio:** `(μ_p − r_f) / σ_p × √252` — the primary risk-adjusted performance metric
- **Max Drawdown:** Largest peak-to-trough decline — tail risk measure
- **Calmar Ratio:** CAGR / |Max Drawdown| — return per unit of worst loss
- **Sortino Ratio:** Only penalises downside volatility
- **Turnover:** Average rebalancing cost proxy — important for practical feasibility
- **VaR / CVaR 95%:** Daily Value-at-Risk and Expected Shortfall

These metrics are appropriate because raw return prediction R² is often near zero in finance (low signal-to-noise), yet even small predictive edges translate into real portfolio performance when properly aggregated via optimisation.

---

# 8. Project Structure

```
├── data/
│   └── prices.csv              # Cached price data (auto-generated)
├── docs/
│   └── presentation.pptx       # Final presentation slides
├── notebooks/
│   └── portfolio_analysis.ipynb  # Full analysis notebook (run this!)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Download & preprocess price data
│   ├── features.py             # Feature engineering (returns, vol, momentum, RSI)
│   ├── models.py               # ML models (Ridge, RF, GBM, Ensemble)
│   ├── portfolio.py            # Portfolio strategies & walk-forward backtest
│   ├── evaluation.py           # Performance metrics (Sharpe, MDD, Calmar…)
│   ├── visualization.py        # All plotting utilities
│   └── lstm_model.py           # LSTM return predictor (PyTorch)
├── tests/
│   └── test_pipeline.py        # 26 unit & integration tests (pytest)
├── config.py                   # Central configuration (dates, hyperparameters)
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

**Key folders:**
- `src/` — All reusable Python modules, importable from the notebook
- `notebooks/` — The main analysis notebook with all results and visualisations
- `tests/` — Automated tests to verify correctness of each module
- `config.py` — Change any parameter (dates, tickers, hyperparameters) here

---

# 9. Installation

### Option 1 — Google Colab (recommended, no installation needed)

Open the notebook directly in your browser:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub tab**
3. Paste the repository URL and select `notebooks/portfolio_analysis.ipynb`
4. Add this cell at the top and run it:

```python
!git clone https://github.com/maelpertuisot-del/Portfolio-allocation.git
%cd Portfolio-allocation
!pip install -r requirements.txt
import sys; sys.path.insert(0, '.')
```

5. **Runtime → Run all**

### Option 2 — Local installation

```bash
# Clone the repository
git clone https://github.com/maelpertuisot-del/Portfolio-allocation.git
cd Portfolio-allocation

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Or launch the notebook
jupyter notebook notebooks/portfolio_analysis.ipynb

# Run tests
pytest tests/ -v
```
