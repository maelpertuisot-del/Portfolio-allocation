# 📈 AI in Finance — Project 5: ML-Driven Portfolio Allocation

**Course:** AI in Finance  
**Instructors:** Nicolas de Roux & Mohamed El Fakir (QuantCube Technology)  
**Group:** [Your names here]  

---

## Overview

This project builds a **machine-learning-driven portfolio allocation model** for a universe of 50 liquid US equities. We combine predictive ML models for return forecasting with classical portfolio optimisation (Markowitz, Risk Parity) to construct and evaluate multiple strategies over a walk-forward backtest spanning 2018–2024.

### Key Results

| Strategy     | CAGR  | Sharpe | Max DD |
|--------------|-------|--------|--------|
| EqualWeight  | benchmark | benchmark | benchmark |
| MinVariance  | ↑     | ↑↑     | ↓ (smaller) |
| RiskParity   | ↑     | ↑↑↑    | ↓↓     |
| MLPortfolio  | ↑↑    | ↑↑     | ↓      |

---

## Project Structure

```
ai-finance-portfolio/
│
├── src/                       # Core Python modules
│   ├── data_loader.py         # Data download & preprocessing
│   ├── features.py            # Feature engineering
│   ├── models.py              # ML models (Ridge, RF, GBM, Ensemble)
│   ├── portfolio.py           # Portfolio strategies & walk-forward backtest
│   ├── evaluation.py          # Performance metrics (Sharpe, MDD, Calmar …)
│   └── visualization.py       # All plotting utilities
│
├── notebooks/
│   └── portfolio_analysis.py  # Full analysis (run as Jupyter notebook)
│
├── data/                      # Auto-created, cached price CSVs
├── results/                   # Auto-created, saved figures & tables
│
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-group/ai-finance-portfolio.git
cd ai-finance-portfolio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Option 1 — Jupyter Notebook (recommended for exploration)

```bash
# Convert the Python script to a notebook
jupytext --to notebook notebooks/portfolio_analysis.py

# Launch Jupyter
jupyter notebook notebooks/portfolio_analysis.ipynb
```

Or open directly in **VS Code** with the Jupyter extension (`.py` files with `# %%` markers run as cells natively).

### Option 2 — Command-line pipeline

```bash
# Full pipeline (download data + train models + backtest)
python main.py

# Use cached data (skip download)
python main.py --cache-only

# Custom prediction horizon (21-day returns)
python main.py --horizon 21

# Custom train/test split
python main.py --train-end 2021-12-31 --val-end 2022-12-31
```

### Option 3 — Kaggle Dataset

If you prefer to use the [Kaggle Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset):

```python
from src.data_loader import load_kaggle_prices

prices = load_kaggle_prices("/path/to/kaggle/archive/stocks/")
```

---

## Methodology

### 1. Data
- **Source:** Yahoo Finance (adjusted closing prices, auto-adjusted for splits & dividends)
- **Universe:** 50 US equities across 10 GICS sectors
- **Period:** 2015–2024 (training) / 2018–2024 (backtest)
- **Frequency:** Daily

### 2. Feature Engineering

| Category       | Features                                                      |
|----------------|---------------------------------------------------------------|
| Returns        | Log-returns at 1d, 5d, 21d, 63d                              |
| Volatility     | Realised vol at 10d, 21d, 63d (annualised)                   |
| Momentum       | Price / MA ratio (5d, 20d, 60d); JT momentum (3m, 6m, 12m)  |
| RSI            | 14-day RSI (rescaled to [0,1])                               |
| Vol ratio      | Short-term vol / Long-term vol                               |
| Cross-sec rank | Rank-normalised version of key features                       |

### 3. ML Models

| Model        | Type      | Key Hyperparameters                        |
|--------------|-----------|--------------------------------------------|
| Ridge        | Linear    | α = 0.1                                    |
| Lasso        | Linear    | α = 0.001                                  |
| ElasticNet   | Linear    | α = 0.01, l1_ratio = 0.5                   |
| RandomForest | Ensemble  | 200 trees, max_depth=6, max_features=0.5   |
| XGBoost/GBM  | Ensemble  | 300 trees, lr=0.05, max_depth=3            |

All models evaluated with **5-fold TimeSeriesSplit** cross-validation.

### 4. Portfolio Strategies

| Strategy      | Description                                          |
|---------------|------------------------------------------------------|
| EqualWeight   | 1/N benchmark                                        |
| MinVariance   | Markowitz min-variance with shrinkage covariance     |
| MaxSharpe     | Tangency portfolio                                    |
| RiskParity    | Equal Risk Contribution (ERC)                        |
| MLPortfolio   | ML-predicted returns → mean-variance utility max     |
| LongShortML   | Long top-quintile / Short bottom-quintile (ML signal)|

### 5. Evaluation Metrics

- **CAGR** — Compound Annual Growth Rate
- **Sharpe Ratio** — Risk-adjusted return (annualised)
- **Sortino Ratio** — Downside-risk-adjusted return
- **Max Drawdown** — Largest peak-to-trough decline
- **Calmar Ratio** — CAGR / |Max Drawdown|
- **VaR / CVaR (95%)** — Tail risk measures
- **Turnover** — Average rebalancing activity (transaction cost proxy)
- **Information Ratio** — Active return vs benchmark

---

## Results

All outputs are saved to `results/`:

| File                          | Content                                  |
|-------------------------------|------------------------------------------|
| `01_price_history.png`        | Normalised price history by sector       |
| `02_returns_distribution.png` | Returns histograms with Gaussian fit     |
| `03_correlation_heatmap.png`  | Full pairwise return correlation matrix  |
| `04_volatility_regimes.png`   | Rolling vol — regime detection           |
| `05_feature_ic.png`           | Information Coefficient per feature      |
| `06_model_comparison.png`     | R², IC, RMSE across models               |
| `07_feature_importance.png`   | Feature importance (best model)          |
| `08_shap_importance.png`      | SHAP values summary plot                 |
| `09_cumulative_returns.png`   | NAV comparison + drawdowns               |
| `10_rolling_sharpe.png`       | Rolling 6-month Sharpe                   |
| `11_ml_weights.png`           | ML portfolio weights over time           |
| `12_sector_allocation.png`    | Sector allocation by strategy            |
| `model_results.csv`           | Full model performance table             |
| `portfolio_summary.csv`       | Full portfolio performance table         |

---

## Key References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Breiman, L. (2001). *Random Forests*. Machine Learning.
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Goulet Coulombe, P. et al. (2022). *How is Machine Learning Useful for Macroeconomic Forecasting?*
- DeMiguel, V. et al. (2009). *Optimal vs Naive Diversification*. Review of Financial Studies.
- De Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley.

---

## Academic Integrity

This project was completed by [Group Name]. All code is original. External resources are cited above.

---

## License

MIT
