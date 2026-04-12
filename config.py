"""
config.py
=========
Central configuration for the ML Portfolio Allocation project.

Edit this file to change hyperparameters, dates, tickers or paths
without touching the source code.

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
RESULTS_DIR  = ROOT / "results"
PRICES_CACHE = DATA_DIR / "prices.csv"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA = dict(
    start            = "2015-01-01",    # Download start
    end              = "2026-01-01",    # Download end
    missing_threshold= 0.10,            # Drop tickers with >10% NaN
    ffill_limit      = 5,               # Max consecutive forward-fill days
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FEATURES = dict(
    return_periods   = [1, 5, 21, 63],  # Log-return horizons (days)
    vol_windows      = [10, 21, 63],    # Volatility estimation windows
    ma_windows       = [5, 20, 60],     # Moving-average ratio windows
    rsi_window       = 14,              # RSI period
    horizon          = 5,               # Forward-return target horizon (days)
)

# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------
SPLIT = dict(
    train_end        = "2022-12-31",
    val_end          = "2024-06-30",
    # test: everything after val_end
)

# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------
MODELS = dict(
    ridge = dict(alpha=0.1),
    lasso = dict(alpha=0.001),
    elasticnet = dict(alpha=0.01, l1_ratio=0.5),
    random_forest = dict(
        n_estimators     = 200,
        max_depth        = 6,
        max_features     = 0.5,
        min_samples_leaf = 30,
        random_state     = 42,
    ),
    gbm = dict(
        n_estimators   = 300,
        learning_rate  = 0.05,
        max_depth      = 3,
        subsample      = 0.8,
        random_state   = 42,
    ),
    cv_splits = 5,                      # TimeSeriesSplit folds
)

# ---------------------------------------------------------------------------
# Portfolio optimisation
# ---------------------------------------------------------------------------
PORTFOLIO = dict(
    rebalance_freq   = "ME",            # Month-end rebalancing
    lookback_days    = 252,             # Covariance estimation window
    transaction_cost = 0.001,           # 10 bps one-way cost
    cov_shrinkage    = 0.10,            # Ledoit-Wolf diagonal shrinkage
    ml_risk_aversion = 2.0,             # γ in ML mean-variance utility
    ls_quantile      = 0.20,            # Long/Short top/bottom quantile
    rf_annual        = 0.04,            # Risk-free rate (annualised)
)

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
BACKTEST = dict(
    start = "2018-01-01",
    end   = "2026-01-01",
)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
PLOT = dict(
    dpi     = 120,
    figsize = (14, 6),
    palette = [
        "#2196F3", "#F44336", "#4CAF50",
        "#FF9800", "#9C27B0", "#00BCD4", "#795548",
    ],
)
