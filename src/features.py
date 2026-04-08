"""
features.py
===========
Feature engineering for the ML portfolio allocation project.

Feature families
----------------
- Returns          : log-returns at 1d / 5d / 21d / 63d
- Volatility       : rolling realised vol at multiple windows
- Momentum         : price-over-MA ratio, Jegadeesh-Titman signals
- RSI              : Relative Strength Index (14-day)
- Cross-sectional  : rank-normalised versions of the above
- Target           : forward log-returns at 5d / 21d horizons

Key design choices
------------------
- All features are lag-safe: only information up to time t is used
  to predict at time t (no lookahead / data leakage).
- Features are computed at the stock level (wide format: date × ticker).
- `build_ml_dataset` converts to a panel (stock-date × feature) ready
  for scikit-learn estimators.

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def log_returns(prices: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """
    Compute log-returns over `period` trading days.

    r_t = log(P_t / P_{t-period})

    Log-returns are preferred over simple returns because:
    - They are additive over time
    - They are closer to normally distributed
    - They handle compounding naturally
    """
    return np.log(prices / prices.shift(period))


def realised_vol(
    prices: pd.DataFrame,
    window: int = 21,
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Rolling realised volatility = std of daily log-returns over `window` days.
    Annualised by default (×√252).
    """
    daily_ret = log_returns(prices, 1)
    rv = daily_ret.rolling(window, min_periods=window // 2).std()
    return rv * np.sqrt(252) if annualise else rv


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI) in [0, 100].

    RSI > 70  → overbought signal
    RSI < 30  → oversold signal

    Uses Wilder's EMA smoothing.
    """
    delta = prices.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ma_ratio(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Price-over-MA ratio:  P_t / MA_t(window) − 1.

    Positive values indicate price above its moving average (bullish momentum).
    """
    return prices / prices.rolling(window, min_periods=window // 2).mean() - 1


def cross_section_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional (per-date) rank-normalisation to [0, 1].

    Removes common factors and focuses on relative performance,
    which is more stable for portfolio signals.
    """
    return df.rank(axis=1, pct=True)


# ---------------------------------------------------------------------------
# Feature matrix builders
# ---------------------------------------------------------------------------

def build_feature_dict(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute all features as a dictionary {feature_name: wide DataFrame}.

    Each DataFrame has shape (T × N) — same index/columns as `prices`.

    Parameters
    ----------
    prices : pd.DataFrame  (date × ticker), adjusted close prices

    Returns
    -------
    feats : dict of pd.DataFrame
    """
    feats: Dict[str, pd.DataFrame] = {}

    # ── Returns ────────────────────────────────────────────────────────────── #
    for p in [1, 5, 21, 63]:
        feats[f"ret_{p}d"] = log_returns(prices, p)

    # ── Realised volatility ────────────────────────────────────────────────── #
    for w in [10, 21, 63]:
        feats[f"vol_{w}d"] = realised_vol(prices, w)

    # ── Price-over-MA ratio (momentum proxy) ──────────────────────────────── #
    for w in [5, 20, 60]:
        feats[f"ma_ratio_{w}d"] = ma_ratio(prices, w)

    # ── Jegadeesh-Titman momentum (skip-1-month, look back 3 / 6 / 12m) ──── #
    feats["mom_3m"]  = log_returns(prices, 63)  - log_returns(prices, 21)
    feats["mom_6m"]  = log_returns(prices, 126) - log_returns(prices, 21)
    feats["mom_12m"] = log_returns(prices, 252) - log_returns(prices, 21)

    # ── RSI ────────────────────────────────────────────────────────────────── #
    feats["rsi_14"] = rsi(prices, 14) / 100.0   # rescale to [0, 1]

    # ── Volatility ratio (short-term vol / long-term vol) ─────────────────── #
    feats["vol_ratio"] = feats["vol_10d"] / feats["vol_63d"]

    # ── Cross-sectional ranks (remove market-wide effects) ────────────────── #
    for key in ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "mom_3m", "mom_6m"]:
        feats[f"rank_{key}"] = cross_section_rank(feats[key])

    return feats


def build_target(
    prices: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Forward log-return over `horizon` trading days — the prediction target.

    ⚠ This uses future prices (shift(-horizon)).  It must ONLY be used as the
    label y, never as a feature x, to avoid lookahead bias.

    Parameters
    ----------
    horizon : int  prediction horizon in trading days

    Returns
    -------
    fwd_ret : pd.DataFrame  shape (T × N)
    """
    return np.log(prices.shift(-horizon) / prices)


# ---------------------------------------------------------------------------
# Panel dataset (for scikit-learn)
# ---------------------------------------------------------------------------

def build_ml_dataset(
    prices: pd.DataFrame,
    horizon: int = 5,
    feature_names: Optional[List[str]] = None,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a flat panel dataset suitable for scikit-learn estimators.

    Layout
    ------
    - Index : MultiIndex (date, ticker)
    - X     : feature columns
    - y     : forward log-return (target)

    Parameters
    ----------
    prices        : pd.DataFrame (T × N)
    horizon       : forward return horizon in days
    feature_names : subset of features to include (default = all)
    dropna        : drop rows with any NaN

    Returns
    -------
    X : pd.DataFrame  (T*N × F)
    y : pd.Series     (T*N,)
    """
    feats  = build_feature_dict(prices)
    target = build_target(prices, horizon).rename(columns=lambda c: c)

    if feature_names is not None:
        feats = {k: v for k, v in feats.items() if k in feature_names}

    # Stack: wide (T × N) → long (T*N × 1) for each feature
    stacked: Dict[str, pd.Series] = {}
    for name, df in feats.items():
        stacked[name] = df.stack()

    X = pd.DataFrame(stacked)
    X.index.names = ["date", "ticker"]

    y = target.stack()
    y.index.names = ["date", "ticker"]
    y.name = f"fwd_ret_{horizon}d"

    # Align
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    if dropna:
        valid = X.notna().all(axis=1) & y.notna()
        X, y  = X[valid], y[valid]

    return X, y


def get_feature_names() -> List[str]:
    """Return the full list of feature names produced by build_feature_dict."""
    dummy = pd.DataFrame(
        np.random.randn(300, 3),
        index=pd.date_range("2020-01-01", periods=300, freq="B"),
        columns=["A", "B", "C"],
    )
    dummy = np.exp(dummy.cumsum())
    return list(build_feature_dict(dummy).keys())
