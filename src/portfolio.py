"""
portfolio.py
============
Portfolio construction and optimisation.

Strategies implemented
----------------------
1. EqualWeight       — 1/N benchmark (naive diversification)
2. MinVariance       — Markowitz minimum-variance portfolio
3. MaxSharpe         — Markowitz mean-variance (tangency portfolio)
4. RiskParity        — Equal risk contribution (risk-parity)
5. MLPortfolio       — ML-predicted returns fed into mean-variance optimiser
6. LongShortML       — Long winners / short losers based on ML signal

All strategies share a common interface:
    weights = strategy.fit(returns_train).predict(returns_test)

Portfolio rebalancing
---------------------
Portfolios are rebalanced periodically (default: monthly).
Transaction costs are applied at each rebalance.

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

# Annualisation constants
TRADING_DAYS = 252
RF_ANNUAL    = 0.04   # risk-free rate (approx. current level)
RF_DAILY     = RF_ANNUAL / TRADING_DAYS


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PortfolioStrategy(ABC):
    """Abstract base for all portfolio strategies."""

    def __init__(self, name: str, long_only: bool = True):
        self.name      = name
        self.long_only = long_only
        self._weights: Optional[pd.Series] = None

    @abstractmethod
    def compute_weights(
        self,
        returns: pd.DataFrame,
        predicted_returns: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute portfolio weights given historical returns.

        Parameters
        ----------
        returns           : pd.DataFrame (T × N) of past returns
        predicted_returns : optional pd.Series (N,) of ML-predicted expected returns

        Returns
        -------
        weights : pd.Series (N,)  — sum = 1 for long-only, sum ≈ 0 for L/S
        """

    def fit(self, returns: pd.DataFrame, **kwargs) -> "PortfolioStrategy":
        self._weights = self.compute_weights(returns, **kwargs)
        return self

    @property
    def weights(self) -> pd.Series:
        if self._weights is None:
            raise RuntimeError("Call fit() before accessing weights.")
        return self._weights


# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def _cov_matrix(returns: pd.DataFrame, shrinkage: float = 0.1) -> np.ndarray:
    """
    Sample covariance matrix with Ledoit-Wolf-style diagonal shrinkage.

    Shrinkage improves the condition number of the covariance matrix
    and leads to more stable portfolio weights out-of-sample.

    Σ_shrunk = (1 - δ) Σ + δ · diag(Σ)
    """
    cov  = returns.cov().values
    diag = np.diag(np.diag(cov))
    return (1 - shrinkage) * cov + shrinkage * diag


def _portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    var = w @ cov @ w
    return float(np.sqrt(max(var, 1e-12)))


def _portfolio_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    ret = _portfolio_return(w, mu)
    vol = _portfolio_vol(w, cov)
    return (ret - RF_DAILY) / vol if vol > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Equal Weight (1/N)
# ---------------------------------------------------------------------------

class EqualWeightStrategy(PortfolioStrategy):
    """
    Naive 1/N diversification.

    Despite its simplicity, 1/N is a very competitive benchmark that many
    sophisticated strategies fail to beat out-of-sample
    (DeMiguel et al., 2009).
    """

    def __init__(self):
        super().__init__("EqualWeight")

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        n = len(returns.columns)
        return pd.Series(1 / n, index=returns.columns)


# ---------------------------------------------------------------------------
# 2. Minimum Variance
# ---------------------------------------------------------------------------

class MinVarianceStrategy(PortfolioStrategy):
    """
    Markowitz minimum-variance portfolio.

    Minimises portfolio variance without any return assumption.
    Robust to return estimation errors.

    min_w  w'Σw
    s.t.   sum(w) = 1
           w ≥ 0  (long-only)
    """

    def __init__(self, shrinkage: float = 0.1):
        super().__init__("MinVariance")
        self.shrinkage = shrinkage

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        n   = len(returns.columns)
        cov = _cov_matrix(returns, self.shrinkage)

        def objective(w):
            return _portfolio_vol(w, cov)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0, 1)] * n if self.long_only else [(-0.3, 1)] * n
        w0          = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if not result.success:
            log.warning(f"MinVariance optimisation did not converge: {result.message}")

        w = result.x
        w = np.clip(w, 0, 1) if self.long_only else w
        w /= w.sum()

        return pd.Series(w, index=returns.columns)


# ---------------------------------------------------------------------------
# 3. Maximum Sharpe (Tangency portfolio)
# ---------------------------------------------------------------------------

class MaxSharpeStrategy(PortfolioStrategy):
    """
    Tangency portfolio: maximise the Sharpe ratio.

    max_w  (μ'w − r_f) / sqrt(w'Σw)
    s.t.   sum(w) = 1,  w ≥ 0

    Uses historical mean returns as expected return estimates
    (vulnerable to estimation error — use with caution).
    """

    def __init__(self, shrinkage: float = 0.1):
        super().__init__("MaxSharpe")
        self.shrinkage = shrinkage

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        n   = len(returns.columns)
        mu  = returns.mean().values
        cov = _cov_matrix(returns, self.shrinkage)

        def neg_sharpe(w):
            return -_portfolio_sharpe(w, mu, cov)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0, 1)] * n if self.long_only else [(-0.3, 1)] * n
        w0          = np.ones(n) / n

        result = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        w = result.x
        w = np.clip(w, 0, 1) if self.long_only else w
        w /= np.abs(w).sum() if not self.long_only else w.sum()

        return pd.Series(w, index=returns.columns)


# ---------------------------------------------------------------------------
# 4. Risk Parity
# ---------------------------------------------------------------------------

class RiskParityStrategy(PortfolioStrategy):
    """
    Equal Risk Contribution (ERC) — Risk Parity.

    Each asset contributes equally to portfolio volatility.
    More diversified in risk terms than equal-weight.
    Used widely by institutional investors (e.g., Bridgewater All Weather).

    Objective: min_w  sum_i sum_j (RC_i - RC_j)^2
    where RC_i = w_i * (Σw)_i / (w'Σw)  (risk contribution of asset i)
    """

    def __init__(self, shrinkage: float = 0.1):
        super().__init__("RiskParity")
        self.shrinkage = shrinkage

    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        n   = len(returns.columns)
        cov = _cov_matrix(returns, self.shrinkage)

        def risk_contributions(w):
            sigma  = np.sqrt(w @ cov @ w)
            mrc    = cov @ w          # marginal risk contributions
            rc     = w * mrc / sigma  # total risk contributions
            return rc

        def objective(w):
            rc     = risk_contributions(w)
            target = np.full(n, rc.sum() / n)
            return float(np.sum((rc - target) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(1e-6, 1)] * n   # long-only
        w0          = np.ones(n) / n

        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000},
        )

        w = np.clip(result.x, 0, 1)
        w /= w.sum()

        return pd.Series(w, index=returns.columns)


# ---------------------------------------------------------------------------
# 5. ML-enhanced portfolio
# ---------------------------------------------------------------------------

class MLPortfolioStrategy(PortfolioStrategy):
    """
    ML-predicted returns plugged into the Markowitz mean-variance optimiser.

    Instead of using historical sample means (noisy), we use ML predictions
    as expected return estimates, then solve:

    max_w  w' μ̂_ML  −  γ/2 · w'Σw
    s.t.   sum(w) = 1,  w ≥ 0

    The risk-aversion parameter γ controls the return/risk trade-off.
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        shrinkage: float = 0.1,
        long_only: bool = True,
    ):
        super().__init__("MLPortfolio", long_only)
        self.risk_aversion = risk_aversion
        self.shrinkage     = shrinkage

    def compute_weights(
        self,
        returns: pd.DataFrame,
        predicted_returns: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Parameters
        ----------
        returns           : historical returns for covariance estimation
        predicted_returns : ML-predicted expected returns (Series indexed by ticker)
        """
        if predicted_returns is None:
            log.warning("No ML predictions supplied → falling back to MaxSharpe")
            return MaxSharpeStrategy().compute_weights(returns)

        tickers = returns.columns.tolist()
        mu      = predicted_returns.reindex(tickers).fillna(0).values
        cov     = _cov_matrix(returns[tickers], self.shrinkage)
        n       = len(tickers)
        gamma   = self.risk_aversion

        def neg_utility(w):
            ret  = _portfolio_return(w, mu)
            var  = w @ cov @ w
            return -(ret - gamma / 2 * var)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds      = [(0, 1)] * n if self.long_only else [(-0.3, 1)] * n
        w0          = np.ones(n) / n

        result = minimize(
            neg_utility, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        w = np.clip(result.x, 0, 1) if self.long_only else result.x
        w /= w.sum() if self.long_only else np.abs(w).sum()

        return pd.Series(w, index=tickers)


# ---------------------------------------------------------------------------
# 6. Long-Short ML signal
# ---------------------------------------------------------------------------

class LongShortMLStrategy(PortfolioStrategy):
    """
    Long-Short portfolio based on ML signal ranking.

    - Long the top-Q quantile of predicted returns
    - Short the bottom-Q quantile of predicted returns
    - Dollar-neutral: sum(w) ≈ 0

    This strategy isolates the predictive power of the ML signal,
    independently of market direction.
    """

    def __init__(self, quantile: float = 0.2):
        super().__init__("LongShortML", long_only=False)
        self.quantile = quantile

    def compute_weights(
        self,
        returns: pd.DataFrame,
        predicted_returns: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.Series:

        if predicted_returns is None:
            raise ValueError("LongShortML requires predicted_returns")

        tickers = returns.columns.tolist()
        pred    = predicted_returns.reindex(tickers)

        q_low  = pred.quantile(self.quantile)
        q_high = pred.quantile(1 - self.quantile)

        longs  = pred[pred >= q_high].index
        shorts = pred[pred <= q_low].index

        n_long  = len(longs)
        n_short = len(shorts)

        if n_long == 0 or n_short == 0:
            log.warning("LongShort: no longs or shorts, returning equal-weight")
            return pd.Series(1 / len(tickers), index=tickers)

        w = pd.Series(0.0, index=tickers)
        w[longs]  =  0.5 / n_long
        w[shorts] = -0.5 / n_short

        return w


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    strategy: PortfolioStrategy,
    returns: pd.DataFrame,
    rebalance_freq: str = "ME",
    lookback_days: int = 252,
    transaction_cost: float = 0.001,
    ml_predictions: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Rolling walk-forward backtest.

    At each rebalance date:
    1. Estimate covariance from last `lookback_days` of returns
    2. (Optionally) fetch ML predictions for that date
    3. Compute new weights
    4. Apply transaction costs (proportional to turnover)
    5. Track portfolio value

    Parameters
    ----------
    strategy         : PortfolioStrategy instance
    returns          : pd.DataFrame (T × N) of daily returns
    rebalance_freq   : pandas offset string ('ME' = month-end, 'QE' = quarter-end)
    lookback_days    : history window for covariance estimation
    transaction_cost : cost per unit of turnover (e.g. 0.001 = 10 bps)
    ml_predictions   : optional DataFrame (rebalance_dates × N) of ML-predicted returns

    Returns
    -------
    portfolio_returns : pd.Series   (daily portfolio returns)
    weights_df       : pd.DataFrame (rebalance_dates × N)
    """
    rebalance_dates = returns.resample(rebalance_freq).last().index
    rebalance_dates = rebalance_dates[rebalance_dates >= returns.index[lookback_days]]

    daily_returns   = []
    all_weights     = []
    current_weights = None

    for i, date in enumerate(rebalance_dates):
        # ── Training window ──────────────────────────────────────────────── #
        hist = returns.loc[:date].iloc[-lookback_days:]

        # ── ML predictions ────────────────────────────────────────────────── #
        ml_pred = None
        if ml_predictions is not None:
            closest = ml_predictions.index.asof(date)
            if pd.notna(closest):
                ml_pred = ml_predictions.loc[closest]

        # ── New weights ───────────────────────────────────────────────────── #
        new_weights = strategy.compute_weights(hist, predicted_returns=ml_pred)

        # ── Transaction costs ─────────────────────────────────────────────── #
        if current_weights is not None:
            turnover = (new_weights - current_weights.reindex(new_weights.index).fillna(0)).abs().sum()
            tc       = transaction_cost * turnover
        else:
            tc = 0.0

        current_weights = new_weights
        record = new_weights.copy()
        record.name = date
        all_weights.append(record)

        # ── Apply weights until next rebalance ────────────────────────────── #
        end_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1]
        period_ret = returns.loc[date:end_date].iloc[1:]

        if len(period_ret) == 0:
            continue

        # Align weights and returns
        w = new_weights.reindex(period_ret.columns).fillna(0)
        w /= w.abs().sum() if w.abs().sum() > 0 else 1

        port_ret = period_ret @ w
        port_ret.iloc[0] -= tc   # subtract transaction cost on first day

        daily_returns.append(port_ret)

    portfolio_returns = pd.concat(daily_returns).sort_index()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="first")]

    weights_df = pd.DataFrame(all_weights).fillna(0)

    return portfolio_returns, weights_df
