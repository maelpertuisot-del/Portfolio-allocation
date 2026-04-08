"""
evaluation.py
=============
Portfolio performance evaluation metrics.

Metrics implemented
-------------------
- CAGR                    : Compound Annual Growth Rate
- Annualised volatility
- Sharpe ratio            : risk-adjusted return
- Sortino ratio           : downside-risk-adjusted return
- Maximum Drawdown (MDD)  : largest peak-to-trough decline
- Calmar ratio            : CAGR / |MDD|
- Average turnover        : rebalancing activity proxy
- Information ratio       : active return / tracking error
- Value-at-Risk (VaR 95 %)
- Conditional VaR (CVaR 95 %)

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RF_DAILY = 0.04 / TRADING_DAYS


# ---------------------------------------------------------------------------
# Single-metric functions
# ---------------------------------------------------------------------------

def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    n_years = len(returns) / TRADING_DAYS
    cum     = (1 + returns).prod()
    return float(cum ** (1 / n_years) - 1)


def annualised_vol(returns: pd.Series) -> float:
    """Annualised standard deviation of returns."""
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf_daily: float = RF_DAILY) -> float:
    """
    Annualised Sharpe ratio.
    SR = (μ_p − r_f) / σ_p  ×  √252
    """
    excess = returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf_daily: float = RF_DAILY) -> float:
    """
    Sortino ratio — penalises only downside volatility.
    SR_sortino = (μ_p − r_f) / σ_downside  ×  √252
    """
    excess    = returns - rf_daily
    downside  = excess[excess < 0].std()
    if downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown (negative number).
    MDD = min_t  (NAV_t / max_{s≤t} NAV_s  − 1)
    """
    cum_ret = (1 + returns).cumprod()
    peak    = cum_ret.cummax()
    dd      = cum_ret / peak - 1
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio = CAGR / |Max Drawdown|."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return float(cagr(returns) / abs(mdd))


def var_95(returns: pd.Series) -> float:
    """Historical Value-at-Risk at 95 % confidence (1-day)."""
    return float(np.percentile(returns, 5))


def cvar_95(returns: pd.Series) -> float:
    """Conditional Value-at-Risk (Expected Shortfall) at 95 %."""
    cutoff = var_95(returns)
    tail   = returns[returns <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else cutoff


def average_turnover(weights_df: pd.DataFrame) -> float:
    """
    Average one-way turnover across rebalancing periods.
    Turnover = 0.5 × Σ_i |w_{i,t} − w_{i,t-1}|  (annualised)
    """
    if weights_df is None or len(weights_df) < 2:
        return np.nan
    diffs = weights_df.diff().abs().sum(axis=1).iloc[1:]
    return float(diffs.mean() * 0.5)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Information Ratio = active_return / tracking_error.
    Measures skill relative to a benchmark.
    """
    active = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index).fillna(0)
    te     = active.std() * np.sqrt(TRADING_DAYS)
    if te == 0:
        return 0.0
    return float(active.mean() * TRADING_DAYS / te)


# ---------------------------------------------------------------------------
# Full scorecard
# ---------------------------------------------------------------------------

def performance_summary(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    weights_df: Optional[pd.DataFrame] = None,
    name: str = "Portfolio",
) -> pd.Series:
    """
    Compute the full performance scorecard.

    Parameters
    ----------
    returns    : daily portfolio return series
    benchmark  : optional benchmark return series (for IR)
    weights_df : optional weights DataFrame (for turnover)
    name       : label for the result Series

    Returns
    -------
    metrics : pd.Series
    """
    metrics = {
        "CAGR (%)":         round(cagr(returns) * 100, 2),
        "Vol (%)":          round(annualised_vol(returns) * 100, 2),
        "Sharpe":           round(sharpe_ratio(returns), 3),
        "Sortino":          round(sortino_ratio(returns), 3),
        "Max Drawdown (%)": round(max_drawdown(returns) * 100, 2),
        "Calmar":           round(calmar_ratio(returns), 3),
        "VaR 95% (%)":      round(var_95(returns) * 100, 2),
        "CVaR 95% (%)":     round(cvar_95(returns) * 100, 2),
    }

    if benchmark is not None:
        metrics["IR"] = round(information_ratio(returns, benchmark), 3)

    if weights_df is not None:
        metrics["Avg Turnover"] = round(average_turnover(weights_df), 4)

    return pd.Series(metrics, name=name)


def compare_strategies(
    strategy_returns: Dict[str, pd.Series],
    benchmark_name: str = "EqualWeight",
    weights_dict: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Build a comparison table for all strategies.

    Parameters
    ----------
    strategy_returns : dict {strategy_name: daily_return_series}
    benchmark_name   : name of the strategy to use as benchmark for IR
    weights_dict     : optional dict of weights DataFrames for turnover

    Returns
    -------
    summary : pd.DataFrame  (metrics × strategies)
    """
    benchmark = strategy_returns.get(benchmark_name)

    summaries = []
    for name, rets in strategy_returns.items():
        w_df = weights_dict.get(name) if weights_dict else None
        s    = performance_summary(rets, benchmark=benchmark, weights_df=w_df, name=name)
        summaries.append(s)

    return pd.concat(summaries, axis=1)


def cumulative_returns(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """Convert daily returns to cumulative NAV."""
    return starting_value * (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Time series of drawdown from running peak."""
    nav  = cumulative_returns(returns)
    peak = nav.cummax()
    return nav / peak - 1


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """
    Rolling Sharpe ratio over `window` trading days.
    Useful for detecting regime changes.
    """
    excess = returns - RF_DAILY
    return (
        excess.rolling(window).mean() /
        excess.rolling(window).std() *
        np.sqrt(TRADING_DAYS)
    )
