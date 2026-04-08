"""
visualization.py
================
All plotting utilities for the portfolio ML project.

Functions
---------
plot_price_history        : price evolution of selected tickers
plot_returns_distribution : histogram + QQ plot of returns
plot_correlation_heatmap  : pairwise return correlations
plot_feature_importance   : bar chart of feature importances
plot_shap_summary         : SHAP beeswarm plot
plot_cumulative_returns   : portfolio NAV comparison
plot_drawdowns            : drawdown series for all strategies
plot_weights_heatmap      : portfolio weights over time
plot_efficient_frontier   : Monte Carlo efficient frontier
plot_model_comparison     : model metrics bar chart
plot_rolling_sharpe       : rolling Sharpe over time

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from .evaluation import cumulative_returns, drawdown_series, rolling_sharpe

# ── Global style ───────────────────────────────────────────────────────────── #
PALETTE  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4", "#795548"]
FIGSIZE  = (14, 6)
DPI      = 120


def _save_or_show(fig: plt.Figure, path: Optional[str] = None) -> plt.Figure:
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=DPI)
    return fig


# ---------------------------------------------------------------------------
# 1. Price history
# ---------------------------------------------------------------------------

def plot_price_history(
    prices: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    normalise: bool = True,
    title: str = "Normalised Price History",
    path: Optional[str] = None,
) -> plt.Figure:
    """Plot price history (optionally normalised to 100 at start)."""
    tickers = tickers or prices.columns.tolist()[:10]
    data    = prices[tickers].copy()

    if normalise:
        data = data / data.iloc[0] * 100

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, col in enumerate(data.columns):
        ax.plot(data.index, data[col], label=col,
                color=PALETTE[i % len(PALETTE)], linewidth=1.2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalised Price (base = 100)" if normalise else "Price ($)")
    ax.legend(ncol=5, fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 2. Returns distribution
# ---------------------------------------------------------------------------

def plot_returns_distribution(
    returns: pd.DataFrame,
    tickers: Optional[List[str]] = None,
    bins: int = 60,
    path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of daily log-returns for selected tickers."""
    tickers = tickers or returns.columns.tolist()[:6]

    n  = len(tickers)
    nc = min(n, 3)
    nr = (n + nc - 1) // nc

    fig, axes = plt.subplots(nr, nc, figsize=(14, 4 * nr))
    axes      = np.array(axes).flatten() if n > 1 else [axes]

    for ax, ticker in zip(axes, tickers):
        ret = returns[ticker].dropna()
        ax.hist(ret, bins=bins, color="#2196F3", alpha=0.7, density=True, edgecolor="white")
        from scipy.stats import norm
        mu, sigma = ret.mean(), ret.std()
        x = np.linspace(ret.min(), ret.max(), 300)
        ax.plot(x, norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal fit")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(ticker, fontweight="bold")
        ax.set_xlabel("Daily log-return")
        ax.legend(fontsize=8)
        stats_txt = f"μ={mu*252:.1%}  σ={sigma*np.sqrt(252):.1%}"
        ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=8, va="top", ha="left")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Daily Returns Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    returns: pd.DataFrame,
    title: str = "Return Correlation Matrix",
    path: Optional[str] = None,
) -> plt.Figure:
    """Seaborn heatmap of pairwise return correlations."""
    corr = returns.corr()
    n    = len(corr)
    size = max(8, n * 0.4)

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        annot=(n <= 20), fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, square=True,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importance",
    path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart of feature importances."""
    top = importance.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    colors  = ["#2196F3" if v > 0 else "#F44336" for v in top.values]
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 5. Cumulative returns
# ---------------------------------------------------------------------------

def plot_cumulative_returns(
    strategy_returns: Dict[str, pd.Series],
    title: str = "Portfolio Cumulative Returns",
    path: Optional[str] = None,
) -> plt.Figure:
    """Compare cumulative NAV across all strategies."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (name, rets) in enumerate(strategy_returns.items()):
        nav = cumulative_returns(rets)
        style = "--" if "Benchmark" in name or "EqualWeight" in name else "-"
        lw    = 1.5 if style == "--" else 2.0
        ax.plot(nav.index, nav.values, label=name,
                color=PALETTE[i % len(PALETTE)], linewidth=lw, linestyle=style)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(1, color="black", linewidth=0.6, linestyle=":")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 6. Drawdowns
# ---------------------------------------------------------------------------

def plot_drawdowns(
    strategy_returns: Dict[str, pd.Series],
    title: str = "Drawdown",
    path: Optional[str] = None,
) -> plt.Figure:
    """Plot drawdown series for all strategies."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (name, rets) in enumerate(strategy_returns.items()):
        dd = drawdown_series(rets) * 100
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.25, color=PALETTE[i % len(PALETTE)])
        ax.plot(dd.index, dd.values, label=name,
                color=PALETTE[i % len(PALETTE)], linewidth=1.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 7. Weights heatmap
# ---------------------------------------------------------------------------

def plot_weights_heatmap(
    weights_df: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
    path: Optional[str] = None,
) -> plt.Figure:
    """Stacked area chart of portfolio weights over time."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    top_n   = min(15, weights_df.shape[1])
    top_cols = weights_df.mean().nlargest(top_n).index

    weights_df[top_cols].plot.area(ax=ax, stacked=True, alpha=0.85,
                                    colormap="tab20", legend=True)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight")
    ax.set_xlabel("")
    ax.legend(ncol=5, fontsize=7, loc="lower left")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 8. Efficient frontier (Monte Carlo)
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    returns: pd.DataFrame,
    n_simulations: int = 3000,
    highlight: Optional[Dict[str, pd.Series]] = None,
    path: Optional[str] = None,
) -> plt.Figure:
    """
    Monte Carlo efficient frontier with random portfolios.
    Highlights specific strategies if provided.
    """
    mu   = returns.mean().values * 252
    cov  = returns.cov().values  * 252
    n    = len(mu)

    vols, rets, sharpes = [], [], []
    np.random.seed(42)
    for _ in range(n_simulations):
        w = np.random.dirichlet(np.ones(n))
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        vols.append(v)
        rets.append(r)
        sharpes.append((r - 0.04) / v)

    vols    = np.array(vols)
    rets    = np.array(rets)
    sharpes = np.array(sharpes)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(vols * 100, rets * 100, c=sharpes,
                    cmap="viridis", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    if highlight:
        for i, (name, rets_s) in enumerate(highlight.items()):
            ann_ret = rets_s.mean() * 252 * 100
            ann_vol = rets_s.std()  * np.sqrt(252) * 100
            ax.scatter(ann_vol, ann_ret, s=150, zorder=5,
                       color=PALETTE[i % len(PALETTE)],
                       edgecolors="black", linewidths=1.5, label=name)

    ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualised Return (%)",    fontsize=11)
    ax.set_title("Efficient Frontier (Monte Carlo)", fontsize=13, fontweight="bold")
    if highlight:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 9. Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = "test_r2",
    title: Optional[str] = None,
    path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing model performance on a given metric."""
    title = title or f"Model Comparison — {metric}"
    data  = results[metric].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2196F3" if v >= 0 else "#F44336" for v in data.values]
    bars    = ax.bar(data.index, data.values, color=colors, edgecolor="white", width=0.55)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 10. Rolling Sharpe
# ---------------------------------------------------------------------------

def plot_rolling_sharpe(
    strategy_returns: Dict[str, pd.Series],
    window: int = 126,
    title: str = "Rolling 6-Month Sharpe Ratio",
    path: Optional[str] = None,
) -> plt.Figure:
    """Rolling Sharpe ratio for all strategies."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, (name, rets) in enumerate(strategy_returns.items()):
        rs = rolling_sharpe(rets, window).dropna()
        ax.plot(rs.index, rs.values, label=name,
                color=PALETTE[i % len(PALETTE)], linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (annualised)")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 11. Sector allocation
# ---------------------------------------------------------------------------

def plot_sector_allocation(
    weights: pd.Series,
    sector_map: pd.Series,
    title: str = "Sector Allocation",
    path: Optional[str] = None,
) -> plt.Figure:
    """Pie chart of portfolio allocation by sector."""
    sector_w = weights.rename(sector_map).groupby(level=0).sum()
    sector_w = sector_w[sector_w > 0.005].sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sector_w.values,
        labels=sector_w.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=plt.cm.tab20.colors,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_or_show(fig, path)
