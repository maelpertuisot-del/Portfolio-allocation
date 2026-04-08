"""
main.py
=======
Command-line entry point for the portfolio ML pipeline.

Usage
-----
    python main.py                        # full pipeline (download + train + backtest)
    python main.py --cache-only           # use cached data only
    python main.py --no-shap              # skip SHAP computation
    python main.py --horizon 21           # 21-day return prediction horizon
    python main.py --train-end 2021-12-31 # custom train/test split

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────── #
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML Portfolio Allocation Pipeline")
    p.add_argument("--cache-only",  action="store_true",
                   help="Load prices from cache (skip download)")
    p.add_argument("--no-shap",     action="store_true",
                   help="Skip SHAP computation")
    p.add_argument("--horizon",     type=int, default=5,
                   help="Return prediction horizon in days (default: 5)")
    p.add_argument("--train-end",   default="2020-12-31",
                   help="Last date of training set (default: 2020-12-31)")
    p.add_argument("--val-end",     default="2022-06-30",
                   help="Last date of validation set (default: 2022-06-30)")
    p.add_argument("--tickers",     nargs="+", default=None,
                   help="Override ticker list")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Imports ──────────────────────────────────────────────────────────── #
    from src.data_loader  import download_prices, load_prices, UNIVERSE
    from src.features     import build_ml_dataset, log_returns, build_target
    from src.models       import get_all_models, train_and_evaluate
    from src.portfolio    import (EqualWeightStrategy, MinVarianceStrategy,
                                  MaxSharpeStrategy, RiskParityStrategy,
                                  MLPortfolioStrategy, LongShortMLStrategy,
                                  walk_forward_backtest)
    from src.evaluation   import compare_strategies
    from src.visualization import (plot_price_history, plot_returns_distribution,
                                    plot_correlation_heatmap, plot_cumulative_returns,
                                    plot_drawdowns, plot_rolling_sharpe, plot_model_comparison)

    tickers = args.tickers or UNIVERSE

    # ── 1. Data ──────────────────────────────────────────────────────────── #
    log.info("━━━ Step 1: Data Collection ━━━")
    cache = "data/prices.csv"
    if args.cache_only and os.path.exists(cache):
        prices = load_prices(cache)
    else:
        prices = download_prices(tickers, save_path=cache)

    returns = log_returns(prices, 1).dropna()

    # ── 2. EDA plots ─────────────────────────────────────────────────────── #
    log.info("━━━ Step 2: EDA ━━━")
    plot_price_history(prices, path="results/01_price_history.png")
    plot_returns_distribution(returns, path="results/02_returns_distribution.png")
    plot_correlation_heatmap(returns, path="results/03_correlation_heatmap.png")
    plt.close("all")
    log.info("EDA plots saved → results/")

    # ── 3. Feature engineering ────────────────────────────────────────────── #
    log.info("━━━ Step 3: Feature Engineering ━━━")
    X, y = build_ml_dataset(prices, horizon=args.horizon)
    log.info(f"ML dataset: {X.shape[0]} obs × {X.shape[1]} features")

    # ── 4. Train / test split ─────────────────────────────────────────────── #
    log.info("━━━ Step 4: Model Training ━━━")
    dates      = X.index.get_level_values("date")
    train_mask = dates <= args.val_end
    test_mask  = dates >  args.val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    log.info(f"Train: {train_mask.sum()} obs  |  Test: {test_mask.sum()} obs")

    # ── 5. ML models ──────────────────────────────────────────────────────── #
    ml_models = get_all_models()
    results, fitted = train_and_evaluate(
        ml_models, X_train, y_train, X_test, y_test
    )

    log.info("\n" + "=" * 60)
    log.info("MODEL RESULTS:\n" + results.round(4).to_string())
    log.info("=" * 60)
    results.to_csv("results/model_results.csv")

    plot_model_comparison(results, path="results/06_model_comparison.png")
    plt.close("all")

    # ── 6. Portfolio backtest ─────────────────────────────────────────────── #
    log.info("━━━ Step 5: Portfolio Backtesting ━━━")
    best_name  = results["test_r2"].idxmax()
    best_model = fitted[best_name]
    log.info(f"Using best model: {best_name}")

    prices_bt  = prices.loc["2018-01-01":"2024-01-01"]
    returns_bt = log_returns(prices_bt, 1).dropna()

    # Precompute monthly ML predictions
    log.info("Computing ML predictions for backtest …")
    rebalance_dates = returns_bt.resample("ME").last().index
    ml_preds_all = {}

    for date in rebalance_dates:
        try:
            hist   = prices.loc[:date].iloc[-325:]
            X_p, _ = build_ml_dataset(hist, horizon=args.horizon, dropna=True)
            latest = (
                X_p.reset_index().sort_values("date")
                .drop_duplicates("ticker", keep="last")
                .set_index("ticker")
                .drop(columns="date", errors="ignore")
            )
            if not latest.empty:
                ml_preds_all[date] = pd.Series(best_model.predict(latest), index=latest.index)
        except Exception:
            pass

    ml_predictions = pd.DataFrame(ml_preds_all).T

    # Run all strategies
    strategies = {
        "EqualWeight": EqualWeightStrategy(),
        "MinVariance": MinVarianceStrategy(),
        "MaxSharpe":   MaxSharpeStrategy(),
        "RiskParity":  RiskParityStrategy(),
        "MLPortfolio": MLPortfolioStrategy(risk_aversion=2.0),
    }

    all_returns, all_weights = {}, {}
    for name, strat in strategies.items():
        ml_p = ml_predictions if "ML" in name else None
        try:
            r, w = walk_forward_backtest(
                strat, returns_bt, rebalance_freq="ME",
                lookback_days=252, transaction_cost=0.001,
                ml_predictions=ml_p,
            )
            all_returns[name] = r
            all_weights[name] = w
            log.info(f"  {name}: ✓")
        except Exception as e:
            log.warning(f"  {name}: failed — {e}")

    # Align and evaluate
    idx = None
    for r in all_returns.values():
        idx = r.index if idx is None else idx.intersection(r.index)

    all_returns_aligned = {k: v.loc[idx] for k, v in all_returns.items()}
    summary = compare_strategies(all_returns_aligned, "EqualWeight", all_weights)

    log.info("\n" + "=" * 70)
    log.info("PORTFOLIO PERFORMANCE:\n" + summary.round(3).to_string())
    log.info("=" * 70)
    summary.to_csv("results/portfolio_summary.csv")

    # ── 7. Final plots ────────────────────────────────────────────────────── #
    log.info("━━━ Step 6: Saving Final Plots ━━━")
    plot_cumulative_returns(all_returns_aligned, path="results/09_cumulative_returns.png")
    plot_drawdowns(all_returns_aligned,          path="results/10_drawdowns.png")
    plot_rolling_sharpe(all_returns_aligned,     path="results/11_rolling_sharpe.png")
    plt.close("all")

    log.info("\n✅ Pipeline complete!  Results saved to results/")
    log.info(f"   Best Sharpe : {summary.loc['Sharpe'].idxmax()}")
    log.info(f"   Best CAGR   : {summary.loc['CAGR (%)'].idxmax()}")


if __name__ == "__main__":
    main()
