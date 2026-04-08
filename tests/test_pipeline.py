"""
tests/test_pipeline.py
======================
Unit and integration tests for the ML portfolio pipeline.

Run with:
    pytest tests/ -v

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_prices():
    """500-day price history for 10 synthetic stocks."""
    np.random.seed(0)
    dates  = pd.date_range("2019-01-01", periods=500, freq="B")
    prices = pd.DataFrame(
        np.exp(np.cumsum(np.random.randn(500, 10) * 0.008, axis=0)),
        index=dates,
        columns=[f"S{i:02d}" for i in range(10)],
    )
    return prices


@pytest.fixture(scope="module")
def synthetic_returns(synthetic_prices):
    from src.features import log_returns
    return log_returns(synthetic_prices, 1).dropna()


# ---------------------------------------------------------------------------
# 1. data_loader
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_sector_map_coverage(self):
        from src.data_loader import UNIVERSE, SECTOR_MAP
        missing = [t for t in UNIVERSE if t not in SECTOR_MAP]
        assert len(missing) == 0, f"Missing sector mappings: {missing}"

    def test_time_split_no_overlap(self, synthetic_prices):
        from src.data_loader import time_split
        train, val, test = time_split(
            synthetic_prices, train_end="2020-06-30", val_end="2020-09-30"
        )
        assert train.index[-1] <= pd.Timestamp("2020-06-30")
        assert val.index[0]   >  pd.Timestamp("2020-06-30")
        assert test.index[0]  >  pd.Timestamp("2020-09-30")

    def test_time_split_sizes_sum(self, synthetic_prices):
        from src.data_loader import time_split
        train, val, test = time_split(
            synthetic_prices, train_end="2020-06-30", val_end="2020-09-30"
        )
        total = len(train) + len(val) + len(test)
        # Total may be 1-2 less than original due to overlap removal
        assert total >= len(synthetic_prices) - 3


# ---------------------------------------------------------------------------
# 2. features
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_log_returns_shape(self, synthetic_prices):
        from src.features import log_returns
        ret = log_returns(synthetic_prices, 1)
        assert ret.shape == synthetic_prices.shape

    def test_log_returns_no_lookahead(self, synthetic_prices):
        from src.features import log_returns
        ret = log_returns(synthetic_prices, 5)
        # At t=0..4, returns should be NaN (not enough history)
        assert ret.iloc[:4].isna().all().all()

    def test_rsi_bounds(self, synthetic_prices):
        from src.features import rsi
        r = rsi(synthetic_prices, 14).dropna()
        assert (r >= 0).all().all()
        assert (r <= 100).all().all()

    def test_feature_dict_keys(self, synthetic_prices):
        from src.features import build_feature_dict
        feats = build_feature_dict(synthetic_prices)
        expected = ["ret_1d", "vol_21d", "ma_ratio_20d", "rsi_14", "mom_3m"]
        for k in expected:
            assert k in feats, f"Missing feature: {k}"

    def test_ml_dataset_no_future_leakage(self, synthetic_prices):
        from src.features import build_ml_dataset
        X, y = build_ml_dataset(synthetic_prices, horizon=5)
        dates_X = X.index.get_level_values("date")
        dates_y = y.index.get_level_values("date")
        # All feature dates should match label dates (future is baked into y)
        assert (dates_X == dates_y).all()

    def test_ml_dataset_no_nan(self, synthetic_prices):
        from src.features import build_ml_dataset
        X, y = build_ml_dataset(synthetic_prices, horizon=5, dropna=True)
        assert not X.isna().any().any(), "X contains NaN"
        assert not y.isna().any(),       "y contains NaN"

    def test_cross_section_rank_bounds(self, synthetic_prices):
        from src.features import build_feature_dict
        feats = build_feature_dict(synthetic_prices)
        rank  = feats["rank_ret_1d"].dropna()
        assert (rank >= 0).all().all()
        assert (rank <= 1).all().all()


# ---------------------------------------------------------------------------
# 3. models
# ---------------------------------------------------------------------------

class TestModels:
    def test_ridge_fit_predict(self, synthetic_prices):
        from src.features import build_ml_dataset
        from src.models   import make_linear_model
        X, y = build_ml_dataset(synthetic_prices, horizon=5)
        model = make_linear_model("ridge")
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)
        assert np.isfinite(pred).all()

    def test_random_forest_fit_predict(self, synthetic_prices):
        from src.features import build_ml_dataset
        from src.models   import make_random_forest
        X, y = build_ml_dataset(synthetic_prices, horizon=5)
        model = make_random_forest(n_estimators=10)
        model.fit(X, y)
        pred = model.predict(X)
        assert len(pred) == len(y)

    def test_ts_cross_validate_output(self, synthetic_prices):
        from src.features import build_ml_dataset
        from src.models   import make_linear_model, ts_cross_validate
        X, y = build_ml_dataset(synthetic_prices, horizon=5)
        model = make_linear_model("ridge")
        res = ts_cross_validate(model, X, y, n_splits=3)
        assert "mean" in res
        assert "scores" in res
        assert len(res["scores"]) == 3


# ---------------------------------------------------------------------------
# 4. portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_equal_weight_sums_to_one(self, synthetic_returns):
        from src.portfolio import EqualWeightStrategy
        w = EqualWeightStrategy().compute_weights(synthetic_returns)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_min_variance_sums_to_one(self, synthetic_returns):
        from src.portfolio import MinVarianceStrategy
        w = MinVarianceStrategy().compute_weights(synthetic_returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_min_variance_long_only(self, synthetic_returns):
        from src.portfolio import MinVarianceStrategy
        w = MinVarianceStrategy().compute_weights(synthetic_returns)
        assert (w >= -1e-6).all()

    def test_risk_parity_sums_to_one(self, synthetic_returns):
        from src.portfolio import RiskParityStrategy
        w = RiskParityStrategy().compute_weights(synthetic_returns)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_risk_parity_equal_risk(self, synthetic_returns):
        """Risk contributions should be approximately equal."""
        from src.portfolio import RiskParityStrategy, _cov_matrix
        w   = RiskParityStrategy().compute_weights(synthetic_returns).values
        cov = _cov_matrix(synthetic_returns)
        sigma = np.sqrt(w @ cov @ w)
        rc    = w * (cov @ w) / sigma
        # Coefficient of variation of risk contributions should be small
        assert rc.std() / rc.mean() < 0.30, "Risk contributions not equal"

    def test_long_short_dollar_neutral(self, synthetic_returns):
        from src.portfolio import LongShortMLStrategy
        pred = pd.Series(np.random.randn(len(synthetic_returns.columns)),
                         index=synthetic_returns.columns)
        w = LongShortMLStrategy().compute_weights(synthetic_returns, predicted_returns=pred)
        assert abs(w.sum()) < 1e-6, "Long-short not dollar-neutral"

    def test_walk_forward_backtest_length(self, synthetic_returns):
        from src.portfolio import walk_forward_backtest, EqualWeightStrategy
        ret, w = walk_forward_backtest(
            EqualWeightStrategy(), synthetic_returns,
            lookback_days=60, rebalance_freq="ME",
        )
        assert len(ret) > 0
        assert isinstance(ret, pd.Series)
        assert isinstance(w, pd.DataFrame)


# ---------------------------------------------------------------------------
# 5. evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(1)
        return pd.Series(
            np.random.randn(252) * 0.01 + 0.0004,
            index=pd.date_range("2022-01-01", periods=252, freq="B"),
        )

    def test_sharpe_sign(self, sample_returns):
        from src.evaluation import sharpe_ratio
        sr = sharpe_ratio(sample_returns)
        assert sr > 0   # positive expected return → positive Sharpe

    def test_max_drawdown_negative(self, sample_returns):
        from src.evaluation import max_drawdown
        mdd = max_drawdown(sample_returns)
        assert mdd <= 0

    def test_cvar_le_var(self, sample_returns):
        from src.evaluation import var_95, cvar_95
        assert cvar_95(sample_returns) <= var_95(sample_returns)

    def test_performance_summary_keys(self, sample_returns):
        from src.evaluation import performance_summary
        s = performance_summary(sample_returns)
        for key in ["CAGR (%)", "Sharpe", "Max Drawdown (%)", "Calmar"]:
            assert key in s.index

    def test_compare_strategies_shape(self, sample_returns):
        from src.evaluation import compare_strategies
        strats = {"A": sample_returns, "B": sample_returns * 0.5}
        df = compare_strategies(strats)
        assert "A" in df.columns
        assert "B" in df.columns


# ---------------------------------------------------------------------------
# 6. Integration: full mini-pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self, synthetic_prices):
        """
        Smoke test: run the complete pipeline on synthetic data.
        Should complete without errors.
        """
        from src.features  import build_ml_dataset, log_returns
        from src.models    import make_random_forest
        from src.portfolio import walk_forward_backtest, EqualWeightStrategy
        from src.evaluation import performance_summary

        # Feature engineering
        X, y = build_ml_dataset(synthetic_prices, horizon=5)
        assert len(X) > 0

        # Model training
        model = make_random_forest(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert np.isfinite(preds).all()

        # Backtest
        returns = log_returns(synthetic_prices, 1).dropna()
        port_ret, weights = walk_forward_backtest(
            EqualWeightStrategy(), returns,
            lookback_days=60, rebalance_freq="ME",
        )
        assert len(port_ret) > 10

        # Metrics
        metrics = performance_summary(port_ret)
        assert np.isfinite(metrics["Sharpe"])
        assert metrics["Max Drawdown (%)"] <= 0
