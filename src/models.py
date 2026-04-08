"""
models.py
=========
Machine-learning models for return and volatility prediction.

Models implemented
------------------
1. LinearBaseline     — Ridge / Lasso regression (regularised linear)
2. RandomForestModel  — Random Forest (bagging + feature subsampling)
3. GBMModel           — Gradient-Boosted Trees (XGBoost / LightGBM)
4. EnsembleModel      — Simple average of multiple base estimators

All models follow the scikit-learn API (fit / predict / score).

Training protocol
-----------------
- TimeSeriesSplit cross-validation for hyperparameter search
  (no random K-fold which would cause data leakage on time series)
- StandardScaler applied inside each fold to prevent leakage
- Early stopping for GBM models

Course: AI in Finance — Nicolas de Roux & Mohamed El Fakir
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: time-aware cross-validation scorer
# ---------------------------------------------------------------------------

def ts_cross_validate(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    metric: str = "r2",
) -> Dict[str, Any]:
    """
    Walk-forward (TimeSeriesSplit) cross-validation.

    Uses sklearn's TimeSeriesSplit which ensures that training data always
    precedes validation data — correct protocol for financial time series.

    Parameters
    ----------
    estimator : sklearn Pipeline
    X, y      : features and target
    n_splits  : number of CV folds
    metric    : 'r2' or 'mse'

    Returns
    -------
    dict with 'scores', 'mean', 'std'
    """
    tss    = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tss.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        estimator.fit(X_tr, y_tr)
        pred = estimator.predict(X_va)

        if metric == "r2":
            s = r2_score(y_va, pred)
        else:
            s = -mean_squared_error(y_va, pred, squared=False)

        scores.append(s)
        log.debug(f"  Fold {fold+1}/{n_splits}  {metric}={s:.4f}")

    return {"scores": scores, "mean": np.mean(scores), "std": np.std(scores)}


# ---------------------------------------------------------------------------
# 1. Ridge / Lasso / ElasticNet (regularised linear)
# ---------------------------------------------------------------------------

def make_linear_model(
    model_type: str = "ridge",
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> Pipeline:
    """
    Regularised linear regression pipeline.

    Ridge (L2):  shrinks all coefficients — good baseline, handles multicollinearity.
    Lasso (L1):  performs variable selection (sparse solution).
    ElasticNet:  blend of L1 + L2 — recommended when features are correlated.

    Parameters
    ----------
    model_type : 'ridge' | 'lasso' | 'elasticnet'
    alpha      : regularisation strength
    l1_ratio   : ElasticNet mixing parameter (0 = Ridge, 1 = Lasso)
    """
    if model_type == "ridge":
        reg = Ridge(alpha=alpha)
    elif model_type == "lasso":
        reg = Lasso(alpha=alpha, max_iter=5000)
    elif model_type == "elasticnet":
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  reg),
    ])


# ---------------------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------------------

def make_random_forest(
    n_estimators: int = 300,
    max_depth: Optional[int] = 8,
    max_features: float = 0.5,
    min_samples_leaf: int = 20,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """
    Random Forest regressor pipeline.

    Key ideas (from course):
    - Bagging: each tree trained on a bootstrap sample → variance reduction
    - Feature subsampling at each split → decorrelates trees
    - Ensemble of many trees → stable, low-variance predictions
    - Hyperparameters tuned via TimeSeriesSplit CV

    Parameters
    ----------
    n_estimators    : number of trees
    max_depth       : max tree depth (None = fully grown)
    max_features    : fraction of features considered at each split
    min_samples_leaf: min samples per leaf (regularisation)
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("model", rf)])


# ---------------------------------------------------------------------------
# 3. Gradient Boosting (XGBoost / LightGBM / sklearn)
# ---------------------------------------------------------------------------

def make_gbm(
    library: str = "xgboost",
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.6,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Gradient Boosting regression pipeline.

    Key ideas (from course):
    - Boosting: sequential fitting, each tree corrects residuals of previous ones
    - Reduces bias (vs. bagging which reduces variance)
    - Regularisation via learning rate, max_depth, subsampling

    Parameters
    ----------
    library : 'xgboost' | 'lightgbm' | 'sklearn'
    """
    if library == "xgboost" and HAS_XGB:
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
        )
    elif library == "lightgbm" and HAS_LGB:
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    else:
        log.info(f"Falling back to sklearn GradientBoostingRegressor")
        model = GradientBoostingRegressor(
            n_estimators=min(n_estimators, 200),
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )

    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# ---------------------------------------------------------------------------
# 4. Ensemble (simple model averaging)
# ---------------------------------------------------------------------------

class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Simple averaging ensemble of multiple base estimators.

    Combining diverse models reduces both bias and variance
    compared to any single model.
    """

    def __init__(self, estimators: List[Tuple[str, Pipeline]]):
        self.estimators = estimators

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleRegressor":
        for name, est in self.estimators:
            log.debug(f"  Fitting {name} …")
            est.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.stack(
            [est.predict(X) for _, est in self.estimators], axis=1
        )
        return preds.mean(axis=1)


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

def get_all_models() -> Dict[str, Pipeline]:
    """
    Return all models for comparison as a dictionary {name: pipeline}.
    """
    models: Dict[str, Any] = {
        "Ridge":      make_linear_model("ridge",      alpha=0.1),
        "Lasso":      make_linear_model("lasso",      alpha=0.001),
        "ElasticNet": make_linear_model("elasticnet", alpha=0.01, l1_ratio=0.5),
        "RandomForest": make_random_forest(
            n_estimators=200, max_depth=6, max_features=0.5, min_samples_leaf=30
        ),
        "GBM": make_gbm(
            library="xgboost" if HAS_XGB else "sklearn",
            n_estimators=300, learning_rate=0.05, max_depth=3
        ),
    }

    # Ensemble of the best non-linear models
    ensemble_base = [
        ("rf",  make_random_forest(n_estimators=200, max_depth=6)),
        ("gbm", make_gbm(n_estimators=300, learning_rate=0.05, max_depth=3)),
        ("en",  make_linear_model("elasticnet", alpha=0.01)),
    ]
    models["Ensemble"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  EnsembleRegressor(ensemble_base)),
    ])
    # Note: scaler inside EnsembleRegressor → remove outer one
    models["Ensemble"] = EnsembleRegressor(ensemble_base)

    return models


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_and_evaluate(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_cv_splits: int = 5,
) -> pd.DataFrame:
    """
    Fit all models, run time-series CV, and evaluate on held-out test set.

    Returns a summary DataFrame with CV and test metrics.
    """
    records = []
    fitted  = {}

    for name, model in models.items():
        log.info(f"Training {name} …")

        # Time-series cross-validation (in-sample)
        cv_res = ts_cross_validate(
            model if hasattr(model, "fit") else model,
            X_train, y_train,
            n_splits=n_cv_splits,
        )

        # Refit on full training set
        model.fit(X_train, y_train)
        fitted[name] = model

        # Out-of-sample test performance
        pred_test = model.predict(X_test)
        test_r2   = r2_score(y_test, pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

        # IC (Information Coefficient) = rank correlation between pred and actual
        ic = pd.Series(pred_test).corr(pd.Series(y_test.values), method="spearman")

        records.append({
            "model":    name,
            "cv_r2":    cv_res["mean"],
            "cv_r2_std":cv_res["std"],
            "test_r2":  test_r2,
            "test_rmse":test_rmse,
            "ic":       ic,
        })
        log.info(
            f"  {name:15s}  CV R²={cv_res['mean']:+.4f}±{cv_res['std']:.4f}"
            f"  Test R²={test_r2:+.4f}  IC={ic:.4f}"
        )

    results = pd.DataFrame(records).set_index("model")
    return results, fitted


def get_feature_importance(model: Pipeline, feature_names: List[str]) -> pd.Series:
    """
    Extract feature importances from tree-based models.

    Falls back to coefficient magnitude for linear models.
    """
    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=feature_names)
    elif hasattr(estimator, "coef_"):
        imp = pd.Series(np.abs(estimator.coef_), index=feature_names)
    else:
        raise ValueError(f"Model {type(estimator).__name__} has no feature importances")

    return imp.sort_values(ascending=False)
