from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def evaluate_regression(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Compute regression metrics for a fitted model."""
    predictions = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, predictions),
        "MSE": mean_squared_error(y, predictions),
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "R2": r2_score(y, predictions),
    }


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a baseline linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int | None = None,
) -> RandomForestRegressor:
    """Train a random forest regressor for non-linear prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 5,
) -> XGBRegressor:
    """Train an XGBoost regressor for gradient-boosted prediction."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: Path) -> Path:
    """Persist a model to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> Any:
    """Load a persisted model from disk."""
    return joblib.load(path)
