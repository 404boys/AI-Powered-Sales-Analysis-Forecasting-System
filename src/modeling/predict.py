"""Shared prediction helpers for trained LightGBM sales models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.build_features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS


def prepare_model_row(row: pd.Series, model) -> pd.DataFrame:
    """Match the feature order and categorical levels used during training."""
    missing = sorted(set(FEATURE_COLUMNS).difference(row.index))
    if missing:
        raise ValueError(f"Prediction row is missing model features: {missing}")

    X = pd.DataFrame([{column: row[column] for column in FEATURE_COLUMNS}])
    saved_categories = getattr(model.booster_, "pandas_categorical", None) or []
    category_position = 0

    for column in CATEGORICAL_COLUMNS:
        if column in X.columns:
            if category_position < len(saved_categories):
                levels = saved_categories[category_position]
                X[column] = pd.Categorical(X[column], categories=levels)
            else:
                X[column] = X[column].astype("category")
            category_position += 1

    return X


def predict_sales(model, row: pd.Series) -> float:
    """Predict sales and reverse the log1p target transformation."""
    X = prepare_model_row(row, model)
    prediction = model.predict(X, num_iteration=model.best_iteration_)
    return float(max(0.0, np.expm1(prediction[0])))
