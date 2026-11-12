"""Train and evaluate a LightGBM Rossmann sales forecasting model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

# Make the project src/ directory importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.build_features import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_features,
)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate error metrics; RMSPE is valid here because targets are positive."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "RMSPE": float(np.sqrt(np.mean(((actual - predicted) / actual) ** 2))),
        "MAE": float(np.mean(np.abs(actual - predicted))),
        "RMSE": float(np.sqrt(np.mean((actual - predicted) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="Data/Cleaned/store_sales_cleaned.csv",
        help="Path to the cleaned Rossmann CSV file.",
    )
    parser.add_argument("--output-dir", default="artifacts/model")
    parser.add_argument("--validation-days", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1200)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare the data.
    raw = pd.read_csv(data_path, low_memory=False)
    featured = build_features(raw)

    # Reserve the final N calendar days for validation; never randomly shuffle dates.
    last_date = featured["Date"].max()
    validation_start = last_date - pd.Timedelta(days=args.validation_days - 1)
    train = featured.loc[featured["Date"] < validation_start].copy()
    valid = featured.loc[featured["Date"] >= validation_start].copy()

    if train.empty or valid.empty:
        raise ValueError("The selected validation window leaves an empty train or validation set.")

    # Convert categorical predictors consistently for LightGBM.
    for column in CATEGORICAL_COLUMNS:
        train[column] = train[column].astype("category")
        valid[column] = valid[column].astype("category")

    X_train = train[FEATURE_COLUMNS]
    X_valid = valid[FEATURE_COLUMNS]
    y_train = np.log1p(train["Sales"])
    y_valid = np.log1p(valid["Sales"])

    # Customers is deliberately excluded: future customer counts are not known
    # when a real forecast is made. Sales is the target, never a predictor.
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(100)],
    )

    # Convert log-scale predictions back to the original sales scale.
    predictions = np.maximum(
        0,
        np.expm1(model.predict(X_valid, num_iteration=model.best_iteration_)),
    )
    actual = valid["Sales"].to_numpy()
    metrics = calculate_metrics(actual, predictions)

    # A simple seasonal baseline: use the same store's sales seven calendar days ago.
    lag_7 = valid["Sales_Lag_7"].to_numpy()
    baseline_mask = np.isfinite(lag_7)
    baseline_metrics = calculate_metrics(actual[baseline_mask], lag_7[baseline_mask])
    metrics["Baseline_7DayLag_RMSPE"] = baseline_metrics["RMSPE"]
    metrics["Baseline_7DayLag_MAE"] = baseline_metrics["MAE"]

    metrics.update(
        {
            "validation_start": validation_start.strftime("%Y-%m-%d"),
            "validation_end": last_date.strftime("%Y-%m-%d"),
            "validation_days": int(args.validation_days),
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "forecast_type": "one-day-ahead rolling validation using observed prior sales",
            "excluded_predictors": ["Customers", "Sales", "Open"],
        }
    )

    # Save reproducible outputs locally; model artifacts should not be committed
    # to GitHub unless the team intentionally chooses to version large binaries.
    joblib.dump(model, output_dir / "lightgbm_sales_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "Date": valid["Date"].to_numpy(),
            "Store": valid["Store"].to_numpy(),
            "Actual_Sales": actual,
            "Predicted_Sales": predictions,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)

    print("\nValidation metrics")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and evaluation files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
