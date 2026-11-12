"""Train and evaluate a LightGBM Rossmann sales forecasting model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Make the project src/ directory importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.build_features import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    build_features,
)


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate error metrics for positive sales targets."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.size == 0:
        raise ValueError("Cannot calculate metrics for an empty evaluation set.")
    if np.any(actual <= 0):
        raise ValueError("RMSPE requires strictly positive actual sales values.")

    errors = actual - predicted
    return {
        "RMSPE": float(np.sqrt(np.mean((errors / actual) ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
    }


def validate_input(raw: pd.DataFrame) -> None:
    """Fail early with readable messages when the cleaned CSV is not model-ready."""
    required = {
        "Store", "Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday",
        "StoreType", "Assortment", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Cleaned dataset is missing required columns: {missing}")
    if raw.empty:
        raise ValueError("The cleaned dataset contains no rows.")
    if raw[["Store", "Date"]].duplicated().any():
        raise ValueError("The cleaned dataset has duplicate Store-Date combinations.")
    sales = pd.to_numeric(raw["Sales"], errors="coerce")
    if sales.isna().any() or (sales <= 0).any():
        raise ValueError("Sales must contain positive numeric values only.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a chronological LightGBM sales forecasting model."
    )
    parser.add_argument(
        "--data",
        default="Data/Cleaned/store_sales_cleaned.csv",
        help="Path to the cleaned Rossmann CSV file.",
    )
    parser.add_argument("--output-dir", default="artifacts/model")
    parser.add_argument("--validation-days", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1200)
    args = parser.parse_args()

    if args.validation_days < 1:
        parser.error("--validation-days must be at least 1.")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1.")

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {data_path.resolve()}\n"
            "Expected Data/Cleaned/store_sales_cleaned.csv from the repository root."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cleaned data from: {data_path.resolve()}")
    raw = pd.read_csv(data_path, low_memory=False)
    validate_input(raw)

    print("Building calendar-based lag and rolling features...")
    featured = build_features(raw)
    missing_features = sorted(set(FEATURE_COLUMNS).difference(featured.columns))
    if missing_features:
        raise ValueError(f"Feature engineering did not create: {missing_features}")

    # Reserve the final N calendar days for validation; never randomly shuffle dates.
    last_date = featured["Date"].max()
    validation_start = last_date - pd.Timedelta(days=args.validation_days - 1)
    train = featured.loc[featured["Date"] < validation_start].copy()
    valid = featured.loc[featured["Date"] >= validation_start].copy()

    if train.empty or valid.empty:
        raise ValueError(
            "The selected validation window leaves an empty train or validation set. "
            "Reduce --validation-days or check the dataset date range."
        )

    # Convert categorical predictors consistently for LightGBM.
    for column in CATEGORICAL_COLUMNS:
        train[column] = train[column].astype("category")
        valid[column] = pd.Categorical(
            valid[column],
            categories=train[column].cat.categories,
        )

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
        importance_type="gain",
    )

    print(
        f"Training on {len(train):,} rows; validating on {len(valid):,} rows "
        f"({validation_start:%Y-%m-%d} to {last_date:%Y-%m-%d})."
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    # Convert log-scale predictions back to the original sales scale.
    predictions = np.maximum(
        0,
        np.expm1(model.predict(X_valid, num_iteration=model.best_iteration_)),
    )
    actual = valid["Sales"].to_numpy()
    metrics = calculate_metrics(actual, predictions)

    # Seasonal baseline: same store's sales seven calendar days earlier.
    lag_7 = valid["Sales_Lag_7"].to_numpy(dtype=float)
    baseline_mask = np.isfinite(lag_7)
    if baseline_mask.any():
        # Compare model and baseline on the exact same rows. Some stores may
        # not have seven days of history at the start of the validation window.
        matched_model_metrics = calculate_metrics(
            actual[baseline_mask],
            predictions[baseline_mask],
        )
        baseline_metrics = calculate_metrics(
            actual[baseline_mask],
            lag_7[baseline_mask],
        )
        metrics["Model_On_BaselineRows_RMSPE"] = matched_model_metrics["RMSPE"]
        metrics["Model_On_BaselineRows_MAE"] = matched_model_metrics["MAE"]
        metrics["Baseline_7DayLag_RMSPE"] = baseline_metrics["RMSPE"]
        metrics["Baseline_7DayLag_MAE"] = baseline_metrics["MAE"]
        metrics["baseline_rows"] = int(baseline_mask.sum())
    else:
        metrics["Model_On_BaselineRows_RMSPE"] = None
        metrics["Model_On_BaselineRows_MAE"] = None
        metrics["Baseline_7DayLag_RMSPE"] = None
        metrics["Baseline_7DayLag_MAE"] = None
        metrics["baseline_rows"] = 0

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
            "data_rows": int(len(raw)),
            "store_count": int(raw["Store"].nunique()),
            "data_start": pd.to_datetime(raw["Date"]).min().strftime("%Y-%m-%d"),
            "data_end": pd.to_datetime(raw["Date"]).max().strftime("%Y-%m-%d"),
        }
    )

    # Save reproducible outputs locally; model binaries are not committed to GitHub.
    joblib.dump(model, output_dir / "lightgbm_sales_model.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv",
        index=False,
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
