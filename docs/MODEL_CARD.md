# Model Card — Rossmann Daily Sales Forecasting

## Model identity

- **Project:** AI-Powered Sales Analysis and Forecasting System
- **Model family:** LightGBM gradient-boosted decision trees
- **Task:** Regression; estimate daily sales for a store-day
- **Target:** `Sales`, trained on `log1p(Sales)`
- **Training entry point:** `src/modeling/train_lightgbm.py`
- **Feature builder:** `src/features/build_features.py`

## Intended use

This is an academic prototype for exploring store-level retail sales patterns and demonstrating a machine-learning forecasting workflow. It can support classroom demonstrations of feature engineering, chronological validation, model evaluation, and scenario exploration.

It is not a production inventory, staffing, financial, or promotion-optimization system. Predictions should not be used as guaranteed business outcomes.

## Data and scope

The pipeline expects the cleaned Rossmann CSV at `Data/Cleaned/store_sales_cleaned.csv`. The development dataset contains store-day records with positive sales and open stores. The data covers multiple stores and dates, and includes calendar, promotion, holiday, store-type, assortment, competition, and Promo2 information.

The training script validates required columns, positive numeric sales, and unique Store-Date records.

## Predictors and leakage controls

The feature set includes:
- Store and calendar characteristics.
- Promotion and holiday indicators.
- Store, assortment, competition, and Promo2 metadata.
- Previous-day, previous-week, and previous-fortnight sales.
- Shifted rolling means and standard deviation.

The target `Sales` is not a predictor. `Customers` is excluded because future customer counts are not known at forecast time. `Open` is excluded because the cleaned data contains only open, positive-sales rows.

Calendar history is reindexed to daily dates within each store's observed span. Missing dates are assigned zero sales for lag construction. The source data cannot distinguish a closed day from an open zero-sales day omitted during cleaning, so this is an explicit approximation.

Rolling features shift the history by one day before aggregation, preventing the current day's target from entering its own predictors.

## Validation protocol

The default evaluation reserves the final 42 calendar days chronologically. It does not randomly shuffle dates.

This is **one-day-ahead rolling validation**: actual sales from earlier dates in the validation window can become historical inputs for later validation days. It is not a fixed-origin forecast of all 42 days made at the start of the window.

Metrics reported by the training script:
- RMSPE
- MAE
- RMSE
- Seven-day-lag baseline RMSPE and MAE

Only values generated in the local `artifacts/model/metrics.json` after a successful training run should be reported. No benchmark values are pre-filled in this model card.

## What-If Simulator

The app lets a user change promotion, school-holiday, and state-holiday values for a selected historical store-day and compare model predictions. Historical lag features remain fixed.

This is a model-based sensitivity scenario, not a causal estimate. The result does not prove that changing a promotion or holiday would cause the displayed sales difference.

## Known limitations

1. The cleaned file omits closed and zero-sales days; the model does not estimate whether a store will be open.
2. Missing dates are treated as zero-sales history for lag construction.
3. The validation protocol uses observed previous sales during the validation period and therefore represents rolling one-day-ahead evaluation, not a multi-week fixed-origin forecast.
4. The dataset is historical and may not represent current store operations or market conditions.
5. Feature importance is model-specific and should not be interpreted as causal influence.
6. The What-If simulator is not a causal experiment.

## Before any real-world use

A production system would require a data pipeline that preserves all store-calendar dates, explicit store-open status, a forecast horizon matching the business decision, monitoring for drift, retraining controls, error analysis by store and season, and review by domain experts.
