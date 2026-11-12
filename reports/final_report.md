# AI-Powered Sales Analysis and Forecasting System
## Project Report

## 1. Executive summary

This project analyzes historical retail sales and builds a store-level daily sales forecasting prototype using the Rossmann Store Sales dataset. It combines exploratory data analysis, a reproducible dataset-quality audit, calendar-based feature engineering, a LightGBM regression model, model evaluation, and an interactive Streamlit application.

The application supports historical sales exploration, recursive future forecasts, promotion/holiday What-If scenarios, and model-performance review. The project is an academic prototype; it is not connected to a live retailer system.

## 2. Business problem and objectives

Retail sales vary with promotions, holidays, store characteristics, competition, and recent demand. Forecasting daily store sales can help retailers plan inventory, staffing, and promotional activity.

Project objectives:

1. Explore historical sales patterns and data quality.
2. Build a reproducible feature-engineering pipeline.
3. Train a machine-learning model without using future customer counts or the current target as predictors.
4. Evaluate the model with a chronological holdout and a simple weekly baseline.
5. Present historical data, forecasts, and scenario comparisons in a usable application.

## 3. Dataset profile and quality audit

The committed cleaned dataset is `Data/Cleaned/store_sales_cleaned.csv`.

| Dataset characteristic | Verified value |
|---|---:|
| Rows | 844,338 |
| Columns | 24 |
| Stores | 1,115 |
| Date range | 2013-01-01 to 2015-07-31 |
| Duplicate Store-Date rows | 0 |
| Invalid dates | 0 |
| Invalid, zero, or negative Sales rows | 0 |
| Missing PromoInterval values | 423,292 |
| Unobserved store-days within each store's observed date span | 204,372 |

The missing `PromoInterval` values are retained in the source and treated as the `None` category during feature engineering. They represent stores without a Promo2 interval schedule in this dataset, rather than a missing numeric sales value.

The cleaned file contains open, positive-sales store-days only. It does not contain a complete calendar record of closed stores or open days with zero sales. Therefore, the observed sales totals must not be described as complete all-store calendar totals.

A reproducible audit is available:

```powershell
python src/data/validate_dataset.py
```

It writes `reports/data_quality_audit.md` and `reports/data_quality_audit.json`. Unobserved store-days are reported as descriptive gaps; the source does not tell us whether each gap was a closed day or an open zero-sales day.

## 4. Feature engineering

The feature pipeline is implemented in `src/features/build_features.py`.

It creates:

- Calendar features: year, month, day, day of year, week of year, day of week, and weekend flag.
- Store and business features: store identifier, store type, assortment, promotion, holiday, competition, and Promo2 metadata.
- Historical demand features: one-day, seven-day, and fourteen-day sales lags.
- Rolling demand features: shifted seven-day and fourteen-day means and a shifted seven-day standard deviation.
- Log-scaled competition distance: `log1p(CompetitionDistance)`.

For each store, sales history is reindexed to daily dates within that store's observed span. Missing dates are represented as zero-sales history for lag construction. Rolling calculations shift the history by one day before aggregation, so the current day's target cannot enter its own predictors.

The target `Sales` is never used as a predictor. `Customers` is excluded because future customer counts are not known when a forecast is made. `Open` is excluded because the cleaned dataset contains open, positive-sales records only.

## 5. Machine-learning model

The model is a LightGBM gradient-boosted decision-tree regressor. It is trained on `log1p(Sales)`, and predictions are converted back to the original sales scale with `expm1`.

Training is implemented in `src/modeling/train_lightgbm.py`. The default configuration uses up to 1,200 boosting rounds with early stopping. The final 42 calendar days are reserved chronologically for the default validation run; dates are not randomly shuffled.

The validation design is **one-day-ahead rolling evaluation**. Actual sales from earlier dates in the validation window may be used as history for later validation dates. This is not a fixed-origin 42-day forecast made at the beginning of the validation period.

## 6. Preliminary real-data smoke-test results

The real-data GitHub Actions smoke test successfully trained a small model using 40 boosting rounds and a 14-day validation window. The run used the committed dataset and evaluated 13,420 validation rows from 2015-07-18 to 2015-07-31.

| Metric | 40-round model | Seven-day lag baseline |
|---|---:|---:|
| RMSPE | 13.87% | 36.07% |
| MAE | 700.72 | 2,083.16 |

The model and baseline were evaluated on the same 13,420 rows. The smoke-test run used all 40 requested rounds; it did not establish the final best iteration for the default 1,200-round training configuration.

These values are **preliminary smoke-test results**, not the final full-training result. The final report should be updated after running the default training command locally and reviewing `artifacts/model/metrics.json`. Do not present the smoke-test scores as the final 42-day evaluation.

Metrics used:

- **RMSPE:** root mean squared percentage error, suitable here because validation sales are positive.
- **MAE:** average absolute prediction error in the dataset's sales units.
- **RMSE:** square root of mean squared error, which penalizes larger errors more strongly.

## 7. Streamlit application

Run the app after training a model:

```powershell
streamlit run app/app.py
```

The app contains five pages:

1. **Overview:** recorded sales totals, daily trend, and store-type comparison.
2. **Sales Explorer:** historical sales for a selected store.
3. **Future Forecast:** recursive daily forecasts for 7, 14, or 28 days after the selected store's last recorded date.
4. **What-If Simulator:** changes promotion, school-holiday, and state-holiday assumptions for a historical store-day and compares predictions.
5. **Model Performance:** validation metrics, a matched-row seven-day baseline comparison, and gain-based feature importance.

The Future Forecast page feeds each predicted day's sales into the lag history for subsequent days. It assumes the store remains open and holds the selected promotion and holiday assumptions constant across the horizon. It is a demonstration based on historical 2013–2015 data, not a live 2026 forecast, and it does not currently quantify prediction intervals.

The What-If simulator also holds historical lag features fixed. Its output is a model-based scenario, not a causal estimate of the impact of changing a promotion or holiday.

## 8. Testing and reproducibility

The repository includes unit tests for calendar feature engineering, leakage safeguards, dataset audit calculations, and recursive forecasting.

Run locally:

```powershell
python -m pytest -q
python src/data/validate_dataset.py
python src/modeling/train_lightgbm.py
streamlit run app/app.py
```

GitHub Actions provides:
- Lightweight Python unit tests and syntax checks.
- A real-data smoke workflow that audits the committed CSV, trains a small LightGBM model, and checks all five Streamlit pages.

A successful smoke workflow verifies that the tested pipeline and app paths execute on the committed data. The full default training run and an in-browser demonstration on the student's own machine should still be completed before final submission.

## 9. Limitations and future improvements

1. The cleaned dataset omits closed and zero-sales store-days; the model does not predict whether a store will be open.
2. Missing calendar dates are approximated as zero-sales history for lag features.
3. Validation uses observed prior sales during the validation window and is not a fixed-origin multi-week forecast.
4. Future forecast assumptions are held constant over the selected horizon, and forecast uncertainty is not yet quantified.
5. The What-If simulator is not a causal experiment.
6. The data is historical and does not represent current store operations.

Potential future work includes explicit store-open modelling, known future promotion calendars, holiday-calendar integration, prediction intervals, store-level error diagnostics, and retraining/monitoring workflows.

## 10. Conclusion

The project provides an end-to-end academic workflow from cleaned retail data and exploratory analysis through feature engineering, LightGBM forecasting, evaluation, and an interactive application. Its key differentiators are calendar-correct lag features, leakage safeguards, recursive multi-day forecasting, and an interactive What-If simulator. All reported performance claims should remain tied to reproducible training outputs and the documented limitations of the dataset and validation design.
