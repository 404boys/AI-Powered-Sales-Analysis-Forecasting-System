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

Training is implemented in `src/modeling/train_lightgbm.py`. The default configuration uses up to 1,200 boosting rounds with early stopping. The final 42 calendar days are reserved chronologically for validation; dates are not randomly shuffled.

The validation design is **one-day-ahead rolling evaluation**. Actual sales from earlier dates in the validation window may be used as history for later validation dates. This is not a fixed-origin 42-day forecast made at the beginning of the validation period.

## 6. Full default model evaluation

The default training configuration completed successfully in GitHub Actions using the committed cleaned dataset. It trained a LightGBM model with 1,200 boosting rounds and evaluated the final 42 calendar days.

| Evaluation detail | Verified value |
|---|---:|
| Training rows | 804,056 |
| Validation rows | 40,282 |
| Validation period | 2015-06-20 to 2015-07-31 |
| Boosting rounds used | 1,200 |
| Best iteration | 1,200 |
| Model family | LightGBMRegressor |
| Input features | 27 |

Early stopping did not trigger before the configured maximum; the run used all 1,200 rounds.

### Model versus seven-day lag baseline

| Metric | LightGBM model | Seven-day lag baseline |
|---|---:|---:|
| RMSPE | **11.14%** | 41.02% |
| MAE | **539.12** | 2,191.87 |
| Evaluation rows | 40,282 | 40,282 |

Both model and baseline metrics were calculated on the exact same 40,282 validation rows. MAE values are in the dataset's sales units.

The model outperformed the simple weekly baseline on both RMSPE and MAE in this validation run. These are measured results from the full default training configuration, not estimates. The run's metrics and model artifacts were uploaded as the `rossmann-full-model-and-evaluation` GitHub Actions artifact.

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

The successful real-data GitHub Actions workflow:
- Runs the unit test suite.
- Audits the committed cleaned CSV.
- Trains the default 1,200-round model with a 42-day chronological validation window.
- Opens all five Streamlit pages through automated app tests.
- Uploads the trained model, metrics, feature importance, validation predictions, and audit outputs as a workflow artifact.

To reproduce locally from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python src/data/validate_dataset.py
python src/modeling/train_lightgbm.py
streamlit run app/app.py
```

The generated model files are excluded from Git by default. A successful CI run verifies the full tested pipeline on the committed dataset; the student should still launch the app locally and rehearse the demonstration on the presentation machine.

## 9. Limitations and future improvements

1. The cleaned dataset omits closed and zero-sales store-days; the model does not predict whether a store will be open.
2. Missing calendar dates are approximated as zero-sales history for lag features.
3. Validation uses observed prior sales during the validation window and is not a fixed-origin multi-week forecast.
4. Future forecast assumptions are held constant over the selected horizon, and forecast uncertainty is not yet quantified.
5. The What-If simulator is not a causal experiment.
6. The data is historical and does not represent current store operations.

Potential future work includes explicit store-open modelling, known future promotion calendars, holiday-calendar integration, prediction intervals, store-level error diagnostics, and retraining/monitoring workflows.

## 10. Conclusion

The project provides an end-to-end academic workflow from cleaned retail data and exploratory analysis through feature engineering, LightGBM forecasting, evaluation, and an interactive application. Its key differentiators are calendar-correct lag features, leakage safeguards, recursive multi-day forecasting, and an interactive What-If simulator. The full default model training and all five application-page checks have completed successfully, with the dataset and validation limitations documented.
