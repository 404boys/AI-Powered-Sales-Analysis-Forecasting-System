# Forecasting and What-If Analysis

This guide covers the dataset audit, LightGBM training pipeline, recursive future forecast, and interactive Streamlit application.

## 1. Environment setup

Use Python 3.10 or 3.11 for predictable compatibility.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Audit the dataset

Run this from the repository root:

```powershell
python src/data/validate_dataset.py
```

The command writes a Markdown and JSON quality report under `reports/`.

## 3. Train the model

Ensure the cleaned dataset is available at:

`Data/Cleaned/store_sales_cleaned.csv`

Run a smaller local smoke test:

```powershell
python src/modeling/train_lightgbm.py --n-estimators 300
```

For the default evaluation configuration:

```powershell
python src/modeling/train_lightgbm.py
```

The default validation period is the final 42 calendar days. Training writes these local artifacts to `artifacts/model/`:

- `lightgbm_sales_model.joblib`
- `metrics.json`
- `feature_importance.csv`
- `validation_predictions.csv`

Generated model artifacts are excluded from Git by default.

## 4. Run the app

```powershell
streamlit run app/app.py
```

## App sections

- **Overview:** recorded sales trend and store-type comparison.
- **Sales Explorer:** historical sales for a selected store.
- **Future Forecast:** recursive 7-, 14-, or 28-day forecasts. Each predicted day becomes historical input for the next forecast day.
- **What-If Simulator:** change promotion and holiday assumptions for a historical store-day and compare model predictions.
- **Model Performance:** validation metrics, matched-row seven-day baseline comparison, and gain-based feature importance.

## Important modelling notes

- `Customers` is excluded because future customer counts are not known at forecast time.
- Calendar-based lag features are built on a complete daily calendar for each store's observed date span. Missing dates in the cleaned open-day-only file are treated as zero-sales history.
- Validation is one-day-ahead rolling evaluation: actual prior-day sales become available as the next day is predicted. It is not a fixed-origin 42-day forecast.
- The Future Forecast page uses recursive predictions and assumes the store remains open. Promotion and holiday assumptions are held constant across the selected horizon.
- The Future Forecast page is a historical-data demonstration using 2013–2015 data, not a live current-date forecast.
- The What-If tool is a model scenario, not a causal estimate. Historical lag features are held constant while the selected assumptions change.
- The cleaned source contains open, positive-sales records only; overview totals do not include closed-store days.

## Troubleshooting

If the app says the model is missing, run the training command first. If local training is slow on a lower-powered laptop, use `--n-estimators 300` for an initial smoke test, then run the default configuration when possible.
