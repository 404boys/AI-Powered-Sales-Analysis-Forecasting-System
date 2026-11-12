# AI-Powered Sales Analysis and Forecasting System

An end-to-end retail analytics and machine-learning project using the Rossmann Store Sales dataset. It combines exploratory analysis, reproducible data-quality checks, LightGBM forecasting, and an interactive Streamlit application.

## Project goals

1. Understand historical store-level sales patterns.
2. Validate and prepare an analysis-ready dataset.
3. Engineer calendar, store, promotion, and historical-sales features.
4. Train and evaluate a machine-learning model with chronological validation.
5. Explore historical sales, future forecasts, and promotion/holiday scenarios.

## Main features

- **Data analysis:** data cleaning, exploratory analysis, and visualizations.
- **Dataset audit:** a Python CLI that exports Markdown and JSON quality reports.
- **Forecasting:** LightGBM regression on a log-transformed sales target.
- **Time-aware features:** prior-day, prior-week, prior-fortnight, and shifted rolling sales features.
- **Recursive future forecasting:** 7-, 14-, or 28-day forecasts that feed predicted sales into subsequent lag features.
- **Model evaluation:** RMSPE, MAE, RMSE, a matched-row seven-day-lag baseline comparison, and gain-based feature importance.
- **Interactive website:** Overview, Sales Explorer, Future Forecast, What-If Simulator, and Model Performance pages.

## Repository structure

```text
.
├── Data/
│   └── Cleaned/
│       └── store_sales_cleaned.csv
├── Notebooks/
│   ├── deep_analysis.ipynb
│   └── superstore_sales_eda.ipynb
├── app/
│   └── app.py
├── docs/
│   ├── MODEL_CARD.md
│   ├── PROJECT_ARCHITECTURE.md
│   ├── PROJECT_DEMO_GUIDE.md
│   ├── VIVA_QUESTIONS.md
│   └── WEBSITE_DEPLOYMENT.md
├── reports/
│   ├── final_report.md
│   └── model_evaluation.md
├── src/
│   ├── data/
│   │   └── validate_dataset.py
│   ├── features/
│   │   └── build_features.py
│   └── modeling/
│       ├── forecast.py
│       ├── predict.py
│       └── train_lightgbm.py
├── tests/
│   ├── test_dataset_validation.py
│   ├── test_features.py
│   └── test_forecast.py
├── .github/workflows/
│   ├── python-ci.yml
│   └── model-smoke-test.yml
├── README.md
├── README_FORECASTING.md
└── requirements.txt
```

## Dataset

The forecasting pipeline expects the cleaned Rossmann CSV at `Data/Cleaned/store_sales_cleaned.csv`. The committed cleaned data contains open, positive-sales store-days. Dashboard totals therefore represent recorded rows and are not complete calendar totals including closed stores.

## Setup (Windows PowerShell)

Run these commands from the repository root:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended for predictable compatibility with the machine-learning dependencies.

## 1. Run the dataset quality audit

```powershell
python src/data/validate_dataset.py
```

This creates `reports/data_quality_audit.md` and `reports/data_quality_audit.json`. The audit reports missing values, duplicate Store-Date keys, invalid/zero/negative sales, date coverage, and unobserved store-days within each store's date span. Since the cleaned source contains only open, positive-sales rows, an unobserved store-day is a descriptive gap—not automatically a data error.

## 2. Run the automated tests

```powershell
python -m pytest -q
```

GitHub Actions runs unit tests and Python syntax checks on pushes and pull requests. A separate real-data smoke-test workflow audits the committed dataset, trains a small model, and checks all five Streamlit pages.

## 3. Train the forecasting model

First run a smaller local smoke test:

```powershell
python src/modeling/train_lightgbm.py --n-estimators 300
```

Then run the full training configuration:

```powershell
python src/modeling/train_lightgbm.py
```

Optional arguments:

- `--data`: path to the cleaned CSV.
- `--output-dir`: output directory for local model artifacts.
- `--validation-days`: number of final calendar days reserved for validation (default: 42).
- `--n-estimators`: maximum boosting rounds (default: 1200; early stopping may stop sooner).

Training creates local files in `artifacts/model/`:

- `lightgbm_sales_model.joblib`
- `metrics.json`
- `feature_importance.csv`
- `validation_predictions.csv`

These generated model files are excluded from Git by default.

## 4. Run the website

After training completes successfully:

```powershell
streamlit run app/app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

| Page | Purpose |
|---|---|
| Overview | Recorded sales trends and store-type comparisons |
| Sales Explorer | Historical sales for a selected store |
| Future Forecast | Recursive 7-, 14-, or 28-day store forecasts using selected assumptions |
| What-If Simulator | Compare predictions after changing promotion and holiday assumptions |
| Model Performance | Validation metrics, matched baseline comparison, and feature importance |

The Future Forecast page assumes the selected store stays open and holds the chosen promotion and holiday assumptions constant over the horizon. It is a demonstration using historical 2013–2015 data, not a live 2026 forecast. The What-If page is a model scenario, not a causal estimate.

## Modelling approach and limitations

- The target is `Sales`, transformed with `log1p` for training and converted back with `expm1` for predictions.
- `Customers` is excluded because future customer counts are not known at forecast time.
- `Sales` is never used as a predictor, and `Open` is excluded because the cleaned data contains only open, positive-sales rows.
- The final 42 calendar days are held out chronologically; dates are not randomly shuffled.
- Validation is **one-day-ahead rolling evaluation**. Actual sales from earlier dates, including earlier dates within the validation window, can contribute to later validation-day lag features. This is not a fixed-origin 42-day forecast.
- Missing dates within each store's observed span are treated as zero-sales history days for lag construction. The source cannot distinguish closed days from open days omitted due to zero sales.
- Feature importance is model-specific and does not establish causation.

## Project status

The repository includes the cleaned dataset, analysis notebooks, data audit, feature pipeline, recursive forecasting code, tests, Streamlit app, and project documentation. A 40-round, 14-day real-data smoke test has completed; its preliminary metrics are in [the model evaluation report](reports/model_evaluation.md). Run the default 1,200-round/42-day training locally before presenting final model results.

## Further documentation

- [Forecasting guide](README_FORECASTING.md)
- [Final project report](reports/final_report.md)
- [Model card](docs/MODEL_CARD.md)
- [Architecture](docs/PROJECT_ARCHITECTURE.md)
- [Demo guide](docs/PROJECT_DEMO_GUIDE.md)
- [Viva questions](docs/VIVA_QUESTIONS.md)
- [Website deployment](docs/WEBSITE_DEPLOYMENT.md)
- [Model evaluation report](reports/model_evaluation.md)
