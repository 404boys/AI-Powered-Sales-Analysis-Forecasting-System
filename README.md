# AI-Powered Sales Analysis and Forecasting System

An end-to-end retail analytics and machine-learning project using the Rossmann Store Sales dataset. The project combines data preparation and exploratory analysis with a LightGBM sales forecasting model and an interactive Streamlit application.

## Project goals

1. Understand historical store-level sales patterns.
2. Prepare a reliable, analysis-ready dataset.
3. Engineer calendar, store, promotion, and historical-sales features.
4. Train and evaluate a machine-learning model using a chronological validation split.
5. Explore store-level predictions and interactive promotion/holiday scenarios.

## Main features

- **Data analysis:** cleaning, quality checks, exploratory analysis, and visualizations (see the existing notebooks and analysis folders).
- **Forecasting:** LightGBM regression on a log-transformed sales target.
- **Time-aware features:** calendar variables, prior-day/prior-week/prior-fortnight sales, and shifted rolling statistics.
- **Model evaluation:** RMSPE, MAE, RMSE, and a 7-day-lag baseline.
- **Interactive website:** Streamlit pages for an overview, store-level sales exploration, What-If simulation, and model performance.
- **What-If analysis:** compare model predictions after changing promotion and holiday assumptions for a selected historical store-day.

## Repository structure

```text
.
├── Data/
│   └── Cleaned/
│       └── store_sales_cleaned.csv
├── Notebooks/                 # Existing data analysis notebooks
├── app/
│   └── app.py                 # Streamlit application
├── reports/
│   └── model_evaluation.md
├── src/
│   ├── features/
│   │   └── build_features.py
│   └── modeling/
│       └── train_lightgbm.py
├── tests/
│   └── test_features.py
├── README.md
├── README_FORECASTING.md
└── requirements.txt
```

## Dataset

The forecasting pipeline expects the cleaned CSV at:

`Data/Cleaned/store_sales_cleaned.csv`

Expected fields include store/date/sales, promotion and holiday indicators, store type and assortment, competition metadata, and Promo2 metadata. The training script validates required columns, positive sales, and unique Store-Date rows before training.

The cleaned file used during development contains open, positive-sales store-days. Consequently, dashboard totals represent recorded rows and must not be interpreted as a complete total including closed stores.

## Setup (Windows PowerShell)

Open the repository root in VS Code. Then run:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 or 3.11 is recommended for the most predictable compatibility with the ML ecosystem. If you already have another supported Python version and package installation succeeds, try the smoke test before changing your installation.

## Train the forecasting model

Run from the repository root:

```powershell
python src/modeling/train_lightgbm.py --n-estimators 300
```

This smaller run is a first smoke test. After it completes successfully, run the full training configuration:

```powershell
python src/modeling/train_lightgbm.py
```

Optional arguments:

- `--data`: path to the cleaned CSV.
- `--output-dir`: output directory for local model artifacts.
- `--validation-days`: number of final calendar days reserved for validation (default: 42).
- `--n-estimators`: maximum number of boosting rounds (default: 1200; early stopping may stop sooner).

Training creates these local files in `artifacts/model/`:

- `lightgbm_sales_model.joblib`
- `metrics.json`
- `feature_importance.csv`
- `validation_predictions.csv`

Generated model binaries and predictions are ignored by Git and are not committed by default.

## Run the website

After a successful training run:

```powershell
streamlit run app/app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). Open it in your browser.

### Website pages

| Page | Purpose |
|---|---|
| Overview | Recorded sales trend, store-type comparison, and headline dataset counts |
| Sales Explorer | Historical sales and selected operating details for one store |
| What-If Simulator | Compare original and adjusted model predictions for promotion and holiday inputs |
| Model Performance | Validation metrics and top model feature importances |

## Modelling approach

- The target is `Sales`, transformed with `log1p` for training and converted back with `expm1` for predictions.
- `Customers` is excluded because future customer counts are not known when forecasting.
- `Sales` is the target and is never included as a predictor.
- `Open` is excluded because the cleaned file contains open, positive-sales records only.
- The final 42 calendar days are held out chronologically; dates are not randomly shuffled.
- Validation is **one-day-ahead rolling evaluation**. Actual sales from earlier dates, including earlier dates within the validation window, can contribute to later validation-day lag features. This is not a fixed-origin 42-day forecast.
- Missing dates within each store's observed date span are treated as zero-sales history days when calendar lags are built. The cleaned data cannot distinguish a closed day from an open day omitted due to zero sales; this assumption is documented in the evaluation report.
- The What-If simulator holds historical lag features fixed. It is a model scenario, not a causal estimate of the effect of a promotion or holiday.

## Tests

Run the lightweight feature-engineering tests:

```powershell
python -m pytest -q
```

GitHub Actions also runs the test suite and a Python syntax compilation check on pushes and pull requests.

## Project status

The analysis and forecasting/application code are in the repository. Model performance values should be reported only after running the training script and reviewing the generated `artifacts/model/metrics.json`. A successful GitHub code commit does not, by itself, mean the model has been trained or the website has been run.

## Evaluation report

See [reports/model_evaluation.md](reports/model_evaluation.md) for the validation protocol, metrics to record, and modelling limitations.
