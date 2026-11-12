# Project Architecture

## End-to-end flow

```mermaid
flowchart TD
    A[Cleaned Rossmann CSV] --> B[Input validation]
    B --> C[Calendar feature engineering]
    C --> D[Chronological train and validation split]
    D --> E[LightGBM training with early stopping]
    E --> F[Validation metrics and baseline]
    E --> G[Saved model artifact]
    F --> H[Model Performance page]
    G --> I[Streamlit application]
    C --> I
    I --> J[Overview]
    I --> K[Sales Explorer]
    I --> L[Future Forecast]
    I --> M[What-If Simulator]
    I --> H
```

## Main modules

| Path | Responsibility |
|---|---|
| `Data/Cleaned/store_sales_cleaned.csv` | Cleaned store-day source data |
| `src/data/validate_dataset.py` | Reproducible CSV quality audit with Markdown and JSON outputs |
| `src/features/build_features.py` | Date handling, calendar history, lag/rolling features, and feature validation |
| `src/modeling/train_lightgbm.py` | Input checks, chronological split, model training, evaluation, and artifact saving |
| `src/modeling/predict.py` | Shared model-row preparation and inverse-log prediction helper |
| `src/modeling/forecast.py` | Recursive multi-day forecast generation using predicted sales as later lag history |
| `app/app.py` | Streamlit interface, historical exploration, forecasting, and What-If scenarios |
| `tests/test_features.py` | Unit tests for calendar lags, leakage prevention, and invalid inputs |
| `tests/test_dataset_validation.py` | Unit tests for dataset audit calculations and report output |
| `artifacts/model/` | Local model, metrics, feature importance, and validation predictions |

## Training outputs

After training, the following files are generated locally:

- `lightgbm_sales_model.joblib`: fitted model.
- `metrics.json`: evaluation metrics and validation metadata.
- `feature_importance.csv`: model feature importance values.
- `validation_predictions.csv`: actual and predicted values for validation rows.

These are generated artifacts and should not be committed by default. The model can be recreated from the source code and cleaned dataset.

## Reproducibility

Run commands from the repository root:

```powershell
python -m pytest -q
python src/modeling/train_lightgbm.py --n-estimators 300
python src/modeling/train_lightgbm.py
streamlit run app/app.py
```

The 300-estimator command is a first local smoke test. The default training run uses up to 1,200 estimators with early stopping. GitHub Actions also has a real-data smoke workflow that trains the default model on the committed CSV and checks all five app pages.

## Validation boundaries

The training split is chronological. Validation is rolling one-day-ahead: earlier actual validation-day sales may become lag inputs for subsequent validation dates. Do not describe it as a fixed-origin 42-day-ahead forecast.

The data is open-positive-only, and missing dates inside a store's observed span are treated as zero-sales history. Refer to the model card for the implications.
