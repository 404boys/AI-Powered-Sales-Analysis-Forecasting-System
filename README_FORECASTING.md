# Forecasting and What-If Analysis

This repository includes a LightGBM forecasting pipeline and a Streamlit app.

## 1. Environment setup

Use Python 3.10 or 3.11 for the smoothest compatibility with the project dependencies.

\`\`\`bash
python -m venv .venv
\`\`\`

Windows PowerShell:

\`\`\`powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
\`\`\`

## 2. Train the model

Make sure the cleaned dataset is available at:

\`Data/Cleaned/store_sales_cleaned.csv\`

From the repository root, run:

\`\`\`powershell
python src/modeling/train_lightgbm.py
\`\`\`

The default validation period is the final 42 calendar days in the dataset. To use a smaller model for a first local test:

\`\`\`powershell
python src/modeling/train_lightgbm.py --n-estimators 300
\`\`\`

Training creates local files in \`artifacts/model/\`:
- \`lightgbm_sales_model.joblib\`
- \`metrics.json\`
- \`feature_importance.csv\`
- \`validation_predictions.csv\`

These generated artifacts are not committed to GitHub by default.

## 3. Run the app

\`\`\`powershell
streamlit run app/app.py
\`\`\`

## App sections

- **Overview:** dataset-level sales trend and store-type comparison.
- **Sales Explorer:** historical sales for a selected store.
- **What-If Simulator:** compare model predictions after changing promotion and holiday assumptions for a selected historical store-day.
- **Model Performance:** validation metrics, a matched-row comparison against a seven-day seasonal baseline, and gain-based feature importance.

## Important modelling notes

- \`Customers\` is excluded from predictors because future customer counts are not known at forecast time.
- Calendar-based lag features are built on a complete daily calendar for each store's observed date span. Missing dates in the cleaned open-day-only file are treated as zero-sales history days.
- The current validation protocol is one-day-ahead rolling evaluation: actual prior-day sales become available as the next day is predicted. It is not a fixed-origin 42-day forecast.
- The What-If tool is a model scenario, not a causal estimate. Historical lag features are held constant while the selected assumptions change.
- The cleaned source contains open, positive-sales records only; the overview must not be interpreted as including closed-store sales.

## Troubleshooting

If the app says the model is missing, run the training command first. If training is slow on a lower-powered laptop, use \`--n-estimators 300\` for a quick smoke test, then train the full model later.
