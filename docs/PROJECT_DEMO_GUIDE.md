# Project Demonstration Guide

## Project title
**AI-Powered Sales Analysis and Forecasting System**

## One-minute explanation
This project analyzes historical Rossmann retail sales and builds a daily store-level forecasting prototype. It uses a leakage-aware LightGBM model with calendar, store, promotion, competition, and historical-sales features. A Streamlit application presents historical sales, recursive multi-day forecasts, What-If scenarios, and model evaluation.

## Suggested live demonstration (6–8 minutes)

1. **Introduce the business problem.** Retailers need to estimate store-level daily sales to support inventory, staffing, and promotion planning.
2. **Show the data analysis.** Open the existing notebooks and explain the data quality checks, exploratory analysis, and key patterns.
3. **Show the dataset audit.** Run `python src/data/validate_dataset.py` and explain the missing PromoInterval values and unobserved store-days.
4. **Explain feature engineering.** Show `src/features/build_features.py`. Explain calendar-correct lags and shifted rolling statistics; today's target is not used in today's features.
5. **Explain the model.** Show `src/modeling/train_lightgbm.py`. Explain LightGBM, the log-transformed target, chronological validation, early stopping, and why `Customers` is excluded.
6. **Open the Streamlit app.** Demonstrate Overview, Sales Explorer, Future Forecast, What-If Simulator, and Model Performance.
7. **Demonstrate Future Forecast.** Choose a store and 7-day horizon. Explain that predictions are fed recursively into later lag features and that the store is assumed open.
8. **Demonstrate What-If.** Change promotion or holiday assumptions and compare the model's predictions. Explain that this is not a causal estimate.
9. **Explain evaluation.** Show the generated `artifacts/model/metrics.json`. The repository includes a 40-round, 14-day real-data smoke result; clearly distinguish it from the final default 1,200-round, 42-day training run.

## The differentiating features

- **Recursive Future Forecast:** predicts multiple days in sequence and feeds each prediction into later lag features.
- **What-If Simulator:** changes selected promotion and holiday inputs for a historical store-day and compares the model's predictions.

Neither feature should be described as a causal or guaranteed business outcome. The future forecast assumes the store stays open and holds the selected assumptions constant.

## Team-of-four task split

| Member | Suggested responsibility | Evidence to present |
|---|---|---|
| 1 | Business problem, dataset, data quality, and cleaning | Dataset profile, audit output, cleaning workflow |
| 2 | Exploratory data analysis and business insights | Existing notebooks and visualizations |
| 3 | Feature engineering, LightGBM, and evaluation | Feature code, training process, metrics, baseline |
| 4 | Streamlit application, Future Forecast, and What-If | App walkthrough, forecast output, scenario comparison |

All members should understand the overall pipeline and limitations. Adjust this split to match the work each member actually completed.

## Important evaluation language

- Use the actual scores from a successful run of the current training script.
- Identify the smoke-test metrics as preliminary; do not call them the final 42-day results.
- State that validation is one-day-ahead rolling evaluation, not a fixed-origin 42-day forecast.
- Do not claim that the What-If simulator estimates causal impact.
- The cleaned CSV contains open, positive-sales store-days only.
- Missing dates inside each store's observed span are treated as zero-sales history for lag construction.

## Pre-submission checklist

- [ ] Run `python -m pytest -q`.
- [ ] Run `python src/data/validate_dataset.py`.
- [ ] Run the full default model training command and save its output.
- [ ] Review `artifacts/model/metrics.json` and the matched baseline comparison.
- [ ] Start the app and visit all five pages.
- [ ] Test at least two Future Forecast horizons and two What-If scenarios.
- [ ] Ensure the report uses final measured metrics, not smoke-test metrics as final results.
- [ ] Rehearse the presentation and agree on who explains each section.
