# Project Demonstration Guide

## Project title
**AI-Powered Sales Analysis and Forecasting System**

## One-minute explanation
This project uses the Rossmann Store Sales dataset to understand historical retail sales patterns and build a machine-learning model that predicts daily sales for individual stores. The forecasting pipeline uses LightGBM with calendar, store, promotion, competition, and historical-sales features. A Streamlit application makes the results explorable and includes a What-If Simulator that lets a user change promotion and holiday assumptions and compare the model's predictions.

## Suggested live demonstration (5–7 minutes)

1. **Introduce the business problem.** Retailers need to estimate store-level daily sales to support staffing, inventory, and promotion planning.
2. **Show the data-analysis work.** Open the existing notebooks and explain data quality checks, exploratory analysis, and the key patterns found.
3. **Explain feature engineering.** Show `src/features/build_features.py`. Explain calendar variables, previous-day/previous-week/previous-fortnight sales, and rolling statistics. Emphasize that the historical features are shifted so today's target is not used to predict today's sales.
4. **Explain the model.** Show `src/modeling/train_lightgbm.py`. Explain the log-transformed target, LightGBM, chronological holdout, early stopping, and why `Customers` is excluded.
5. **Open the Streamlit app.** Demonstrate Overview, Sales Explorer, What-If Simulator, and Model Performance.
6. **Explain evaluation.** Read the measured values from the generated `artifacts/model/metrics.json`. Compare the model with the seven-day lag baseline. Never quote example or unverified metrics.
7. **Discuss limitations and future work.** Explain the open-positive-only cleaned data, the missing-date zero-sales assumption, and that the What-If comparison is not a causal experiment.

## The differentiating feature: What-If Simulator

The simulator starts from a historical store-day and lets the user change promotion, school-holiday, and state-holiday inputs. It then compares the model's original prediction with the scenario prediction.

The historical lag features are deliberately held fixed. This isolates the model's response to the selected input changes, but it does not prove that a real-world intervention would cause the same change. Explain this distinction during the presentation.

## Team-of-four task split

| Member | Suggested responsibility | Evidence to present |
|---|---|---|
| 1 | Business problem, dataset, data quality, and cleaning | Dataset description, cleaning workflow, quality checks |
| 2 | Exploratory data analysis and visual insights | Existing notebooks and visualizations |
| 3 | Feature engineering and LightGBM model | Feature code, training process, validation protocol |
| 4 | Streamlit application, What-If simulator, and final integration | App walkthrough, scenario comparison, documentation |

All members should understand the overall pipeline and be able to explain the limitations. Adjust this split to match the actual work each member completed.

## Important evaluation language

- Report only the scores produced by a successful run of the current training script.
- State clearly that the validation is one-day-ahead rolling evaluation, not a fixed-origin 42-day forecast.
- Do not claim that the What-If simulator estimates causal impact.
- The cleaned CSV contains open, positive-sales store-days. Dashboard totals are totals of recorded rows, not complete totals including closed stores.
- Missing dates within a store's observed span are treated as zero-sales history days for lag construction. The source does not identify whether each missing date was closed or an open zero-sales day.

## Pre-submission checklist

- [ ] Run the feature tests with `python -m pytest -q`.
- [ ] Train the model and save the terminal output.
- [ ] Check `artifacts/model/metrics.json` and compare with the baseline.
- [ ] Start the app and visit all four pages.
- [ ] Test at least two What-If scenarios.
- [ ] Ensure no model metrics are invented or copied from an old draft.
- [ ] Rehearse the presentation and agree on who explains each section.
