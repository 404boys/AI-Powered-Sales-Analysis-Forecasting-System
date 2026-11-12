# LightGBM Model Evaluation

## Evaluation status

**Status: Preliminary real-data smoke test completed.**

A real-data smoke test completed successfully in GitHub Actions using the committed cleaned CSV. It trained a 40-round LightGBM model with a 14-calendar-day chronological validation window. This verifies that the data pipeline and model-training path execute on the actual repository dataset.

**This is not the final full training result.** The default local training configuration uses up to 1,200 rounds and a 42-day validation window. Run that configuration before final submission and replace or supplement the preliminary results below with the generated `artifacts/model/metrics.json`.

## Dataset and validation design

- Dataset: `Data/Cleaned/store_sales_cleaned.csv`
- Rows: 844,338
- Stores: 1,115
- Date range: 2013-01-01 to 2015-07-31
- Model: LightGBM regressor trained on `log1p(Sales)`
- Smoke-test validation: final 14 calendar days, held out chronologically
- Validation rows: 13,420
- Validation dates: 2015-07-18 to 2015-07-31
- Training rows: 830,918
- Best iteration: 40 (the smoke test was capped at 40 rounds and did not reach early stopping)

## Preliminary real-data smoke-test results

| Metric | 40-round model | Seven-day lag baseline |
|---|---:|---:|
| RMSPE | 13.87% | 36.07% |
| MAE | 700.72 | 2,083.16 |
| Evaluation rows | 13,420 | 13,420 |

The model and baseline were evaluated on the same 13,420 rows, making the comparison directly comparable. Values are in the dataset's sales units where applicable.

Interpretation: in this short smoke-test configuration, the LightGBM model outperformed the simple seven-day lag baseline on both RMSPE and MAE. This should be treated as preliminary evidence only; it does not establish final performance under the default 42-day/1,200-round configuration.

## Metrics to report after full training

| Metric | Meaning |
|---|---|
| RMSPE | Root mean squared percentage error on positive-sales validation rows |
| MAE | Mean absolute error in the dataset's sales units |
| RMSE | Root mean squared error in the dataset's sales units |
| Model-on-baseline-rows metrics | Model error on the exact rows where a seven-day lag is available |
| Seven-day lag baseline | Error from predicting each store's sales using its sales seven calendar days earlier |

The training script saves full-run metrics to `artifacts/model/metrics.json`, feature importance to `artifacts/model/feature_importance.csv`, and validation predictions to `artifacts/model/validation_predictions.csv`.

## Forecast protocol and limitations

The default training split reserves the final 42 calendar days chronologically. Validation is **one-day-ahead rolling evaluation**: for each validation date, lag features may use actual sales from earlier dates, including earlier dates within the validation window. It is not a fixed-origin 42-day forecast made at the beginning of the window.

The cleaned input contains open, positive-sales rows only. Dates absent from a store's observed rows are treated as zero-sales history days for calendar lag construction. The source does not distinguish a closed day from an open day omitted because of zero sales.

The model excludes `Customers`, `Sales`, and `Open` from predictors. Future customer counts are not known at forecast time. The Future Forecast page uses recursive predictions as later lag history, assumes the store remains open, and holds selected promotion/holiday assumptions constant. The What-If simulator holds historical lag features fixed; neither interface provides a causal estimate.

## Feature importance

The training script exports gain-based feature importance. Gain sums the improvement in the model objective attributed to splits using a feature. It is model-specific, does not establish causality, and does not show the direction of a feature's effect.

## Reproducibility

From the repository root:

```powershell
python -m pip install -r requirements.txt
python src/data/validate_dataset.py
python src/modeling/train_lightgbm.py
```

Generated model files are written to `artifacts/model/` locally and are not committed by default.
