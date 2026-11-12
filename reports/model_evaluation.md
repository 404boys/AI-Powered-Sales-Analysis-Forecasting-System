# LightGBM Model Evaluation

## Evaluation status

**Status: Full default model training and application smoke tests completed successfully.**

The default training configuration completed in GitHub Actions against the committed cleaned dataset. It used up to 1,200 boosting rounds and the final 42 calendar days for chronological validation. All five Streamlit pages also completed the automated app smoke test.

## Dataset and validation design

- Dataset: `Data/Cleaned/store_sales_cleaned.csv`
- Rows: 844,338
- Stores: 1,115
- Date range: 2013-01-01 to 2015-07-31
- Model: LightGBMRegressor trained on `log1p(Sales)`
- Training rows: 804,056
- Validation rows: 40,282
- Validation dates: 2015-06-20 to 2015-07-31
- Maximum boosting rounds: 1,200
- Best iteration: 1,200 (early stopping did not trigger before the maximum)

## Full default model results

| Metric | LightGBM model | Seven-day lag baseline |
|---|---:|---:|
| RMSPE | **11.14%** | 41.02% |
| MAE | **539.12** | 2,191.87 |
| RMSE | 790.73 | Not exported for baseline |
| Evaluation rows | 40,282 | 40,282 |

The model and baseline were evaluated on the exact same 40,282 validation rows. MAE and RMSE values are in the dataset's sales units.

The model outperformed the seven-day lag baseline on RMSPE and MAE in this run. These values are measured from the full default training configuration. The trained model, metrics, feature importance, validation predictions, and data audit are available in the GitHub Actions artifact named `rossmann-full-model-and-evaluation` for the successful full model workflow run.

## Metric definitions

- **RMSPE:** root mean squared percentage error on positive-sales validation rows.
- **MAE:** mean absolute error in the dataset's sales units.
- **RMSE:** root mean squared error in the dataset's sales units.
- **Seven-day lag baseline:** predicts each store's sales using its sales seven calendar days earlier.
- **Matched-row comparison:** model and baseline are evaluated on the same rows where a seven-day lag is available.

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
python -m pytest -q
python src/data/validate_dataset.py
python src/modeling/train_lightgbm.py
```

Generated model files are written to `artifacts/model/` locally and are not committed by default.
