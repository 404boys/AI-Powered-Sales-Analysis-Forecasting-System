# Initial LightGBM Model Evaluation

## Dataset and validation setup

- Dataset: cleaned Rossmann store sales data
- Input rows: 844,338
- Stores: 1,115
- Date coverage: 2013-01-01 to 2015-07-31
- Training rows: 804,056
- Validation rows: 40,282
- Validation window: 2015-06-20 to 2015-07-31
- Model: LightGBM regressor trained on \`log1p(Sales)\`
- Best iteration: 1,198
- Random date splitting was not used.

## Initial validation results

| Metric | LightGBM | 7-calendar-day lag baseline |
|---|---:|---:|
| RMSPE | 10.94% | 41.02% |
| MAE | 535.16 | 2,191.87 |
| RMSE | 782.53 | Not reported |

RMSPE is calculated on positive-sales validation rows. MAE and RMSE are expressed in the dataset's sales units.

## Forecast protocol and limitations

This is a **one-day-ahead rolling validation**. For each validation date, the lag features may use actual sales from earlier dates, including earlier dates in the validation window. This reflects a process where each day's actual sales become available before forecasting the next day.

It is not a fixed-origin 42-day forecast. A true multi-day forecast must generate future lag features recursively or use a separate horizon-aware design.

The cleaned input contains open, positive-sales rows only. For calendar lag construction, dates absent from a store's observed rows are treated as zero-sales history days. The source does not distinguish a closed day from an open day omitted because of zero sales, so this is an explicit assumption.

The model excludes \`Customers\`, \`Sales\`, and \`Open\` from predictors. Future customer counts are not known at forecast time. The model's What-If simulator holds historical lag features fixed and changes the selected promotion/holiday inputs. Its output is a model scenario, not a causal estimate.

## Initial feature-importance ranking

LightGBM split-count importance ranks the following features highest:

1. Store
2. DayOfYear
3. Day
4. DayOfWeek
5. Sales_Lag_1
6. Sales_Lag_14
7. Sales_Lag_7
8. Sales_Rolling_14_Mean
9. Sales_Rolling_7_Std
10. Sales_Rolling_7_Mean
11. Promo
12. Year

Split-count importance indicates how frequently a feature was used in tree splits. It does not establish causality or the direction of a feature's effect.

## Reproducibility

From the repository root:

\`\`\`powershell
pip install -r requirements.txt
python src/modeling/train_lightgbm.py
\`\`\`

Generated model files are written to \`artifacts/model/\` locally and are not committed by default.
