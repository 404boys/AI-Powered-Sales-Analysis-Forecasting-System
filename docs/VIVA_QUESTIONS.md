# Project Viva Preparation

Use these as concise study notes. Be prepared to explain the actual code and results rather than memorizing answers.

## Business and dataset

### 1. What problem does the project solve?
It explores historical retail sales and predicts daily sales for individual stores. Such estimates can support operational planning, although this prototype is not connected to a live retailer system.

### 2. Why did you choose the Rossmann Store Sales dataset?
It contains store-level daily sales together with calendar, promotion, store, competition, and holiday information, making it suitable for retail forecasting and business analysis.

### 3. What is one important limitation of the cleaned dataset?
The version used by the forecasting pipeline contains only open, positive-sales store-days. It does not provide a complete record of closed days and open days with zero sales.

## Feature engineering

### 4. What is a lag feature?
A lag feature is a previous observation used as an input. For example, `Sales_Lag_7` represents the same store's sales seven calendar days earlier.

### 5. Why do you use calendar-based lags?
A row-based shift would mean the previous recorded row, not necessarily the previous calendar day. Reindexing each store's history to daily dates makes a seven-day lag correspond to seven calendar days.

### 6. How do you avoid target leakage in rolling features?
The history series is shifted by one day before calculating rolling statistics. Therefore, the current day's sales do not enter the current day's predictors.

### 7. Why is `Customers` excluded?
The number of customers for a future day is not known when the forecast is made. Using it would make the model less realistic for advance planning.

### 8. Why transform competition distance?
The logarithm transformation reduces the influence of very large distances and can make the numeric feature easier for a tree model to use.

## Machine learning and evaluation

### 9. Why LightGBM?
LightGBM is a gradient-boosted decision-tree framework that handles nonlinear relationships and tabular data efficiently. It also supports categorical features and missing numeric values.

### 10. Why use `log1p(Sales)`?
Sales can be highly skewed. The log transformation compresses large values during training. Predictions are converted back to the original scale with `expm1`.

### 11. Why is the split chronological?
A random split can mix future and past dates. A chronological holdout better reflects the task of predicting future sales from earlier observations.

### 12. What does early stopping do?
It monitors validation performance and stops adding trees when the monitored score no longer improves for the configured patience period. This can reduce unnecessary training and overfitting.

### 13. What is RMSPE?
Root Mean Squared Percentage Error is the square root of the average squared relative error. It emphasizes proportional errors and is suitable here because the validation targets are strictly positive.

### 14. Why compare against a seven-day lag baseline?
Retail sales often have a weekly pattern. Predicting sales from the same store seven days earlier is a simple reference point. The ML model should be compared with this baseline rather than evaluated in isolation.

### 15. What does one-day-ahead rolling validation mean?
When predicting a validation day, the feature history may include actual sales from earlier validation days. It represents daily forecasting where yesterday's actual sales become available before predicting the next day. It is not a single forecast made 42 days in advance.

## Application and What-If analysis

### 16. What does the What-If Simulator do?
It changes selected promotion and holiday inputs for a historical store-day, runs the model again, and compares the resulting prediction with the original prediction.

### 17. Is the What-If result causal?
No. It is a model-based scenario comparison, not a causal experiment. Historical lag features are held fixed, and the model does not prove that changing a promotion will cause the displayed sales difference.

### 18. Why build a Streamlit app?
Streamlit makes it possible to present data exploration, model outputs, and scenario controls in an interactive browser interface without building a separate front-end and API stack.

## Be ready to show

- The cleaned dataset and its limitations.
- The feature-building function.
- The model training and chronological split.
- The actual generated metrics and baseline comparison.
- The four application pages.
- A What-If scenario and its limitations.

**Important:** Fill in actual metric values only after training the model successfully. Never memorize or present unverified scores.
