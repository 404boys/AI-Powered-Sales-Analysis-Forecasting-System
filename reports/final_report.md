# AI-Powered Sales Analysis & Forecasting System
## Exploratory Data Analysis & Data Preprocessing Report

**Author:** Data Analytics & ML Engineering Team  
**Date:** September 2026  
**Environment:** Python 3.14 | Pandas | NumPy | Matplotlib | Seaborn  

---

## 1. Executive Summary

This report documents the end-to-end data audit, data engineering, cleaning, and exploratory data analysis (EDA) conducted on multi-table retail transactional records. The primary objective is to clean raw business operations data, identify customer demand patterns, analyze promotional elasticity, and engineer temporal signals to serve as a robust baseline for predictive time-series machine learning models (e.g., XGBoost, LightGBM, and Prophet).

### Key Project Metrics
* **Raw Transaction Records:** 1,017,209 rows across 1,115 unique retail outlets.
* **Date Coverage:** January 1, 2013 to July 31, 2015 (942 calendar days).
* **Operational Pruning:** 172,871 non-trading rows (`Open == 0` and zero-sales days) pruned.
* **Revenue Preservation:** **100.00%** of total historical gross revenue ($5.87 billion) retained.
* **Clean Modeling Records:** 844,338 fully validated transactions with 24 engineered features.
* **Data Integrity:** **0 missing values**, **0 duplicate records**, **0 negative transactions**.

---

## 2. Business Problem & Analytical Objectives

Retail enterprises face demand volatility driven by promotional calendars, seasonal spikes, and competitor proximity. Inaccurate sales forecasts lead to overstocking costs or stockout revenue loss.

### Core Objectives:
1. **Transaction Integrity Audit:** Detect structural data anomalies, duplicate checkouts, and mismatched operational flags.
2. **Systematic Imputation:** Resolve high missingness across competitor and promotional metadata using business logic.
3. **Behavioral Profiling:** Measure the revenue lift of active promotions and compare sales efficiency across store formats.
4. **Feature Enrichment:** Construct cyclical temporal features and log-scale geographic metrics for downstream AI/ML forecasting algorithms.

---

## 3. Data Architecture & Schema Validation

### 3.1 Dataset Integration
The raw operational records were constructed by joining daily store checkouts (`train.csv`) with store metadata (`store.csv`) using `Store` as the relational key, producing a unified schema of 1,017,209 rows and 18 columns.

### 3.2 Data Quality Audit Findings
* **Missing Value Mechanism (MAR):** $100\%$ of missing values in `PromoInterval`, `Promo2SinceWeek`, and `Promo2SinceYear` (508,031 rows, 49.94%) correspond strictly to stores where `Promo2 == 0`. These represent structural non-participation rather than random data loss.
* **Competitor Inception Gaps:** `CompetitionOpenSinceMonth` and `Year` exhibited 31.79% missingness; `CompetitionDistance` had 2,642 missing records (0.26%).
* **Operational Sanity Violations:** 54 instances showed active status (`Open == 1`) with zero gross sales (`Sales == 0`), representing store refurbishment or data logging dropouts.
* **Temporal Continuity:** 934 out of 1,115 stores possess a full 942-day historical timeline; remaining stores entered operations in later phases.

---

## 4. Data Cleaning & Feature Engineering Pipeline

### 4.1 Transformation & Imputation Strategy
1. **Temporal Parsing:** Cast `Date` from raw object strings to native `datetime64[us]` and sorted chronologically per store.
2. **Geographical Median Imputation:** Imputed missing `CompetitionDistance` values with the sample median (2,330 meters) to avoid bias from long-tail rural outliers (max: 75,860m).
3. **Structural Default Assignment:** Assigned zero (`0`) to unrecorded competitor start years/months and `Promo2` timeline trackers; assigned `"None"` to missing `PromoInterval` categories.
4. **Operational Pruning:** Filtered out all closed store days (`Open == 0`) and zero-revenue records, reducing the dataset to 844,338 active trading rows while maintaining 100% gross revenue fidelity.
5. **Categorical Standardization:** Mapped `StateHoliday` to clean discrete categories `['0', 'a', 'b', 'c']` and converted `StoreType` and `Assortment` to native categorical dtypes.

### 4.2 Machine Learning Feature Generation
* **Temporal Features:** Extracted `Year`, `Month`, `Day`, `WeekOfYear`, and a binary `IsWeekend` flag (`DayOfWeek` $\in [6, 7]$).
* **Non-Linear Geographic Scaling:** Applied log-transformation:
  $$\text{Log\_CompetitionDistance} = \ln(1 + \text{CompetitionDistance})$$
  This stabilizes variance and addresses heavy right skewness ($9.77\%$ IQR outliers).
* **Basket Productivity:** Computed historical baseline:
  $$\text{SalesPerCustomer} = \frac{\text{Sales}}{\text{Customers}}$$
  Showing a median basket of 9.25 currency units per transaction.

---

## 5. Exploratory Data Analysis & Business Insights

### 5.1 Sales Distribution
* Daily store revenue exhibits a unimodal, right-skewed distribution centered between **5,500 and 7,000 currency units**.
* High revenue outliers (> 20,000) reflect major promotional campaigns and flagship urban store volumes rather than measurement errors.
* *Modeling Implication:* Tree-based gradient boosters (XGBoost/LightGBM) or optimizing root mean squared percentage error (RMSPE) on log-transformed targets are recommended.

### 5.2 Promotional Revenue Impact
* Active promotional days (`Promo == 1`) produce a significant upward shift in median sales and customer basket sizes compared to non-promotional trading days.
* *Business Action:* Promotions are a dominant driver of sales velocity; scheduling algorithms must prioritize promotional calendars.

### 5.3 Store Classification Archetypes
* **StoreType b:** Generates the highest mean daily revenue despite representing only ~1.5% of total store locations (15,830 records).
* **StoreTypes a & d:** Represent over 85% of the store network, acting as predictable baseline revenue generators.

### 5.4 Macro Seasonality & Annual Trends
* A prominent and repeatable demand peak occurs every **December**, corresponding to holiday shopping volumes.
* Demand stabilizes post-holiday (January–February), establishing predictable annual cyclicality.

### 5.5 Footfall vs. Revenue Elasticity
* Customer footfall and gross sales demonstrate a strong linear Pearson correlation coefficient of **$r = 0.8236$**.
* Footfall represents the primary operational predictor of total sales performance.

---

## 6. Recommendations for Downstream ML Modeling

1. **Target Transformation:** Train forecasting models on $\ln(1 + \text{Sales})$ to prevent high-revenue spikes from dominating loss functions.
2. **Evaluation Metric:** Use **RMSPE (Root Mean Square Percentage Error)** to align optimization with retail industry standards.
3. **Feature Priority:** Include `Promo`, `Customers` (or estimated footfall lags), `StoreType`, `WeekOfYear`, and `Log_CompetitionDistance` as primary features.
4. **Time-Series Split:** Use temporal cross-validation (e.g., train on 2013–2014, validate on 2015) to prevent future data leakage.

---
*Certified Clean Dataset Location:* `Data/Cleaned/store_sales_cleaned.csv`  
*Generated Visualization Assets:* `Visualizations/`