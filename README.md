# AI-Powered Sales Analysis & Forecasting System
> An enterprise-grade data audit, cleaning, and exploratory data analysis pipeline on retail sales transactions.

## 📌 Project Overview
- **Objective:** Perform end-to-end data cleaning, anomaly detection, and exploratory data analysis to prepare retail sales data for predictive AI/ML modeling.
- **Dataset:** Historical retail store records (1M+ rows) including daily sales, customer footfall, promotional campaigns, and store metadata.

## 🛠️ Key Pipeline Steps
1. **Data Ingestion & Merge:** Relational join of transaction logs and store metadata.
2. **Data Quality Audit:** Handled missingness in promotional schedules (MAR) and evaluated structural tracking gaps.
3. **Operational Filtering:** Pruned closed store days (`Open == 0`) and zero-sales records while ensuring 100% gross revenue preservation.
4. **Feature Engineering:** Extracted temporal cycles (`Month_Sin`, `Month_Cos`), weekend indicators, and log-transformed competition distances.
5. **Exploratory Data Analysis:** Generated high-resolution visualization charts covering sales distributions, promotional lifts, store archetype performance, and customer correlations.

## 🚀 Execution
Run the Jupyter notebook sequentially:
```bash
Notebooks/superstore_sales_eda.ipynb