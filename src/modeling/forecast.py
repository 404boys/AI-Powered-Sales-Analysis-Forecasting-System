"""Recursive multi-day store sales forecasting."""

from __future__ import annotations

import pandas as pd

from features.build_features import build_features
from modeling.predict import predict_sales


def forecast_store(
    raw_history: pd.DataFrame,
    model,
    store_id: int,
    horizon: int = 7,
    promo: int = 0,
    school_holiday: int = 0,
    state_holiday: str = "0",
) -> pd.DataFrame:
    """Forecast future daily sales recursively for one store.

    Future predictions are fed back into the lag history for subsequent days.
    Store metadata is copied from the latest observed row. The caller must make
    explicit that the store is assumed open and that calendar assumptions are
    held constant over the selected horizon.
    """
    if horizon < 1 or horizon > 28:
        raise ValueError("Forecast horizon must be between 1 and 28 days.")
    if promo not in (0, 1) or school_holiday not in (0, 1):
        raise ValueError("Promo and SchoolHoliday must be either 0 or 1.")
    if state_holiday not in {"0", "a", "b", "c"}:
        raise ValueError("StateHoliday must be one of: 0, a, b, c.")

    required = {
        "Store", "Date", "Sales", "Promo", "StateHoliday", "SchoolHoliday",
        "StoreType", "Assortment", "CompetitionDistance",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    }
    missing = sorted(required.difference(raw_history.columns))
    if missing:
        raise ValueError(f"Raw history is missing required columns: {missing}")

    history = raw_history.loc[raw_history["Store"] == store_id].copy()
    if history.empty:
        raise ValueError(f"No historical rows found for store {store_id}.")

    history["Date"] = pd.to_datetime(history["Date"], errors="raise")
    history = history.sort_values("Date").reset_index(drop=True)
    last_observed_date = history["Date"].max()
    template = history.iloc[-1].copy()
    forecast_rows: list[dict] = []

    for day_number in range(1, horizon + 1):
        future_date = last_observed_date + pd.Timedelta(days=day_number)
        future_row = template.copy()
        future_row["Date"] = future_date
        future_row["Sales"] = 0.0
        future_row["Promo"] = int(promo)
        future_row["SchoolHoliday"] = int(school_holiday)
        future_row["StateHoliday"] = str(state_holiday)
        if "Customers" in future_row.index:
            future_row["Customers"] = float("nan")

        extended = pd.concat(
            [history, future_row.to_frame().T],
            ignore_index=True,
        )
        featured = build_features(extended)
        prediction_row = featured.loc[featured["Date"] == future_date].iloc[-1]
        prediction = predict_sales(model, prediction_row)

        # Feed the prediction into history so later forecast days use recursive lags.
        future_row["Sales"] = prediction
        history = pd.concat(
            [history, future_row.to_frame().T],
            ignore_index=True,
        )
        forecast_rows.append(
            {
                "Date": future_date,
                "Store": store_id,
                "Predicted_Sales": prediction,
                "Promo": int(promo),
                "SchoolHoliday": int(school_holiday),
                "StateHoliday": str(state_holiday),
            }
        )

    return pd.DataFrame(forecast_rows)
