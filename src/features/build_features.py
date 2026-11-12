"""Leakage-safe feature engineering for Rossmann daily sales forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_COLUMNS = [
    "Store",
    "DayOfWeek",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "PromoInterval",
]

FEATURE_COLUMNS = [
    "Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday",
    "StoreType", "Assortment", "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    "Year", "Month", "Day", "DayOfYear", "WeekOfYear", "IsWeekend",
    "Log_CompetitionDistance", "Sales_Lag_1", "Sales_Lag_7",
    "Sales_Lag_14", "Sales_Rolling_7_Mean", "Sales_Rolling_14_Mean",
    "Sales_Rolling_7_Std",
]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build calendar-based lag and rolling features for observed open days.

    The input contains only positive-sales/open records. Missing dates inside
    each store's observed active span are represented as zero-sales history
    days so that a 7-day lag means seven calendar days, not seven recorded rows.
    Only original observed rows are returned as prediction targets.
    """
    required = {"Store", "Date", "Sales"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    if data.empty:
        raise ValueError("Input dataset is empty.")
    if data[["Store", "Date"]].isna().any().any():
        raise ValueError("Store and Date must not contain missing values.")

    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    frame["Sales"] = pd.to_numeric(frame["Sales"], errors="coerce")
    if frame["Sales"].isna().any() or not np.isfinite(frame["Sales"]).all():
        raise ValueError("Sales must contain finite numeric values.")
    if (frame["Sales"] < 0).any():
        raise ValueError("Sales cannot contain negative values.")

    frame = frame.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Each store must have at most one target row per calendar date.
    if frame.duplicated(["Store", "Date"]).any():
        raise ValueError("Input contains duplicate Store-Date combinations.")

    # Build one complete daily sales history per store, within its observed span.
    calendar_parts = []
    for store, group in frame[["Store", "Date", "Sales"]].groupby("Store", sort=False):
        group = group.set_index("Date")
        daily_index = pd.date_range(group.index.min(), group.index.max(), freq="D")
        # Reindex on calendar dates, filling unobserved dates with zero history.
        daily_sales = group["Sales"].reindex(daily_index, fill_value=0).astype(float)
        calendar_parts.append(
            pd.DataFrame(
                {
                    "Store": store,
                    "Date": daily_index,
                    "HistorySales": daily_sales.to_numpy(),
                }
            )
        )

    calendar = pd.concat(calendar_parts, ignore_index=True)
    calendar = calendar.sort_values(["Store", "Date"]).reset_index(drop=True)

    # Shift before rolling so today's target can never enter today's features.
    history = calendar.groupby("Store", sort=False)["HistorySales"]
    calendar["Sales_Lag_1"] = history.shift(1)
    calendar["Sales_Lag_7"] = history.shift(7)
    calendar["Sales_Lag_14"] = history.shift(14)
    calendar["Sales_Rolling_7_Mean"] = history.transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).mean()
    )
    calendar["Sales_Rolling_14_Mean"] = history.transform(
        lambda values: values.shift(1).rolling(14, min_periods=5).mean()
    )
    calendar["Sales_Rolling_7_Std"] = history.transform(
        lambda values: values.shift(1).rolling(7, min_periods=3).std()
    )
    calendar = calendar.drop(columns="HistorySales")

    # Join history features back only to rows with a real positive-sales target.
    frame = frame.merge(
        calendar,
        on=["Store", "Date"],
        how="left",
        validate="one_to_one",
    )

    # Derive calendar fields from Date to keep them consistent.
    frame["Year"] = frame["Date"].dt.year
    frame["Month"] = frame["Date"].dt.month
    frame["Day"] = frame["Date"].dt.day
    frame["DayOfYear"] = frame["Date"].dt.day_of_year
    frame["WeekOfYear"] = frame["Date"].dt.isocalendar().week.astype("int16")
    frame["DayOfWeek"] = frame["Date"].dt.day_of_week + 1
    frame["IsWeekend"] = (frame["Date"].dt.dayofweek >= 5).astype("int8")

    # Log-transform competition distance to reduce the effect of extreme distances.
    # Keep missing distances as NaN; LightGBM can handle missing numeric values.
    if "CompetitionDistance" in frame.columns:
        frame["Log_CompetitionDistance"] = np.log1p(
            pd.to_numeric(frame["CompetitionDistance"], errors="coerce")
        )

    # A missing PromoInterval means the store has no Promo2 interval schedule.
    if "PromoInterval" in frame.columns:
        frame["PromoInterval"] = frame["PromoInterval"].fillna("None")

    # Rossmann uses zero as a missing-value code for these store metadata fields.
    for column in [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].replace(0, float("nan"))

    if "StateHoliday" in frame.columns:
        frame["StateHoliday"] = frame["StateHoliday"].fillna("0").astype(str)

    return frame
