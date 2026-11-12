"""Tests for recursive future store forecasts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.modeling.forecast import forecast_store


class ConstantLogModel:
    """Small deterministic model double that records received feature rows."""

    def __init__(self):
        self.booster_ = SimpleNamespace(
            pandas_categorical=[
                [1],
                [1, 2, 3, 4, 5, 6, 7],
                ["0", "a", "b", "c"],
                ["a"],
                ["a"],
                ["None"],
            ]
        )
        self.best_iteration_ = 1
        self.seen = []

    def predict(self, X, num_iteration=None):
        self.seen.append(X.copy())
        return np.array([np.log1p(100.0)])


def sample_history():
    return pd.DataFrame(
        {
            "Store": [1, 1, 1],
            "Date": ["2015-01-01", "2015-01-02", "2015-01-03"],
            "Sales": [80.0, 90.0, 95.0],
            "Promo": [0, 0, 1],
            "StateHoliday": ["0", "0", "0"],
            "SchoolHoliday": [0, 0, 0],
            "StoreType": ["a", "a", "a"],
            "Assortment": ["a", "a", "a"],
            "CompetitionDistance": [500, 500, 500],
            "CompetitionOpenSinceMonth": [0, 0, 0],
            "CompetitionOpenSinceYear": [0, 0, 0],
            "Promo2": [0, 0, 0],
            "Promo2SinceWeek": [0, 0, 0],
            "Promo2SinceYear": [0, 0, 0],
            "PromoInterval": [None, None, None],
            "Customers": [10, 11, 12],
        }
    )


def test_forecast_returns_future_dates_and_recursive_lag_history():
    model = ConstantLogModel()
    forecast = forecast_store(
        sample_history(),
        model,
        store_id=1,
        horizon=2,
        promo=1,
        school_holiday=0,
        state_holiday="0",
    )

    assert forecast["Date"].tolist() == [
        pd.Timestamp("2015-01-04"),
        pd.Timestamp("2015-01-05"),
    ]
    assert forecast["Predicted_Sales"].tolist() == pytest.approx([100.0, 100.0])
    assert model.seen[0]["Sales_Lag_1"].iloc[0] == 95.0
    assert model.seen[1]["Sales_Lag_1"].iloc[0] == pytest.approx(100.0)
    assert model.seen[1]["Promo"].iloc[0] == 1


def test_forecast_rejects_horizon_above_28_days():
    model = ConstantLogModel()
    try:
        forecast_store(sample_history(), model, store_id=1, horizon=29)
    except ValueError as error:
        assert "between 1 and 28" in str(error)
    else:
        raise AssertionError("Expected an invalid horizon to be rejected.")
