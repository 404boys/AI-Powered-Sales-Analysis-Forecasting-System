"""Unit tests for leakage-safe calendar feature engineering."""

import pandas as pd
import pytest

from src.features.build_features import build_features


def make_store_data() -> pd.DataFrame:
    """Create a tiny store history with deliberate missing calendar dates."""
    return pd.DataFrame(
        {
            "Store": [1, 1, 1, 2, 2],
            "Date": [
                "2015-01-01",
                "2015-01-03",
                "2015-01-08",
                "2015-01-01",
                "2015-01-02",
            ],
            "Sales": [100.0, 300.0, 800.0, 50.0, 60.0],
            "CompetitionDistance": [500, 500, 500, 1000, 1000],
            "PromoInterval": [None, None, None, "Jan,Apr,Jul,Oct", "Jan,Apr,Jul,Oct"],
            "CompetitionOpenSinceMonth": [0, 0, 0, 3, 3],
            "CompetitionOpenSinceYear": [0, 0, 0, 2012, 2012],
            "Promo2SinceWeek": [0, 0, 0, 10, 10],
            "Promo2SinceYear": [0, 0, 0, 2013, 2013],
            "StateHoliday": [0, 0, 0, 0, 0],
        }
    )


def test_calendar_lag_uses_calendar_days_not_previous_observed_row() -> None:
    featured = build_features(make_store_data())
    store_one = featured.loc[featured["Store"] == 1].set_index("Date")

    # Jan 2 is absent from the source; the Jan 3 one-day lag must therefore be 0.
    assert store_one.loc[pd.Timestamp("2015-01-03"), "Sales_Lag_1"] == 0

    # Jan 1 is exactly seven calendar days before Jan 8.
    assert store_one.loc[pd.Timestamp("2015-01-08"), "Sales_Lag_7"] == 100


def test_features_do_not_include_current_day_target() -> None:
    data = make_store_data()
    featured = build_features(data)
    jan_three = featured.loc[
        (featured["Store"] == 1) & (featured["Date"] == pd.Timestamp("2015-01-03"))
    ].iloc[0]

    assert jan_three["Sales_Lag_1"] == 0
    assert jan_three["Sales_Lag_7"] != jan_three["Sales"]


def test_log_competition_distance_is_created() -> None:
    featured = build_features(make_store_data())
    import numpy as np

    expected = np.log1p(500)
    assert featured.loc[featured["Store"] == 1, "Log_CompetitionDistance"].iloc[0] == pytest.approx(expected)


def test_missing_promo_interval_is_labelled_none() -> None:
    featured = build_features(make_store_data())
    assert (featured.loc[featured["Store"] == 1, "PromoInterval"] == "None").all()


def test_duplicate_store_date_is_rejected() -> None:
    data = make_store_data()
    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate Store-Date"):
        build_features(duplicated)
