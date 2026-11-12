"""Tests for the cleaned dataset quality audit."""

import pandas as pd
import pytest

from src.data.validate_dataset import audit_dataset, render_markdown


def sample_data():
    return pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "Date": ["2015-01-01", "2015-01-03", "2015-01-01"],
            "Sales": [100.0, 150.0, 200.0],
            "PromoInterval": [None, "Jan,Apr,Jul,Oct", None],
        }
    )


def test_audit_reports_basic_counts_and_date_span():
    audit = audit_dataset(sample_data())

    assert audit["row_count"] == 3
    assert audit["store_count"] == 2
    assert audit["date_start"] == "2015-01-01"
    assert audit["date_end"] == "2015-01-03"
    assert audit["duplicate_store_date_rows"] == 0
    assert audit["missing_values_by_column"] == {"PromoInterval": 2}


def test_audit_counts_unobserved_days_without_calling_them_errors():
    audit = audit_dataset(sample_data())

    assert audit["calendar_store_days_within_each_store_span"] == 4
    assert audit["unobserved_store_days_within_spans"] == 1
    assert "not proof of bad data" in audit["interpretation_note"]


def test_audit_rejects_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        audit_dataset(pd.DataFrame({"Store": [1], "Sales": [100]}))


def test_markdown_report_contains_quality_checks():
    report = render_markdown(audit_dataset(sample_data()))

    assert "# Cleaned Dataset Quality Audit" in report
    assert "Duplicate Store-Date rows" in report
