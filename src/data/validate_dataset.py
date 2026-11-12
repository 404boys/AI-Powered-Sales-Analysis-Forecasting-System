"""Create a reproducible quality audit for the cleaned Rossmann dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Store", "Date", "Sales"}


def audit_dataset(frame: pd.DataFrame) -> dict:
    """Return descriptive quality checks without silently modifying source data."""
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Dataset contains no rows.")

    dates = pd.to_datetime(frame["Date"], errors="coerce")
    sales = pd.to_numeric(frame["Sales"], errors="coerce")
    valid_dates = dates.dropna()

    store_count = int(frame["Store"].nunique(dropna=True))
    duplicate_store_dates = int(
        frame.loc[:, ["Store", "Date"]].duplicated().sum()
    )

    calendar_store_days = None
    observed_rows = int(len(frame))
    if dates.notna().all() and frame["Store"].notna().all():
        spans = (
            pd.DataFrame({"Store": frame["Store"], "Date": dates})
            .groupby("Store")["Date"]
            .agg(["min", "max"])
        )
        calendar_store_days = int(
            ((spans["max"] - spans["min"]).dt.days + 1).sum()
        )

    missing_by_column = {
        str(column): int(count)
        for column, count in frame.isna().sum().items()
        if int(count) > 0
    }

    return {
        "row_count": observed_rows,
        "column_count": int(frame.shape[1]),
        "store_count": store_count,
        "date_start": valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        "date_end": valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None,
        "invalid_date_rows": int(dates.isna().sum()),
        "missing_store_rows": int(frame["Store"].isna().sum()),
        "duplicate_store_date_rows": duplicate_store_dates,
        "invalid_sales_rows": int(sales.isna().sum()),
        "zero_sales_rows": int((sales == 0).sum()),
        "negative_sales_rows": int((sales < 0).sum()),
        "missing_values_by_column": missing_by_column,
        "observed_rows": observed_rows,
        "calendar_store_days_within_each_store_span": calendar_store_days,
        "unobserved_store_days_within_spans": (
            max(calendar_store_days - observed_rows, 0)
            if calendar_store_days is not None
            else None
        ),
        "interpretation_note": (
            "The cleaned source may intentionally contain only open, positive-sales "
            "store-days. Unobserved store-days are descriptive gaps, not proof of bad data."
        ),
    }


def render_markdown(audit: dict) -> str:
    """Render the audit summary as a readable Markdown report."""
    lines = [
        "# Cleaned Dataset Quality Audit",
        "",
        "> Automatically generated from the cleaned CSV. Review the interpretation "
        "of missing store-days before treating them as data defects.",
        "",
        "## Dataset overview",
        "",
        f"- Rows: {audit['row_count']:,}",
        f"- Columns: {audit['column_count']:,}",
        f"- Stores: {audit['store_count']:,}",
        f"- Date range: {audit['date_start']} to {audit['date_end']}",
        "",
        "## Validation checks",
        "",
        "| Check | Count |",
        "|---|---:|",
        f"| Invalid dates | {audit['invalid_date_rows']:,} |",
        f"| Missing store IDs | {audit['missing_store_rows']:,} |",
        f"| Duplicate Store-Date rows | {audit['duplicate_store_date_rows']:,} |",
        f"| Invalid sales values | {audit['invalid_sales_rows']:,} |",
        f"| Zero-sales rows | {audit['zero_sales_rows']:,} |",
        f"| Negative-sales rows | {audit['negative_sales_rows']:,} |",
        f"| Unobserved store-days within each store's date span | "
        f"{audit['unobserved_store_days_within_spans'] if audit['unobserved_store_days_within_spans'] is not None else 'N/A'} |",
        "",
        "## Missing values",
        "",
    ]
    if audit["missing_values_by_column"]:
        lines.extend(
            f"- {column}: {count:,}"
            for column, count in audit["missing_values_by_column"].items()
        )
    else:
        lines.append("- No missing values.")
    lines.extend(["", "## Interpretation", "", audit["interpretation_note"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the cleaned Rossmann sales dataset."
    )
    parser.add_argument(
        "--data",
        default="Data/Cleaned/store_sales_cleaned.csv",
        help="Path to the cleaned CSV file.",
    )
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path.resolve()}")

    frame = pd.read_csv(data_path, low_memory=False)
    audit = audit_dataset(frame)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "data_quality_audit.json"
    markdown_path = output_dir / "data_quality_audit.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(audit), encoding="utf-8")

    print(render_markdown(audit))
    print(f"\nSaved: {json_path.resolve()}")
    print(f"Saved: {markdown_path.resolve()}")


if __name__ == "__main__":
    main()
