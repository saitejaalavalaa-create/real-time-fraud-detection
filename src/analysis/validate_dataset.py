"""Validate the synthetic fraud-detection dataset.

This module loads the generated transaction dataset, checks data quality rules,
and prints a clean validation report. It is designed to support a production-style
ML project and to catch issues before training or deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "amount",
    "merchant_id",
    "merchant_category",
    "country",
    "device_id",
    "payment_method",
    "account_age_days",
    "card_age_days",
    "is_new_device",
    "is_international",
    "failed_transactions_24h",
    "transactions_1h",
    "transactions_24h",
    "average_customer_amount",
    "distance_from_last_transaction_km",
    "is_fraud",
]

EXPECTED_ROW_COUNT = 200_000
EXPECTED_FRAUD_COUNT = 3_000
EXPECTED_FRAUD_RATE = 0.015


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the synthetic transaction dataset from the repository data directory.

    Args:
        data_path: Optional path to the dataset. If absent, the repository's
            ``data/transactions.csv`` path is used.

    Returns:
        pandas.DataFrame: The loaded transaction data.

    Raises:
        FileNotFoundError: If the dataset does not exist.
    """
    if data_path is None:
        project_root = Path(__file__).resolve().parents[2]
        resolved_path = project_root / "data" / "transactions.csv"
    else:
        resolved_path = Path(data_path)

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Dataset not found at "
            f"{resolved_path}. Please run: python src/data/generate_transactions.py"
        )

    return pd.read_csv(resolved_path)


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validate the dataset schema and required columns.

    Args:
        df: Input dataframe.

    Returns:
        dict: A dictionary summarizing schema validation results.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in REQUIRED_COLUMNS]

    schema_report = {
        "expected_row_count": EXPECTED_ROW_COUNT,
        "actual_row_count": len(df),
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "has_required_columns": len(missing_columns) == 0,
    }

    return schema_report


def validate_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Perform a data-quality validation across the synthetic transaction dataset.

    Args:
        df: Input dataframe.

    Returns:
        dict: A dictionary containing validation metrics and pass/fail flags.
    """
    report: dict[str, Any] = {}

    report["row_count"] = len(df)
    report["duplicate_transaction_ids"] = df["transaction_id"].duplicated().sum()
    report["duplicate_rows"] = df.duplicated().sum()
    report["missing_values"] = df.isna().sum().to_dict()

    report["null_transaction_ids"] = int(df["transaction_id"].isna().sum())
    report["null_customer_ids"] = int(df["customer_id"].isna().sum())
    report["invalid_amounts"] = int((df["amount"] <= 0).sum())
    report["invalid_fraud_labels"] = int(~df["is_fraud"].isin([0, 1]).all())

    try:
        parsed = pd.to_datetime(df["timestamp"], errors="raise")
        report["timestamp_parsing_valid"] = True
        report["timestamp_min"] = parsed.min()
        report["timestamp_max"] = parsed.max()
    except (TypeError, ValueError):
        report["timestamp_parsing_valid"] = False
        report["timestamp_min"] = None
        report["timestamp_max"] = None

    report["fraud_count"] = int(df["is_fraud"].sum())
    report["fraud_percentage"] = float((df["is_fraud"].mean()) * 100 if len(df) else 0.0)
    report["legitimate_count"] = int((1 - df["is_fraud"]).sum())
    report["categorical_distributions"] = {
        "merchant_category": df["merchant_category"].value_counts().to_dict(),
        "country": df["country"].value_counts().to_dict(),
        "payment_method": df["payment_method"].value_counts().to_dict(),
    }

    report["dtypes"] = df.dtypes.astype(str).to_dict()
    return report


def generate_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for the transaction dataset.

    Args:
        df: Input dataframe.

    Returns:
        dict: Summary statistics used in the human-readable validation report.
    """
    fraud_mask = df["is_fraud"] == 1
    legitimate_mask = df["is_fraud"] == 0

    summary = {
        "total_transactions": len(df),
        "total_fraud_transactions": int(fraud_mask.sum()),
        "total_legitimate_transactions": int(legitimate_mask.sum()),
        "fraud_percentage": float((fraud_mask.mean() * 100) if len(df) else 0.0),
        "amount_mean": float(df["amount"].mean()),
        "amount_median": float(df["amount"].median()),
        "amount_std": float(df["amount"].std()),
        "amount_min": float(df["amount"].min()),
        "amount_max": float(df["amount"].max()),
        "fraud_amount_mean": float(df.loc[fraud_mask, "amount"].mean()),
        "legitimate_amount_mean": float(df.loc[legitimate_mask, "amount"].mean()),
        "international_transaction_percentage": float(
            (df["is_international"].mean() * 100) if len(df) else 0.0
        ),
        "new_device_percentage": float(
            (df["is_new_device"].mean() * 100) if len(df) else 0.0
        ),
    }

    return summary


def print_validation_report(
    schema_report: dict[str, Any],
    quality_report: dict[str, Any],
    summary_stats: dict[str, Any],
) -> None:
    """Print a professional, human-readable validation report."""
    print("=" * 88)
    print("Synthetic Transaction Dataset Validation Report")
    print("=" * 88)
    print(f"Total transactions: {summary_stats['total_transactions']:,}")
    print(f"Total fraud transactions: {summary_stats['total_fraud_transactions']:,}")
    print(f"Total legitimate transactions: {summary_stats['total_legitimate_transactions']:,}")
    print(f"Fraud percentage: {summary_stats['fraud_percentage']:.2f}%")
    print(f"Date range: {quality_report['timestamp_min']} to {quality_report['timestamp_max']}")
    print("\nSummary statistics:")
    print(f"  Amount mean: ${summary_stats['amount_mean']:.2f}")
    print(f"  Amount median: ${summary_stats['amount_median']:.2f}")
    print(f"  Amount std: ${summary_stats['amount_std']:.2f}")
    print(f"  Amount min: ${summary_stats['amount_min']:.2f}")
    print(f"  Amount max: ${summary_stats['amount_max']:.2f}")
    print(f"  Fraud avg amount: ${summary_stats['fraud_amount_mean']:.2f}")
    print(f"  Legit avg amount: ${summary_stats['legitimate_amount_mean']:.2f}")
    print(f"  International transaction percentage: {summary_stats['international_transaction_percentage']:.2f}%")
    print(f"  New-device percentage: {summary_stats['new_device_percentage']:.2f}%")

    print("\nSchema checks:")
    print(f"  Required columns present: {'Yes' if schema_report['has_required_columns'] else 'No'}")
    if schema_report["missing_columns"]:
        print(f"  Missing columns: {schema_report['missing_columns']}")
    if schema_report["extra_columns"]:
        print(f"  Extra columns: {schema_report['extra_columns']}")

    print("\nData quality checks:")
    print(f"  Duplicate transaction IDs: {quality_report['duplicate_transaction_ids']}")
    print(f"  Duplicate rows: {quality_report['duplicate_rows']}")
    print(f"  Null transaction IDs: {quality_report['null_transaction_ids']}")
    print(f"  Null customer IDs: {quality_report['null_customer_ids']}")
    print(f"  Invalid amounts (> 0 required): {quality_report['invalid_amounts']}")
    print(f"  Invalid fraud labels: {quality_report['invalid_fraud_labels']}")
    print(f"  Timestamp parsing valid: {'Yes' if quality_report['timestamp_parsing_valid'] else 'No'}")
    print(f"  Missing values total: {sum(quality_report['missing_values'].values())}")
    print(f"  Fraud count: {quality_report['fraud_count']:,}")
    print(f"  Fraud percentage: {quality_report['fraud_percentage']:.2f}%")

    print("\nCategorical distribution samples:")
    for key, values in quality_report["categorical_distributions"].items():
        print(f"  {key}: {dict(list(values.items())[:5])}")

    print("\nData types:")
    for key, value in quality_report["dtypes"].items():
        print(f"  {key}: {value}")
    print("=" * 88)


def assert_critical_rules(df: pd.DataFrame) -> None:
    """Assert critical data-quality rules for the project dataset.

    Args:
        df: Input dataframe.

    Raises:
        AssertionError: If a required rule fails.
    """
    assert len(df) == EXPECTED_ROW_COUNT, (
        f"Expected exactly {EXPECTED_ROW_COUNT} rows, found {len(df)}."
    )
    assert df["is_fraud"].sum() == EXPECTED_FRAUD_COUNT, (
        f"Expected exactly {EXPECTED_FRAUD_COUNT} fraud records, found {df['is_fraud'].sum()}."
    )
    fraud_rate = df["is_fraud"].mean()
    assert abs(fraud_rate - EXPECTED_FRAUD_RATE) < 0.005, (
        f"Fraud rate expected to be approximately {EXPECTED_FRAUD_RATE:.3f}, "
        f"found {fraud_rate:.4f}."
    )
    assert df["transaction_id"].is_unique, "transaction_id must be unique."
    assert not df.isna().any().any(), "No missing values are allowed in the dataset."
    assert (df["amount"] > 0).all(), "amount must be greater than zero."
    assert set(df["is_fraud"].unique()) <= {0, 1}, (
        "is_fraud must contain only 0 and 1 values."
    )


def main() -> None:
    """Run the validation workflow for the generated transaction dataset."""
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    schema_report = validate_schema(df)
    quality_report = validate_data_quality(df)
    summary_stats = generate_summary_statistics(df)

    assert_critical_rules(df)
    print_validation_report(schema_report, quality_report, summary_stats)


if __name__ == "__main__":
    main()
