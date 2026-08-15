"""Exploratory data analysis for the synthetic fraud-detection dataset.

This module loads the generated transaction dataset, computes summary statistics,
compares fraudulent and legitimate behavior, and saves a set of matplotlib plots
under the repository ``reports/figures`` directory.

The analysis is intentionally descriptive and does not implement model training,
train/test splitting, or inference logic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMN = "is_fraud"
FIGURE_DIR = Path("reports") / "figures"


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the transaction dataset from the repository data directory.

    Args:
        data_path: Optional custom dataset path. If omitted, the repository's
            ``data/transactions.csv`` file is used.

    Returns:
        pandas.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset cannot be found.
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


def analyze_class_distribution(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze the fraud class distribution and imbalance ratio.

    Args:
        df: Input dataframe.

    Returns:
        dict: Class distribution metrics.
    """
    fraud_count = int(df[TARGET_COLUMN].sum())
    legitimate_count = int((1 - df[TARGET_COLUMN]).sum())
    total_count = len(df)
    fraud_percentage = (fraud_count / total_count) * 100 if total_count else 0.0
    class_imbalance_ratio = legitimate_count / fraud_count if fraud_count else np.nan

    return {
        "fraud_count": fraud_count,
        "legitimate_count": legitimate_count,
        "fraud_percentage": fraud_percentage,
        "class_imbalance_ratio": class_imbalance_ratio,
    }


def analyze_numeric_features(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float]]:
    """Compute summary statistics for the provided numeric features.

    Args:
        df: Input dataframe.
        feature_columns: Numeric columns to summarize.

    Returns:
        dict: A mapping of feature name to summary statistics.
    """
    stats: dict[str, dict[str, float]] = {}

    for column in feature_columns:
        if column not in df.columns:
            continue
        series = df[column]
        stats[column] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return stats


def analyze_categorical_features(df: pd.DataFrame, category_columns: list[str]) -> dict[str, pd.DataFrame]:
    """Summarize fraud rates for categorical features.

    Args:
        df: Input dataframe.
        category_columns: Categorical columns to evaluate.

    Returns:
        dict: A mapping of feature name to a fraud-rate summary dataframe.
    """
    results: dict[str, pd.DataFrame] = {}

    for column in category_columns:
        if column not in df.columns:
            continue

        summary = (
            df.groupby(column, dropna=False)[TARGET_COLUMN]
            .agg(["count", "sum", "mean"])
            .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
            .reset_index()
        )
        summary["fraud_rate_pct"] = summary["fraud_rate"] * 100
        summary = summary.sort_values("fraud_rate", ascending=False)
        results[column] = summary

    return results


def analyze_fraud_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """Compare fraudulent and legitimate transactions across key risk features.

    Args:
        df: Input dataframe.

    Returns:
        dict: Feature comparison statistics for fraud vs legitimate transactions.
    """
    numeric_features = [
        "amount",
        "transactions_1h",
        "transactions_24h",
        "failed_transactions_24h",
        "account_age_days",
        "card_age_days",
        "average_customer_amount",
        "distance_from_last_transaction_km",
    ]

    comparison: dict[str, Any] = {}
    fraud_mask = df[TARGET_COLUMN] == 1
    legitimate_mask = df[TARGET_COLUMN] == 0

    for feature in numeric_features:
        fraud_series = df.loc[fraud_mask, feature]
        legit_series = df.loc[legitimate_mask, feature]

        comparison[feature] = {
            "fraud": {
                "mean": float(fraud_series.mean()),
                "median": float(fraud_series.median()),
                "std": float(fraud_series.std()),
                "min": float(fraud_series.min()),
                "max": float(fraud_series.max()),
            },
            "legitimate": {
                "mean": float(legit_series.mean()),
                "median": float(legit_series.median()),
                "std": float(legit_series.std()),
                "min": float(legit_series.min()),
                "max": float(legit_series.max()),
            },
        }

    comparison["is_new_device"] = {
        "fraud": float(df.loc[fraud_mask, "is_new_device"].mean() * 100),
        "legitimate": float(df.loc[legitimate_mask, "is_new_device"].mean() * 100),
    }
    comparison["is_international"] = {
        "fraud": float(df.loc[fraud_mask, "is_international"].mean() * 100),
        "legitimate": float(df.loc[legitimate_mask, "is_international"].mean() * 100),
    }

    return comparison


def create_visualizations(df: pd.DataFrame, output_dir: str | Path | None = None) -> dict[str, str]:
    """Create and save a set of EDA plots for the transaction dataset.

    Args:
        df: Input dataframe.
        output_dir: Directory where plots are saved. Defaults to ``reports/figures``.

    Returns:
        dict: A mapping of plot names to saved file paths.
    """
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / FIGURE_DIR
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, str] = {}

    # 1. Class distribution
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=["steelblue", "firebrick"])
    ax.set_title("Fraud vs Legitimate Class Distribution")
    ax.set_ylabel("Transactions")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    fig.tight_layout()
    path = output_dir / "class_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["class_distribution.png"] = str(path)

    # 2. Amount by fraud status
    fig, ax = plt.subplots(figsize=(8, 5))
    df[df[TARGET_COLUMN] == 0]["amount"].hist(alpha=0.7, bins=40, label="Legitimate", ax=ax)
    df[df[TARGET_COLUMN] == 1]["amount"].hist(alpha=0.7, bins=40, label="Fraud", ax=ax)
    ax.set_title("Transaction Amount Distribution by Fraud Status")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "amount_by_fraud_status.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["amount_by_fraud_status.png"] = str(path)

    # 3. Fraud rate by merchant category
    merchant_summary = (
        df.groupby("merchant_category")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .rename(columns={"count": "transaction_count", "mean": "fraud_rate"})
        .reset_index()
    )
    merchant_summary = merchant_summary.sort_values("fraud_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(merchant_summary["merchant_category"], merchant_summary["fraud_rate"] * 100, color="darkorange")
    ax.set_title("Fraud Rate by Merchant Category")
    ax.set_xlabel("Merchant Category")
    ax.set_ylabel("Fraud Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = output_dir / "fraud_rate_by_merchant_category.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["fraud_rate_by_merchant_category.png"] = str(path)

    # 4. Fraud rate by payment method
    payment_summary = (
        df.groupby("payment_method")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .rename(columns={"count": "transaction_count", "mean": "fraud_rate"})
        .reset_index()
    )
    payment_summary = payment_summary.sort_values("fraud_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(payment_summary["payment_method"], payment_summary["fraud_rate"] * 100, color="seagreen")
    ax.set_title("Fraud Rate by Payment Method")
    ax.set_xlabel("Payment Method")
    ax.set_ylabel("Fraud Rate (%)")
    fig.tight_layout()
    path = output_dir / "fraud_rate_by_payment_method.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["fraud_rate_by_payment_method.png"] = str(path)

    # 5. Fraud rate by international transaction
    international_summary = (
        df.groupby("is_international")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .rename(columns={"count": "transaction_count", "mean": "fraud_rate"})
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Domestic", "International"], international_summary["fraud_rate"] * 100, color=["cornflowerblue", "tomato"])
    ax.set_title("Fraud Rate by International Transaction")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Fraud Rate (%)")
    fig.tight_layout()
    path = output_dir / "fraud_rate_by_international.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["fraud_rate_by_international.png"] = str(path)

    # 6. Fraud rate by new device
    new_device_summary = (
        df.groupby("is_new_device")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .rename(columns={"count": "transaction_count", "mean": "fraud_rate"})
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Known Device", "New Device"], new_device_summary["fraud_rate"] * 100, color=["slategray", "crimson"])
    ax.set_title("Fraud Rate by New Device")
    ax.set_xlabel("Device Status")
    ax.set_ylabel("Fraud Rate (%)")
    fig.tight_layout()
    path = output_dir / "fraud_rate_by_new_device.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["fraud_rate_by_new_device.png"] = str(path)

    # 7. Fraud rate by transaction velocity
    velocity_bins = pd.cut(df["transactions_1h"], bins=[-1, 1, 3, 5, 10, 50], labels=["0-1", "2-3", "4-5", "6-10", ">10"])
    velocity_summary = (
        pd.DataFrame({"velocity_bucket": velocity_bins, TARGET_COLUMN: df[TARGET_COLUMN]})
        .groupby("velocity_bucket")[TARGET_COLUMN]
        .agg(["count", "mean"])
        .rename(columns={"count": "transaction_count", "mean": "fraud_rate"})
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(velocity_summary["velocity_bucket"], velocity_summary["fraud_rate"] * 100, color="purple")
    ax.set_title("Fraud Rate by Transaction Velocity")
    ax.set_xlabel("Transactions in Last Hour")
    ax.set_ylabel("Fraud Rate (%)")
    fig.tight_layout()
    path = output_dir / "fraud_rate_by_velocity.png"
    fig.savefig(path)
    plt.close(fig)
    saved_paths["fraud_rate_by_velocity.png"] = str(path)

    return saved_paths


def _pearson_and_spearman(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, pd.DataFrame]:
    """Calculate Pearson and Spearman correlations to the fraud label."""
    correlations: dict[str, pd.DataFrame] = {}

    pearson = df[numeric_columns + [TARGET_COLUMN]].corr(method="pearson")
    spearman = df[numeric_columns + [TARGET_COLUMN]].corr(method="spearman")

    correlations["pearson"] = pearson[[TARGET_COLUMN]].sort_values(TARGET_COLUMN, ascending=False)
    correlations["spearman"] = spearman[[TARGET_COLUMN]].sort_values(TARGET_COLUMN, ascending=False)

    return correlations


def print_eda_report(
    class_distribution: dict[str, Any],
    numeric_summary: dict[str, dict[str, float]],
    categorical_summary: dict[str, pd.DataFrame],
    fraud_pattern_summary: dict[str, Any],
    correlation_summary: dict[str, pd.DataFrame],
) -> None:
    """Print a concise exploratory data analysis summary to the terminal."""
    print("=" * 100)
    print("Fraud Dataset Exploratory Data Analysis")
    print("=" * 100)
    print(f"Fraud transactions: {class_distribution['fraud_count']:,}")
    print(f"Legitimate transactions: {class_distribution['legitimate_count']:,}")
    print(f"Fraud percentage: {class_distribution['fraud_percentage']:.2f}%")
    print(f"Class imbalance ratio (legitimate / fraud): {class_distribution['class_imbalance_ratio']:.2f}:1")

    print("\nKey numeric feature summary (fraud vs legitimate):")
    for feature, values in fraud_pattern_summary.items():
        if feature in {"is_new_device", "is_international"}:
            print(
                f"  {feature}: fraud={values['fraud']:.2f}%, legitimate={values['legitimate']:.2f}%"
            )
        else:
            print(
                f"  {feature}: "
                f"fraud mean={values['fraud']['mean']:.2f}, "
                f"legit mean={values['legitimate']['mean']:.2f}, "
                f"fraud median={values['fraud']['median']:.2f}, "
                f"legit median={values['legitimate']['median']:.2f}"
            )

    print("\nTop fraud-rate categories:")
    for name, summary in categorical_summary.items():
        print(f"  {name}:")
        print(summary.head(5).to_string(index=False))

    print("\nStrongest numerical relationships with is_fraud (Pearson):")
    print(correlation_summary["pearson"].head(10).to_string())
    print("\nStrongest numerical relationships with is_fraud (Spearman):")
    print(correlation_summary["spearman"].head(10).to_string())
    print("=" * 100)


def main() -> None:
    """Run the EDA workflow for the synthetic transaction dataset."""
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    numeric_columns = [
        "amount",
        "transactions_1h",
        "transactions_24h",
        "failed_transactions_24h",
        "account_age_days",
        "card_age_days",
        "average_customer_amount",
        "distance_from_last_transaction_km",
    ]
    categorical_columns = ["merchant_category", "payment_method", "country"]

    class_distribution = analyze_class_distribution(df)
    numeric_summary = analyze_numeric_features(df, numeric_columns)
    categorical_summary = analyze_categorical_features(df, categorical_columns)
    fraud_pattern_summary = analyze_fraud_patterns(df)
    correlation_summary = _pearson_and_spearman(df, numeric_columns)

    saved_plots = create_visualizations(df)
    print_eda_report(
        class_distribution=class_distribution,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        fraud_pattern_summary=fraud_pattern_summary,
        correlation_summary=correlation_summary,
    )
    print("\nSaved plots:")
    for name, path in saved_plots.items():
        print(f"  {name} -> {path}")


if __name__ == "__main__":
    main()
