"""Generate a realistic synthetic transaction dataset for fraud detection.

This module creates a synthetic financial transaction dataset designed for a
real-time fraud detection and risk decisioning project. The generator is
structured to reflect realistic fraud behavior rather than assigning fraud labels
randomly.

The data includes transaction-level signals such as amount, device information,
merchant context, account age, country, transaction velocity, failed payment
behavior, and distance from the previous transaction. Fraud labels are assigned
only to the highest-risk minority of rows in order to maintain a realistic
approximate fraud rate of 1.5%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_transactions(
    n_transactions: int = 200_000,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a synthetic fraud-detection dataset and save it to CSV.

    Args:
        n_transactions: Number of synthetic transactions to generate.
        seed: Random seed for reproducible output.
        output_path: Destination CSV path. If not provided, the dataset is saved
            under the repository's data directory.

    Returns:
        pandas.DataFrame: The generated transaction dataset.
    """
    rng = np.random.default_rng(seed)

    # Keep the project rooted to the repository directory so the script works when
    # executed from any location, including Codespaces terminals.
    if output_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / "data" / "transactions.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthetic customer and merchant counts. The purpose is to create repeated
    # customer behavior with realistic amounts, velocities, and device usage.
    n_customers = max(4000, int(n_transactions / 20))
    n_merchants = max(2000, int(n_transactions / 40))
    n_devices = max(6000, int(n_transactions / 15))

    countries = ["US", "CA", "GB", "DE", "FR", "ES", "AU", "SG", "JP", "IN", "BR"]
    merchant_categories = [
        "grocery",
        "restaurants",
        "fuel",
        "transport",
        "utilities",
        "healthcare",
        "retail",
        "travel",
        "entertainment",
        "electronics",
        "digital_goods",
        "subscriptions",
        "cash_advance",
        "gaming",
        "gift_cards",
    ]

    # Merchant category should be a useful, but not deterministic, signal. We give
    # a modest probability boost to categories that are more often associated with
    # suspicious behavior, while still allowing legitimate transactions in every
    # category and fraud to appear in every category with a low probability.
    merchant_category_risk = {
        "grocery": 0.05,
        "restaurants": 0.06,
        "fuel": 0.07,
        "transport": 0.08,
        "utilities": 0.05,
        "healthcare": 0.04,
        "retail": 0.09,
        "travel": 0.16,
        "entertainment": 0.12,
        "electronics": 0.18,
        "digital_goods": 0.20,
        "subscriptions": 0.15,
        "cash_advance": 0.22,
        "gaming": 0.26,
        "gift_cards": 0.24,
    }

    category_probs = np.array(
        [
            0.16,
            0.14,
            0.08,
            0.08,
            0.09,
            0.08,
            0.12,
            0.06,
            0.06,
            0.04,
            0.03,
            0.02,
            0.02,
            0.01,
            0.01,
        ],
        dtype=float,
    )

    # Create customer and merchant identifiers in a synthetic format; no real
    # customer or financial information is used.
    customer_ids = [f"CUST_{i:07d}" for i in range(1, n_customers + 1)]
    merchant_ids = [f"MERCHANT_{i:07d}" for i in range(1, n_merchants + 1)]
    device_ids = [f"DEV_{i:07d}" for i in range(1, n_devices + 1)]

    customer_home_country = rng.choice(countries, size=n_customers)
    customer_base_amount = np.clip(rng.lognormal(mean=3.2, sigma=0.65, size=n_customers), 5, 8000)

    # Assign each transaction a customer and a time window.
    transaction_time_start = pd.Timestamp("2024-01-01 00:00:00")
    transaction_time_end = pd.Timestamp("2024-12-31 23:59:00")
    time_seconds = np.random.default_rng(seed + 1).integers(
        0,
        int((transaction_time_end - transaction_time_start).total_seconds()) + 1,
        size=n_transactions,
    )
    timestamps = transaction_time_start + pd.to_timedelta(time_seconds, unit="s")

    customer_idx = rng.integers(0, n_customers, size=n_transactions)
    merchant_idx = rng.integers(0, n_merchants, size=n_transactions)

    # Generate a realistic base transaction distribution. Normal transactions are
    # the majority; later we identify a smaller set of high-risk transactions by
    # examining suspicious patterns that are typical in fraud.
    base_amounts = np.clip(
        rng.lognormal(mean=3.1, sigma=0.75, size=n_transactions),
        2.0,
        15000.0,
    )
    payment_methods = np.array(["credit_card", "debit_card", "wallet", "bank_transfer"], dtype=object)
    payment_method = rng.choice(payment_methods, size=n_transactions)

    account_age_days = np.clip(rng.lognormal(mean=4.9, sigma=0.9, size=n_transactions), 2, 5000)
    card_age_days = np.clip(rng.lognormal(mean=4.2, sigma=0.9, size=n_transactions), 1, 4000)

    # Device behavior is important in fraud modeling. A new device is often a
    # meaningful risk indicator, especially when combined with velocity, travel,
    # or unusual merchant activity.
    is_new_device = rng.random(n_transactions) < 0.10
    device_used = rng.integers(0, n_devices, size=n_transactions)
    device_id = np.array([device_ids[int(i)] for i in device_used], dtype=object)

    # Customers have home countries; some transactions are international based on
    # destination country, which is a strong fraud signal when combined with
    # device or amount anomalies.
    home_country = customer_home_country[customer_idx]
    international_flag = rng.random(n_transactions) < 0.08
    country = np.array(home_country, dtype=object)
    foreign_country_pool = np.array(countries, dtype=object)
    foreign_idx = np.where(international_flag)[0]
    country[foreign_idx] = rng.choice(foreign_country_pool, size=len(foreign_idx))

    merchant_category = rng.choice(merchant_categories, size=n_transactions, p=category_probs)
    merchant_category_risk_vector = np.array(
        [merchant_category_risk[category] for category in merchant_category],
        dtype=float,
    )

    # Transaction counts over 1 hour and 24 hours capture velocity, a classic fraud
    # signal. Fraudulent transactions often show surprisingly high activity in a
    # short period.
    transactions_1h = rng.poisson(1.8, size=n_transactions)
    transactions_24h = rng.poisson(5.5, size=n_transactions)
    failed_transactions_24h = rng.poisson(0.5, size=n_transactions)

    # A customer-specific average amount captures normal spending, while large
    # deviations from that pattern become suspicious when combined with other risk
    # signals.
    average_customer_amount = customer_base_amount[customer_idx]
    amount = base_amounts * rng.uniform(0.7, 1.5, size=n_transactions)
    amount = np.clip(amount, 1.0, 25000.0)

    # Distance from the previous transaction helps model travel fraud patterns.
    distance_from_last_transaction_km = np.clip(
        rng.gamma(shape=2.0, scale=45.0, size=n_transactions),
        0.0,
        5000.0,
    )

        # Build a probabilistic fraud propensity from realistic transaction signals.
    # Unlike a deterministic top-risk selection, this introduces overlap between
    # legitimate and fraudulent transactions so models must generalize.
    amount_z = (
        np.log1p(amount) - np.mean(np.log1p(amount))
    ) / np.std(np.log1p(amount))

    velocity_1h = np.log1p(transactions_1h)
    velocity_24h = np.log1p(transactions_24h)
    failed_signal = np.log1p(failed_transactions_24h)
    distance_signal = np.log1p(distance_from_last_transaction_km)
    young_account_signal = (account_age_days < 30).astype(float)

    risk_score = (
        0.65 * amount_z
        + 0.55 * is_new_device.astype(float)
        + 0.65 * international_flag.astype(float)
        + 0.55 * failed_signal
        + 0.60 * velocity_1h
        + 0.45 * velocity_24h
        + 0.50 * distance_signal
        + 0.45 * young_account_signal
        + 0.30 * merchant_category_risk_vector
    )

    # Add unobserved randomness so fraud is not a deterministic function
    # of the exact features used by the model.
    latent_score = 2.0 * risk_score + rng.normal(
        loc=0.0,
        scale=0.45,
        size=n_transactions,
    )

    # Calibrate fraud probability to approximately 1.5% prevalence.
    target_fraud_rate = 0.015

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    lower = -15.0
    upper = 5.0

    for _ in range(60):
        midpoint = (lower + upper) / 2.0
        fraud_probability = sigmoid(latent_score + midpoint)

        if fraud_probability.mean() > target_fraud_rate:
            upper = midpoint
        else:
            lower = midpoint

    intercept = (lower + upper) / 2.0
    fraud_probability = sigmoid(latent_score + intercept)

    # Sample fraud probabilistically instead of forcing exact labels.
    is_fraud = rng.random(n_transactions) < fraud_probability



    

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN_{i:09d}" for i in range(1, n_transactions + 1)],
            "customer_id": np.array([customer_ids[i] for i in customer_idx], dtype=object),
            "timestamp": timestamps,
            "amount": amount.astype(float),
            "merchant_id": np.array([merchant_ids[i] for i in merchant_idx], dtype=object),
            "merchant_category": merchant_category,
            "country": country,
            "device_id": device_id,
            "payment_method": payment_method,
            "account_age_days": account_age_days.astype(float),
            "card_age_days": card_age_days.astype(float),
            "is_new_device": is_new_device.astype(bool),
            "is_international": international_flag.astype(bool),
            "failed_transactions_24h": failed_transactions_24h.astype(int),
            "transactions_1h": transactions_1h.astype(int),
            "transactions_24h": transactions_24h.astype(int),
            "average_customer_amount": average_customer_amount.astype(float),
            "distance_from_last_transaction_km": distance_from_last_transaction_km.astype(float),
            "is_fraud": is_fraud.astype(bool),
        }
    )

    # Enforce a consistent output order matching the specification.
    columns = [
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
    df = df[columns]

    # Save CSV in the repository data directory. This is the only generated data
    # artifact meant to be kept beyond the script itself.
    df.to_csv(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial transaction data for fraud detection modeling."
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=200_000,
        help="Number of synthetic transactions to generate (default: 200000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/transactions.csv",
        help="Destination path for the generated CSV file.",
    )
    return parser.parse_args()


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise generation summary after the data is created."""
    fraud_count = int(df["is_fraud"].sum())
    fraud_percentage = (fraud_count / len(df)) * 100 if len(df) else 0.0

    print("Synthetic transaction dataset generated successfully.")
    print(f"Number of transactions: {len(df):,}")
    print(f"Number of fraudulent transactions: {fraud_count:,}")
    print(f"Fraud percentage: {fraud_percentage:.2f}%")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Basic amount statistics:")
    print(df["amount"].describe().to_string())


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    output_path = Path(args.output)

    # Ensure the output path resolves relative to the repository root when the
    # script is invoked from anywhere in the workspace.
    if not output_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        output_path = repo_root / output_path

    dataset = generate_transactions(
        n_transactions=args.transactions,
        seed=args.seed,
        output_path=output_path,
    )
    print_summary(dataset)


if __name__ == "__main__":
    main()
