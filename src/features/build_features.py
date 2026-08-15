"""Build a reusable feature-engineering and train/test split pipeline.

This module loads the synthetic fraud dataset, engineers timestamp-derived
features, separates the target from the predictors, applies a scikit-learn
ColumnTransformer with one-hot encoding for categorical variables, and saves the
split datasets plus metadata required for downstream modelling.

The workflow is designed to prevent data leakage by fitting preprocessing only
on the training data and by keeping the transformation logic in reusable
functions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "is_fraud"
IDENTIFIER_COLUMNS = ["transaction_id", "customer_id", "merchant_id", "device_id"]
NUMERIC_FEATURES = [
    "amount",
    "account_age_days",
    "card_age_days",
    "failed_transactions_24h",
    "transactions_1h",
    "transactions_24h",
    "average_customer_amount",
    "distance_from_last_transaction_km",
    "is_new_device",
    "is_international",
]
CATEGORICAL_FEATURES = ["merchant_category", "country", "payment_method"]


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the transaction dataset from the repository data directory.

    Args:
        data_path: Optional custom dataset path. If omitted, the repository's
            ``data/transactions.csv`` file is used.

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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered feature columns and return a ready-to-model dataset.

    Args:
        df: Input transaction dataframe.

    Returns:
        pandas.DataFrame: Dataframe with raw identifiers removed and timestamp-
            derived features added.
    """
    working_df = df.copy()

    # Convert timestamp to pandas datetime to avoid using the raw timestamp value as
    # a direct model feature. The derived temporal features capture seasonal and
    # behavioral patterns without leaking exact event times.
    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], errors="raise")
    working_df["hour"] = working_df["timestamp"].dt.hour
    working_df["day_of_week"] = working_df["timestamp"].dt.dayofweek
    working_df["day_of_month"] = working_df["timestamp"].dt.day
    working_df["month"] = working_df["timestamp"].dt.month
    working_df["is_weekend"] = working_df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int)

    # Remove identifiers that are not valid direct predictors while preserving the
    # target and the engineered temporal features needed for modeling.
    feature_columns = [
        col
        for col in working_df.columns
        if col not in IDENTIFIER_COLUMNS + ["timestamp"]
    ]
    return working_df[feature_columns]


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets with stratification.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Proportion of the data reserved for testing.
        random_state: Random state for reproducibility.
        stratify: Whether to stratify the split by target class.

    Returns:
        tuple: ``X_train, X_test, y_train, y_test``.
    """
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for numerical and categorical inputs.

    The numerical features are standardized using ``StandardScaler``. The
    categorical features use ``OneHotEncoder`` with ``handle_unknown='ignore'``.
    This preprocessing is fitted only on the training data to prevent data
    leakage.

    Args:
        X_train: Training feature matrix used to infer the feature layout.

    Returns:
        ColumnTransformer: The configured preprocessing transformer.
    """
    feature_names = list(X_train.columns)
    numerical = [col for col in feature_names if col in NUMERIC_FEATURES]
    categorical = [col for col in feature_names if col in CATEGORICAL_FEATURES]

    # Include derived timestamp features in the numeric set because they are
    # ordinal and can be used by the model directly after scaling.
    timestamp_features = ["hour", "day_of_week", "day_of_month", "month", "is_weekend"]
    numerical = list(dict.fromkeys(numerical + timestamp_features))
    categorical = [col for col in categorical if col in feature_names]

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("scaler", StandardScaler())]),
                numerical,
            ),
            (
                "categorical",
                Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return transformer


def _save_array(array: np.ndarray, path: Path) -> None:
    """Persist a dense NumPy array to disk.

    Args:
        array: Feature matrix to save.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_processed_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    transformed_feature_names: list[str],
    metadata_path: Path,
) -> None:
    """Save processed data and split metadata to the data/processed directory.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_test: Test target vector.
        transformed_feature_names: Preprocessed feature names after encoding.
        metadata_path: Destination path for metadata JSON.
    """
    processed_dir = metadata_path.parent
    processed_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    _save_array(X_train_processed, processed_dir / "X_train.npy")
    _save_array(X_test_processed, processed_dir / "X_test.npy")
    np.save(processed_dir / "y_train.npy", y_train.to_numpy())
    np.save(processed_dir / "y_test.npy", y_test.to_numpy())

    metadata = {
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_fraud_records": int(y_train.sum()),
        "test_fraud_records": int(y_test.sum()),
        "training_fraud_percentage": float((y_train.mean() * 100) if len(y_train) else 0.0),
        "test_fraud_percentage": float((y_test.mean() * 100) if len(y_test) else 0.0),
        "feature_names_after_preprocessing": transformed_feature_names,
        "transformed_feature_count": int(X_train_processed.shape[1]),
        "raw_feature_count": int(X_train.shape[1]),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    """Run the full feature engineering and data-splitting workflow."""
    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    feature_frame = create_features(df)
    X = feature_frame.drop(columns=[TARGET_COLUMN])
    y = feature_frame[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42, stratify=True)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    transformed_feature_names = list(preprocessor.get_feature_names_out())
    metadata_path = Path("data/processed/dataset_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    _save_array(X_train_processed, Path("data/processed/X_train.npy"))
    _save_array(X_test_processed, Path("data/processed/X_test.npy"))
    np.save(Path("data/processed/y_train.npy"), y_train.to_numpy())
    np.save(Path("data/processed/y_test.npy"), y_test.to_numpy())

    metadata = {
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_fraud_records": int(y_train.sum()),
        "test_fraud_records": int(y_test.sum()),
        "training_fraud_percentage": float((y_train.mean() * 100) if len(y_train) else 0.0),
        "test_fraud_percentage": float((y_test.mean() * 100) if len(y_test) else 0.0),
        "feature_names_after_preprocessing": transformed_feature_names,
        "transformed_feature_count": int(X_train_processed.shape[1]),
        "raw_feature_count": int(X_train.shape[1]),
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Feature engineering and split pipeline completed.")
    print(f"Original rows: {len(df):,}")
    print(f"Number of raw features: {X.shape[1]}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Training fraud count: {y_train.sum():,}")
    print(f"Test fraud count: {y_test.sum():,}")
    print(f"Training fraud percentage: {y_train.mean() * 100:.2f}%")
    print(f"Test fraud percentage: {y_test.mean() * 100:.2f}%")
    print(f"Transformed feature count: {X_train_processed.shape[1]}")


if __name__ == "__main__":
    main()
