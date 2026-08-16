from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path("artifacts/models/random_forest.joblib")
PREPROCESSOR_PATH = Path("artifacts/preprocessing/preprocessor.joblib")
THRESHOLD = 0.45


def load_artifacts():
    """Load the trained model and fitted preprocessing transformer."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/models/train_models.py first."
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. "
            "Run python src/features/build_features.py first."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def predict_fraud(model, preprocessor, transaction):
    """Predict fraud probability from a raw transaction dictionary."""

    import pandas as pd

    transaction_df = pd.DataFrame([transaction])

    transaction_df["timestamp"] = pd.to_datetime(
        transaction_df["timestamp"],
        errors="raise",
    )

    transaction_df["hour"] = transaction_df["timestamp"].dt.hour
    transaction_df["day_of_week"] = transaction_df["timestamp"].dt.dayofweek
    transaction_df["day_of_month"] = transaction_df["timestamp"].dt.day
    transaction_df["month"] = transaction_df["timestamp"].dt.month
    transaction_df["is_weekend"] = (
        transaction_df["timestamp"]
        .dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    transaction_df = transaction_df.drop(
        columns=[
            "transaction_id",
            "customer_id",
            "merchant_id",
            "device_id",
            "timestamp",
        ],
        errors="ignore",
    )

    processed_features = preprocessor.transform(transaction_df)

    fraud_probability = float(
        model.predict_proba(processed_features)[0, 1]
    )

    decision = (
        "REVIEW"
        if fraud_probability >= THRESHOLD
        else "APPROVE"
    )

    return {
        "fraud_probability": fraud_probability,
        "threshold": THRESHOLD,
        "decision": decision,
    }