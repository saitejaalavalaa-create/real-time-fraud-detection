from fastapi import FastAPI
from pydantic import BaseModel
import csv
from datetime import datetime, timezone
from pathlib import Path
from src.monitoring.monitor_predictions import (
    load_predictions,
    generate_monitoring_report,
)

from src.inference.predict import load_artifacts, predict_fraud


app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="1.0.0",
)

model, preprocessor = load_artifacts()
LOG_PATH = Path("monitoring/predictions.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    timestamp: str
    amount: float
    merchant_id: str
    merchant_category: str
    country: str
    device_id: str
    payment_method: str
    account_age_days: float
    card_age_days: float
    is_new_device: bool
    is_international: bool
    failed_transactions_24h: int
    transactions_1h: int
    transactions_24h: int
    average_customer_amount: float
    distance_from_last_transaction_km: float
@app.get("/metrics")
def metrics():
    df = load_predictions()
    return generate_monitoring_report(df)


@app.get("/")
def root():
    return {
        "service": "Real-Time Fraud Detection API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
    }

def log_prediction(transaction, result):
    file_exists = LOG_PATH.exists()

    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "logged_at",
                "transaction_id",
                "amount",
                "fraud_probability",
                "threshold",
                "decision",
            ])

        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            transaction.transaction_id,
            transaction.amount,
            result["fraud_probability"],
            result["threshold"],
            result["decision"],
        ])
@app.post("/predict")
def predict(transaction: Transaction):
    result = predict_fraud(
        model,
        preprocessor,
        transaction.model_dump(),
    )

    log_prediction(transaction, result)

    return result