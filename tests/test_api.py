from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "RandomForestClassifier"


def test_predict_low_risk_transaction():
    transaction = {
        "transaction_id": "TXN_TEST_LOW",
        "customer_id": "CUST_TEST_LOW",
        "timestamp": "2024-08-16 10:15:00",
        "amount": 18.0,
        "merchant_id": "MERCHANT_TEST_LOW",
        "merchant_category": "grocery",
        "country": "US",
        "device_id": "DEV_TEST_LOW",
        "payment_method": "credit_card",
        "account_age_days": 500,
        "card_age_days": 300,
        "is_new_device": False,
        "is_international": False,
        "failed_transactions_24h": 0,
        "transactions_1h": 1,
        "transactions_24h": 3,
        "average_customer_amount": 25.0,
        "distance_from_last_transaction_km": 4.0,
    }

    response = client.post("/predict", json=transaction)

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "threshold" in data
    assert "decision" in data

    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["threshold"] == 0.45
    assert data["decision"] in ["APPROVE", "REVIEW"]


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "approve_count" in data
    assert "review_count" in data
    assert "review_rate" in data
    assert "average_fraud_probability" in data
    assert "recent_predictions_1h" in data
    assert "recent_review_rate_1h" in data