# Real-Time Fraud Detection System

An end-to-end machine learning system for detecting potentially fraudulent financial transactions in real time.

The project covers synthetic transaction generation, exploratory analysis, feature engineering, imbalanced classification, model comparison, threshold tuning, real-time inference with FastAPI, prediction logging, and lightweight model monitoring.

## Key Features

* Synthetic dataset with 200,000 financial transactions and approximately 1.5% fraud prevalence
* Feature engineering for transaction amount, velocity, device behavior, international activity, account history, and transaction distance
* Logistic Regression, Random Forest, and XGBoost model comparison
* Model evaluation using ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrices
* Decision-threshold tuning for imbalanced fraud detection
* Real-time fraud scoring through a FastAPI REST API
* `APPROVE` and `REVIEW` risk decisions
* Prediction logging for production monitoring
* API monitoring metrics including prediction volume, review rate, and average fraud probability

## System Architecture

The system follows an end-to-end machine learning workflow from transaction generation to real-time fraud monitoring.

```text
Synthetic Transaction Data
          |
          v
Exploratory Data Analysis
          |
          v
Feature Engineering
          |
          v
Preprocessing Pipeline
          |
          v
Model Training
(Logistic Regression / Random Forest / XGBoost)
          |
          v
Model Evaluation + Threshold Tuning
          |
          v
Random Forest Model
          |
          v
FastAPI Real-Time Inference
          |
          v
Fraud Probability
          |
          v
Threshold = 0.45
      /         \
     v           v
 APPROVE       REVIEW
     \           /
      v         v
 Prediction Logging
          |
          v
Monitoring Metrics
```

### Current Inference Flow

A transaction is submitted to the `/predict` endpoint. The saved preprocessing pipeline transforms the raw transaction into the same feature representation used during model training.

The trained Random Forest model generates a fraud probability, and the tuned `0.45` decision threshold converts that probability into an `APPROVE` or `REVIEW` decision.

Each prediction is written to `monitoring/predictions.csv`, allowing the monitoring component to track prediction volume, review rate, and average fraud probability.

## Model Performance

The models were evaluated on a stratified test set of 40,000 transactions. Because fraud represents only about 1.5% of the dataset, PR-AUC, precision, recall, and F1 are more informative than accuracy alone.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0557 | 0.7189 | 0.1034 | 0.8569 | 0.1333 |
| Random Forest | 0.1856 | 0.1976 | 0.1914 | 0.8539 | **0.1381** |
| XGBoost | 0.0879 | 0.5196 | 0.1503 | 0.8385 | 0.1330 |

Random Forest achieved the highest PR-AUC and was selected as the production inference model.

### Threshold Tuning

Using the default classification threshold is not always appropriate for highly imbalanced fraud detection. Threshold analysis was therefore performed to evaluate the trade-off between precision and recall.

For the Random Forest model, a threshold of `0.45` produced:

* Precision: **0.1675**
* Recall: **0.2794**
* F1 Score: **0.2095**
* True Positives: **164**
* False Positives: **815**
* False Negatives: **423**

The `0.45` threshold is used by the real-time inference layer to convert model probabilities into `APPROVE` or `REVIEW` decisions.

## API Usage

The trained model is exposed through a FastAPI application for real-time fraud scoring.

### Start the API

```bash
uvicorn api.main:app --reload
```

The API runs locally at port `8000`.

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "model": "RandomForestClassifier"
}
```

### Fraud Prediction

Send a transaction to the `/predict` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "transaction_id": "TXN_TEST_001",
  "customer_id": "CUST_TEST_001",
  "timestamp": "2024-08-16 14:30:00",
  "amount": 250.0,
  "merchant_id": "MERCHANT_TEST_001",
  "merchant_category": "electronics",
  "country": "US",
  "device_id": "DEV_TEST_001",
  "payment_method": "credit_card",
  "account_age_days": 120,
  "card_age_days": 90,
  "is_new_device": true,
  "is_international": false,
  "failed_transactions_24h": 2,
  "transactions_1h": 3,
  "transactions_24h": 8,
  "average_customer_amount": 45.0,
  "distance_from_last_transaction_km": 120.0
}'
```

Example response:

```json
{
  "fraud_probability": 0.8175,
  "threshold": 0.45,
  "decision": "REVIEW"
}
```

Transactions with a fraud probability greater than or equal to `0.45` are sent for review; lower-risk transactions are approved.

### Monitoring Metrics

```bash
curl http://127.0.0.1:8000/metrics
```

Example response:

```json
{
  "total_predictions": 2,
  "approve_count": 1,
  "review_count": 1,
  "review_rate": 0.5,
  "average_fraud_probability": 0.4172,
  "recent_predictions_1h": 2,
  "recent_review_rate_1h": 0.5
}
```

Prediction events are logged to:

```text
monitoring/predictions.csv
```
## Project Structure

```text
real-time-fraud-detection/
├── api/
│   └── main.py
├── artifacts/
│   ├── models/
│   └── preprocessing/
├── data/
│   ├── processed/
│   └── transactions.csv
├── monitoring/
│   └── predictions.csv
├── reports/
│   ├── figures/
│   ├── model_metrics.csv
│   ├── threshold_analysis.csv
│   └── threshold_recommendation.json
├── src/
│   ├── analysis/
│   │   ├── explore_dataset.py
│   │   └── validate_dataset.py
│   ├── data/
│   │   └── generate_transactions.py
│   ├── features/
│   │   └── build_features.py
│   ├── inference/
│   │   └── predict.py
│   ├── models/
│   │   ├── train_models.py
│   │   └── tune_threshold.py
│   └── monitoring/
│       └── monitor_predictions.py
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository and move into the project directory:

```bash
git clone <repository-url>
cd real-time-fraud-detection
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the ML Pipeline

### 1. Generate synthetic transactions

```bash
python src/data/generate_transactions.py
```

### 2. Build features and preprocessing artifacts

```bash
python src/features/build_features.py
```

### 3. Train and evaluate models

```bash
python src/models/train_models.py
```

### 4. Tune classification thresholds

```bash
python src/models/tune_threshold.py
```

### 5. Start the real-time API

```bash
uvicorn api.main:app --reload
```

### 6. Run the monitoring report

After sending predictions through the API:

```bash
python src/monitoring/monitor_predictions.py
```