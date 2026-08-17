# Real-Time Fraud Detection System

An end-to-end machine learning system for detecting potentially fraudulent financial transactions using transaction behavior, device information, payment activity, transaction velocity, and geographic signals.

The project includes synthetic data generation, exploratory analysis, feature engineering, imbalanced classification, model comparison, threshold tuning, FastAPI inference, Docker containerization, AWS deployment, prediction logging, and monitoring.

Source / Model Precision Recall F1 ROC-AUC Your Logistic Regression 0.056 0.719 0.103 0.857 Your Random Forest 0.186 0.198 0.191 0.854 Your XGBoost 0.088 0.520 0.150 0.839 Scientific Reports RF ~0.82 ~0.83 ~0.82 ~0.98 Scientific Reports XGBoost lower precision than RF ~0.84 comparable ~0.97 XGBoost ~0.92 ~0.95 ~0.93 ~1.00

## Dataset

The project uses a synthetic transaction dataset containing approximately:

* **200,000 transactions**
* **~1.5% fraudulent transactions**
* **~98.5% legitimate transactions**

Fraud labels are generated probabilistically from combinations of suspicious transaction signals rather than a single deterministic rule.

### Target

```text
0 / False  → Legitimate transaction
1 / True   → Fraudulent transaction
```

### Important Features

The dataset includes behavioral and transaction-level features such as:

```text
amount
merchant_category
country
payment_method
account_age_days
card_age_days
is_new_device
is_international
failed_transactions_24h
transactions_1h
transactions_24h
average_customer_amount
distance_from_last_transaction_km
```

## Class Imbalance

Fraud detection is highly imbalanced.

Only about **1.5% of transactions are fraudulent**, so standard accuracy alone is not sufficient to evaluate the models.

For example, a model predicting every transaction as legitimate could achieve approximately 98.5% accuracy while detecting zero fraud.

Because of this, the project emphasizes:

```text
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Confusion Matrix
```

## Data Preparation

The target variable is separated from the predictive features.

Identifiers such as transaction ID, customer ID, merchant ID, and device ID are excluded from direct model training.

Timestamp information is transformed into:

```text
hour
day_of_week
day_of_month
month
is_weekend
```

Numerical features are standardized with `StandardScaler`.

Categorical variables are transformed using `OneHotEncoder`.

The final preprocessing pipeline produces approximately **45 transformed model features**.

## Train/Test Split

The dataset is split using a stratified train/test split:

```text
Training data: 80%
Testing data:  20%
```

Stratification preserves the fraud rate in both datasets.

Example:

```text
Training rows: ~160,000
Testing rows:  ~40,000
Fraud rate:    ~1.5%
```

## Models

Three supervised classification algorithms are compared.

### Logistic Regression

Provides a simple and interpretable baseline model.

### Random Forest

Combines multiple decision trees and captures nonlinear relationships between transaction behaviors.

### XGBoost

Gradient-boosted trees are evaluated as another nonlinear fraud-detection approach.

## Model Performance

The models are evaluated on the original imbalanced test distribution rather than artificially balancing the test set.

## Model Performance Comparison

| Source / Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Our Logistic Regression | 0.056 | 0.719 | 0.103 | 0.857 |
| Our Random Forest | 0.186 | 0.198 | 0.191 | 0.854 |
| Our XGBoost | 0.088 | 0.520 | 0.150 | 0.839 |
| Scientific Reports - Random Forest | ~0.82 | ~0.83 | ~0.82 | ~0.98 |
| Scientific Reports - XGBoost | Lower than RF | ~0.84 | Comparable | ~0.97 |
| XGBoost Benchmark | ~0.92 | ~0.95 | ~0.93 | ~1.00 |

> **Note:** External benchmark results are shown only for comparison. They were obtained using different datasets, preprocessing methods, class distributions, sampling strategies, and experimental settings, so they are not directly comparable to this project's results.

The fraud prevalence is approximately `0.015`, meaning a random classifier would have a PR-AUC near `0.015`.

The best model therefore achieves a PR-AUC roughly **9× higher than the random baseline**.

## Why Accuracy Is Not the Main Metric

Consider an imbalanced dataset with:

```text
98.5% legitimate
1.5% fraudulent
```

A classifier predicting every transaction as legitimate would achieve approximately:

```text
Accuracy ≈ 98.5%
Fraud Recall = 0%
```

That model would be useless for fraud detection.

For this reason, the project prioritizes PR-AUC, precision, recall, and F1 rather than optimizing accuracy alone.

## Threshold Tuning

The default probability threshold of `0.50` is not automatically optimal for fraud detection.

The project evaluates multiple thresholds to understand the trade-off between:

```text
False positives
        ↕
Fraud detection recall
```

The deployed Random Forest currently uses:

```text
Decision threshold = 0.45
```

Prediction logic:

```text
Fraud probability < 0.45  → APPROVE
Fraud probability ≥ 0.45  → REVIEW
```

At this operating point, the model prioritizes a balance between fraud detection and investigation volume.

Thresholds can be adjusted based on business requirements.

For example:

```text
Lower threshold
→ higher recall
→ more fraud detected
→ more false-positive reviews

Higher threshold
→ higher precision
→ fewer false alarms
→ more fraud may be missed
```

## Real-Time Inference

Transactions are scored through a FastAPI endpoint.

```text
Raw Transaction
      |
      v
Feature Engineering
      |
      v
Saved Preprocessing Pipeline
      |
      v
Random Forest
      |
      v
Fraud Probability
      |
      v
Threshold = 0.45
   /            \
  v              v
APPROVE         REVIEW
```

## Example Prediction

Example high-risk transaction:

```json
{
  "amount": 250.0,
  "is_new_device": true,
  "is_international": false,
  "failed_transactions_24h": 2,
  "transactions_1h": 3,
  "transactions_24h": 8,
  "distance_from_last_transaction_km": 120.0
}
```

Example response:

```json
{
  "fraud_probability": 0.8175,
  "threshold": 0.45,
  "decision": "REVIEW"
}
```

Example low-risk prediction:

```json
{
  "fraud_probability": 0.0169,
  "threshold": 0.45,
  "decision": "APPROVE"
}
```

## API Endpoints

```text
GET  /health
POST /predict
GET  /metrics
```

Swagger documentation is available through:

```text
/docs
```

## Monitoring

Every prediction is logged for lightweight production monitoring.

The monitoring layer tracks:

```text
Total predictions
Approved transactions
Transactions sent for review
Review rate
Average fraud probability
Predictions during the last hour
Recent review rate
```

## Deployment

The application is containerized with Docker and deployed on AWS.

```text
GitHub
   |
   v
Docker
   |
   v
Amazon ECR
   |
   v
Amazon EC2
   |
   v
FastAPI
   |
   v
Fraud Detection API
```

AWS services used:

* Amazon EC2
* Amazon ECR
* IAM roles
* Security Groups

## Tech Stack

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
Random Forest
Logistic Regression
FastAPI
Uvicorn
Docker
Amazon ECR
Amazon EC2
Pytest
Matplotlib
Git/GitHub
```

## Future Improvements

Potential extensions include:

* SHAP-based fraud explanations
* HTTPS and custom domain
* GitHub Actions CI/CD
* automated Docker builds and ECR deployments
* model drift monitoring
* feature drift detection
* real-time streaming with Kafka
* Redis-based feature storage
* fraud investigation dashboard
