"""Train and evaluate baseline fraud-detection models.

This module loads the processed train/test arrays, trains three models designed
for highly imbalanced fraud data, evaluates them using classification metrics,
and saves the results plus model artifacts for further comparison.

The workflow is intentionally model-focused and does not perform threshold tuning,
SMOTE, or other imbalance strategies yet. Instead, it uses class-weighting and
balanced model selection based primarily on PR-AUC.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = Path("artifacts") / "models"


def load_processed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the processed training and test arrays from disk.

    Returns:
        tuple: ``(X_train, X_test, y_train, y_test)`` as dense NumPy arrays.

    Raises:
        FileNotFoundError: If any required processed file is missing.
    """
    required_files = [
        PROCESSED_DIR / "X_train.npy",
        PROCESSED_DIR / "X_test.npy",
        PROCESSED_DIR / "y_train.npy",
        PROCESSED_DIR / "y_test.npy",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed training data. Please generate the dataset first via: "
            "python src/features/build_features.py"
        )

    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    return X_train, X_test, y_train, y_test


def build_models(y_train: np.ndarray) -> dict[str, Any]:
    """Build a dictionary of models configured for imbalanced classification.

    Args:
        y_train: Training labels used to compute class-balance weights.

    Returns:
        dict: A mapping of model names to initialized model objects.
    """
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            n_jobs=None,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=5,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Evaluate a fit model on the held-out test set.

    Args:
        model: Trained scikit-learn or XGBoost model.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        dict: Model evaluation metrics and confusion matrix results.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Save model comparison metrics to a CSV file.

    Args:
        results: A list of result dictionaries, one per model.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def save_confusion_matrix(model_name: str, cm: np.ndarray, output_path: Path) -> None:
    """Save a confusion matrix plot for a model.

    Args:
        model_name: Label used in the filename.
        cm: Confusion matrix array.
        output_path: Destination PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    img = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_yticklabels(["Legitimate", "Fraud"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="black")

    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_model(model: Any, filepath: Path) -> None:
    """Persist a trained model to disk.

    Args:
        model: Trained model to save.
        filepath: Destination joblib file.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def print_model_report(results: list[dict[str, Any]]) -> None:
    """Print a readable evaluation summary for each model.

    Args:
        results: A list of result dictionaries from the evaluation step.
    """
    print("=" * 120)
    print("Fraud Model Evaluation Report")
    print("=" * 120)

    for result in results:
        print(f"Model: {result['model_name']}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1: {result['f1']:.4f}")
        print(f"  ROC-AUC: {result['roc_auc']:.4f}")
        print(f"  PR-AUC: {result['pr_auc']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print("  Confusion matrix:")
        print(result["confusion_matrix"])
        print()


def main() -> None:
    """Train the baseline fraud-detection models and save artifacts."""
    try:
        X_train, X_test, y_train, y_test = load_processed_data()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    models = build_models(y_train)
    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            result = {
                "model_name": model_name,
                **metrics,
            }
            results.append(result)

            artifact_path = ARTIFACTS_DIR / f"{model_name.replace('_', '_')}.joblib"
            if model_name == "logistic_regression":
                artifact_path = ARTIFACTS_DIR / "logistic_regression.joblib"
            elif model_name == "random_forest":
                artifact_path = ARTIFACTS_DIR / "random_forest.joblib"
            elif model_name == "xgboost":
                artifact_path = ARTIFACTS_DIR / "xgboost.joblib"
            save_model(model, artifact_path)

            confusion_plot_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
            save_confusion_matrix(model_name, metrics["confusion_matrix"], confusion_plot_path)

            if best_result is None or metrics["pr_auc"] > best_result["pr_auc"]:
                best_result = result.copy()

        except Exception as exc:  # pragma: no cover - defensive runtime error handling.
            print(f"Error training or evaluating {model_name}: {exc}", file=sys.stderr)
            continue

    if not results:
        raise RuntimeError("No models were successfully trained and evaluated.")

    save_results(results, REPORTS_DIR / "model_metrics.csv")

    if best_result is None:
        raise RuntimeError("No valid best model could be determined from the evaluation results.")

    best_model_name = best_result["model_name"]
    best_model_summary = {
        "best_model": best_model_name,
        "metrics": {
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1": best_result["f1"],
            "roc_auc": best_result["roc_auc"],
            "pr_auc": best_result["pr_auc"],
        },
    }
    with (REPORTS_DIR / "best_model.json").open("w", encoding="utf-8") as file:
        json.dump(best_model_summary, file, indent=2)

    print_model_report(results)
    print(f"Best model selected by PR-AUC: {best_model_name}")
    print(f"Best PR-AUC: {best_result['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
