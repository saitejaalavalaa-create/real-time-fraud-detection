"""
Threshold Tuning Script for Fraud Detection Models

This script evaluates different classification thresholds for fraud detection models
and identifies optimal thresholds based on various metrics (F1, recall, precision).

Usage:
    python src/models/tune_threshold.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import load
from sklearn.metrics import confusion_matrix, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

# Set matplotlib backend for reproducibility
plt.switch_backend("Agg")

# Configuration
SEED = 42
np.random.seed(SEED)

# Paths
ARTIFACTS_DIR = Path("artifacts/models")
DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models_and_data():
    """
    Load trained models and test data.

    Returns:
        dict: Dictionary with models and test data
            - models: dict with model_name -> model object
            - X_test: numpy array of shape (n_samples, n_features)
            - y_test: numpy array of shape (n_samples,)
    """
    print("Loading models and data...")

    models = {}
    model_names = ["logistic_regression", "random_forest", "xgboost"]

    for model_name in model_names:
        model_path = ARTIFACTS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        models[model_name] = load(model_path)
        print(f"  ✓ Loaded {model_name}")

    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    print(f"  ✓ Loaded X_test: {X_test.shape}")
    print(f"  ✓ Loaded y_test: {y_test.shape}")

    return {"models": models, "X_test": X_test, "y_test": y_test}


def get_predictions_proba(model, X_test):
    """
    Get fraud probability predictions from a model.

    Args:
        model: Trained classification model
        X_test: Test features array

    Returns:
        numpy array: Fraud probabilities (1D array)
    """
    proba = model.predict_proba(X_test)
    # Handle binary classification (proba is shape (n_samples, 2))
    return proba[:, 1]


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics from confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)

    Returns:
        dict: Dictionary containing:
            - tp: True positives
            - tn: True negatives
            - fp: False positives
            - fn: False negatives
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1: Harmonic mean of precision and recall
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Handle zero-division safely
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_thresholds(y_true, y_proba, thresholds):
    """
    Evaluate model performance across different classification thresholds.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: List/array of thresholds to evaluate

    Returns:
        pd.DataFrame: Results with columns for each metric and threshold
    """
    results = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred)
        metrics["threshold"] = float(threshold)
        results.append(metrics)

    return pd.DataFrame(results)


def find_best_threshold(df_results, metric="f1"):
    """
    Find the threshold with the best metric value.

    Args:
        df_results: DataFrame with threshold evaluation results
        metric: Metric to optimize (default: 'f1')

    Returns:
        tuple: (best_threshold, best_value)
    """
    idx_best = df_results[metric].idxmax()
    return df_results.loc[idx_best, "threshold"], df_results.loc[idx_best, metric]


def find_recall_constrained_threshold(df_results, min_recall=0.95):
    """
    Find the lowest threshold that achieves at least min_recall.

    Args:
        df_results: DataFrame with threshold evaluation results
        min_recall: Minimum recall to achieve

    Returns:
        tuple: (threshold, recall) or (None, None) if not achievable
    """
    candidates = df_results[df_results["recall"] >= min_recall]
    if len(candidates) > 0:
        # Return the highest threshold that still achieves min_recall
        idx_best = candidates["threshold"].idxmax()
        return candidates.loc[idx_best, "threshold"], candidates.loc[idx_best, "recall"]
    return None, None


def plot_precision_recall_vs_threshold(results_dict, output_path):
    """
    Create a plot comparing precision and recall across thresholds for all models.

    Args:
        results_dict: Dictionary with model_name -> DataFrame of results
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Precision and Recall vs Threshold", fontsize=14, fontweight="bold")

    model_names = list(results_dict.keys())
    for idx, model_name in enumerate(model_names):
        df = results_dict[model_name]
        ax = axes[idx]

        ax.plot(df["threshold"], df["precision"], marker="o", label="Precision", linewidth=2)
        ax.plot(df["threshold"], df["recall"], marker="s", label="Recall", linewidth=2)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(model_name.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved precision-recall plot: {output_path}")
    plt.close()


def plot_f1_vs_threshold(results_dict, output_path):
    """
    Create a plot comparing F1 scores across thresholds for all models.

    Args:
        results_dict: Dictionary with model_name -> DataFrame of results
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("F1 Score vs Threshold", fontsize=14, fontweight="bold")

    model_names = list(results_dict.keys())
    for idx, model_name in enumerate(model_names):
        df = results_dict[model_name]
        ax = axes[idx]

        ax.plot(df["threshold"], df["f1"], marker="o", color="green", linewidth=2)
        ax.scatter(
            df["threshold"].iloc[df["f1"].idxmax()],
            df["f1"].max(),
            color="red",
            s=100,
            zorder=5,
            label=f"Best F1: {df['f1'].max():.3f}",
        )
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_title(model_name.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved F1 plot: {output_path}")
    plt.close()


def print_report(results_dict, recommendations):
    """
    Print a formatted terminal report of threshold analysis results.

    Args:
        results_dict: Dictionary with model_name -> DataFrame of results
        recommendations: Dictionary with model_name -> recommendation dict
    """
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS REPORT".center(80))
    print("=" * 80)

    for model_name in results_dict.keys():
        df = results_dict[model_name]
        rec = recommendations[model_name]

        print(f"\n{'Model: ' + model_name.upper():-^80}")
        print(f"\n{'Best F1 Threshold':30s} {rec['f1_threshold']:.2f}")
        print(f"{'  - Precision':30s} {rec['f1_metrics']['precision']:.4f}")
        print(f"{'  - Recall':30s} {rec['f1_metrics']['recall']:.4f}")
        print(f"{'  - F1 Score':30s} {rec['f1_metrics']['f1']:.4f}")
        print(f"{'  - True Positives':30s} {rec['f1_metrics']['tp']}")
        print(f"{'  - False Positives':30s} {rec['f1_metrics']['fp']}")
        print(f"{'  - False Negatives':30s} {rec['f1_metrics']['fn']}")

        if rec["recall_95_threshold"] is not None:
            print(f"\n{'95% Recall Threshold':30s} {rec['recall_95_threshold']:.2f}")
            print(f"{'  - Precision':30s} {rec['recall_95_metrics']['precision']:.4f}")
            print(f"{'  - Recall':30s} {rec['recall_95_metrics']['recall']:.4f}")
            print(f"{'  - F1 Score':30s} {rec['recall_95_metrics']['f1']:.4f}")
            print(f"{'  - True Positives':30s} {rec['recall_95_metrics']['tp']}")
            print(f"{'  - False Positives':30s} {rec['recall_95_metrics']['fp']}")
            print(f"{'  - False Negatives':30s} {rec['recall_95_metrics']['fn']}")
        else:
            print(f"\n{'95% Recall Threshold':30s} NOT ACHIEVABLE")

        # Summary statistics
        print(f"\n{'Threshold Range':30s} {df['threshold'].min():.2f} - {df['threshold'].max():.2f}")
        print(f"{'Max Precision':30s} {df['precision'].max():.4f}")
        print(f"{'Max Recall':30s} {df['recall'].max():.4f}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("FRAUD DETECTION THRESHOLD TUNING".center(80))
    print("=" * 80 + "\n")

    # Load data
    data = load_models_and_data()
    models = data["models"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Define thresholds to evaluate
    thresholds = np.concatenate([
    np.arange(0.01, 0.10, 0.01),
    np.arange(0.10, 0.95, 0.05),
])
    print(f"\nEvaluating thresholds: {thresholds}\n")

    # Evaluate all models
    all_results = {}
    recommendations = {}

    for model_name, model in models.items():
        print(f"Processing {model_name}...")
        y_proba = get_predictions_proba(model, X_test)
        df_results = evaluate_thresholds(y_test, y_proba, thresholds)
        all_results[model_name] = df_results

        # Find best thresholds
        best_f1_threshold, best_f1_value = find_best_threshold(df_results, metric="f1")
        recall_95_threshold, recall_95_value = find_recall_constrained_threshold(df_results, min_recall=0.95)

        # Get metrics for recommendations
        f1_idx = df_results[df_results["threshold"] == best_f1_threshold].index[0]
        f1_metrics = df_results.loc[f1_idx].to_dict()

        if recall_95_threshold is not None:
            recall_95_idx = df_results[df_results["threshold"] == recall_95_threshold].index[0]
            recall_95_metrics = df_results.loc[recall_95_idx].to_dict()
        else:
            recall_95_metrics = None

        recommendations[model_name] = {
            "f1_threshold": float(best_f1_threshold),
            "f1_metrics": {k: v for k, v in f1_metrics.items() if k != "threshold"},
            "recall_95_threshold": float(recall_95_threshold) if recall_95_threshold is not None else None,
            "recall_95_metrics": {k: v for k, v in recall_95_metrics.items() if k != "threshold"}
            if recall_95_metrics is not None
            else None,
        }

    # Save results to CSV
    print("\nSaving results...")
    combined_results = []
    for model_name, df in all_results.items():
        df_copy = df.copy()
        df_copy["model"] = model_name
        combined_results.append(df_copy)

    df_all = pd.concat(combined_results, ignore_index=True)
    csv_path = REPORTS_DIR / "threshold_analysis.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"  ✓ Saved threshold analysis: {csv_path}")

    # Create plots
    print("\nGenerating plots...")
    plot_precision_recall_vs_threshold(all_results, FIGURES_DIR / "precision_recall_vs_threshold.png")
    plot_f1_vs_threshold(all_results, FIGURES_DIR / "f1_vs_threshold.png")

    # Print report
    print_report(all_results, recommendations)

    # Save recommendations
    recommendations_path = REPORTS_DIR / "threshold_recommendation.json"
    with open(recommendations_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"\n✓ Saved recommendations: {recommendations_path}\n")


if __name__ == "__main__":
    main()
