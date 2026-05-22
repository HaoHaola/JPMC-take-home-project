"""
Classification pipeline for the Census income project.

This script covers Part 1 + classification:
1. Load and preprocess the data.
2. Build the encoded classification dataframe.
3. Train/dev/test split.
4. Logistic Regression baseline with threshold tuning.
5. Histogram Gradient Boosting with threshold tuning.
6. Optional top-feature Gradient Boosting using permutation importance.
7. Optional neural network model if PyTorch is installed and --run-nn is passed.

Run example:
python classification_model.py \
    --data census-bureau.data \
    --columns census-bureau.columns \
    --output-dir outputs_classification
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import build_classification_xy, ensure_output_dir, prepare_model_dataframe


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=random_state,
        stratify=y,
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def tune_threshold(y_true: pd.Series, y_proba: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """Select the threshold that maximizes class-1 F1 on the dev set."""
    thresholds = np.arange(0.10, 0.91, 0.05)
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_class_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            }
        )

    threshold_results = pd.DataFrame(rows)
    best_threshold = float(
        threshold_results.loc[threshold_results["f1_class_1"].idxmax(), "threshold"]
    )
    return best_threshold, threshold_results


def evaluate_predictions(y_true, y_pred, y_proba, title: str) -> Dict[str, float]:
    print(f"\n===== {title} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_proba))
    print("PR-AUC:", average_precision_score(y_true, y_proba))
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "model": title,
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision_class_1": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_class_1": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_class_1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def run_logistic_regression(X_train, X_dev, X_test, y_train, y_dev, y_test, output_dir: Path):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)

    y_dev_proba = model.predict_proba(X_dev_scaled)[:, 1]
    best_threshold, threshold_results = tune_threshold(y_dev, y_dev_proba)
    threshold_results.to_csv(output_dir / "logistic_threshold_results.csv", index=False)

    # Final model: combine train and dev, refit scaler on combined set, evaluate once on test.
    X_train_final = pd.concat([X_train, X_dev], axis=0)
    y_train_final = pd.concat([y_train, y_dev], axis=0)

    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_test_scaled_final = scaler_final.transform(X_test)

    final_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    final_model.fit(X_train_final_scaled, y_train_final)

    y_test_proba = final_model.predict_proba(X_test_scaled_final)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    metrics = evaluate_predictions(
        y_test,
        y_test_pred,
        y_test_proba,
        f"Logistic Regression Test Performance, threshold={best_threshold:.2f}",
    )
    metrics["threshold"] = best_threshold
    return metrics


def run_gradient_boosting(X_train, X_dev, X_test, y_train, y_dev, y_test, output_dir: Path):
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=29,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42,
    )

    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight_train)

    y_dev_proba = model.predict_proba(X_dev)[:, 1]
    best_threshold, threshold_results = tune_threshold(y_dev, y_dev_proba)
    threshold_results.to_csv(output_dir / "gb_threshold_results.csv", index=False)

    X_train_final = pd.concat([X_train, X_dev], axis=0)
    y_train_final = pd.concat([y_train, y_dev], axis=0)
    sample_weight_final = compute_sample_weight(class_weight="balanced", y=y_train_final)

    final_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=29,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42,
    )
    final_model.fit(X_train_final, y_train_final, sample_weight=sample_weight_final)

    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    metrics = evaluate_predictions(
        y_test,
        y_test_pred,
        y_test_proba,
        f"Gradient Boosting Test Performance, threshold={best_threshold:.2f}",
    )
    metrics["threshold"] = best_threshold
    return final_model, metrics


def run_top_feature_gradient_boosting(
    trained_gb,
    X: pd.DataFrame,
    y: pd.Series,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    output_dir: Path,
    top_n: int = 30,
):
    """Use permutation importance from the dev set, then retrain a simpler model using top N encoded features."""
    print("\nComputing permutation importance. This may take a few minutes...")
    importance = permutation_importance(
        trained_gb,
        X_dev,
        y_dev,
        scoring="roc_auc",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    feature_importance.to_csv(output_dir / "gb_permutation_importance.csv", index=False)

    top_features = feature_importance.head(top_n)["feature"].tolist()
    X_top = X[top_features]

    X_train, X_dev_top, X_test, y_train, y_dev_top, y_test = split_data(X_top, y)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=500,
        max_leaf_nodes=27,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=42,
    )
    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight_train)

    y_dev_proba = model.predict_proba(X_dev_top)[:, 1]
    best_threshold, threshold_results = tune_threshold(y_dev_top, y_dev_proba)
    threshold_results.to_csv(output_dir / "gb_top_features_threshold_results.csv", index=False)

    X_train_final = pd.concat([X_train, X_dev_top], axis=0)
    y_train_final = pd.concat([y_train, y_dev_top], axis=0)
    sample_weight_final = compute_sample_weight(class_weight="balanced", y=y_train_final)
    model.fit(X_train_final, y_train_final, sample_weight=sample_weight_final)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    metrics = evaluate_predictions(
        y_test,
        y_test_pred,
        y_test_proba,
        f"Top-{top_n} Feature Gradient Boosting Test Performance, threshold={best_threshold:.2f}",
    )
    metrics["threshold"] = best_threshold
    return metrics


def run_neural_network_optional(X, y, output_dir: Path, epochs: int = 30):
    """Optional MLP version of the classification model. Runs only if PyTorch is installed."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("\nPyTorch is not installed, so the neural network section was skipped.")
        return None

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_dev_tensor = torch.tensor(X_dev_scaled, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=512, shuffle=True)
    dev_loader = DataLoader(TensorDataset(X_dev_tensor, y_dev_tensor), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=1024, shuffle=False)

    class IncomeMLP(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            return self.model(x)

    def evaluate_loader(model, loader, threshold=0.5):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                logits = model(xb)
                batch_probs = torch.sigmoid(logits).cpu().numpy().flatten()
                probs.extend(batch_probs)
                labels.extend(yb.numpy().flatten())
        probs = np.array(probs)
        labels = np.array(labels).astype(int)
        preds = (probs >= threshold).astype(int)
        return labels, preds, probs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IncomeMLP(X_train_tensor.shape[1]).to(device)

    num_negative = int((y_train == 0).sum())
    num_positive = int((y_train == 1).sum())
    pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_dev_auc = -np.inf
    best_state = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        y_dev_true, _, y_dev_proba = evaluate_loader(model, dev_loader)
        dev_auc = roc_auc_score(y_dev_true, y_dev_proba)
        print(f"Epoch {epoch + 1:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Dev AUC: {dev_auc:.4f}")
        if dev_auc > best_dev_auc:
            best_dev_auc = dev_auc
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    y_dev_true, _, y_dev_proba = evaluate_loader(model, dev_loader)
    best_threshold, threshold_results = tune_threshold(pd.Series(y_dev_true), y_dev_proba)
    threshold_results.to_csv(output_dir / "nn_threshold_results.csv", index=False)

    y_test_true, y_test_pred, y_test_proba = evaluate_loader(model, test_loader, best_threshold)
    metrics = evaluate_predictions(
        y_test_true,
        y_test_pred,
        y_test_proba,
        f"Neural Network Test Performance, threshold={best_threshold:.2f}",
    )
    metrics["threshold"] = best_threshold
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="census-bureau.data")
    parser.add_argument("--columns", default="census-bureau.columns")
    parser.add_argument("--output-dir", default="outputs_classification")
    parser.add_argument("--skip-feature-selection", action="store_true")
    parser.add_argument("--run-nn", action="store_true", help="Run the optional PyTorch neural network model.")
    parser.add_argument("--nn-epochs", type=int, default=30)
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir)

    df_model = prepare_model_dataframe(args.data, args.columns)
    X, y = build_classification_xy(df_model)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label distribution:")
    print(y.value_counts(normalize=True).rename("percentage") * 100)

    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y)

    metrics = []
    metrics.append(run_logistic_regression(X_train, X_dev, X_test, y_train, y_dev, y_test, output_dir))
    trained_gb, gb_metrics = run_gradient_boosting(X_train, X_dev, X_test, y_train, y_dev, y_test, output_dir)
    metrics.append(gb_metrics)

    if not args.skip_feature_selection:
        metrics.append(run_top_feature_gradient_boosting(trained_gb, X, y, X_dev, y_dev, output_dir))

    if args.run_nn:
        nn_metrics = run_neural_network_optional(X, y, output_dir, epochs=args.nn_epochs)
        if nn_metrics is not None:
            metrics.append(nn_metrics)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_dir / "classification_model_comparison.csv", index=False)
    print("\nSaved model comparison to:", output_dir / "classification_model_comparison.csv")
    print(metrics_df)


if __name__ == "__main__":
    main()
