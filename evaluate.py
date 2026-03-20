"""Run all baselines and print a markdown results table."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from dataset import load_datasets
from baselines import (
    train_majority,
    train_logistic,
    train_lstm,
    predict_lstm,
)


def extract_numpy(dataset):
    """Extract flat features and labels as numpy arrays."""
    X = np.array([s[0].numpy() for s in dataset.samples])
    y = np.array([s[3] for s in dataset.samples])
    return X, y


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fold_recall = recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
    raise_recall = recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    return acc, f1, fold_recall, raise_recall


def main():
    print("Loading datasets...")
    train_ds, val_ds = load_datasets()

    X_train, y_train = extract_numpy(train_ds)
    X_val, y_val = extract_numpy(val_ds)

    results = []

    # 1. Majority class
    print("\nTraining Majority Class...")
    maj = train_majority(X_train, y_train)
    y_pred_maj = maj.predict(X_val)
    results.append(("Majority Class", *compute_metrics(y_val, y_pred_maj)))

    # 2. Logistic Regression
    print("Training Logistic Regression...")
    lr = train_logistic(X_train, y_train)
    y_pred_lr = lr.predict(X_val)
    results.append(("Logistic Regression", *compute_metrics(y_val, y_pred_lr)))

    # 3. LSTM
    print("Training LSTM...")
    lstm = train_lstm(train_ds)
    y_pred_lstm = predict_lstm(lstm, val_ds)
    results.append(("LSTM", *compute_metrics(y_val, y_pred_lstm)))

    # Print results table
    print("\n## Results\n")
    print("| Model | Accuracy | Macro F1 | Fold Recall | Raise Recall |")
    print("|-------|----------|----------|-------------|--------------|")
    for name, acc, f1, fr, rr in results:
        print(f"| {name} | {acc:.4f} | {f1:.4f} | {fr:.4f} | {rr:.4f} |")


if __name__ == "__main__":
    main()
