"""Run baselines on processed splits and save metrics artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

from dataset import load_datasets
from baselines import train_logistic, train_lstm, train_majority, predict_lstm


LABEL_NAMES = ["Fold", "Call", "Raise"]


def extract_numpy(dataset):
    X = np.array([s[0].numpy() for s in dataset.samples], dtype=np.float32)
    y = np.array([s[3] for s in dataset.samples], dtype=np.int64)
    return X, y


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    per_class = {
        LABEL_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(3)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fold_recall": float(recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]),
        "raise_recall": float(recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def evaluate_predictions(y_true, y_pred, model_name, split_name):
    metrics = compute_metrics(y_true, y_pred)
    print(
        f"[{split_name}] {model_name:<20} "
        f"acc={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        f"raise_recall={metrics['raise_recall']:.4f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/metrics")
    parser.add_argument("--prefix", type=str, default="small_baselines")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed datasets...")
    train_ds, val_ds, test_ds = load_datasets(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )

    X_train, y_train = extract_numpy(train_ds)
    X_val, y_val = extract_numpy(val_ds)
    X_test, y_test = extract_numpy(test_ds)

    split_targets = {
        "val": (X_val, y_val, val_ds),
        "test": (X_test, y_test, test_ds),
    }

    results = {}

    print("\nTraining Majority Class...")
    maj = train_majority(X_train, y_train)
    results["Majority Class"] = {}
    for split_name, (X_split, y_split, _) in split_targets.items():
        y_pred = maj.predict(X_split)
        results["Majority Class"][split_name] = evaluate_predictions(
            y_split, y_pred, "Majority Class", split_name
        )

    print("\nTraining Logistic Regression...")
    lr = train_logistic(X_train, y_train)
    results["Logistic Regression"] = {}
    for split_name, (X_split, y_split, _) in split_targets.items():
        y_pred = lr.predict(X_split)
        results["Logistic Regression"][split_name] = evaluate_predictions(
            y_split, y_pred, "Logistic Regression", split_name
        )

    print("\nTraining LSTM...")
    lstm = train_lstm(train_ds)
    results["LSTM"] = {}
    for split_name, (_, y_split, ds_split) in split_targets.items():
        y_pred = predict_lstm(lstm, ds_split)
        results["LSTM"][split_name] = evaluate_predictions(y_split, y_pred, "LSTM", split_name)

    print("\n## Validation Results\n")
    print("| Model | Accuracy | Macro F1 | Fold Recall | Raise Recall |")
    print("|-------|----------|----------|-------------|--------------|")
    for model_name in ["Majority Class", "Logistic Regression", "LSTM"]:
        m = results[model_name]["val"]
        print(
            f"| {model_name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['fold_recall']:.4f} | {m['raise_recall']:.4f} |"
        )

    metrics_path = out_dir / f"{args.prefix}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
