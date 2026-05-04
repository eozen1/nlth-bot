"""Run H1/H2/H3 experiment suite and save consolidated results."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, recall_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import predict_lstm, train_lstm
from src.dataset import load_datasets
from src.transformer import predict_transformer, train_transformer


LABEL_NAMES = ["Fold", "Call", "Raise"]


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
    }


def dataset_labels(dataset):
    return [sample[3] for sample in dataset.samples]


def summarize_history(history):
    if not history:
        return {"best_val_macro_f1": None, "best_val_loss": None, "best_epoch": None}
    best_f1 = max(history, key=lambda x: x["val_macro_f1"])
    best_loss = min(history, key=lambda x: x["val_loss"])
    return {
        "best_val_macro_f1": float(best_f1["val_macro_f1"]),
        "best_val_loss": float(best_loss["val_loss"]),
        "best_epoch": int(best_f1["epoch"]),
    }


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_eval_transformer(train_ds, val_ds, test_ds, args, *, loss_type, history_only):
    model, history = train_transformer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        patience=args.patience,
        loss_type=loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        include_flat_features=not history_only,
        seed=args.seed,
    )
    y_val = dataset_labels(val_ds)
    y_test = dataset_labels(test_ds)
    y_pred_val = predict_transformer(model, val_ds, batch_size=args.batch_size)
    y_pred_test = predict_transformer(model, test_ds, batch_size=args.batch_size)
    return {
        "val": compute_metrics(y_val, y_pred_val),
        "test": compute_metrics(y_test, y_pred_test),
        "training": summarize_history(history),
        "loss_type": loss_type,
        "history_only": history_only,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal_alpha",
        type=float,
        nargs=3,
        default=None,
        metavar=("ALPHA_FOLD", "ALPHA_CALL", "ALPHA_RAISE"),
    )
    parser.add_argument("--lstm_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="results/metrics/final_hypotheses_results.json")
    args = parser.parse_args()

    set_global_seed(args.seed)

    print("Loading processed datasets...")
    train_ds, val_ds, test_ds = load_datasets(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )
    y_val = dataset_labels(val_ds)
    y_test = dataset_labels(test_ds)

    print("\nTraining LSTM baseline (shared for H1)...")
    lstm = train_lstm(train_ds, num_epochs=args.lstm_epochs, seed=args.seed)
    lstm_results = {
        "val": compute_metrics(y_val, predict_lstm(lstm, val_ds)),
        "test": compute_metrics(y_test, predict_lstm(lstm, test_ds)),
    }

    print("\nTraining Transformer (CE, full context) ...")
    ce_full = train_eval_transformer(
        train_ds,
        val_ds,
        test_ds,
        args,
        loss_type="ce",
        history_only=False,
    )

    print("\nTraining Transformer (Focal, full context) ...")
    focal_full = train_eval_transformer(
        train_ds,
        val_ds,
        test_ds,
        args,
        loss_type="focal",
        history_only=False,
    )

    print("\nTraining Transformer (CE, history-only) ...")
    ce_history_only = train_eval_transformer(
        train_ds,
        val_ds,
        test_ds,
        args,
        loss_type="ce",
        history_only=True,
    )

    h1_relative_gain = (
        (ce_full["val"]["macro_f1"] - lstm_results["val"]["macro_f1"])
        / max(lstm_results["val"]["macro_f1"], 1e-8)
    )
    h2_raise_recall_gain = focal_full["val"]["raise_recall"] - ce_full["val"]["raise_recall"]
    h3_val_loss_delta = ce_full["training"]["best_val_loss"] - ce_history_only["training"]["best_val_loss"]

    results = {
        "config": {
            "epochs": args.epochs,
            "lstm_epochs": args.lstm_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "patience": args.patience,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "seed": args.seed,
        },
        "experiments": {
            "lstm_baseline": lstm_results,
            "transformer_ce_full_context": ce_full,
            "transformer_focal_full_context": focal_full,
            "transformer_ce_history_only": ce_history_only,
        },
        "hypotheses": {
            "H1": {
                "statement": "Transformer Macro F1 >= 10% relative gain over LSTM",
                "relative_macro_f1_gain_val": h1_relative_gain,
                "supported": bool(h1_relative_gain >= 0.10),
            },
            "H2": {
                "statement": "Focal Loss increases Raise recall over CE",
                "raise_recall_gain_val": h2_raise_recall_gain,
                "supported": bool(h2_raise_recall_gain > 0.0),
            },
            "H3": {
                "statement": "Adding context features lowers validation loss vs history-only",
                "val_loss_delta_context_minus_history_only": h3_val_loss_delta,
                "supported": bool(h3_val_loss_delta < 0.0),
            },
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Hypothesis Outcomes (validation) ===")
    print(
        f"H1 relative Macro F1 gain (Transformer CE full vs LSTM): "
        f"{100.0 * h1_relative_gain:.2f}%"
    )
    print(f"H2 Raise recall gain (Focal vs CE): {h2_raise_recall_gain:.4f}")
    print(
        "H3 best val loss delta (context - history_only): "
        f"{h3_val_loss_delta:.4f} (negative supports H3)"
    )
    print(f"\nSaved consolidated results: {out_path}")


if __name__ == "__main__":
    main()
