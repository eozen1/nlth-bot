"""Train Transformer for H1 and compare against LSTM baseline."""

import argparse
import json
import sys
from pathlib import Path

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
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal_alpha",
        type=float,
        nargs=3,
        default=None,
        metavar=("ALPHA_FOLD", "ALPHA_CALL", "ALPHA_RAISE"),
    )
    parser.add_argument(
        "--history_only",
        action="store_true",
        help="Ignore flat context features and use only action history sequence.",
    )
    parser.add_argument("--out_metrics", type=str, default="results/metrics/h1_transformer_vs_lstm.json")
    parser.add_argument("--out_history", type=str, default="results/metrics/h1_transformer_training_history.json")
    parser.add_argument("--out_ckpt", type=str, default="checkpoints/h1_transformer.pt")
    args = parser.parse_args()

    print("Loading processed datasets...")
    train_ds, val_ds, test_ds = load_datasets(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
    )
    y_val = dataset_labels(val_ds)
    y_test = dataset_labels(test_ds)

    print("\nTraining Transformer...")
    transformer_model, history = train_transformer(
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
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        include_flat_features=not args.history_only,
    )

    y_pred_val_t = predict_transformer(transformer_model, val_ds, batch_size=args.batch_size)
    y_pred_test_t = predict_transformer(transformer_model, test_ds, batch_size=args.batch_size)

    print("\nTraining LSTM baseline...")
    lstm_model = train_lstm(train_ds)
    y_pred_val_l = predict_lstm(lstm_model, val_ds)
    y_pred_test_l = predict_lstm(lstm_model, test_ds)

    results = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "patience": args.patience,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "focal_alpha": args.focal_alpha,
            "history_only": args.history_only,
        },
        "Transformer": {
            "val": compute_metrics(y_val, y_pred_val_t),
            "test": compute_metrics(y_test, y_pred_test_t),
        },
        "LSTM": {
            "val": compute_metrics(y_val, y_pred_val_l),
            "test": compute_metrics(y_test, y_pred_test_l),
        },
    }
    results["h1_macro_f1_relative_gain_on_val"] = (
        (results["Transformer"]["val"]["macro_f1"] - results["LSTM"]["val"]["macro_f1"])
        / max(results["LSTM"]["val"]["macro_f1"], 1e-8)
    )

    out_metrics = Path(args.out_metrics)
    out_history = Path(args.out_history)
    out_ckpt = Path(args.out_ckpt)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_history.parent.mkdir(parents=True, exist_ok=True)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    out_metrics.write_text(json.dumps(results, indent=2))
    out_history.write_text(json.dumps(history, indent=2))
    torch.save(transformer_model.state_dict(), out_ckpt)

    print("\nValidation Macro F1:")
    print(f"  Transformer: {results['Transformer']['val']['macro_f1']:.4f}")
    print(f"  LSTM:        {results['LSTM']['val']['macro_f1']:.4f}")
    print(
        "  Relative gain (H1): "
        f"{100.0 * results['h1_macro_f1_relative_gain_on_val']:.2f}%"
    )
    print(f"\nSaved metrics:  {out_metrics}")
    print(f"Saved history:  {out_history}")
    print(f"Saved checkpoint: {out_ckpt}")


if __name__ == "__main__":
    main()
