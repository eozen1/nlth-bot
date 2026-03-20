"""Three baseline models: majority class, logistic regression, LSTM."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


# ── sklearn baselines ──────────────────────────────────────────────


def train_majority(X_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_train, y_train)
    return clf


def train_logistic(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


# ── LSTM model ─────────────────────────────────────────────────────


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequences, lengths):
        # Sort by descending length for pack_padded_sequence
        lengths = lengths.cpu()
        sorted_idx = torch.argsort(lengths, descending=True)
        sorted_seq = sequences[sorted_idx]
        sorted_len = lengths[sorted_idx]

        packed = pack_padded_sequence(sorted_seq, sorted_len, batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        logits = self.fc(h_n.squeeze(0))

        # Unsort to original order
        _, unsort_idx = torch.sort(sorted_idx)
        logits = logits[unsort_idx]
        return logits


def train_lstm(
    train_dataset,
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
) -> LSTMClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_dim=7, hidden_dim=hidden_dim, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        n_batches = 0
        for flat_feat, sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}")

    return model


def predict_lstm(model: LSTMClassifier, dataset, batch_size: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []

    model.eval()
    with torch.no_grad():
        for flat_feat, sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            logits = model(sequences, lengths)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds)
