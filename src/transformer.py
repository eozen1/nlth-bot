"""Causal Transformer baseline for poker action prediction."""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class CausalTransformerClassifier(nn.Module):
    def __init__(
        self,
        seq_input_dim: int = 7,
        flat_input_dim: int = 8,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        num_classes: int = 3,
        include_flat_features: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.include_flat_features = include_flat_features
        self.seq_proj = nn.Linear(seq_input_dim, d_model)
        if include_flat_features:
            self.flat_proj = nn.Linear(flat_input_dim, d_model)
            head_input_dim = 2 * d_model
        else:
            self.flat_proj = None
            head_input_dim = d_model
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _causal_mask(self, seq_len: int, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, flat_feat, sequences, lengths):
        batch_size, seq_len, _ = sequences.shape
        x = self.seq_proj(sequences)
        x = x + self.pos_embed[:, :seq_len, :]

        causal_mask = self._causal_mask(seq_len, sequences.device)
        pad_mask = (
            torch.arange(seq_len, device=sequences.device)[None, :]
            >= lengths.to(sequences.device)[:, None]
        )

        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        # Gather final valid hidden state for each sequence.
        idx = (lengths.to(sequences.device) - 1).clamp(min=0, max=seq_len - 1)
        seq_repr = h[torch.arange(batch_size, device=sequences.device), idx]
        if self.include_flat_features:
            flat_repr = self.flat_proj(flat_feat)
            logits = self.head(torch.cat([seq_repr, flat_repr], dim=-1))
        else:
            logits = self.head(seq_repr)
        return logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha[targets] * loss
        return loss.mean()


def predict_transformer(model, dataset, batch_size: int = 256):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for flat_feat, sequences, lengths, _labels in loader:
            flat_feat = flat_feat.to(device)
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            logits = model(flat_feat, sequences, lengths)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_transformer(
    train_dataset,
    val_dataset,
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    patience: int = 3,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    focal_alpha=None,
    include_flat_features: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalTransformerClassifier(
        seq_input_dim=7,
        flat_input_dim=train_dataset.samples[0][0].shape[0],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=train_dataset.max_seq_len,
        num_classes=3,
        include_flat_features=include_flat_features,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Expected 'ce' or 'focal'.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    no_improve = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for flat_feat, sequences, lengths, labels in train_loader:
            flat_feat = flat_feat.to(device)
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(flat_feat, sequences, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_losses = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for flat_feat, sequences, lengths, labels in val_loader:
                flat_feat = flat_feat.to(device)
                sequences = sequences.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                logits = model(flat_feat, sequences, lengths)
                val_losses.append(criterion(logits, labels).item())

                y_true.append(labels.cpu().numpy())
                y_pred.append(logits.argmax(dim=1).cpu().numpy())

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        val_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": avg_val_loss,
                "val_macro_f1": val_macro_f1,
            }
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_macro_f1={val_macro_f1:.4f} loss={loss_type}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, history
