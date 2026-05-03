"""Load processed JSONL poker datasets for modeling."""

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

FOLD, CALL, RAISE = 0, 1, 2
ACTION_DIM = 3
SEQ_TOKEN_DIM = 7  # action one-hot(3) + token features(4)

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _history_token_to_vector(token: List[float]) -> List[float]:
    """Convert [action_id, f1, f2, f3, f4] into [action_oh(3), f1..f4]."""
    if len(token) < 5:
        return [0.0] * SEQ_TOKEN_DIM

    action_id = int(token[0])
    action_oh = [0.0] * ACTION_DIM
    if 0 <= action_id < ACTION_DIM:
        action_oh[action_id] = 1.0

    features = []
    for value in token[1:5]:
        if value is None:
            features.append(0.0)
        else:
            features.append(float(value))

    return action_oh + features


def _example_to_sample(example: dict) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    flat = example.get("flat_features", {})
    flat_vec = [
        float(flat.get("pot_norm", 0.0)),
        float(flat.get("pos_norm", 0.0)),
        float(flat.get("round_norm", 0.0)),
        float(flat.get("active_norm", 0.0)),
        float(flat.get("to_call_bb", 0.0)),
        float(flat.get("stack_bb", 0.0)),
        float(flat.get("pot_odds", 0.0)),
        float(flat.get("spr", 0.0) if flat.get("spr") is not None else 0.0),
    ]

    history = example.get("history_tokens", [])
    seq_vecs = [_history_token_to_vector(tok) for tok in history]

    if seq_vecs:
        seq_tensor = torch.tensor(seq_vecs, dtype=torch.float32)
    else:
        seq_tensor = torch.zeros(0, SEQ_TOKEN_DIM, dtype=torch.float32)

    seq_len = seq_tensor.shape[0]
    label = int(example["label_id"])

    return torch.tensor(flat_vec, dtype=torch.float32), seq_tensor, seq_len, label


def load_jsonl_samples(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            samples.append(_example_to_sample(example))
    return samples


class PokerDataset(Dataset):
    def __init__(self, samples, max_seq_len: int):
        self.samples = samples
        self.max_seq_len = max(max_seq_len, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_feat, seq_tensor, seq_len, label = self.samples[idx]
        padded = torch.zeros(self.max_seq_len, SEQ_TOKEN_DIM, dtype=torch.float32)
        actual_len = min(seq_len, self.max_seq_len)
        if actual_len > 0:
            padded[:actual_len] = seq_tensor[:actual_len]
        clamped_len = max(actual_len, 1)
        return flat_feat, padded, clamped_len, label


def load_datasets(
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    test_path: str | Path | None = None,
):
    """Load train/val/test processed JSONL splits as PokerDataset objects."""
    train_path = Path(train_path or (DEFAULT_PROCESSED_DIR / "small_train.jsonl"))
    val_path = Path(val_path or (DEFAULT_PROCESSED_DIR / "small_val.jsonl"))
    test_path = Path(test_path or (DEFAULT_PROCESSED_DIR / "small_test.jsonl"))

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed split: {path}. Run scripts/preprocess_phh.py first."
            )

    train_samples = load_jsonl_samples(train_path)
    val_samples = load_jsonl_samples(val_path)
    test_samples = load_jsonl_samples(test_path)

    all_samples = train_samples + val_samples + test_samples
    max_seq_len = max((sample[2] for sample in all_samples), default=1)
    max_seq_len = max(max_seq_len, 1)

    print(
        "Loaded processed samples: "
        f"{len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test"
    )
    print(f"Max sequence length: {max_seq_len}")

    return (
        PokerDataset(train_samples, max_seq_len),
        PokerDataset(val_samples, max_seq_len),
        PokerDataset(test_samples, max_seq_len),
    )
