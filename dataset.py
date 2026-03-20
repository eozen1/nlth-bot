"""Parse .phh (TOML) poker hand files and build a PyTorch Dataset."""

import os
import tomllib
import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Action labels
FOLD, CALL, RAISE = 0, 1, 2


def parse_phh(path: str) -> List[Tuple[torch.Tensor, torch.Tensor, int, int]]:
    """Parse a single .phh file and return decision-point samples.

    Returns list of (flat_features[4], prior_actions[N,7], seq_len, label).
    """
    with open(path, "rb") as f:
        hand = tomllib.load(f)

    actions_raw = hand.get("actions", [])
    num_players = hand.get("num_players", 6)
    starting_stacks = hand.get("starting_stacks", [])
    total_chips = sum(starting_stacks) if starting_stacks else num_players * 10000

    # Track game state
    active = [True] * num_players
    current_bets = [0] * num_players
    pot = 0
    betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river

    # Collect prior action history (for LSTM sequences)
    prior_actions: List[List[float]] = []  # each entry: [action_oh(3) + features(4)]
    samples = []

    for action_str in actions_raw:
        parts = action_str.strip().split()
        if not parts:
            continue

        # New betting round: deal board cards
        if parts[0] == "d" and len(parts) >= 2 and parts[1] == "db":
            # Collect bets into pot
            pot += sum(current_bets)
            current_bets = [0] * num_players
            betting_round = min(betting_round + 1, 3)
            continue

        # Player actions: pN <action> [amount]
        if not parts[0].startswith("p") or len(parts[0]) < 2:
            continue
        try:
            player_idx = int(parts[0][1:]) - 1  # 1-indexed to 0-indexed
        except ValueError:
            continue
        if player_idx < 0 or player_idx >= num_players:
            continue
        if len(parts) < 2:
            continue

        action_type = parts[1]

        # Compute features at this decision point
        num_active = sum(active)
        pot_with_bets = pot + sum(current_bets)
        pot_norm = pot_with_bets / (2 * total_chips) if total_chips > 0 else 0
        pos_norm = player_idx / max(num_players - 1, 1)
        round_norm = betting_round / 3
        active_norm = num_active / max(num_players, 1)

        features = [pot_norm, pos_norm, round_norm, active_norm]

        if action_type == "f":
            label = FOLD
        elif action_type == "cc":
            label = CALL
        elif action_type == "cbr":
            label = RAISE
        else:
            # Skip non-decision actions (deal hole cards, show, muck, etc.)
            continue

        flat_feat = torch.tensor(features, dtype=torch.float32)

        # Build sequence of prior actions for LSTM
        if prior_actions:
            seq_tensor = torch.tensor(prior_actions, dtype=torch.float32)
        else:
            seq_tensor = torch.zeros(0, 7, dtype=torch.float32)
        seq_len = len(prior_actions)

        samples.append((flat_feat, seq_tensor, seq_len, label))

        # Record this action into prior history
        action_oh = [0.0, 0.0, 0.0]
        action_oh[label] = 1.0
        prior_actions.append(action_oh + features)

        # Update game state
        if action_type == "f":
            active[player_idx] = False
        elif action_type == "cbr" and len(parts) >= 3:
            try:
                amount = int(parts[2])
                current_bets[player_idx] = amount
            except ValueError:
                pass

    return samples


class PokerDataset(Dataset):
    def __init__(self, samples, max_seq_len: int):
        self.samples = samples
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_feat, seq_tensor, seq_len, label = self.samples[idx]

        # Pad sequence to max_seq_len
        padded = torch.zeros(self.max_seq_len, 7, dtype=torch.float32)
        actual_len = min(seq_tensor.shape[0], self.max_seq_len)
        if actual_len > 0:
            padded[:actual_len] = seq_tensor[:actual_len]

        # Clamp seq_length to at least 1 (pack_padded_sequence requires > 0)
        clamped_len = max(actual_len, 1)

        return flat_feat, padded, clamped_len, label


def load_datasets(
    data_dir: str = DATA_DIR, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[PokerDataset, PokerDataset]:
    """Parse all .phh files and return train/val PokerDatasets, split by hand."""
    phh_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".phh")]
    )
    if not phh_files:
        raise FileNotFoundError(f"No .phh files in {data_dir}. Run setup_data.py first.")

    # Parse all hands
    hands = []  # list of (hand_samples)
    for path in phh_files:
        samples = parse_phh(path)
        if samples:
            hands.append(samples)

    # Split by hand
    random.seed(seed)
    random.shuffle(hands)
    split_idx = int(len(hands) * train_ratio)
    train_hands = hands[:split_idx]
    val_hands = hands[split_idx:]

    train_samples = [s for hand in train_hands for s in hand]
    val_samples = [s for hand in val_hands for s in hand]

    # Find max sequence length across all samples
    all_samples = train_samples + val_samples
    max_seq_len = max((s[2] for s in all_samples), default=1)
    max_seq_len = max(max_seq_len, 1)

    print(f"Parsed {len(hands)} hands → {len(train_samples)} train / {len(val_samples)} val samples")
    print(f"Max sequence length: {max_seq_len}")

    return PokerDataset(train_samples, max_seq_len), PokerDataset(val_samples, max_seq_len)
