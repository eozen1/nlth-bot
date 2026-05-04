from pathlib import Path
from collections import Counter
import argparse
import ast
import json
import random
import re


LABEL_CODE_TO_NAME = {
    "f": "Fold",
    "cc": "Call",
    "cbr": "Raise",
}

LABEL_NAME_TO_ID = {
    "Fold": 0,
    "Call": 1,
    "Raise": 2,
}

STREET_NAMES = {
    0: "preflop",
    1: "flop",
    2: "turn",
    3: "river",
}


def safe_parse_phh(path: Path):
    text = path.read_text(errors="ignore")
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)

    out = {}
    try:
        tree = ast.parse(text)
        nodes = tree.body
    except SyntaxError:
        nodes = []

    if nodes:
        for node in nodes:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                key = node.targets[0].id
                try:
                    out[key] = ast.literal_eval(node.value)
                except Exception:
                    pass
        return out

    # Fallback for PHH variants containing non-literal metadata lines (e.g., time=00:01:12).
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.isidentifier():
            continue
        try:
            out[key] = ast.literal_eval(value)
        except Exception:
            continue

    return out


def parse_int_or_none(x):
    try:
        return int(x)
    except Exception:
        return None


def compute_split_boundaries(n, train_ratio, val_ratio):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def collect_hand_files(data_dir: Path, extensions):
    files = []
    for ext in extensions:
        files.extend(data_dir.rglob(f"*{ext}"))
    # dedupe while preserving deterministic ordering
    return sorted(set(files))


def build_examples_from_hand(path: Path, data_dir: Path, max_history_len: int):
    hand = safe_parse_phh(path)
    actions = hand.get("actions", [])

    starting_stacks = hand.get("starting_stacks", [])
    players = hand.get("players", [])
    blinds = hand.get("blinds_or_straddles", [])
    antes = hand.get("antes", [])
    min_bet = hand.get("min_bet", 100)

    num_players = len(starting_stacks) or len(players) or max(len(blinds), len(antes), 2)
    total_chips = sum(starting_stacks) if starting_stacks else num_players * 10000

    starting_stacks = (starting_stacks + [10000] * num_players)[:num_players]
    blinds = (blinds + [0] * num_players)[:num_players]
    antes = (antes + [0] * num_players)[:num_players]

    big_blind = max([b for b in blinds if isinstance(b, (int, float))] + [min_bet, 1])

    # State tracking
    active = [True] * num_players
    current_bets = [float(v) for v in blinds]
    committed_total = [float(antes[i] + blinds[i]) for i in range(num_players)]
    pot = float(sum(antes))
    street = 0

    # Sequence history stores [action_id, pot_norm, pos_norm, round_norm, active_norm]
    history_tokens = []
    examples = []

    hand_stats = {
        "num_examples": 0,
        "known_label_counts": Counter(),
        "unknown_code_counts": Counter(),
        "max_history_before_decision": 0,
    }

    rel_path = str(path.relative_to(data_dir))

    for decision_idx, action in enumerate(actions):
        parts = action.strip().split()
        if not parts:
            continue

        if parts[0] == "d":
            if len(parts) >= 2 and parts[1] == "db":
                pot += sum(current_bets)
                current_bets = [0.0] * num_players
                street = min(street + 1, 3)
            continue

        if not parts[0].startswith("p") or len(parts) < 2:
            continue

        actor = parse_int_or_none(parts[0][1:])
        if actor is None:
            continue
        actor -= 1
        if actor < 0 or actor >= num_players:
            continue

        code = parts[1]
        amount = parse_int_or_none(parts[2]) if len(parts) >= 3 else None

        num_active = sum(active)
        pot_with_bets = pot + sum(current_bets)
        to_call = max(current_bets) - current_bets[actor]
        if to_call < 0:
            to_call = 0.0

        stack_remaining = max(float(starting_stacks[actor]) - committed_total[actor], 0.0)
        pot_norm = (pot_with_bets / (2.0 * total_chips)) if total_chips > 0 else 0.0
        pos_norm = actor / max(num_players - 1, 1)
        round_norm = street / 3.0
        active_norm = num_active / max(num_players, 1)
        stack_bb = stack_remaining / float(big_blind)
        to_call_bb = to_call / float(big_blind)
        pot_odds_denom = pot_with_bets + to_call
        pot_odds = (to_call / pot_odds_denom) if (to_call > 0 and pot_odds_denom > 0) else 0.0
        spr = (stack_remaining / pot_with_bets) if pot_with_bets > 0 else None

        label_name = LABEL_CODE_TO_NAME.get(code)
        if label_name is not None:
            hand_stats["max_history_before_decision"] = max(
                hand_stats["max_history_before_decision"], len(history_tokens)
            )
            history_tail = history_tokens[-max_history_len:] if max_history_len > 0 else []

            example = {
                "hand_id": rel_path,
                "decision_index": decision_idx,
                "actor": actor,
                "actor_name": players[actor] if actor < len(players) else f"p{actor + 1}",
                "street": STREET_NAMES.get(street, "river"),
                "flat_features": {
                    "pot_norm": pot_norm,
                    "pos_norm": pos_norm,
                    "round_norm": round_norm,
                    "active_norm": active_norm,
                    "to_call_bb": to_call_bb,
                    "stack_bb": stack_bb,
                    "pot_odds": pot_odds,
                    "spr": spr,
                },
                "history_tokens": history_tail,
                "label": label_name,
                "label_id": LABEL_NAME_TO_ID[label_name],
                "action_code": code,
            }
            examples.append(example)
            hand_stats["num_examples"] += 1
            hand_stats["known_label_counts"][label_name] += 1

            history_tokens.append(
                [
                    LABEL_NAME_TO_ID[label_name],
                    pot_norm,
                    pos_norm,
                    round_norm,
                    active_norm,
                ]
            )
        else:
            hand_stats["unknown_code_counts"][code] += 1

        # Update state after action
        if code == "f":
            active[actor] = False
        elif code == "cc":
            target = max(current_bets)
            delta = max(0.0, target - current_bets[actor])
            current_bets[actor] = target
            committed_total[actor] += delta
        elif code == "cbr" and amount is not None:
            raise_to = float(amount)
            delta = max(0.0, raise_to - current_bets[actor])
            current_bets[actor] = raise_to
            committed_total[actor] += delta

    return examples, hand_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--prefix", type=str, default="small")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_history_len", type=int, default=64)
    parser.add_argument(
        "--extensions",
        type=str,
        default=".phh,.phhs",
        help="Comma-separated hand-history extensions to include.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio > 0, train+val < 1")

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    files = collect_hand_files(data_dir, extensions)
    if not files:
        raise FileNotFoundError(
            f"No hand files found in {data_dir} with extensions: {extensions}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(files)

    n_train, n_val, n_test = compute_split_boundaries(
        len(files), args.train_ratio, args.val_ratio
    )
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    split_files = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    split_paths = {
        split: out_dir / f"{args.prefix}_{split}.jsonl" for split in split_files.keys()
    }

    summary = {
        "num_phh_files": len(files),
        "split_hand_counts": {
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        },
        "split_example_counts": {"train": 0, "val": 0, "test": 0},
        "split_label_counts": {"train": {}, "val": {}, "test": {}},
        "unknown_action_counts": {},
        "max_history_before_decision": 0,
        "config": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "max_history_len": args.max_history_len,
        },
    }

    global_unknown = Counter()

    for split, split_list in split_files.items():
        label_counts = Counter()
        out_path = split_paths[split]
        with out_path.open("w", encoding="utf-8") as writer:
            for idx, path in enumerate(split_list):
                examples, hand_stats = build_examples_from_hand(
                    path=path, data_dir=data_dir, max_history_len=args.max_history_len
                )
                for ex in examples:
                    writer.write(json.dumps(ex) + "\n")
                summary["split_example_counts"][split] += len(examples)
                label_counts.update(hand_stats["known_label_counts"])
                global_unknown.update(hand_stats["unknown_code_counts"])
                summary["max_history_before_decision"] = max(
                    summary["max_history_before_decision"],
                    hand_stats["max_history_before_decision"],
                )

                if (idx + 1) % 2000 == 0:
                    print(
                        f"[{split}] processed {idx + 1}/{len(split_list)} hands "
                        f"-> {summary['split_example_counts'][split]} examples"
                    )

        summary["split_label_counts"][split] = dict(label_counts)

    summary["unknown_action_counts"] = dict(global_unknown.most_common())
    summary["total_example_count"] = sum(summary["split_example_counts"].values())
    summary["total_label_counts"] = dict(
        (
            Counter(summary["split_label_counts"]["train"])
            + Counter(summary["split_label_counts"]["val"])
            + Counter(summary["split_label_counts"]["test"])
        )
    )

    vocab = {
        "label_code_to_name": LABEL_CODE_TO_NAME,
        "label_name_to_id": LABEL_NAME_TO_ID,
        "street_names": STREET_NAMES,
        "history_token_schema": [
            "action_id",
            "pot_norm",
            "pos_norm",
            "round_norm",
            "active_norm",
        ],
        "flat_feature_names": [
            "pot_norm",
            "pos_norm",
            "round_norm",
            "active_norm",
            "to_call_bb",
            "stack_bb",
            "pot_odds",
            "spr",
        ],
    }

    summary_path = out_dir / f"{args.prefix}_preprocess_summary.json"
    vocab_path = out_dir / f"{args.prefix}_vocab.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    vocab_path.write_text(json.dumps(vocab, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {split_paths['train']}")
    print(f"Wrote: {split_paths['val']}")
    print(f"Wrote: {split_paths['test']}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {vocab_path}")


if __name__ == "__main__":
    main()
