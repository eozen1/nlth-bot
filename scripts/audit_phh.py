from pathlib import Path
from collections import Counter, defaultdict
import argparse
import ast
import json
import re
import statistics as stats


ACTION_MAP = {
    "f": "Fold",
    "cc": "Call",
    "cbr": "Raise",
}


def collect_hand_files(data_dir: Path, extensions):
    files = []
    for ext in extensions:
        files.extend(data_dir.rglob(f"*{ext}"))
    return sorted(set(files))


def safe_parse_phh(path: Path):
    """
    Parse PHH-like assignment files without using exec().
    Handles lowercase true/false from PHH files.
    """
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
                name = node.targets[0].id
                try:
                    out[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
        return out

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


def classify_action(action: str):
    parts = action.split()
    if not parts:
        return None

    # Deal actions are not decision labels.
    if parts[0] == "d":
        return None

    if len(parts) >= 2 and parts[0].startswith("p"):
        return ACTION_MAP.get(parts[1], "Other")

    return "Other"


def audit_file(path: Path):
    obj = safe_parse_phh(path)

    actions = obj.get("actions", [])
    players = obj.get("players", [])
    starting_stacks = obj.get("starting_stacks", [])
    blinds = obj.get("blinds_or_straddles", [])
    antes = obj.get("antes", [])

    labels = []
    decision_count = 0
    board_deals = 0
    hole_deals = 0
    max_history_before_decision = 0
    history_len = 0

    for action in actions:
        parts = action.split()
        if not parts:
            continue

        if parts[0] == "d":
            if len(parts) >= 2 and parts[1] == "dh":
                hole_deals += 1
            elif len(parts) >= 2 and parts[1] == "db":
                board_deals += 1
            history_len += 1
            continue

        label = classify_action(action)
        if label is not None:
            decision_count += 1
            labels.append(label)
            max_history_before_decision = max(max_history_before_decision, history_len)
            history_len += 1

    return {
        "path": str(path),
        "num_actions": len(actions),
        "num_decisions": decision_count,
        "labels": labels,
        "num_players": len(players) if players else len(starting_stacks),
        "has_finishing_stacks": "finishing_stacks" in obj,
        "has_starting_stacks": bool(starting_stacks),
        "has_blinds": bool(blinds),
        "has_antes": bool(antes),
        "hole_deals": hole_deals,
        "board_deals": board_deals,
        "max_history_before_decision": max_history_before_decision,
    }


def percentile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = int(round((p / 100) * (len(xs) - 1)))
    return xs[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="audit_summary.json")
    parser.add_argument(
        "--extensions",
        type=str,
        default=".phh,.phhs",
        help="Comma-separated hand-history extensions to include.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    extensions = [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
    files = collect_hand_files(data_dir, extensions)

    print(f"Found {len(files):,} hand files under {data_dir} with {extensions}")

    global_labels = Counter()
    per_top_dir = defaultdict(Counter)
    decision_counts = []
    history_lengths = []
    malformed = []

    for i, path in enumerate(files):
        try:
            info = audit_file(path)
        except Exception as e:
            malformed.append({"path": str(path), "error": str(e)})
            continue

        labels = Counter(info["labels"])
        global_labels.update(labels)

        try:
            top = path.relative_to(data_dir).parts[0]
        except Exception:
            top = "ROOT"

        per_top_dir[top].update(labels)

        decision_counts.append(info["num_decisions"])
        history_lengths.append(info["max_history_before_decision"])

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1:,}/{len(files):,} files")

    total_decisions = sum(global_labels.values())

    summary = {
        "num_phh_files": len(files),
        "num_malformed_files": len(malformed),
        "total_decision_points": total_decisions,
        "class_counts": dict(global_labels),
        "class_percentages": {
            k: round(v / total_decisions, 6) if total_decisions else 0.0
            for k, v in global_labels.items()
        },
        "decisions_per_hand": {
            "mean": round(stats.mean(decision_counts), 3) if decision_counts else None,
            "p50": percentile(decision_counts, 50),
            "p90": percentile(decision_counts, 90),
            "p95": percentile(decision_counts, 95),
            "p99": percentile(decision_counts, 99),
            "max": max(decision_counts) if decision_counts else None,
        },
        "history_length_before_decision": {
            "mean": round(stats.mean(history_lengths), 3) if history_lengths else None,
            "p50": percentile(history_lengths, 50),
            "p90": percentile(history_lengths, 90),
            "p95": percentile(history_lengths, 95),
            "p99": percentile(history_lengths, 99),
            "max": max(history_lengths) if history_lengths else None,
        },
        "per_top_directory_class_counts": {
            k: dict(v) for k, v in sorted(per_top_dir.items())
        },
        "malformed_examples": malformed[:20],
    }

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()