from pathlib import Path
from collections import Counter, defaultdict
import argparse
import ast
import json
import re


KNOWN_ACTIONS = {
    "f": "Fold",
    "cc": "Call",
    "cbr": "Raise",
}


def safe_parse_phh(path: Path):
    text = path.read_text(errors="ignore")
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)

    tree = ast.parse(text)
    out = {}

    for node in tree.body:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--out", type=str, default="results/metrics/action_vocab_small.json"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.rglob("*.phh"))

    action_code_counts = Counter()
    known_label_counts = Counter()
    unknown_code_counts = Counter()
    examples = defaultdict(list)
    per_source_counts = defaultdict(Counter)

    for path in files:
        hand = safe_parse_phh(path)
        actions = hand.get("actions", [])

        try:
            source = path.relative_to(data_dir).parts[0]
        except Exception:
            source = "ROOT"

        for action in actions:
            parts = action.strip().split()
            if len(parts) < 2:
                continue

            if parts[0] == "d":
                continue

            if not parts[0].startswith("p"):
                continue

            code = parts[1]
            action_code_counts[code] += 1
            per_source_counts[source][code] += 1

            if code in KNOWN_ACTIONS:
                known_label_counts[KNOWN_ACTIONS[code]] += 1
            else:
                unknown_code_counts[code] += 1
                if len(examples[code]) < 20:
                    examples[code].append(
                        {
                            "file": str(path),
                            "action": action,
                        }
                    )

    summary = {
        "num_phh_files": len(files),
        "action_code_counts": dict(action_code_counts.most_common()),
        "known_label_counts": dict(known_label_counts),
        "unknown_code_counts": dict(unknown_code_counts.most_common()),
        "unknown_examples": dict(examples),
        "per_source_action_code_counts": {
            k: dict(v.most_common()) for k, v in sorted(per_source_counts.items())
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
