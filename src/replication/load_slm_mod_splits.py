"""Load and seed-42 stratified-split a single SLM-Mod CSV from Agam Goyal's
2026-05-02 release (folder ``reddit_removals_1k``).

Per the SLM-Mod NAACL 2025 paper Section 3.1 + 3.4: balanced 80/20 train/test
split where the test set has equal removed and approved counts. Per Agam's
2026-05-04 email: torch.manual_seed(42) and np.random.seed(42).

CSV columns: body, subreddit, removed, context.
Output JSONL row: {id, subreddit, body, parent_body, label}.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np


SEED = 42
TRAIN_FRAC = 0.8


def _stable_id(subreddit, idx, body):
    h = hashlib.md5(f"{subreddit}|{idx}|{body[:64]}".encode("utf-8")).hexdigest()[:10]
    return f"slm_{subreddit}_{idx:04d}_{h}"


def parse_agam_csv(csv_path):
    """Read one Agam CSV; return list of dicts in our standard schema.

    Agam's `removed` column is the string 'True' or 'False'. We map to our
    label vocabulary 'removed'/'approved' to match build_dataset.py.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            removed_str = r["removed"].strip()
            if removed_str not in ("True", "False"):
                raise ValueError(
                    f"{csv_path}:{idx}: unexpected `removed` value {removed_str!r}; "
                    f"expected 'True' or 'False'"
                )
            label = "removed" if removed_str == "True" else "approved"
            rows.append({
                "id": _stable_id(r["subreddit"], idx, r["body"]),
                "subreddit": r["subreddit"],
                "body": r["body"],
                "parent_body": r["context"],
                "label": label,
            })
    return rows


def stratified_split(rows, seed=SEED, train_frac=TRAIN_FRAC):
    """Stratified 80/20 split: 80/20 within each label class. Deterministic
    given seed. Matches the SLM-Mod paper's balanced-test-set protocol.
    """
    import torch  # local import; keeps bare CSV-only use cheap
    torch.manual_seed(seed)
    np.random.seed(seed)

    by_label = {"removed": [], "approved": []}
    for r in rows:
        by_label[r["label"]].append(r)

    train, test = [], []
    for label in ("removed", "approved"):
        bucket = by_label[label]
        idx = np.random.permutation(len(bucket))
        split = int(round(train_frac * len(bucket)))
        train.extend(bucket[i] for i in idx[:split])
        test.extend(bucket[i] for i in idx[split:])
    # Shuffle the merged splits so labels are interleaved (deterministic).
    train_idx = np.random.permutation(len(train))
    test_idx = np.random.permutation(len(test))
    train = [train[i] for i in train_idx]
    test = [test[i] for i in test_idx]
    return train, test


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to one Agam CSV (e.g. changemyview.csv)")
    p.add_argument("--out-dir", required=True,
                   help="Output directory; train.jsonl + test.jsonl + counts.json land here")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    args = p.parse_args()

    rows = parse_agam_csv(args.csv)
    if not rows:
        sys.exit(f"No rows parsed from {args.csv}")

    train, test = stratified_split(rows, seed=args.seed, train_frac=args.train_frac)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")
    write_jsonl(train, train_path)
    write_jsonl(test, test_path)

    def _balance(split):
        c = {"removed": 0, "approved": 0}
        for r in split:
            c[r["label"]] += 1
        return c

    counts = {
        "input_csv": os.path.abspath(args.csv),
        "n_total": len(rows),
        "n_train": len(train),
        "n_test": len(test),
        "train_balance": _balance(train),
        "test_balance": _balance(test),
        "seed": args.seed,
        "train_frac": args.train_frac,
    }
    with open(os.path.join(args.out_dir, "counts.json"), "w") as f:
        json.dump(counts, f, indent=2)

    print(f"{args.csv}: {len(rows)} rows -> {len(train)} train + {len(test)} test")
    print(f"  train balance: {counts['train_balance']}")
    print(f"  test balance:  {counts['test_balance']}")


if __name__ == "__main__":
    main()
