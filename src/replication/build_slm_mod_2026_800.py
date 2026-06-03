"""Build 800-row stratified sub-samples of the 2026 SLM-Mod cube training sets.

Phase D.1 of the Section 4.5 SLM-Mod replication: training-set-size confound elimination.
The existing 2026 cube cells trained on 8000 rows, while the 2017 cube cells
trained on 800. To make the era-axis comparison apples-to-apples on training-data
scale, this script sub-samples the 2026 train.jsonl files (deterministic seed=42
stratified) to 800 rows each (400 removed + 400 approved), preserving the test
set unchanged.

Inputs are the existing 2026 random_split train/test JSONL pairs at
``~/data/dataset_2026/<sub>/random_split/{train,test}.jsonl``. Outputs are
written to ``~/data/dataset_2026_800/<sub>/random_split/{train,test}.jsonl``,
where train.jsonl is the 800-row sub-sample and test.jsonl is a copy of the
canonical n=2000 test set so the eval is identical to the existing cube cells.

Usage:
    python3 build_slm_mod_2026_800.py \\
        --in-root ~/data/dataset_2026 \\
        --out-root ~/data/dataset_2026_800 \\
        --subs changemyview politics
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path


def stratified_subsample(rows, n_per_class, seed=42):
    """Return n_per_class examples from each label class, deterministic in seed."""
    by_label = {"removed": [], "approved": []}
    for r in rows:
        if r["label"] in by_label:
            by_label[r["label"]].append(r)
    rng = random.Random(seed)
    out = []
    for label in ("removed", "approved"):
        bucket = list(by_label[label])
        rng.shuffle(bucket)
        if len(bucket) < n_per_class:
            raise ValueError(
                f"Need {n_per_class} {label} rows, only have {len(bucket)}"
            )
        out.extend(bucket[:n_per_class])
    rng.shuffle(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-root", default="~/data/dataset_2026",
                   help="Root with <sub>/random_split/{train,test}.jsonl")
    p.add_argument("--out-root", default="~/data/dataset_2026_800",
                   help="Output root; train.jsonl gets sub-sampled, test.jsonl copied")
    p.add_argument("--subs", nargs="+", default=["changemyview", "politics"])
    p.add_argument("--n-per-class", type=int, default=400,
                   help="800 rows total / 2 classes = 400 per class")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    in_root = Path(os.path.expanduser(args.in_root))
    out_root = Path(os.path.expanduser(args.out_root))

    for sub in args.subs:
        in_dir = in_root / sub / "random_split"
        out_dir = out_root / sub / "random_split"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_in = in_dir / "train.jsonl"
        test_in = in_dir / "test.jsonl"
        if not train_in.exists():
            raise FileNotFoundError(f"missing input: {train_in}")

        with open(train_in) as f:
            train_rows = [json.loads(line) for line in f]

        sample = stratified_subsample(train_rows, args.n_per_class, seed=args.seed)
        train_out = out_dir / "train.jsonl"
        with open(train_out, "w") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        balance = sum(1 for r in sample if r["label"] == "removed")
        print(f"{sub}: wrote {len(sample)} train rows "
              f"({balance} removed / {len(sample) - balance} approved) -> {train_out}")

        if test_in.exists():
            shutil.copy(test_in, out_dir / "test.jsonl")
            with open(test_in) as f:
                n_test = sum(1 for _ in f)
            print(f"{sub}: copied test set unchanged ({n_test} rows) -> {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
