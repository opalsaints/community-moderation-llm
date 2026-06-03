"""Build the SLM-Mod 2017 dataset for Section 4.5 thesis replication.

Wraps load_slm_mod_splits.py over a configurable list of subreddits, emitting
``<out-root>/<sub>/slm_mod/{train,test}.jsonl`` matching what
``finetune_v3.py --template slm-mod`` consumes.

Default scope is the Path B-2 subs (changemyview, politics) per locked plan
2026-05-04. Override via --subs.
"""
import argparse
import json
import os
import subprocess
import sys


DEFAULT_SUBS = ["changemyview", "politics"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv-dir",
        required=True,
        help="Directory containing per-sub CSVs (e.g. Agam_Materials/reddit_removals_1k)",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Root directory; produces <out-root>/<sub>/slm_mod/{train,test}.jsonl",
    )
    p.add_argument(
        "--subs",
        nargs="+",
        default=DEFAULT_SUBS,
        help=f"Subs to process (default: {DEFAULT_SUBS})",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    loader = os.path.join(here, "load_slm_mod_splits.py")
    if not os.path.exists(loader):
        sys.exit(f"Cannot find sibling script: {loader}")

    summary = {}
    for sub in args.subs:
        csv_path = os.path.join(args.csv_dir, f"{sub}.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"CSV missing for {sub}: {csv_path}")
        out_dir = os.path.join(args.out_root, sub, "slm_mod")

        cmd = [
            sys.executable,
            loader,
            "--csv", csv_path,
            "--out-dir", out_dir,
            "--seed", str(args.seed),
            "--train-frac", str(args.train_frac),
        ]
        print(f"\n=== Building {sub} ===")
        result = subprocess.run(cmd, check=True)
        del result  # silence unused-var lint

        counts_path = os.path.join(out_dir, "counts.json")
        with open(counts_path) as f:
            summary[sub] = json.load(f)

    # Top-level summary across all subs.
    summary_path = os.path.join(args.out_root, "build_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary: {summary_path}")
    for sub, c in summary.items():
        print(
            f"  {sub:>16}: total={c['n_total']:>4} | "
            f"train={c['n_train']:>4} ({c['train_balance']['removed']}/{c['train_balance']['approved']}) | "
            f"test={c['n_test']:>4} ({c['test_balance']['removed']}/{c['test_balance']['approved']})"
        )


if __name__ == "__main__":
    main()
