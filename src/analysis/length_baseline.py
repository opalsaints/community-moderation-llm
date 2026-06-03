#!/usr/bin/env python3
"""Text-length baseline for each of the 15 2026 subreddits.

For each sub:
  - Load train.jsonl and test.jsonl from the random_split directory.
  - Compute per-example word-count length (whitespace tokens).
  - On the train set, find the (threshold, direction) pair that maximises
    balanced accuracy. `direction = "short_is_removed"` means predict
    removed when length <= threshold; `direction = "long_is_removed"` means
    predict removed when length >= threshold.
  - Apply the chosen (threshold, direction) on the test set and report
    accuracy, macro-F1, Cohen's kappa, and removed-class precision/recall.

Outputs:
  <out>/<sub>.json     per-sub results
  <out>/summary.csv    one row per sub
  <out>/compare.csv    R4_stacked vs length-baseline per sub (if --r4-metrics-dir given)

Usage (on Snellius):
    python scripts/length_baseline.py \
        --dataset-dir ~/data/dataset_2026 \
        --output-dir  ~/data/results/finetuned_2026/length_baseline \
        --r4-metrics-dir ~/data/results/finetuned_2026
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from compute_metrics import basic_metrics, cohens_kappa, confusion_matrix  # noqa: E402
from imbalanced_resample import NATURAL_RATES, macro_f1  # noqa: E402

SUBS = list(NATURAL_RATES.keys())


def word_len(body: str) -> int:
    return len(body.split()) if body else 0


def load_rows(path: Path) -> list[tuple[int, str]]:
    """Return list of (word_length, label) pairs."""
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append((word_len(r.get("body", "")), r.get("label", "")))
    return out


def balanced_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    """Mean of TPR and TNR (robust to class imbalance)."""
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


def score_threshold(
    rows: list[tuple[int, str]], threshold: int, direction: str,
) -> tuple[int, int, int, int]:
    """Count TP/FP/TN/FN (removed = positive) for a given rule."""
    tp = fp = tn = fn = 0
    for length, label in rows:
        pred_removed = (length <= threshold) if direction == "short_is_removed" else (length >= threshold)
        label_removed = (label == "removed")
        if pred_removed and label_removed:
            tp += 1
        elif pred_removed and not label_removed:
            fp += 1
        elif not pred_removed and not label_removed:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def best_threshold(rows: list[tuple[int, str]]) -> dict:
    """Sweep all integer thresholds in {1..max_len+1}, return best (threshold, direction)."""
    max_len = max((ln for ln, _ in rows), default=0)
    candidates = list(range(1, max_len + 2))
    best = {"bal_acc": -1.0}
    for direction in ("short_is_removed", "long_is_removed"):
        for thr in candidates:
            tp, fp, tn, fn = score_threshold(rows, thr, direction)
            ba = balanced_accuracy(tp, fp, tn, fn)
            if ba > best["bal_acc"]:
                best = {
                    "threshold": thr,
                    "direction": direction,
                    "bal_acc": round(ba, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                }
    return best


def evaluate_test(
    rows: list[tuple[int, str]], threshold: int, direction: str,
) -> dict:
    tp, fp, tn, fn = score_threshold(rows, threshold, direction)
    bm = basic_metrics(tp, fp, tn, fn)
    return {
        "accuracy": bm["accuracy"],
        "f1_removed": bm["f1"],
        "f1_macro": round(macro_f1(tp, fp, tn, fn), 4),
        "kappa": cohens_kappa(tp, fp, tn, fn),
        "precision": bm["precision"],
        "recall": bm["recall"],
        "balanced_accuracy": round(balanced_accuracy(tp, fp, tn, fn), 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def load_r4_metrics(metrics_dir: Path, sub: str) -> dict | None:
    p = metrics_dir / f"r4_stacked_{sub}_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True,
                    help="Root of dataset_2026; expects <sub>/random_split/{train,test}.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--r4-metrics-dir", default=None,
                    help="If given, also emit compare.csv (length vs R4_stacked).")
    ap.add_argument("--subs", nargs="*", default=SUBS)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r4_dir = Path(args.r4_metrics_dir) if args.r4_metrics_dir else None

    summary_rows = []
    compare_rows = []

    for sub in args.subs:
        train_path = dataset_dir / sub / "random_split" / "train.jsonl"
        test_path = dataset_dir / sub / "random_split" / "test.jsonl"
        if not train_path.exists() or not test_path.exists():
            print(f"[skip] {sub}: train/test missing ({train_path}, {test_path})",
                  file=sys.stderr)
            continue
        train = load_rows(train_path)
        test = load_rows(test_path)
        if not train or not test:
            print(f"[skip] {sub}: empty files", file=sys.stderr)
            continue

        fit = best_threshold(train)
        test_metrics = evaluate_test(test, fit["threshold"], fit["direction"])

        per_sub = {
            "sub": sub,
            "length_unit": "words",
            "n_train": len(train),
            "n_test": len(test),
            "fit_on_train": fit,
            "test": test_metrics,
        }
        (out_dir / f"{sub}.json").write_text(json.dumps(per_sub, indent=2))
        print(
            f"[ok] {sub}: thr={fit['threshold']} {fit['direction']}  "
            f"train_bal_acc={fit['bal_acc']:.3f}  "
            f"test_acc={test_metrics['accuracy']:.3f}  "
            f"test_kappa={test_metrics['kappa']:.3f}",
            file=sys.stderr,
        )

        summary_rows.append({
            "sub": sub,
            "threshold_words": fit["threshold"],
            "direction": fit["direction"],
            "train_bal_acc": fit["bal_acc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "test_f1_removed": test_metrics["f1_removed"],
            "test_kappa": test_metrics["kappa"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "n_test": len(test),
        })

        if r4_dir is not None:
            r4 = load_r4_metrics(r4_dir, sub) or {}
            r4_acc = r4.get("accuracy")
            r4_f1 = r4.get("macro_f1") or r4.get("f1_macro") or r4.get("f1")
            r4_kappa = r4.get("cohens_kappa") or r4.get("kappa")
            compare_rows.append({
                "sub": sub,
                "length_acc": test_metrics["accuracy"],
                "r4_acc": r4_acc,
                "length_f1_macro": test_metrics["f1_macro"],
                "r4_f1_macro": r4_f1,
                "length_kappa": test_metrics["kappa"],
                "r4_kappa": r4_kappa,
                "delta_acc": (round(r4_acc - test_metrics["accuracy"], 4)
                              if r4_acc is not None else None),
                "delta_kappa": (round(r4_kappa - test_metrics["kappa"], 4)
                                if r4_kappa is not None else None),
            })

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[ok] summary -> {csv_path}", file=sys.stderr)

    if r4_dir is not None and compare_rows:
        cmp_path = out_dir / "compare.csv"
        with cmp_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
            w.writeheader()
            w.writerows(compare_rows)
        print(f"[ok] compare -> {cmp_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
