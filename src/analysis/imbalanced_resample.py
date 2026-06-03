#!/usr/bin/env python3
"""Imbalanced-rate resampling for R4_stacked per-sub predictions.

For each of the 15 subs, loads the per-sub predictions.jsonl and, for each
target removal rate in {0.01, 0.05, 0.10, natural_rate[sub]}, performs
N stratified resamples (default 1000, seed 42): keep all approved, downsample
removed class to match the target rate. Report mean and 2.5 / 97.5 percentile
95% CIs on accuracy, macro-F1, Cohen's kappa, precision (removed), recall
(removed).

Natural removal rates are hardcoded from thesis.tex Table 1 (Section 3.1), which
is the authoritative source for per-sub 2026 mod-removal proportions.

Usage (on Snellius):
    python scripts/imbalanced_resample.py \
        --predictions-dir ~/data/results/finetuned_2026 \
        --output-dir     ~/data/results/finetuned_2026/imbalanced_metrics

Output:
    <out>/<sub>.json   full per-sub results (nested rate -> metric -> {mean, ci_low, ci_high})
    <out>/summary.csv  flat one-row-per (sub, rate, metric) table
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean as _mean

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from compute_metrics import (  # noqa: E402
    basic_metrics,
    cohens_kappa,
    confusion_matrix,
    resample_at_rate,
)

# Natural mod-removal rates per sub, sourced from thesis.tex Table 1 (Section 3.1).
# These are the "Removal %" column values, converted to proportions.
NATURAL_RATES: dict[str, float] = {
    "AskHistorians": 0.285,
    "askscience": 0.101,
    "science": 0.039,
    "legaladvice": 0.124,
    "personalfinance": 0.039,
    "relationships": 0.035,
    "AmItheAsshole": 0.014,
    "changemyview": 0.030,
    "explainlikeimfive": 0.017,
    "Games": 0.051,
    "news": 0.028,
    "TwoXChromosomes": 0.024,
    "politics": 0.009,
    "antiai": 0.012,
    "aiwars": 0.005,
}

STANDARD_RATES = [("0.01", 0.01), ("0.05", 0.05), ("0.10", 0.10)]


def percentile(vals: list[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy default) over floats."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def macro_f1(tp: int, fp: int, tn: int, fn: int) -> float:
    """Macro-averaged F1 over {removed, approved}."""
    def f1_one(tp_: int, fp_: int, fn_: int) -> float:
        if tp_ == 0:
            return 0.0
        prec = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        rec = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Removed as positive, then approved as positive (swap TP<->TN, FP<->FN).
    f1_removed = f1_one(tp, fp, fn)
    f1_approved = f1_one(tn, fn, fp)
    return 0.5 * (f1_removed + f1_approved)


def load_predictions(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def resample_metrics(
    predictions: list[dict],
    rate: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Stratified-downsample to target rate n_boot times; return mean + 95% CIs."""
    rng = random.Random(seed)
    records: dict[str, list[float]] = {
        "accuracy": [],
        "f1_macro": [],
        "f1_removed": [],
        "kappa": [],
        "precision": [],
        "recall": [],
    }
    sizes: list[int] = []
    for _ in range(n_boot):
        s = resample_at_rate(predictions, rate, rng)
        sizes.append(len(s))
        tp, fp, tn, fn, _ = confusion_matrix(s)
        b = basic_metrics(tp, fp, tn, fn)
        records["accuracy"].append(b["accuracy"])
        records["f1_macro"].append(macro_f1(tp, fp, tn, fn))
        records["f1_removed"].append(b["f1"])
        records["kappa"].append(cohens_kappa(tp, fp, tn, fn))
        records["precision"].append(b["precision"])
        records["recall"].append(b["recall"])

    out = {}
    for metric_name, vals in records.items():
        out[metric_name] = {
            "mean": round(_mean(vals), 4),
            "ci_low": round(percentile(vals, 2.5), 4),
            "ci_high": round(percentile(vals, 97.5), 4),
        }
    out["rate"] = rate
    out["n_bootstrap"] = n_boot
    out["n_in"] = len(predictions)
    out["n_resampled_mean"] = round(_mean(sizes), 1) if sizes else 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions-dir",
        required=True,
        help="Dir containing r4_stacked_<sub>_predictions.jsonl files.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Dir for per-sub JSON + summary.csv.",
    )
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--subs",
        nargs="*",
        default=list(NATURAL_RATES.keys()),
        help="Subreddits to run (defaults to all 15).",
    )
    ap.add_argument(
        "--run-tag",
        default="r4_stacked",
        help="Prefix of predictions files: <run_tag>_<sub>_predictions.jsonl.",
    )
    args = ap.parse_args()

    pred_dir = Path(args.predictions_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for sub in args.subs:
        if sub not in NATURAL_RATES:
            print(f"[skip] {sub}: not in NATURAL_RATES table", file=sys.stderr)
            continue
        path = pred_dir / f"{args.run_tag}_{sub}_predictions.jsonl"
        if not path.exists():
            print(f"[skip] {sub}: predictions file missing at {path}", file=sys.stderr)
            continue

        preds = load_predictions(path)
        per_sub = {
            "sub": sub,
            "run_tag": args.run_tag,
            "n_predictions": len(preds),
            "natural_rate": NATURAL_RATES[sub],
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "rates": {},
        }
        rates = STANDARD_RATES + [("natural", NATURAL_RATES[sub])]
        for label, rate in rates:
            print(
                f"[{sub} @ {label} = {rate:.3f}] bootstrapping {args.n_bootstrap} resamples...",
                file=sys.stderr,
            )
            r = resample_metrics(preds, rate, args.n_bootstrap, args.seed)
            r["label"] = label
            per_sub["rates"][label] = r
            for m_name, m_val in r.items():
                if not isinstance(m_val, dict):
                    continue
                summary_rows.append({
                    "sub": sub,
                    "rate_label": label,
                    "rate": rate,
                    "metric": m_name,
                    "mean": m_val["mean"],
                    "ci_low": m_val["ci_low"],
                    "ci_high": m_val["ci_high"],
                })
        (out_dir / f"{sub}.json").write_text(json.dumps(per_sub, indent=2))
        print(f"[ok] {sub} -> {out_dir / (sub + '.json')}", file=sys.stderr)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sub", "rate_label", "rate", "metric", "mean", "ci_low", "ci_high"],
        )
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[ok] summary -> {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
