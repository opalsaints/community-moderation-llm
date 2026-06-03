#!/usr/bin/env python3
"""
Compute extended evaluation metrics from prediction files.

Takes prediction JSONL files (from prompt_eval.py or finetune_lora.py) and computes:
- Confusion matrix, accuracy, F1, precision, recall
- Cohen's kappa (agreement beyond chance)
- Bootstrap 95% confidence intervals for F1, precision, recall
- McNemar's test (pairwise comparison between two models)

Usage:
    # Single prediction file
    python compute_metrics.py --predictions results/prompted/model_sub_predictions.jsonl

    # Batch: all prediction files in results directory
    python compute_metrics.py --results-dir ~/data/results/

    # Pairwise McNemar's test between two models
    python compute_metrics.py --mcnemar pred_a.jsonl pred_b.jsonl
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def load_predictions(filepath):
    """Load predictions from JSONL file."""
    preds = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def confusion_matrix(predictions):
    """Compute confusion matrix from prediction records.

    Each record has 'label' (removed/approved) and 'prediction' (True/False).
    True = should be removed, False = should stay.
    """
    tp = fp = tn = fn = 0
    unparseable = 0

    for p in predictions:
        pred = p.get("prediction", "")
        label = p.get("label", "")

        if pred == "unparseable" or pred not in ("True", "False"):
            unparseable += 1
            continue

        pred_removed = pred == "True"
        label_removed = label == "removed"

        if pred_removed and label_removed:
            tp += 1
        elif pred_removed and not label_removed:
            fp += 1
        elif not pred_removed and not label_removed:
            tn += 1
        else:
            fn += 1

    return tp, fp, tn, fn, unparseable


def basic_metrics(tp, fp, tn, fn):
    """Compute accuracy, F1, precision, recall from confusion matrix."""
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def cohens_kappa(tp, fp, tn, fn):
    """Compute Cohen's kappa for binary classification.

    Measures agreement between model predictions and ground truth,
    correcting for agreement expected by chance.
    """
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0

    # Observed agreement
    p_o = (tp + tn) / total

    # Expected agreement by chance
    p_pred_pos = (tp + fp) / total
    p_true_pos = (tp + fn) / total
    p_pred_neg = (tn + fn) / total
    p_true_neg = (tn + fp) / total
    p_e = p_pred_pos * p_true_pos + p_pred_neg * p_true_neg

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1 - p_e)
    return round(kappa, 4)


def bootstrap_ci(predictions, metric_fn, n_bootstrap=2000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric.

    Args:
        predictions: list of prediction records
        metric_fn: function(predictions) -> float
        n_bootstrap: number of bootstrap samples
        ci: confidence level
        seed: random seed for reproducibility

    Returns:
        (lower, upper, point_estimate)
    """
    rng = random.Random(seed)
    n = len(predictions)
    if n == 0:
        return (0.0, 0.0, 0.0)

    point = metric_fn(predictions)
    scores = []

    for _ in range(n_bootstrap):
        sample = rng.choices(predictions, k=n)
        scores.append(metric_fn(sample))

    scores.sort()
    alpha = 1 - ci
    lo_idx = int(math.floor(alpha / 2 * n_bootstrap))
    hi_idx = int(math.ceil((1 - alpha / 2) * n_bootstrap)) - 1
    lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
    hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

    return (round(scores[lo_idx], 4), round(scores[hi_idx], 4), round(point, 4))


def metric_f1(predictions):
    """Compute F1 from a list of prediction records."""
    tp, fp, tn, fn, _ = confusion_matrix(predictions)
    m = basic_metrics(tp, fp, tn, fn)
    return m["f1"]


def metric_precision(predictions):
    tp, fp, tn, fn, _ = confusion_matrix(predictions)
    m = basic_metrics(tp, fp, tn, fn)
    return m["precision"]


def metric_recall(predictions):
    tp, fp, tn, fn, _ = confusion_matrix(predictions)
    m = basic_metrics(tp, fp, tn, fn)
    return m["recall"]


def metric_accuracy(predictions):
    tp, fp, tn, fn, _ = confusion_matrix(predictions)
    m = basic_metrics(tp, fp, tn, fn)
    return m["accuracy"]


def mcnemars_test(preds_a, preds_b):
    """Run McNemar's test comparing two models on the same test set.

    Compares whether two classifiers have the same error rate.
    Uses chi-squared approximation with continuity correction.

    Args:
        preds_a: predictions from model A (list of dicts with 'id', 'label', 'prediction')
        preds_b: predictions from model B (same format, same test set)

    Returns:
        dict with b, c (discordant counts), chi2, p_value
    """
    # Align by ID
    a_by_id = {p["id"]: p for p in preds_a}
    b_by_id = {p["id"]: p for p in preds_b}
    common_ids = set(a_by_id.keys()) & set(b_by_id.keys())

    if not common_ids:
        return {"error": "no overlapping IDs between prediction files"}

    # Count discordant pairs
    # b = A correct, B wrong
    # c = A wrong, B correct
    b_count = 0  # A right, B wrong
    c_count = 0  # A wrong, B right

    for cid in common_ids:
        pa = a_by_id[cid]
        pb = b_by_id[cid]
        label = pa["label"]

        a_correct = _is_correct(pa["prediction"], label)
        b_correct = _is_correct(pb["prediction"], label)

        if a_correct and not b_correct:
            b_count += 1
        elif not a_correct and b_correct:
            c_count += 1

    # Chi-squared with continuity correction
    denom = b_count + c_count
    if denom == 0:
        return {
            "b": b_count,
            "c": c_count,
            "chi2": 0.0,
            "p_value": 1.0,
            "n_common": len(common_ids),
        }

    chi2 = (abs(b_count - c_count) - 1) ** 2 / denom

    # p-value from chi-squared distribution with 1 df
    # Using survival function approximation
    p_value = _chi2_sf(chi2, df=1)

    return {
        "b": b_count,
        "c": c_count,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "n_common": len(common_ids),
        "significant_0.05": p_value < 0.05,
        "significant_0.01": p_value < 0.01,
    }


def _is_correct(prediction, label):
    """Check if a prediction matches the label."""
    if prediction == "unparseable" or prediction not in ("True", "False"):
        return False
    pred_removed = prediction == "True"
    label_removed = label == "removed"
    return pred_removed == label_removed


def _chi2_sf(x, df=1):
    """Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularized incomplete gamma function approximation.
    For df=1, this is 2 * (1 - Phi(sqrt(x))) where Phi is standard normal CDF.
    """
    if x <= 0:
        return 1.0
    # For df=1: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    # Using error function: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    z = math.sqrt(x)
    return 2 * (1 - _normal_cdf(z))


def _normal_cdf(x):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compute_auc_roc(predictions):
    """Compute ROC-AUC from confidence scores.

    Uses confidence_removed as the predicted probability of removal.
    Pure Python trapezoidal integration over the ROC curve.
    Returns None if confidence scores are unavailable.
    """
    scored = []
    for p in predictions:
        conf = p.get("confidence_removed")
        if conf is None:
            continue
        label_removed = 1 if p.get("label") == "removed" else 0
        scored.append((conf, label_removed))

    if len(scored) < 10:
        return None

    # Sort by confidence descending (highest P(removed) first)
    scored.sort(key=lambda x: -x[0])

    total_pos = sum(y for _, y in scored)
    total_neg = len(scored) - total_pos
    if total_pos == 0 or total_neg == 0:
        return None

    auc = 0.0
    tp = 0
    fp = 0
    prev_tpr = 0.0
    prev_fpr = 0.0

    for conf, label in scored:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / total_pos
        fpr = fp / total_neg
        # Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr = tpr
        prev_fpr = fpr

    return round(auc, 4)


def resample_at_rate(predictions, removal_rate, rng):
    """Resample predictions to simulate a target removal rate.

    Keeps all approved predictions, downsamples removed predictions
    so that removed / total = removal_rate.
    """
    removed = [p for p in predictions if p.get("label") == "removed"]
    approved = [p for p in predictions if p.get("label") == "approved"]

    if not approved or not removed or removal_rate <= 0 or removal_rate >= 1:
        return predictions

    # How many removed to keep: n_rem / (n_rem + n_app) = rate
    # n_rem = rate * n_app / (1 - rate)
    n_rem = int(round(removal_rate * len(approved) / (1 - removal_rate)))
    n_rem = max(1, min(n_rem, len(removed)))

    sampled_removed = rng.sample(removed, n_rem)
    return sampled_removed + approved


def evaluate_at_rate(predictions, removal_rate, n_runs=30, seed=42):
    """Evaluate at a simulated removal rate over multiple resampled runs.

    Returns dict with mean and std for F1, precision, recall, accuracy.
    """
    rng = random.Random(seed)
    run_metrics = {"f1": [], "precision": [], "recall": [], "accuracy": []}

    for _ in range(n_runs):
        resampled = resample_at_rate(predictions, removal_rate, rng)
        tp, fp, tn, fn, _ = confusion_matrix(resampled)
        m = basic_metrics(tp, fp, tn, fn)
        for k in run_metrics:
            run_metrics[k].append(m[k])

    result = {}
    for k, vals in run_metrics.items():
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        result[k + "_mean"] = round(mean, 4)
        result[k + "_std"] = round(std, 4)

    result["removal_rate"] = removal_rate
    result["n_runs"] = n_runs
    return result


def compute_all_metrics(predictions, n_bootstrap=2000, seed=42):
    """Compute all metrics for a single set of predictions."""
    tp, fp, tn, fn, unparseable = confusion_matrix(predictions)
    basic = basic_metrics(tp, fp, tn, fn)
    kappa = cohens_kappa(tp, fp, tn, fn)

    # Bootstrap CIs
    f1_ci = bootstrap_ci(predictions, metric_f1, n_bootstrap, seed=seed)
    prec_ci = bootstrap_ci(predictions, metric_precision, n_bootstrap, seed=seed)
    rec_ci = bootstrap_ci(predictions, metric_recall, n_bootstrap, seed=seed)
    acc_ci = bootstrap_ci(predictions, metric_accuracy, n_bootstrap, seed=seed)

    auc = compute_auc_roc(predictions)

    return {
        "accuracy": basic["accuracy"],
        "f1": basic["f1"],
        "precision": basic["precision"],
        "recall": basic["recall"],
        "cohens_kappa": kappa,
        "auc_roc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "unparseable": unparseable,
        "total_evaluated": tp + fp + tn + fn,
        "total_attempted": len(predictions),
        "bootstrap_ci": {
            "n_bootstrap": n_bootstrap,
            "confidence_level": 0.95,
            "f1": {"lower": f1_ci[0], "upper": f1_ci[1], "point": f1_ci[2]},
            "precision": {"lower": prec_ci[0], "upper": prec_ci[1], "point": prec_ci[2]},
            "recall": {"lower": rec_ci[0], "upper": rec_ci[1], "point": rec_ci[2]},
            "accuracy": {"lower": acc_ci[0], "upper": acc_ci[1], "point": acc_ci[2]},
        },
    }


def find_prediction_files(results_dir):
    """Find all *_predictions.jsonl files in results directory tree."""
    results_dir = Path(results_dir)
    files = []
    for p in sorted(results_dir.rglob("*_predictions.jsonl")):
        files.append(p)
    return files


def infer_metadata(pred_path):
    """Infer model/subreddit/condition from prediction filename or companion JSON."""
    pred_path = Path(pred_path)
    # Try companion summary JSON (same name without _predictions.jsonl)
    stem = pred_path.name.replace("_predictions.jsonl", "")
    summary_json = pred_path.parent / f"{stem}.json"

    meta = {}
    if summary_json.exists():
        with open(summary_json) as f:
            data = json.load(f)
        meta["model"] = data.get("model", "")
        meta["subreddit"] = data.get("subreddit", "")
        meta["method"] = data.get("method", data.get("condition", ""))

    # Infer from directory name if not in JSON
    if not meta.get("method"):
        parent = pred_path.parent.name
        if parent in ("prompted", "finetuned", "test_ft", "test_vllm"):
            meta["method"] = parent

    return meta


def process_single(pred_path, output_path=None, n_bootstrap=2000, seed=42):
    """Process a single prediction file and save enhanced metrics."""
    predictions = load_predictions(pred_path)
    if not predictions:
        print(f"  WARNING: no predictions in {pred_path}")
        return None

    metrics = compute_all_metrics(predictions, n_bootstrap=n_bootstrap, seed=seed)
    meta = infer_metadata(pred_path)
    metrics.update(meta)
    metrics["source_predictions"] = str(pred_path)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved: {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute extended metrics from prediction files"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--predictions",
        help="Single predictions JSONL file",
    )
    mode.add_argument(
        "--results-dir",
        help="Batch process all prediction files in directory tree",
    )
    mode.add_argument(
        "--mcnemar",
        nargs=2,
        metavar=("PRED_A", "PRED_B"),
        help="McNemar's test between two prediction files",
    )

    parser.add_argument(
        "--output", "-o",
        help="Output file path (single mode) or directory (batch mode)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap samples (default: 2000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--imbalanced",
        action="store_true",
        help="Run imbalanced evaluation at simulated removal rates",
    )
    parser.add_argument(
        "--removal-rates",
        default="0.05,0.10",
        help="Comma-separated removal rates for imbalanced eval (default: 0.05,0.10)",
    )
    parser.add_argument(
        "--natural-rates",
        help="JSON file mapping subreddit -> {removal_rate: float} for per-sub natural rate eval",
    )
    parser.add_argument(
        "--n-imbalanced-runs",
        type=int,
        default=30,
        help="Number of resampling runs per rate (default: 30)",
    )

    args = parser.parse_args()

    if args.predictions:
        # Single file mode
        pred_path = args.predictions
        if not os.path.exists(pred_path):
            print(f"ERROR: {pred_path} not found")
            sys.exit(1)

        output = args.output
        if not output:
            output = pred_path.replace("_predictions.jsonl", "_metrics.json")

        print(f"Processing: {pred_path}")
        metrics = process_single(pred_path, output, args.n_bootstrap, args.seed)
        if metrics:
            print(f"\n  F1:        {metrics['f1']:.4f} [{metrics['bootstrap_ci']['f1']['lower']:.4f}, {metrics['bootstrap_ci']['f1']['upper']:.4f}]")
            print(f"  Precision: {metrics['precision']:.4f} [{metrics['bootstrap_ci']['precision']['lower']:.4f}, {metrics['bootstrap_ci']['precision']['upper']:.4f}]")
            print(f"  Recall:    {metrics['recall']:.4f} [{metrics['bootstrap_ci']['recall']['lower']:.4f}, {metrics['bootstrap_ci']['recall']['upper']:.4f}]")
            print(f"  Accuracy:  {metrics['accuracy']:.4f} [{metrics['bootstrap_ci']['accuracy']['lower']:.4f}, {metrics['bootstrap_ci']['accuracy']['upper']:.4f}]")
            print(f"  Kappa:     {metrics['cohens_kappa']:.4f}")
            if metrics.get("auc_roc") is not None:
                print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

            if args.imbalanced:
                predictions = load_predictions(pred_path)
                rates = [float(r) for r in args.removal_rates.split(",")]
                imbalanced = {}
                for rate in rates:
                    res = evaluate_at_rate(predictions, rate, n_runs=args.n_imbalanced_runs, seed=args.seed)
                    imbalanced[str(rate)] = res
                    print(f"\n  Imbalanced @ {rate:.0%}: F1={res['f1_mean']:.4f} +/- {res['f1_std']:.4f}  "
                          f"Prec={res['precision_mean']:.4f}  Rec={res['recall_mean']:.4f}")
                metrics["imbalanced_eval"] = imbalanced
                # Re-save with imbalanced results
                with open(output, "w") as f:
                    json.dump(metrics, f, indent=2)

    elif args.results_dir:
        # Batch mode
        results_dir = args.results_dir
        output_dir = args.output or results_dir

        pred_files = find_prediction_files(results_dir)
        if not pred_files:
            print(f"No prediction files found in {results_dir}")
            sys.exit(1)

        print(f"Found {len(pred_files)} prediction files\n")

        # Load natural rates if provided
        natural_rates = {}
        if args.natural_rates and os.path.exists(args.natural_rates):
            with open(args.natural_rates) as f:
                nr_data = json.load(f)
            for sub, info in nr_data.items():
                if isinstance(info, dict):
                    natural_rates[sub] = info.get("removal_rate", 0)
                else:
                    natural_rates[sub] = float(info)
            print(f"Loaded natural rates for {len(natural_rates)} subreddits\n")

        imbalanced_rates = []
        if args.imbalanced:
            imbalanced_rates = [float(r) for r in args.removal_rates.split(",")]

        all_metrics = {}
        for pf in pred_files:
            rel = pf.relative_to(Path(results_dir))
            out_path = Path(output_dir) / str(rel).replace(
                "_predictions.jsonl", "_metrics.json"
            )
            print(f"Processing: {rel}")
            metrics = process_single(str(pf), str(out_path), args.n_bootstrap, args.seed)
            if metrics:
                key = str(rel).replace("_predictions.jsonl", "")
                auc_str = f"  AUC={metrics['auc_roc']:.4f}" if metrics.get("auc_roc") is not None else ""
                print(f"  F1={metrics['f1']:.4f}  Kappa={metrics['cohens_kappa']:.4f}{auc_str}")

                # Imbalanced evaluation
                if imbalanced_rates:
                    predictions = load_predictions(str(pf))
                    imbalanced = {}
                    sub = metrics.get("subreddit", "")

                    for rate in imbalanced_rates:
                        res = evaluate_at_rate(predictions, rate, n_runs=args.n_imbalanced_runs, seed=args.seed)
                        imbalanced[str(rate)] = res

                    # Also evaluate at natural rate if available
                    if sub and sub in natural_rates:
                        nat_rate = natural_rates[sub]
                        res = evaluate_at_rate(predictions, nat_rate, n_runs=args.n_imbalanced_runs, seed=args.seed)
                        imbalanced["natural"] = res
                        imbalanced["natural"]["removal_rate"] = nat_rate

                    metrics["imbalanced_eval"] = imbalanced

                    # Show summary
                    parts = []
                    for rate in imbalanced_rates:
                        r = imbalanced[str(rate)]
                        parts.append(f"{rate:.0%}: F1={r['f1_mean']:.4f}")
                    if "natural" in imbalanced:
                        r = imbalanced["natural"]
                        parts.append(f"nat({natural_rates[sub]:.1%}): F1={r['f1_mean']:.4f}")
                    print(f"  Imbalanced: {', '.join(parts)}")

                    # Re-save with imbalanced results
                    with open(str(out_path), "w") as f:
                        json.dump(metrics, f, indent=2)

                all_metrics[key] = metrics
            print()

        # Save summary
        summary_path = Path(output_dir) / "all_metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        print(f"Processed {len(all_metrics)} prediction files")

    elif args.mcnemar:
        # McNemar's test mode
        path_a, path_b = args.mcnemar
        for p in [path_a, path_b]:
            if not os.path.exists(p):
                print(f"ERROR: {p} not found")
                sys.exit(1)

        preds_a = load_predictions(path_a)
        preds_b = load_predictions(path_b)

        print(f"Model A: {path_a} ({len(preds_a)} predictions)")
        print(f"Model B: {path_b} ({len(preds_b)} predictions)")

        result = mcnemars_test(preds_a, preds_b)

        if "error" in result:
            print(f"\nERROR: {result['error']}")
            sys.exit(1)

        print(f"\nMcNemar's Test:")
        print(f"  Common test samples: {result['n_common']}")
        print(f"  A correct, B wrong:  {result['b']}")
        print(f"  A wrong, B correct:  {result['c']}")
        print(f"  Chi-squared:         {result['chi2']:.4f}")
        print(f"  p-value:             {result['p_value']:.6f}")
        print(f"  Significant (0.05):  {result['significant_0.05']}")
        print(f"  Significant (0.01):  {result['significant_0.01']}")

        if args.output:
            meta_a = infer_metadata(path_a)
            meta_b = infer_metadata(path_b)
            output = {
                "model_a": meta_a,
                "model_b": meta_b,
                "source_a": path_a,
                "source_b": path_b,
                **result,
            }
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
