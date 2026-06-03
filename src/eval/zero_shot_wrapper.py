#!/usr/bin/env python3
"""
Zero-shot Qwen 3 14B spot-check across a small set of subreddits.

Loads the vLLM engine once, then iterates over a list of subreddits, running
SLM-Mod-style zero-shot inference on each sub's test.jsonl. Produces one
metrics file per sub plus a summary table. Metrics include accuracy, macro-F1,
Cohen's kappa, and AUROC (computed inline from per-prediction logprob-derived
confidence scores, no sklearn dependency).

This wraps prompt construction with the same SLM-Mod template used in
``finetune_v3.py``, so the zero-shot condition is directly comparable to the
per-sub fine-tuned runs from Phase 2.1.

In the narrowed Phase 2 scope (2026-04-15), this is run as a 3-sub spot-check
on changemyview, AskHistorians, and antiai. The pilot never measured the
FT-vs-zero-shot gap on Qwen 3 14B specifically (pilot zero-shot used Gemma 2 9B
/ Llama 3.1 8B / Mistral NeMo 12B / Qwen 2.5 7B), so this fills that gap.

Usage (3-sub spot-check, defaults point at the 2026 rebuild paths):
    python zero_shot_wrapper.py \
        --model Qwen/Qwen3-14B \
        --subs changemyview AskHistorians antiai \
        --output-dir ~/data/results/zero_shot_2026

    # Full 15-sub sweep (not in current scope, kept for flexibility):
    python zero_shot_wrapper.py \
        --sub-list-file ~/data/dataset_2026/final_subs_2026.txt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# finetune_v3 lives in ../train and provides the prompt construction, so the
# zero-shot baseline stays in lock-step with the fine-tuned runs.
_HERE = Path(__file__).resolve().parent
_TRAIN = _HERE.parent / "train"
for _p in (_HERE, _TRAIN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from finetune_v3 import (
    compute_metrics,
    find_cache_dir,
    format_inference_prompt,
    load_jsonl,
    load_tokenizer,
    parse_response,
)

from vllm import LLM, SamplingParams


def compute_auroc(confidences, labels):
    """AUROC from confidence-of-removed scores and string labels.

    Uses the Mann-Whitney U formulation with tie-averaged ranks so ties in
    the confidence score don't bias the estimate. Returns None when AUROC
    is undefined (no valid scores, or only one class present).
    """
    pairs = [
        (conf, 1 if lab == "removed" else 0)
        for conf, lab in zip(confidences, labels)
        if conf is not None
    ]
    if not pairs:
        return None
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    ranks = [0.0] * len(pairs_sorted)
    i = 0
    while i < len(pairs_sorted):
        j = i
        while j + 1 < len(pairs_sorted) and pairs_sorted[j + 1][0] == pairs_sorted[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-indexed ranks
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs_sorted) if y == 1)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return round(u_stat / (n_pos * n_neg), 4)


def resolve_sub_list(args):
    if args.subs:
        return list(args.subs)
    if args.sub_list_file:
        path = Path(os.path.expanduser(args.sub_list_file))
        if not path.exists():
            raise FileNotFoundError(f"Sub list file not found: {path}")
        subs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                subs.append(line)
        return subs
    raise ValueError("Provide either --subs or --sub-list-file")


def run_sub(llm, tokenizer, sub, args, sampling_params):
    dataset_dir = Path(os.path.expanduser(args.dataset_root)) / sub / args.split_name
    test_path = dataset_dir / "test.jsonl"
    rules_path = Path(os.path.expanduser(args.rules_root)) / sub / "rules.txt"

    if not test_path.exists():
        print(f"  [{sub}] SKIP: no test.jsonl at {test_path}")
        return None
    if not rules_path.exists():
        print(f"  [{sub}] SKIP: no rules at {rules_path}")
        return None

    rules_text = rules_path.read_text().strip()
    test_data = load_jsonl(test_path)
    if len(test_data) > args.max_test_samples:
        test_data = test_data[:args.max_test_samples]

    prompts = [
        format_inference_prompt(
            item, sub, rules_text, tokenizer,
            args.model, args.template, args.max_comment_tokens,
        )
        for item in test_data
    ]

    print(f"  [{sub}] {len(prompts)} prompts, running inference...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    predictions = []
    labels = []
    confidences = []
    raw_outputs = []
    for output, item in zip(outputs, test_data):
        generated = output.outputs[0]
        text = generated.text.strip()
        pred = parse_response(text)
        label = item["label"]

        confidence = None
        if generated.logprobs and len(generated.logprobs) > 0:
            first_token_logprobs = generated.logprobs[0]
            true_logprob = None
            false_logprob = None
            for token_id, logprob_obj in first_token_logprobs.items():
                decoded = logprob_obj.decoded_token.strip()
                if decoded == "True":
                    true_logprob = logprob_obj.logprob
                elif decoded == "False":
                    false_logprob = logprob_obj.logprob
            if true_logprob is not None and false_logprob is not None:
                true_prob = math.exp(true_logprob)
                false_prob = math.exp(false_logprob)
                confidence = round(true_prob / (true_prob + false_prob), 4)
            elif true_logprob is not None:
                confidence = round(math.exp(true_logprob), 4)
            elif false_logprob is not None:
                confidence = round(1.0 - math.exp(false_logprob), 4)

        predictions.append(pred)
        labels.append(label)
        confidences.append(confidence)
        raw_outputs.append({
            "id": item.get("id"),
            "label": label,
            "prediction": pred,
            "raw": text,
            "confidence_removed": confidence,
        })

    metrics = compute_metrics(predictions, labels)
    auroc = compute_auroc(confidences, labels)
    n_valid_conf = sum(1 for c in confidences if c is not None)
    metrics.update({
        "auroc": auroc,
        "n_with_confidence": n_valid_conf,
        "model": args.model,
        "subreddit": sub,
        "template": args.template,
        "condition": "zero_shot",
        "eval_seconds": round(elapsed, 1),
        "engine": "vllm",
        "constrained_decoding": False,
    })

    auroc_str = f"{auroc:.4f}" if auroc is not None else "  n/a "
    print(f"  [{sub}] acc={metrics['accuracy']:.4f} "
          f"f1={metrics['f1']:.4f} "
          f"kappa={metrics['cohens_kappa']:.4f} "
          f"auroc={auroc_str} "
          f"({elapsed:.1f}s)")

    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / f"zero_shot_{sub}_metrics.json"
    predictions_file = out_dir / f"zero_shot_{sub}_predictions.jsonl"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(predictions_file, "w") as f:
        for rec in raw_outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Qwen 3 14B baseline over N subs")
    parser.add_argument("--model", default="Qwen/Qwen3-14B",
                        help="HuggingFace model ID (default: Qwen/Qwen3-14B)")
    parser.add_argument("--dataset-root", default="~/data/dataset_2026",
                        help="Root directory containing per-sub dataset folders "
                             "(default: ~/data/dataset_2026 from the 2026 rebuild)")
    parser.add_argument("--rules-root", default="~/data/rules_2026",
                        help="Root directory containing per-sub rules folders "
                             "(default: ~/data/rules_2026 from the 2026 rebuild)")
    parser.add_argument("--split-name", default="random_split",
                        help="Split directory name under <dataset-root>/<sub>/")
    parser.add_argument("--output-dir", default="~/data/results/zero_shot_2026",
                        help="Where to write per-sub metrics and summary")
    parser.add_argument("--subs", nargs="+", default=None,
                        help="Explicit list of subreddit names")
    parser.add_argument("--sub-list-file", default=None,
                        help="File with one subreddit name per line "
                             "(e.g. ~/data/dataset/final_subs_2026.txt)")
    parser.add_argument("--template", default="slm-mod",
                        choices=["slm-mod", "enriched", "no-chat"],
                        help="Prompt template (default: slm-mod)")
    parser.add_argument("--max-test-samples", type=int, default=2000)
    parser.add_argument("--max-comment-tokens", type=int, default=512)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    subs = resolve_sub_list(args)
    print(f"=== Zero-Shot Wrapper ===")
    print(f"Model:     {args.model}")
    print(f"Template:  {args.template}")
    print(f"Subs ({len(subs)}): {', '.join(subs)}")

    if args.cache_dir is None:
        args.cache_dir = find_cache_dir(args.model)
    print(f"Cache dir: {args.cache_dir}")

    tokenizer = load_tokenizer(args.model, args.cache_dir)

    print(f"\nLoading vLLM engine...")
    llm = LLM(
        model=args.model,
        download_dir=args.cache_dir,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        config_format="hf",
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=5,
        logprobs=5,
    )

    summary = []
    for sub in subs:
        print(f"\n--- r/{sub} ---")
        try:
            metrics = run_sub(llm, tokenizer, sub, args, sampling_params)
            if metrics is not None:
                summary.append(metrics)
        except Exception as e:
            print(f"  [{sub}] ERROR: {type(e).__name__}: {e}")

    # Summary table
    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "zero_shot_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"model": args.model, "template": args.template,
                   "results": summary}, f, indent=2)

    print(f"\n{'='*72}")
    print(f"{'Subreddit':<25s} {'Acc':>8s} {'F1':>8s} {'Kappa':>8s} "
          f"{'AUROC':>8s} {'Unparse':>8s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for m in summary:
        auroc_val = m.get("auroc")
        auroc_cell = f"{auroc_val:>8.4f}" if auroc_val is not None else f"{'n/a':>8s}"
        print(f"{m['subreddit']:<25s} {m['accuracy']:>8.4f} "
              f"{m['f1']:>8.4f} {m['cohens_kappa']:>8.4f} "
              f"{auroc_cell} {m['unparseable']:>8d}")
    print(f"\nSummary written to {summary_file}")


if __name__ == "__main__":
    main()
