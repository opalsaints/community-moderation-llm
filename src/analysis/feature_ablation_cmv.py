#!/usr/bin/env python3
"""
CMV feature ablation: mask feature groups from the enriched prompt and
re-evaluate the pilot A5 adapter on CMV 2024 test set.

The enriched template used in the pilot exposes four item-level fields plus
the optional parent context:
  - post_title            -> 'post_metadata' group
  - is_top_level          -> 'structural' group
  - parent_body           -> 'structural' group
  - account_age_days      -> 'author_history' group
  - author_is_first       -> 'author_history' group

'temporal' features (hour_of_day, day_of_week, etc.) are computed by the
feature pipeline but are not present in the enriched prompt template, so
they cannot be ablated inference-only. Temporal ablation would require a
template change and retraining -- handled in Phase 3.2 (selective retraining)
if any inference-masked group shows a meaningful delta.

Usage:
    python feature_ablation_cmv.py \
        --model Qwen/Qwen3-14B \
        --adapter-dir ~/data/results/pilot/A5_adapter \
        --dataset-dir ~/data/dataset/changemyview/enriched_v2 \
        --rules-file ~/data/rules/changemyview/rules.txt \
        --output-dir ~/data/results/feature_ablation_cmv
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TRAIN = _HERE.parent / "train"  # finetune_v3 lives in src/train/
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
from vllm.lora.request import LoRARequest


# Feature groups that can actually be masked inside the enriched prompt.
# Values are "null sentinels" that build_user_message() maps to neutral
# language ("unknown", "reply", etc.).
MASK_GROUPS = {
    "baseline": {},  # no masking
    "post_metadata": {
        "post_title": "",
    },
    "structural": {
        "is_top_level": False,
        "parent_body": "",
    },
    "author_history": {
        "account_age_days": -1,
        "author_is_first": False,
    },
    "all_masked": {
        "post_title": "",
        "is_top_level": False,
        "parent_body": "",
        "account_age_days": -1,
        "author_is_first": False,
    },
}


def apply_mask(item, mask_fields):
    masked = copy.copy(item)
    for key, val in mask_fields.items():
        masked[key] = val
    return masked


def run_condition(llm, tokenizer, test_data, rules_text, args,
                  sampling_params, lora_request, condition_name, mask_fields):
    masked_data = [apply_mask(item, mask_fields) for item in test_data]
    prompts = [
        format_inference_prompt(
            item, args.subreddit, rules_text, tokenizer,
            args.model, "enriched", args.max_comment_tokens,
        )
        for item in masked_data
    ]

    print(f"  [{condition_name}] {len(prompts)} prompts, running inference...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    elapsed = time.time() - start

    predictions = []
    labels = []
    raw = []
    for output, item in zip(outputs, masked_data):
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
        raw.append({
            "id": item.get("id"),
            "label": label,
            "prediction": pred,
            "raw": text,
            "confidence_removed": confidence,
        })

    metrics = compute_metrics(predictions, labels)
    metrics.update({
        "condition": condition_name,
        "masked_fields": list(mask_fields.keys()),
        "model": args.model,
        "adapter_dir": str(args.adapter_dir),
        "subreddit": args.subreddit,
        "template": "enriched",
        "eval_seconds": round(elapsed, 1),
        "engine": "vllm",
    })

    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / f"ablation_{condition_name}_metrics.json"
    predictions_file = out_dir / f"ablation_{condition_name}_predictions.jsonl"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(predictions_file, "w") as f:
        for rec in raw:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  [{condition_name}] acc={metrics['accuracy']:.4f} "
          f"f1={metrics['f1']:.4f} "
          f"kappa={metrics['cohens_kappa']:.4f} "
          f"({elapsed:.1f}s)")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Feature ablation on CMV via prompt masking (inference only)")
    parser.add_argument("--model", default="Qwen/Qwen3-14B",
                        help="Base model ID (must match the adapter's base)")
    parser.add_argument("--adapter-dir", required=True,
                        help="Path to trained LoRA adapter (e.g. A5_adapter/)")
    parser.add_argument("--dataset-dir", required=True,
                        help="Enriched_v2 directory with train.jsonl / test.jsonl")
    parser.add_argument("--rules-file", required=True,
                        help="Path to CMV rules.txt")
    parser.add_argument("--subreddit", default="changemyview")
    parser.add_argument("--output-dir", default="~/data/results/feature_ablation_cmv")
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--max-comment-tokens", type=int, default=512)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="Must match the adapter's rank (pilot used r=16)")
    parser.add_argument("--groups", nargs="+", default=list(MASK_GROUPS.keys()),
                        help=f"Which groups to ablate. Available: {list(MASK_GROUPS.keys())}")
    args = parser.parse_args()

    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.adapter_dir = os.path.expanduser(args.adapter_dir)
    args.rules_file = os.path.expanduser(args.rules_file)
    if args.cache_dir is None:
        args.cache_dir = find_cache_dir(args.model)

    print(f"=== CMV Feature Ablation ===")
    print(f"Model:       {args.model}")
    print(f"Adapter:     {args.adapter_dir}")
    print(f"Dataset:     {args.dataset_dir}")
    print(f"Cache dir:   {args.cache_dir}")
    print(f"Groups:      {', '.join(args.groups)}")

    # Load data
    test_data = load_jsonl(Path(args.dataset_dir) / "test.jsonl")
    if len(test_data) > args.max_test_samples:
        test_data = test_data[:args.max_test_samples]
    print(f"Test size:   {len(test_data)}")

    with open(args.rules_file) as f:
        rules_text = f.read().strip()

    tokenizer = load_tokenizer(args.model, args.cache_dir)

    print(f"\nLoading vLLM + LoRA adapter...")
    llm = LLM(
        model=args.model,
        download_dir=args.cache_dir,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        config_format="hf",
        enable_lora=True,
        max_lora_rank=args.lora_rank,
    )
    lora_request = LoRARequest(
        lora_name="A5_ablation",
        lora_int_id=1,
        lora_path=args.adapter_dir,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5,
        logprobs=5,
    )

    results = []
    for group_name in args.groups:
        if group_name not in MASK_GROUPS:
            print(f"  Unknown group '{group_name}', skipping")
            continue
        print(f"\n--- group: {group_name} ---")
        metrics = run_condition(
            llm, tokenizer, test_data, rules_text, args,
            sampling_params, lora_request,
            group_name, MASK_GROUPS[group_name],
        )
        results.append(metrics)

    # Summary
    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"model": args.model, "adapter_dir": args.adapter_dir,
                   "subreddit": args.subreddit, "conditions": results}, f, indent=2)

    baseline = next((r for r in results if r["condition"] == "baseline"), None)

    print(f"\n{'='*70}")
    print(f"{'Condition':<20s} {'Acc':>8s} {'F1':>8s} {'Kappa':>8s} "
          f"{'Δ Acc':>10s} {'Δ F1':>10s}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        dacc = r["accuracy"] - baseline["accuracy"] if baseline else 0
        df1 = r["f1"] - baseline["f1"] if baseline else 0
        print(f"{r['condition']:<20s} {r['accuracy']:>8.4f} "
              f"{r['f1']:>8.4f} {r['cohens_kappa']:>8.4f} "
              f"{dacc:>+10.4f} {df1:>+10.4f}")
    print(f"\nSummary written to {summary_file}")


if __name__ == "__main__":
    main()
