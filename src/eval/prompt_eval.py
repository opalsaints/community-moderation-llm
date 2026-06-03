#!/usr/bin/env python3
"""
Zero-shot prompting evaluation for content moderation.

Uses vLLM for batched inference with chat templates and constrained decoding.
Classifies test set comments as True (should be removed) or False (should stay)
using instruction-tuned models with proper chat formatting.

Supports two conditions:
  --with-rules: prompt includes subreddit rules
  --without-rules: prompt includes only subreddit name (default)

Usage:
    python prompt_eval.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset-dir ./dataset/AskHistorians/random_split \
        --output-dir ./results/prompted/ \
        --subreddit AskHistorians

    python prompt_eval.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --dataset-dir ./dataset/AskHistorians/random_split \
        --output-dir ./results/prompted/ \
        --subreddit AskHistorians \
        --with-rules --rules-file ./rules/AskHistorians.txt
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


HF_CACHE_DIRS = [
    "/projects/2/managed_datasets/hf_cache_dir",
    os.environ.get("HF_CACHE_DIR", "./hf_cache"),
]


def find_cache_dir(model_id, cache_dirs=HF_CACHE_DIRS):
    """Find which cache directory contains the model."""
    cache_name = f"models--{model_id.replace('/', '--')}"
    for d in cache_dirs:
        model_path = os.path.join(d, cache_name, "snapshots")
        if os.path.exists(model_path) and os.listdir(model_path):
            return d
    return cache_dirs[0]


def load_jsonl(filepath):
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def build_system_prompt(subreddit, rules_text=None):
    """Build the system/instruction portion of the prompt."""
    if rules_text:
        return (
            f"You are a content moderator for the Reddit community r/{subreddit}.\n\n"
            f"The community has the following rules:\n{rules_text}\n\n"
            f"Determine whether a comment violates the community's rules or norms. "
            f"Answer with 'True' if the comment should be removed, or 'False' if it should stay."
        )
    return (
        f"You are a content moderator for the Reddit community r/{subreddit}.\n\n"
        f"Determine whether a comment violates the community's rules or norms. "
        f"Answer with 'True' if the comment should be removed, or 'False' if it should stay."
    )


def build_user_message(comment, parent_body=""):
    """Build the user message containing the comment to classify."""
    parts = []
    if parent_body:
        parts.append(f"Context (preceding comment in thread):\n{parent_body}")
    parts.append(f"Comment:\n{comment}")
    return "\n\n".join(parts)


def build_chat_messages(system_prompt, user_message, model_id):
    """Build chat messages, handling model-specific quirks."""
    model_lower = model_id.lower()

    # Gemma IT does not support system role - embed in user message
    if "gemma" in model_lower:
        combined = f"{system_prompt}\n\n{user_message}"
        return [{"role": "user", "content": combined}]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def truncate_text(text, tokenizer, max_tokens):
    """Truncate text to max_tokens using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text


def compute_metrics(predictions, labels):
    """Compute accuracy, F1, precision, recall for binary classification."""
    tp = fp = tn = fn = 0
    unparseable = 0

    for pred, label in zip(predictions, labels):
        if pred == "unparseable":
            unparseable += 1
            continue
        # True = removed, False = approved
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
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "unparseable": unparseable,
        "total_evaluated": total,
        "total_attempted": len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot prompting evaluation (vLLM)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset-dir", required=True, help="Directory with test.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for results")
    parser.add_argument("--subreddit", required=True, help="Subreddit name")
    parser.add_argument("--with-rules", action="store_true", help="Include subreddit rules")
    parser.add_argument("--rules-file", type=str, default=None, help="Path to rules text file")
    parser.add_argument("--max-test-samples", type=int, default=1000, help="Max test samples (default: 1000)")
    parser.add_argument("--max-comment-tokens", type=int, default=512, help="Max tokens per comment")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache dir (auto-detected)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory fraction")
    args = parser.parse_args()

    # Auto-detect cache directory
    if args.cache_dir is None:
        args.cache_dir = find_cache_dir(args.model)

    model_short = args.model.split("/")[-1]
    condition = "with_rules" if args.with_rules else "without_rules"

    print(f"=== Zero-Shot Prompting (vLLM) ===")
    print(f"Model:      {args.model}")
    print(f"Cache dir:  {args.cache_dir}")
    print(f"Subreddit:  r/{args.subreddit}")
    print(f"Condition:  {condition}")

    # Load tokenizer (for chat template formatting and truncation)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )

    # Load rules
    rules_text = ""
    if args.with_rules and args.rules_file:
        with open(args.rules_file) as f:
            rules_text = f.read().strip()
        print(f"Rules:      {len(rules_text)} chars from {args.rules_file}")

    # Load test data
    dataset_dir = Path(args.dataset_dir)
    test_data = load_jsonl(dataset_dir / "test.jsonl")
    if len(test_data) > args.max_test_samples:
        test_data = test_data[:args.max_test_samples]
    print(f"Samples:    {len(test_data)}")

    # Build all prompts using chat templates
    system_prompt = build_system_prompt(args.subreddit, rules_text if args.with_rules else None)
    prompts = []

    print("Building prompts...")
    for item in test_data:
        comment = truncate_text(item["body"], tokenizer, args.max_comment_tokens)
        parent_body = item.get("parent_body", "")
        if parent_body:
            parent_body = truncate_text(parent_body, tokenizer, args.max_comment_tokens)

        user_msg = build_user_message(comment, parent_body)
        messages = build_chat_messages(system_prompt, user_msg, args.model)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    print(f"Prompts built. Sample length: {len(prompts[0])} chars")

    # Load model via vLLM
    print(f"Loading model via vLLM...")
    llm = LLM(
        model=args.model,
        download_dir=args.cache_dir,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
        config_format="hf",
    )

    # Constrained decoding: model can only output "True" or "False"
    structured_outputs = StructuredOutputsParams(choice=["True", "False"])
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=3,
        logprobs=5,
        structured_outputs=structured_outputs,
    )

    # Batch inference
    print(f"Running inference on {len(prompts)} samples...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    # Process results
    predictions = []
    labels = []
    raw_outputs = []

    for i, (output, item) in enumerate(zip(outputs, test_data)):
        generated = output.outputs[0]
        text = generated.text.strip()
        label = item["label"]

        # Parse prediction (should be "True" or "False" due to constrained decoding)
        if text == "True":
            pred = "True"
        elif text == "False":
            pred = "False"
        else:
            pred = "unparseable"

        # Extract logprobs for confidence score
        confidence = None
        if generated.logprobs:
            first_token_logprobs = generated.logprobs[0]
            # Find logprobs for True and False tokens
            true_logprob = None
            false_logprob = None
            for token_id, logprob_obj in first_token_logprobs.items():
                decoded = logprob_obj.decoded_token.strip()
                if decoded == "True":
                    true_logprob = logprob_obj.logprob
                elif decoded == "False":
                    false_logprob = logprob_obj.logprob

            if true_logprob is not None and false_logprob is not None:
                # Convert to probability of "True" (removed)
                true_prob = math.exp(true_logprob)
                false_prob = math.exp(false_logprob)
                confidence = round(true_prob / (true_prob + false_prob), 4)
            elif true_logprob is not None:
                confidence = round(math.exp(true_logprob), 4)
            elif false_logprob is not None:
                confidence = round(1.0 - math.exp(false_logprob), 4)

        predictions.append(pred)
        labels.append(label)
        raw_outputs.append({
            "id": item.get("id"),
            "label": label,
            "prediction": pred,
            "raw": text,
            "confidence_removed": confidence,
        })

    # Compute metrics
    metrics = compute_metrics(predictions, labels)
    metrics["model"] = args.model
    metrics["subreddit"] = args.subreddit
    metrics["condition"] = condition
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["samples_per_second"] = round(len(test_data) / elapsed, 2) if elapsed > 0 else 0
    metrics["engine"] = "vllm"
    metrics["constrained_decoding"] = True
    metrics["labels"] = "True/False"

    # Print summary
    print(f"\n{'='*60}")
    print(f"Model:      {model_short}")
    print(f"Subreddit:  r/{args.subreddit}")
    print(f"Condition:  {condition}")
    print(f"Samples:    {metrics['total_attempted']}")
    print(f"Time:       {elapsed:.1f}s ({metrics['samples_per_second']} samples/sec)")
    print(f"\nAccuracy:   {metrics['accuracy']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Unparseable:{metrics['unparseable']}")
    print(f"\nTP={metrics['tp']}  FP={metrics['fp']}")
    print(f"FN={metrics['fn']}  TN={metrics['tn']}")

    # Confidence stats
    confs = [r["confidence_removed"] for r in raw_outputs if r["confidence_removed"] is not None]
    if confs:
        print(f"\nConfidence (P(removed)): mean={sum(confs)/len(confs):.3f}, "
              f"min={min(confs):.3f}, max={max(confs):.3f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{model_short}_{args.subreddit}_{condition}.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_file = output_dir / f"{model_short}_{args.subreddit}_{condition}_predictions.jsonl"
    with open(predictions_file, "w") as f:
        for item in raw_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nResults: {results_file}")
    print(f"Predictions: {predictions_file}")


if __name__ == "__main__":
    main()
