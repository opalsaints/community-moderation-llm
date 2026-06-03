#!/usr/bin/env python3
"""
Evaluate Gemini models via Vertex AI on content moderation test sets.

Uses the google-genai SDK with Vertex AI backend for billing through GCP free
trial credits. Produces identical output format (summary JSON + predictions
JSONL) for direct comparison with open-source models via compute_metrics.py.

Supports logprobs for confidence scores and optional thinking mode control.

Prerequisites:
    pip install google-genai
    gcloud auth application-default login

    export GOOGLE_CLOUD_PROJECT=your-gcp-project-id
    export GOOGLE_CLOUD_LOCATION=us-central1
    export GOOGLE_GENAI_USE_VERTEXAI=True

Usage:
    python gemini_eval.py \
        --model gemini-2.5-flash \
        --dataset-dir ./dataset/AskHistorians/random_split \
        --output-dir ./results/gemini/ \
        --subreddit AskHistorians

    python gemini_eval.py \
        --model gemini-2.5-pro \
        --dataset-dir ./dataset/AskHistorians/random_split \
        --output-dir ./results/gemini/ \
        --subreddit AskHistorians \
        --with-rules --rules-file ./rules/AskHistorians/rules.txt

    # Run all subreddits:
    python gemini_eval.py --run-all \
        --model gemini-2.5-flash \
        --data-root ./data \
        --output-dir ./results/gemini/
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Set Vertex AI backend defaults (can be overridden by env vars)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

try:
    from google import genai
    from google.genai.types import GenerateContentConfig, HttpOptions, SafetySetting
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)


AVAILABLE_MODELS = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
}

# Rate limiting defaults (requests per minute) -- conservative for free trial
RPM_LIMITS = {
    "gemini-2.5-flash": 500,
    "gemini-2.5-pro": 100,
    "gemini-2.5-flash-lite": 500,
}

# Optimal thinking budget per model (determined by experiments on r/PublicFreakout):
# - Flash: 128 gives best F1, higher budgets show diminishing returns
# - Pro: 128 minimum (cannot disable thinking)
# - Flash-Lite: does not support thinking
DEFAULT_THINKING_BUDGET = {
    "gemini-2.5-flash": 128,
    "gemini-2.5-pro": 128,
    "gemini-2.5-flash-lite": None,
}

# Safety filters OFF for all categories (experiments showed no effect on
# Gemini 2.5 models where default is already OFF, but explicit for clarity)
SAFETY_OFF = [
    SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
]


def load_jsonl(filepath):
    """Load data from JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_system_prompt(subreddit, rules_text=None):
    """Build the system instruction (matches prompt_eval.py)."""
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
    """Build the user message (matches prompt_eval.py)."""
    parts = []
    if parent_body:
        parts.append(f"Context (preceding comment in thread):\n{parent_body}")
    parts.append(f"Comment:\n{comment}")
    return "\n\n".join(parts)


def truncate_text(text, max_chars=2048):
    """Truncate text by character count."""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def compute_metrics(predictions, labels):
    """Compute accuracy, F1, precision, recall for binary classification."""
    tp = fp = tn = fn = 0
    unparseable = 0

    for pred, label in zip(predictions, labels):
        if pred == "unparseable":
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


def parse_response(text):
    """Parse model response to True/False prediction."""
    text = text.strip()
    if text == "True":
        return "True"
    if text == "False":
        return "False"
    if text.startswith("True"):
        return "True"
    if text.startswith("False"):
        return "False"
    lower = text.lower()
    if lower.startswith("true"):
        return "True"
    if lower.startswith("false"):
        return "False"
    return "unparseable"


def extract_confidence(logprobs_result):
    """Extract P(removed) confidence from logprobs.

    Returns the probability of "True" (removed) from the first token's
    top candidates. Uses softmax over True/False logprobs.
    """
    if not logprobs_result or not logprobs_result.top_candidates:
        return None

    candidates = logprobs_result.top_candidates[0].candidates
    true_logprob = None
    false_logprob = None

    for c in candidates:
        if c.token == "True":
            true_logprob = c.log_probability
        elif c.token == "False":
            false_logprob = c.log_probability

    if true_logprob is not None and false_logprob is not None:
        # Softmax over True/False only
        max_lp = max(true_logprob, false_logprob)
        true_exp = math.exp(true_logprob - max_lp)
        false_exp = math.exp(false_logprob - max_lp)
        return round(true_exp / (true_exp + false_exp), 6)

    if true_logprob is not None:
        return round(math.exp(true_logprob), 6)
    if false_logprob is not None:
        # Only False in top candidates -- P(removed) = 1 - P(False)
        return round(1.0 - math.exp(false_logprob), 6)

    return None


def call_gemini(client, model_id, user_message, config, max_retries=5):
    """Call Gemini API with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=user_message,
                config=config,
            )
            text = ""
            logprobs_result = None

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text.strip()
                logprobs_result = getattr(candidate, "logprobs_result", None)

                # Check for safety blocks
                finish = getattr(candidate, "finish_reason", None)
                if finish and str(finish) not in ("STOP", "MAX_TOKENS", "FinishReason.STOP", "FinishReason.MAX_TOKENS"):
                    return f"BLOCKED:{finish}", None

            usage = getattr(response, "usage_metadata", None)
            thinking_tokens = getattr(usage, "thoughts_token_count", 0) if usage else 0

            return text, logprobs_result

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Resource" in err_str or "quota" in err_str.lower():
                wait = min(2 ** attempt * 5, 60)
                print(f"    Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            elif "500" in err_str or "503" in err_str:
                wait = min(2 ** attempt * 2, 30)
                print(f"    Server error (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error: {e}")
                return f"ERROR:{e}", None

    return "ERROR:max_retries_exceeded", None


def run_evaluation(client, model_id, model_short, subreddit, dataset_dir,
                   output_dir, condition, rules_text, config, max_test_samples,
                   max_comment_chars, rpm):
    """Run evaluation for a single model/subreddit/condition combination."""
    results_file = output_dir / f"{model_short}_{subreddit}_{condition}.json"
    if results_file.exists():
        print(f"  Skipping: {results_file.name} already exists")
        return True

    # Load test data
    test_data = load_jsonl(dataset_dir / "test.jsonl")
    if len(test_data) > max_test_samples:
        test_data = test_data[:max_test_samples]

    system_prompt = build_system_prompt(subreddit, rules_text)

    # Build config with system instruction
    eval_config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        response_logprobs=config.response_logprobs,
        logprobs=config.logprobs,
        thinking_config=config.thinking_config,
    )

    min_interval = 60.0 / rpm

    print(f"\n  === {model_short} / r/{subreddit} / {condition} ===")
    print(f"  Samples: {len(test_data)}, RPM: {rpm}")

    predictions = []
    labels = []
    raw_outputs = []
    errors = 0

    start_time = time.time()
    last_request_time = 0

    for i, item in enumerate(test_data):
        now = time.time()
        elapsed_since_last = now - last_request_time
        if elapsed_since_last < min_interval:
            time.sleep(min_interval - elapsed_since_last)

        comment = truncate_text(item["body"], max_comment_chars)
        parent_body = item.get("parent_body", "")
        if parent_body:
            parent_body = truncate_text(parent_body, max_comment_chars)

        user_msg = build_user_message(comment, parent_body)
        label = item["label"]

        last_request_time = time.time()
        raw_text, logprobs_result = call_gemini(client, model_id, user_msg, eval_config)
        pred = parse_response(raw_text)

        if isinstance(raw_text, str) and (raw_text.startswith("ERROR:") or raw_text.startswith("BLOCKED:")):
            pred = "unparseable"
            errors += 1

        confidence = extract_confidence(logprobs_result)

        predictions.append(pred)
        labels.append(label)
        raw_outputs.append({
            "id": item.get("id"),
            "label": label,
            "prediction": pred,
            "raw": raw_text,
            "confidence_removed": confidence,
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(test_data):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1}/{len(test_data)}] {rate:.1f} samples/sec, "
                  f"ETA {eta:.0f}s, errors={errors}")

    elapsed = time.time() - start_time

    metrics = compute_metrics(predictions, labels)
    metrics["model"] = model_id
    metrics["subreddit"] = subreddit
    metrics["condition"] = condition
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["samples_per_second"] = round(len(test_data) / elapsed, 2) if elapsed > 0 else 0
    metrics["engine"] = "vertex_ai"
    metrics["constrained_decoding"] = True
    metrics["constrained_method"] = "text/x.enum"
    tc = config.thinking_config
    if isinstance(tc, dict):
        metrics["thinking_budget"] = tc.get("thinkingBudget", 0)
    elif tc is not None:
        metrics["thinking_budget"] = getattr(tc, "thinking_budget", 0)
    else:
        metrics["thinking_budget"] = 0
    metrics["labels"] = "True/False"
    metrics["api_errors"] = errors

    # Count confidence scores
    conf_count = sum(1 for r in raw_outputs if r["confidence_removed"] is not None)
    conf_values = [r["confidence_removed"] for r in raw_outputs if r["confidence_removed"] is not None]
    if conf_values:
        metrics["confidence_mean"] = round(sum(conf_values) / len(conf_values), 4)
        metrics["confidence_min"] = round(min(conf_values), 4)
        metrics["confidence_max"] = round(max(conf_values), 4)
        metrics["confidence_count"] = conf_count

    print(f"  Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}  "
          f"Prec: {metrics['precision']:.4f}  Rec: {metrics['recall']:.4f}  "
          f"Unparse: {metrics['unparseable']}  Errors: {errors}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_file = output_dir / f"{model_short}_{subreddit}_{condition}_predictions.jsonl"
    with open(predictions_file, "w") as f:
        for item in raw_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  Results: {results_file.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Gemini Vertex AI evaluation for content moderation")
    parser.add_argument("--model", required=True, choices=list(AVAILABLE_MODELS.keys()),
                        help="Gemini model to use")
    parser.add_argument("--subreddit", type=str, default=None, help="Single subreddit name")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory with test.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for results")
    parser.add_argument("--with-rules", action="store_true", help="Include subreddit rules")
    parser.add_argument("--rules-file", type=str, default=None, help="Path to rules text file")
    parser.add_argument("--max-test-samples", type=int, default=1000,
                        help="Max test samples (default: 1000)")
    parser.add_argument("--max-comment-chars", type=int, default=2048,
                        help="Max chars per comment (default: 2048)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Requests per minute limit (auto-detected per model)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking (Flash only, for ablation)")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Override thinking token budget (default: 128 for Flash/Pro)")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all subreddits (requires --data-root)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root data directory with dataset/ and rules/ subdirs")
    parser.add_argument("--subs-config", type=str, default=None,
                        help="Path to subreddits.json config file")
    args = parser.parse_args()

    model_id = AVAILABLE_MODELS[args.model]
    model_short = args.model
    rpm = args.rpm or RPM_LIMITS.get(args.model, 60)
    output_dir = Path(args.output_dir)

    # Build generation config
    # Thinking budget: 128 is the empirical sweet spot (tested on r/PublicFreakout).
    # Higher budgets show diminishing returns. Flash can disable; Pro cannot.
    default_budget = DEFAULT_THINKING_BUDGET.get(args.model)
    thinking_config = None
    thinking_enabled = False

    if args.model == "gemini-2.5-flash-lite":
        # Flash-Lite does not support thinking
        thinking_config = None
        thinking_enabled = False
    elif args.model == "gemini-2.5-flash":
        if args.no_thinking:
            thinking_config = {"thinkingBudget": 0}
        else:
            budget = args.thinking_budget if args.thinking_budget is not None else default_budget
            thinking_config = {"thinkingBudget": budget}
            thinking_enabled = True
    elif args.model == "gemini-2.5-pro":
        budget = args.thinking_budget if args.thinking_budget is not None else default_budget
        thinking_config = {"thinkingBudget": max(budget, 128)}
        thinking_enabled = True

    # max_output_tokens includes thinking tokens, so thinking models need more
    max_tokens = max(200, (args.thinking_budget or default_budget or 0) + 50) if thinking_enabled else 10

    # Constrained decoding via text/x.enum (equivalent to vLLM StructuredOutputsParams
    # for open-source models). Experiments showed no performance difference vs free text
    # (0 unparseable either way), but used for methodological consistency.
    config = GenerateContentConfig(
        temperature=0,
        max_output_tokens=max_tokens,
        response_logprobs=True,
        logprobs=3,
        thinking_config=thinking_config,
        safety_settings=SAFETY_OFF,
        response_mime_type="text/x.enum",
        response_schema={"type": "STRING", "enum": ["True", "False"]},
    )

    # Initialize client
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    thinking_str = "disabled"
    if thinking_enabled:
        budget = thinking_config.get("thinkingBudget", 0) if thinking_config else 0
        thinking_str = f"enabled (budget={budget})"
    elif args.model == "gemini-2.5-flash-lite":
        thinking_str = "not supported"

    print(f"=== Gemini Vertex AI Evaluation ===")
    print(f"Model:      {model_id}")
    print(f"Thinking:   {thinking_str}")
    print(f"Constrained: text/x.enum [True, False]")
    print(f"Safety:     OFF (all categories)")
    print(f"RPM limit:  {rpm}")
    print(f"Project:    {os.environ.get('GOOGLE_CLOUD_PROJECT', 'not set')}")
    print(f"Location:   {os.environ.get('GOOGLE_CLOUD_LOCATION', 'not set')}")

    if args.run_all:
        if not args.data_root:
            print("ERROR: --run-all requires --data-root")
            sys.exit(1)

        data_root = Path(args.data_root)
        dataset_root = data_root / "dataset"
        rules_root = data_root / "rules"

        # Load subreddit list
        if args.subs_config:
            with open(args.subs_config) as f:
                subs_config = json.load(f)
            all_subs = subs_config.get("subreddits") or (subs_config.get("seen", []) + subs_config.get("unseen", []))
        else:
            all_subs = sorted([
                d.name for d in dataset_root.iterdir()
                if d.is_dir() and (d / "random_split" / "test.jsonl").exists()
            ])

        print(f"Subreddits: {len(all_subs)}")
        print(f"Conditions: without_rules + with_rules")
        print(f"Total jobs: {len(all_subs) * 2}")

        completed = 0
        skipped = 0
        failed = 0

        for sub in all_subs:
            sub_dataset_dir = dataset_root / sub / "random_split"
            if not (sub_dataset_dir / "test.jsonl").exists():
                print(f"  SKIP {sub}: no test.jsonl")
                skipped += 1
                continue

            for condition in ["without_rules", "with_rules"]:
                rules_text = None
                if condition == "with_rules":
                    rules_file = rules_root / sub / "rules.txt"
                    if rules_file.exists():
                        rules_text = rules_file.read_text().strip()
                    else:
                        rules_text = None

                try:
                    ok = run_evaluation(
                        client=client,
                        model_id=model_id,
                        model_short=model_short,
                        subreddit=sub,
                        dataset_dir=sub_dataset_dir,
                        output_dir=output_dir,
                        condition=condition,
                        rules_text=rules_text if condition == "with_rules" else None,
                        config=config,
                        max_test_samples=args.max_test_samples,
                        max_comment_chars=args.max_comment_chars,
                        rpm=rpm,
                    )
                    if ok:
                        completed += 1
                except Exception as e:
                    print(f"  FAILED {sub}/{condition}: {e}")
                    failed += 1

        print(f"\n=== Summary ===")
        print(f"Completed: {completed}")
        print(f"Skipped:   {skipped}")
        print(f"Failed:    {failed}")

    else:
        if not args.subreddit or not args.dataset_dir:
            print("ERROR: --subreddit and --dataset-dir required (or use --run-all)")
            sys.exit(1)

        condition = "with_rules" if args.with_rules else "without_rules"
        rules_text = None
        if args.with_rules and args.rules_file:
            with open(args.rules_file) as f:
                rules_text = f.read().strip()
            print(f"Rules:      {len(rules_text)} chars from {args.rules_file}")

        run_evaluation(
            client=client,
            model_id=model_id,
            model_short=model_short,
            subreddit=args.subreddit,
            dataset_dir=Path(args.dataset_dir),
            output_dir=output_dir,
            condition=condition,
            rules_text=rules_text,
            config=config,
            max_test_samples=args.max_test_samples,
            max_comment_chars=args.max_comment_chars,
            rpm=rpm,
        )


if __name__ == "__main__":
    main()
