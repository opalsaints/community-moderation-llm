#!/usr/bin/env python3
"""
v3 fine-tuning for Reddit content moderation with chat templates.

Key changes from v1/v2:
  - Chat templates (proper instruct model fine-tuning)
  - SLM-Mod-style prompt structure (rules + context + comment)
  - All 7 LoRA target modules (not just attention)
  - Standard LoRA (not DoRA), dropout=0
  - vLLM evaluation with logprobs + Cohen's kappa
  - Three template modes: slm-mod, enriched, no-chat
  - Qwen 3 thinking mode disabled

Two-phase workflow (GPU memory conflict between BnB and vLLM):
  Phase 1: Train and save adapter (--skip-train omitted)
  Phase 2: Load adapter via vLLM and evaluate (--skip-train)

Usage:
    # Training:
    python finetune_v3.py \
        --model Qwen/Qwen3-8B \
        --dataset-dir ~/data/dataset/changemyview/random_split \
        --output-dir ~/data/results/pilot/ \
        --subreddit changemyview \
        --rules-file ~/data/rules/changemyview/rules.txt \
        --template slm-mod --run-tag M2

    # Evaluation (separate SLURM job):
    python finetune_v3.py \
        --model Qwen/Qwen3-8B \
        --dataset-dir ~/data/dataset/changemyview/random_split \
        --output-dir ~/data/results/pilot/ \
        --subreddit changemyview \
        --rules-file ~/data/rules/changemyview/rules.txt \
        --template slm-mod --run-tag M2 --skip-train
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ============================================================
# Constants
# ============================================================

HF_CACHE_DIRS = [
    # Writable scratch first: required for vLLM which creates lock files in the
    # cache dir at load time. The shared SURF cache is read-only for us, and
    # vLLM crashes with PermissionError if we point it at a read-only dir (hit
    # this with Qwen 3 14B during the pilot).
    os.environ.get("HF_CACHE_DIR", "./hf_cache"),
    "/projects/2/managed_datasets/hf_cache_dir",  # shared SURF cache, read-only fallback
]

LORA_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_ALL_MODULES = LORA_ATTN_MODULES + ["gate_proj", "up_proj", "down_proj"]


# ============================================================
# Utility functions (reused from v1/v2)
# ============================================================

def find_cache_dir(model_id, cache_dirs=HF_CACHE_DIRS):
    """Find which cache directory contains the model."""
    cache_name = f"models--{model_id.replace('/', '--')}"
    for d in cache_dirs:
        model_path = os.path.join(d, cache_name, "snapshots")
        if os.path.exists(model_path) and os.listdir(model_path):
            return d
    return cache_dirs[0]


def load_tokenizer(model_id, cache_dir):
    """Load a tokenizer with model-specific fixes applied.

    Mistral NeMo has a known regex bug in its fast tokenizer that produces
    incorrect tokenization and causes training loss to stay flat around 2.5.
    Newer transformers versions accept ``fix_mistral_regex=True``; if that
    flag isn't supported, fall back to the slow tokenizer which doesn't have
    the bug.
    """
    kwargs = dict(cache_dir=cache_dir, local_files_only=True)
    if "mistral" in model_id.lower():
        try:
            return AutoTokenizer.from_pretrained(
                model_id, fix_mistral_regex=True, **kwargs
            )
        except TypeError:
            # Older transformers: fall back to slow tokenizer (no regex bug)
            return AutoTokenizer.from_pretrained(
                model_id, use_fast=False, **kwargs
            )
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_jsonl(filepath):
    with open(filepath) as f:
        return [json.loads(line) for line in f]


def truncate_text(text, tokenizer, max_tokens):
    """Truncate text to max_tokens using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens])
    return text


def parse_response(text):
    """Extract True or False from model output."""
    text = text.strip()
    first_word = text.split()[0] if text.split() else ""
    first_word = first_word.strip(".,!;:'\"")
    if first_word == "True":
        return "True"
    if first_word == "False":
        return "False"
    if "True" in text[:20]:
        return "True"
    if "False" in text[:20]:
        return "False"
    return "unparseable"


def compute_metrics(predictions, labels):
    """Compute accuracy, F1, precision, recall, Cohen's kappa."""
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
    kappa_pe = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (total ** 2) if total else 0
    kappa = (accuracy - kappa_pe) / (1 - kappa_pe) if (1 - kappa_pe) else 0
    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "cohens_kappa": round(kappa, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "unparseable": unparseable,
        "total_evaluated": total,
        "total_attempted": len(predictions),
    }


def compute_auroc(confidences, labels):
    """AUROC from P(removed) scores and string labels. Mann-Whitney U with
    tie-averaged ranks. Returns None when AUROC is undefined (no valid
    scores, or only one class present)."""
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
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs_sorted) if y == 1)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return round(u_stat / (n_pos * n_neg), 4)


# ============================================================
# Prompt construction
# ============================================================

def build_system_prompt(subreddit, rules_text):
    """System prompt for slm-mod and enriched templates."""
    return (
        f"You are a content moderator for the Reddit community r/{subreddit}.\n\n"
        f"The community has the following rules:\n{rules_text}\n\n"
        f"Determine whether the following comment violates the community's rules "
        f"or norms. Answer with 'True' if the comment should be removed, or "
        f"'False' if it should stay."
    )


def build_user_message(item, tokenizer, template, max_comment_tokens):
    """Build user message content. Template controls which features are included."""
    comment = truncate_text(item["body"], tokenizer, max_comment_tokens)
    parent_body = item.get("parent_body", "")
    parts = []

    if template == "enriched":
        post_title = item.get("post_title", "") or "unknown"
        is_top_level = item.get("is_top_level", False)
        author_is_first = item.get("author_is_first", False)

        account_age_days = item.get("account_age_days", -1)
        if account_age_days >= 365:
            account_age = f"{account_age_days / 365:.1f} years"
        elif account_age_days >= 30:
            account_age = f"{account_age_days / 30:.0f} months"
        elif account_age_days > 0:
            account_age = f"{account_age_days:.0f} days"
        else:
            account_age = "unknown"

        meta = "\n".join([
            f"Post title: {post_title}",
            f"Comment type: {'top-level comment' if is_top_level else 'reply'}",
            f"Author account age: {account_age}",
            f"First-time poster in community: {'yes' if author_is_first else 'no'}",
        ])
        parts.append(meta)

    if parent_body:
        parent_body = truncate_text(parent_body, tokenizer, max_comment_tokens // 2)
        parts.append(f"Context (preceding comment):\n{parent_body}")

    parts.append(f"Comment:\n{comment}")
    return "\n\n".join(parts)


def build_chat_messages(system_prompt, user_message, model_id):
    """Build chat message list. Gemma has no system role -- embed in user msg."""
    if "gemma" in model_id.lower():
        return [{"role": "user", "content": f"{system_prompt}\n\n{user_message}"}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def apply_chat_template(tokenizer, messages, model_id, add_generation_prompt=False):
    """Apply chat template with model-specific kwargs (Qwen 3 thinking disabled)."""
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


# Plain-text templates for the no-chat ablation (same content, no chat tokens)

NO_CHAT_TRAIN = """You are a content moderator for the Reddit community r/{subreddit}.

The community has the following rules:
{rules_text}

Determine whether the following comment violates the community's rules or norms. Answer with 'True' if the comment should be removed, or 'False' if it should stay.
{context_block}
Comment: {comment}

Classification: {label}"""

NO_CHAT_INFERENCE = """You are a content moderator for the Reddit community r/{subreddit}.

The community has the following rules:
{rules_text}

Determine whether the following comment violates the community's rules or norms. Answer with 'True' if the comment should be removed, or 'False' if it should stay.
{context_block}
Comment: {comment}

Classification:"""


def _no_chat_context_block(parent_body, tokenizer, max_tokens):
    if not parent_body:
        return ""
    parent_body = truncate_text(parent_body, tokenizer, max_tokens)
    return f"\nContext (preceding comment): {parent_body}"


def _resolve_pooled_context(item, fallback_subreddit, fallback_rules):
    """For pooled training, pull per-example subreddit + rules from the item.

    Pooled examples mix many subreddits, so each example carries its own rules
    in the ``rules_text`` field and its own subreddit in ``subreddit``.
    """
    sub = item.get("subreddit") or fallback_subreddit
    rules = item.get("rules_text") or fallback_rules
    return sub, rules


def format_training_example(item, subreddit, rules_text, tokenizer, model_id,
                            template, max_comment_tokens, include_sub_prefix=True):
    """Format one training example as {"text": formatted_string}."""
    label = "True" if item["label"] == "removed" else "False"

    if template == "no-chat":
        comment = truncate_text(item["body"], tokenizer, max_comment_tokens)
        ctx = _no_chat_context_block(item.get("parent_body", ""), tokenizer,
                                     max_comment_tokens // 2)
        text = NO_CHAT_TRAIN.format(
            subreddit=subreddit, rules_text=rules_text,
            context_block=ctx, comment=comment, label=label,
        )
    elif template == "pooled":
        # Per-example sub + rules; enriched features + subreddit identifier prefix
        sub, rules = _resolve_pooled_context(item, subreddit, rules_text)
        system_prompt = build_system_prompt(sub, rules)
        user_msg = build_user_message(item, tokenizer, "enriched", max_comment_tokens)
        if include_sub_prefix:
            user_msg = f"Subreddit: r/{sub}\n\n{user_msg}"
        messages = build_chat_messages(system_prompt, user_msg, model_id)
        messages.append({"role": "assistant", "content": label})
        text = apply_chat_template(tokenizer, messages, model_id)
    else:
        system_prompt = build_system_prompt(subreddit, rules_text)
        user_msg = build_user_message(item, tokenizer, template, max_comment_tokens)
        messages = build_chat_messages(system_prompt, user_msg, model_id)
        messages.append({"role": "assistant", "content": label})
        text = apply_chat_template(tokenizer, messages, model_id)

    return {"text": text}


def format_training_example_split(item, subreddit, rules_text, tokenizer, model_id,
                                  template, max_comment_tokens, include_sub_prefix=True):
    """Like format_training_example but returns {"prompt": ..., "completion": ...}.

    Used with SFTConfig(completion_only_loss=True) to mask the prompt portion
    from the training loss. The prompt+completion concatenation produces the
    same token sequence as format_training_example's "text" output, so the only
    training-time difference is loss masking.
    """
    label = "True" if item["label"] == "removed" else "False"
    eos = tokenizer.eos_token or ""

    if template == "no-chat":
        comment = truncate_text(item["body"], tokenizer, max_comment_tokens)
        ctx = _no_chat_context_block(item.get("parent_body", ""), tokenizer,
                                     max_comment_tokens // 2)
        prompt = NO_CHAT_INFERENCE.format(
            subreddit=subreddit, rules_text=rules_text,
            context_block=ctx, comment=comment,
        )
        completion = f" {label}"
        return {"prompt": prompt, "completion": completion}

    if template == "pooled":
        sub, rules = _resolve_pooled_context(item, subreddit, rules_text)
        system_prompt = build_system_prompt(sub, rules)
        user_msg = build_user_message(item, tokenizer, "enriched", max_comment_tokens)
        if include_sub_prefix:
            user_msg = f"Subreddit: r/{sub}\n\n{user_msg}"
    else:
        system_prompt = build_system_prompt(subreddit, rules_text)
        user_msg = build_user_message(item, tokenizer, template, max_comment_tokens)

    messages = build_chat_messages(system_prompt, user_msg, model_id)
    prompt = apply_chat_template(tokenizer, messages, model_id,
                                 add_generation_prompt=True)
    completion = label + eos
    return {"prompt": prompt, "completion": completion}


def format_inference_prompt(item, subreddit, rules_text, tokenizer, model_id,
                            template, max_comment_tokens, include_sub_prefix=True):
    """Format one inference prompt (no label, with generation prompt)."""
    if template == "no-chat":
        comment = truncate_text(item["body"], tokenizer, max_comment_tokens)
        ctx = _no_chat_context_block(item.get("parent_body", ""), tokenizer,
                                     max_comment_tokens // 2)
        return NO_CHAT_INFERENCE.format(
            subreddit=subreddit, rules_text=rules_text,
            context_block=ctx, comment=comment,
        )
    elif template == "pooled":
        sub, rules = _resolve_pooled_context(item, subreddit, rules_text)
        system_prompt = build_system_prompt(sub, rules)
        user_msg = build_user_message(item, tokenizer, "enriched", max_comment_tokens)
        if include_sub_prefix:
            user_msg = f"Subreddit: r/{sub}\n\n{user_msg}"
        messages = build_chat_messages(system_prompt, user_msg, model_id)
        return apply_chat_template(tokenizer, messages, model_id,
                                   add_generation_prompt=True)
    else:
        system_prompt = build_system_prompt(subreddit, rules_text)
        user_msg = build_user_message(item, tokenizer, template, max_comment_tokens)
        messages = build_chat_messages(system_prompt, user_msg, model_id)
        return apply_chat_template(tokenizer, messages, model_id,
                                   add_generation_prompt=True)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="v3 fine-tuning for content moderation")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory with train.jsonl and test.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--subreddit", required=True, help="Subreddit name")
    parser.add_argument("--rules-file", required=True, help="Path to rules.txt")
    parser.add_argument("--template", default="slm-mod",
                        choices=["slm-mod", "enriched", "no-chat", "pooled"],
                        help="Prompt template mode (default: slm-mod). "
                             "'pooled' reads subreddit and rules_text from each example.")
    parser.add_argument("--run-tag", default="run",
                        help="Tag for output filenames (e.g. M1, A1)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all",
                        choices=["all", "attention-only"],
                        help="LoRA target modules: 'all' (7 linear layers) or "
                             "'attention-only' (4 q/k/v/o layers)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr-scheduler", default="cosine",
                        choices=["cosine", "linear", "constant"],
                        help="Learning rate scheduler (default: cosine)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor (default: 0.1, 0 to disable)")
    parser.add_argument("--completion-only-loss", action="store_true",
                        help="Mask prompt tokens from the training loss, computing "
                             "gradient only on the assistant's label token(s). "
                             "Addresses loss dilution when ~99%% of tokens are prompt.")
    parser.add_argument("--eval-temperature", type=float, default=0.1,
                        help="Sampling temperature for evaluation (default: 0.1; "
                             "Qwen 3 warns against greedy decoding at temperature=0)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (default: 2048). "
                             "AskHistorians has 2.6%% truncation at 2048; "
                             "override to 2560 for that sub.")
    parser.add_argument("--max-comment-tokens", type=int, default=512)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Limit training samples (for debugging)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, evaluate only (adapter must exist)")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-sub-prefix", dest="include_sub_prefix",
                        action="store_false", default=True,
                        help="For pooled template: skip the 'Subreddit: r/{sub}' "
                             "prefix line. Used for the pooled with/without "
                             "sub-ID ablation. No effect on other templates.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Expand ~ in all path args (defends against shells that don't expand tildes
    # after variable expansion, e.g. ssh with COMMON="... ~/data/...").
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.rules_file = os.path.expanduser(args.rules_file)
    if args.cache_dir:
        args.cache_dir = os.path.expanduser(args.cache_dir)

    if args.cache_dir is None:
        args.cache_dir = find_cache_dir(args.model)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / f"{args.run_tag}_adapter"

    with open(args.rules_file) as f:
        rules_text = f.read().strip()

    # =========================================================
    # PHASE 1: Training (QLoRA + SFTTrainer)
    # =========================================================
    if not args.skip_train:
        adapter_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== v3 Fine-Tuning: {args.run_tag} ===")
        print(f"Model:      {args.model}")
        print(f"Cache dir:  {args.cache_dir}")
        print(f"Subreddit:  r/{args.subreddit}")
        print(f"Template:   {args.template}")
        print(f"LoRA:       r={args.lora_rank} alpha={args.lora_alpha} "
              f"dropout={args.lora_dropout}")
        print(f"Targets:    {LORA_ALL_MODULES if args.target_modules == 'all' else LORA_ATTN_MODULES}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = load_tokenizer(args.model, args.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            cache_dir=args.cache_dir, local_files_only=True,
            quantization_config=bnb_config, device_map="auto",
        )

        # Qwen 3: separate pad from eos to avoid masking the turn-end token
        # during training. <|im_end|> is the chat turn delimiter; <|endoftext|>
        # is safe as pad since it never appears in well-formed chat sequences.
        if "qwen3" in args.model.lower():
            if tokenizer.eos_token != "<|im_end|>":
                tokenizer.eos_token = "<|im_end|>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        elif tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

        model = prepare_model_for_kbit_training(model)
        target_modules = (LORA_ATTN_MODULES if args.target_modules == "attention-only"
                          else LORA_ALL_MODULES)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare training data
        train_data = load_jsonl(dataset_dir / "train.jsonl")
        if args.max_train_samples and len(train_data) > args.max_train_samples:
            train_data = train_data[:args.max_train_samples]

        if args.completion_only_loss:
            train_records = [
                format_training_example_split(
                    item, args.subreddit, rules_text, tokenizer,
                    args.model, args.template, args.max_comment_tokens,
                    include_sub_prefix=args.include_sub_prefix,
                )
                for item in train_data
            ]
            train_dataset = Dataset.from_list(train_records)
            print(f"Training examples: {len(train_data)} (prompt+completion, completion-only loss)")
            print(f"Example prompt (first 500 chars):\n{train_records[0]['prompt'][:500]}...")
            print(f"Example completion: {train_records[0]['completion']!r}")
        else:
            train_texts = [
                format_training_example(
                    item, args.subreddit, rules_text, tokenizer,
                    args.model, args.template, args.max_comment_tokens,
                    include_sub_prefix=args.include_sub_prefix,
                )["text"]
                for item in train_data
            ]
            train_dataset = Dataset.from_dict({"text": train_texts})

            print(f"Training examples: {len(train_data)}")
            print(f"Example (first 500 chars):\n{train_texts[0][:500]}...")

        training_args = SFTConfig(
            output_dir=str(adapter_dir / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type=args.lr_scheduler,
            label_smoothing_factor=args.label_smoothing,
            completion_only_loss=True if args.completion_only_loss else None,
            logging_steps=50,
            save_strategy="epoch",
            bf16=True,
            max_length=args.max_seq_length,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print(f"\nStarting training...")
        start_time = time.time()
        trainer.train()
        train_elapsed = time.time() - start_time
        print(f"Training completed in {train_elapsed:.1f}s")

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"Adapter saved to {adapter_dir}")

        # Save training metadata
        train_meta = {
            "run_tag": args.run_tag,
            "model": args.model,
            "subreddit": args.subreddit,
            "template": args.template,
            "include_sub_prefix": args.include_sub_prefix,
            "seed": args.seed,
            "train_samples": len(train_data),
            "train_seconds": round(train_elapsed, 1),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_modules,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler": args.lr_scheduler,
            "label_smoothing": args.label_smoothing,
            "eval_temperature": args.eval_temperature,
            "max_seq_length": args.max_seq_length,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_file = output_dir / f"{args.run_tag}_train_meta.json"
        with open(meta_file, "w") as f:
            json.dump(train_meta, f, indent=2)
        print(f"Training metadata: {meta_file}")

        print("\nTraining complete. Run eval in a separate job with --skip-train.")
        return

    # =========================================================
    # PHASE 2: Evaluation (vLLM + LoRA adapter)
    # =========================================================
    print(f"\n=== v3 Evaluation: {args.run_tag} ===")

    tokenizer = load_tokenizer(args.model, args.cache_dir)

    test_data = load_jsonl(dataset_dir / "test.jsonl")
    if len(test_data) > args.max_test_samples:
        test_data = test_data[:args.max_test_samples]
    print(f"Test samples: {len(test_data)}")

    prompts = [
        format_inference_prompt(
            item, args.subreddit, rules_text, tokenizer,
            args.model, args.template, args.max_comment_tokens,
            include_sub_prefix=args.include_sub_prefix,
        )
        for item in test_data
    ]
    print(f"Prompts built. Sample length: {len(prompts[0])} chars")
    print(f"First prompt (500 chars):\n{prompts[0][:500]}...")

    print(f"Loading vLLM with LoRA adapter from {adapter_dir}...")
    llm = LLM(
        model=args.model,
        download_dir=args.cache_dir,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
        config_format="hf",
        enable_lora=True,
        max_lora_rank=args.lora_rank,
    )

    lora_request = LoRARequest(
        lora_name=args.run_tag,
        lora_int_id=1,
        lora_path=str(adapter_dir),
    )

    sampling_params = SamplingParams(
        temperature=args.eval_temperature,
        max_tokens=5,
        logprobs=5,
    )

    print(f"Running inference on {len(prompts)} samples...")
    eval_start = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    eval_elapsed = time.time() - eval_start

    # Process results
    predictions = []
    labels = []
    raw_outputs = []

    for output, item in zip(outputs, test_data):
        generated = output.outputs[0]
        text = generated.text.strip()
        label = item["label"]
        pred = parse_response(text)

        # Extract logprob confidence: P(removed) = P(True) / (P(True) + P(False))
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
        raw_outputs.append({
            "id": item.get("id"),
            "label": label,
            "prediction": pred,
            "raw": text,
            "confidence_removed": confidence,
        })

    # Compute and display metrics
    metrics = compute_metrics(predictions, labels)
    metrics["auroc"] = compute_auroc(
        [r["confidence_removed"] for r in raw_outputs], labels
    )
    metrics.update({
        "run_tag": args.run_tag,
        "model": args.model,
        "subreddit": args.subreddit,
        "template": args.template,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "eval_seconds": round(eval_elapsed, 1),
        "engine": "vllm",
    })

    print(f"\n{'='*60}")
    print(f"Run:        {args.run_tag}")
    print(f"Model:      {args.model}")
    print(f"Template:   {args.template}")
    print(f"Eval time:  {eval_elapsed:.1f}s ({len(test_data)} samples)")
    print(f"\nAccuracy:   {metrics['accuracy']:.4f}")
    print(f"F1:         {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Kappa:      {metrics['cohens_kappa']:.4f}")
    auroc_val = metrics.get("auroc")
    print(f"AUROC:      {auroc_val:.4f}" if auroc_val is not None else "AUROC:      n/a")
    print(f"Unparseable:{metrics['unparseable']}")
    print(f"\nTP={metrics['tp']}  FP={metrics['fp']}")
    print(f"FN={metrics['fn']}  TN={metrics['tn']}")

    confs = [r["confidence_removed"] for r in raw_outputs
             if r["confidence_removed"] is not None]
    if confs:
        print(f"\nConfidence (P(removed)): mean={sum(confs)/len(confs):.3f}, "
              f"min={min(confs):.3f}, max={max(confs):.3f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / f"{args.run_tag}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    predictions_file = output_dir / f"{args.run_tag}_predictions.jsonl"
    with open(predictions_file, "w") as f:
        for item in raw_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nResults:     {metrics_file}")
    print(f"Predictions: {predictions_file}")


if __name__ == "__main__":
    main()
