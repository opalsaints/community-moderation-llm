#!/usr/bin/env python3
"""
Claude Sonnet 4.6 evaluation on Reddit moderation, via `claude -p` subprocess (OAuth/Max).

HARD CONSTRAINTS:
- Final inference model is locked to claude-sonnet-4-6. Auto-mode classifier may invoke
  Haiku for routing, but the moderation prediction (the True/False) comes from Sonnet only.
  We verify this per-call by inspecting modelUsage and counting Sonnet's output_tokens.
- No fallback. Errors abort the cell after 5 hard errors.
- Idempotent: refuses to overwrite an existing output file.

Cells are (sub, template, thinking). Same cell -> same system prompt -> cache reuse.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SYSTEM_PROMPT_VARIANTS = {
    "default": (
        "You are a content moderator for the Reddit community r/{sub}.\n\n"
        "{rules_block}"
        "Determine whether a comment violates the community's rules or norms.\n\n"
        "Your ENTIRE response MUST be exactly one of these two words: True or False. "
        "True means remove the comment. False means keep the comment.\n\n"
        "Do NOT include any explanation, reasoning, formatting, markdown, punctuation, "
        "or any other text. Output only the single word."
    ),
    "base_rate": (
        "You are a content moderator for the Reddit community r/{sub}.\n\n"
        "{rules_block}"
        "Determine whether a comment violates the community's rules or norms.\n\n"
        "IMPORTANT: In this evaluation set, roughly half of all comments end up being "
        "removed by human moderators. Calibrate your decisions to that base rate. If you "
        "find yourself almost never marking comments for removal, your threshold is too "
        "strict. Use a balanced ~50% threshold: if you think there is a reasonable chance "
        "(>50%) that this comment violates the rules or norms, mark it for removal.\n\n"
        "Your ENTIRE response MUST be exactly one of these two words: True or False. "
        "True means remove the comment. False means keep the comment.\n\n"
        "Do NOT include any explanation, reasoning, formatting, markdown, punctuation, "
        "or any other text. Output only the single word."
    ),
    "aggressive": (
        "You are a content moderator for the Reddit community r/{sub}.\n\n"
        "{rules_block}"
        "Determine whether a comment violates the community's rules or norms.\n\n"
        "IMPORTANT: Err on the side of removal. If a comment plausibly violates any rule "
        "or community norm, mark it for removal. Under-removal lets harmful or "
        "rule-violating content stay up, which damages the community more than occasional "
        "over-removal.\n\n"
        "Your ENTIRE response MUST be exactly one of these two words: True or False. "
        "True means remove the comment. False means keep the comment.\n\n"
        "Do NOT include any explanation, reasoning, formatting, markdown, punctuation, "
        "or any other text. Output only the single word."
    ),
    "cot": (
        "You are a content moderator for the Reddit community r/{sub}.\n\n"
        "{rules_block}"
        "Determine whether a comment violates the community's rules or norms.\n\n"
        "First, in 1-3 sentences, reason through which rules might apply and whether the "
        "comment actually violates them. Then on a final line, output your verdict in "
        "exactly this format:\n\n"
        "FINAL: True\n"
        "or\n"
        "FINAL: False\n\n"
        "True means remove the comment. False means keep the comment. The FINAL line must "
        "be the last line of your response and contain only one of those two exact strings."
    ),
}


def build_system_prompt(sub, rules_text, variant="default"):
    rules_block = ""
    if rules_text:
        rules_block = f"The community has the following rules:\n{rules_text}\n\n"
    return SYSTEM_PROMPT_VARIANTS[variant].format(sub=sub, rules_block=rules_block)


def build_user_message(row, template):
    parts = []
    parent = row.get("parent_body") or ""
    if parent:
        parts.append(f"Context (preceding comment in thread):\n{parent}")
    if template == "enriched":
        post_title = row.get("post_title") or ""
        is_top = bool(row.get("is_top_level", False))
        age = row.get("account_age_days")
        first = bool(row.get("author_is_first", False))
        if isinstance(age, (int, float)) and age >= 0:
            age_str = f"{int(age)} days"
        else:
            age_str = "unknown"
        parts.append(
            "\n".join([
                f"Post title: {post_title}",
                f"Comment type: {'top-level comment' if is_top else 'reply'}",
                f"Author account age: {age_str}",
                f"First-time poster in community: {'yes' if first else 'no'}",
            ])
        )
    parts.append(f"Comment:\n{row.get('body', '')}")
    return "\n\n".join(parts)


def parse_answer(text):
    if not text:
        return None
    # CoT path: take whatever follows the last FINAL: marker
    for marker in ("FINAL:", "Final:", "final:"):
        if marker in text:
            tail = text.rsplit(marker, 1)[1].strip().lower()
            for ch in ('*', '`', '"', "'", '.', ',', '!', '?', ':', ';', '\n'):
                tail = tail.replace(ch, ' ')
            words = tail.split()
            if words:
                if words[0] == "true":
                    return True
                if words[0] == "false":
                    return False
            break
    t = text.strip().lower()
    for ch in ('*', '`', '"', "'", '.', ',', '!', '?', ':', ';'):
        t = t.replace(ch, '')
    first = t.split()[0] if t.split() else ""
    if first == "true":
        return True
    if first == "false":
        return False
    if "true" in t and "false" not in t:
        return True
    if "false" in t and "true" not in t:
        return False
    return None


def call_claude(user_msg, system_prompt, effort, model, timeout_s=120):
    cmd = [
        "claude", "-p",
        "--model", model,
        "--effort", effort,
        "--output-format", "json",
        "--system-prompt", system_prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=user_msg,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, "", {}, "timeout"

    if proc.returncode != 0:
        return None, proc.stderr[:500], {}, f"exit {proc.returncode}: {proc.stderr[:200]}"

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, proc.stdout[:500], {}, "non-json stdout"

    if data.get("is_error"):
        return None, "", {}, f"api_error: {data.get('api_error_status')}"

    result_text = data.get("result", "")
    if isinstance(result_text, str) and result_text.startswith(("API Error", "Not logged in", "Error:")):
        return None, result_text, {}, f"claude_error: {result_text[:200]}"

    model_usage = data.get("modelUsage", {})
    sonnet_key = next((k for k in model_usage if k.startswith("claude-sonnet")), None)
    haiku_key = next((k for k in model_usage if k.startswith("claude-haiku")), None)
    sonnet_out = model_usage.get(sonnet_key, {}).get("outputTokens", 0) if sonnet_key else 0
    haiku_out = model_usage.get(haiku_key, {}).get("outputTokens", 0) if haiku_key else 0

    if model.startswith("claude-sonnet"):
        if sonnet_out == 0:
            return None, result_text, model_usage, "sonnet produced 0 output tokens"
    elif model.startswith("claude-haiku"):
        if sonnet_out > 0:
            return None, result_text, model_usage, "sonnet ran when haiku requested"
        if haiku_out == 0:
            return None, result_text, model_usage, "haiku produced 0 output tokens"

    return parse_answer(result_text), result_text, model_usage, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--template", choices=["slm_mod", "enriched"], required=True)
    ap.add_argument("--thinking", choices=["off", "on"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="Resume from existing --out JSON if present")
    ap.add_argument("--data-root", default="data/dataset")
    ap.add_argument("--rules-root", default="data/rules")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument(
        "--prompt-variant",
        choices=list(SYSTEM_PROMPT_VARIANTS.keys()),
        default="default",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    resume_state = None
    if out_path.exists():
        if args.resume:
            existing = json.loads(out_path.read_text())
            old_preds = existing.get("predictions", [])
            kept = [
                pp for pp in old_preds
                if pp.get("pred") is not None and not pp.get("error")
            ]
            start_idx = len(old_preds)
            resume_state = {
                "predictions": kept,
                "start_idx": start_idx,
                "sonnet_in": existing.get("sonnet_tokens_in", 0),
                "sonnet_out": existing.get("sonnet_tokens_out", 0),
                "haiku_in": existing.get("haiku_tokens_in", 0),
                "haiku_out": existing.get("haiku_tokens_out", 0),
                "cache_create": existing.get("cache_creation_tokens", 0),
                "cache_read": existing.get("cache_read_tokens", 0),
            }
            print(
                f"RESUME: kept {len(kept)} successful preds (of {len(old_preds)} attempted), restarting at row {start_idx}",
                flush=True,
            )
        else:
            print(f"SKIP: {out_path} exists")
            return 0

    effort = "low" if args.thinking == "off" else "high"

    rules_path = Path(args.rules_root) / args.sub / "rules.txt"
    rules_text = rules_path.read_text() if rules_path.exists() else None
    system_prompt = build_system_prompt(args.sub, rules_text, args.prompt_variant)

    test_path = Path(args.data_root) / args.sub / "enriched_v2" / "test.jsonl"
    n_dropped = 0
    if resume_state is not None:
        n_dropped = resume_state["start_idx"] - len(resume_state["predictions"])
    target_rows = args.n + n_dropped
    rows = []
    with test_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= target_rows:
                break

    print(
        f"=== {args.sub} | {args.template} | thinking={args.thinking} (effort={effort}) | "
        f"model={args.model} | variant={args.prompt_variant} | n={len(rows)} ===",
        flush=True,
    )

    if resume_state is not None:
        predictions = list(resume_state["predictions"])
        start_idx = resume_state["start_idx"]
        sonnet_in = resume_state["sonnet_in"]
        sonnet_out = resume_state["sonnet_out"]
        haiku_in = resume_state["haiku_in"]
        haiku_out = resume_state["haiku_out"]
        cache_create = resume_state["cache_create"]
        cache_read = resume_state["cache_read"]
    else:
        predictions = []
        start_idx = 0
        sonnet_in = sonnet_out = haiku_in = haiku_out = 0
        cache_create = cache_read = 0
    t0 = time.time()
    hard_errors = 0

    for i, row in enumerate(rows):
        if i < start_idx:
            continue
        user_msg = build_user_message(row, args.template)
        pred, raw, usage, err = call_claude(user_msg, system_prompt, effort, args.model)
        if err:
            hard_errors += 1
            print(f"  row {i} HARD ERROR: {err}", flush=True)
            predictions.append({
                "id": row.get("id"),
                "label": row.get("label"),
                "pred": None,
                "raw": raw[:200] if isinstance(raw, str) else "",
                "error": err,
            })
            if hard_errors >= 5:
                print(f"ABORT cell after {hard_errors} hard errors", flush=True)
                break
            continue

        for model_key, m in usage.items():
            if model_key.startswith("claude-sonnet-4-6"):
                sonnet_in += m.get("inputTokens", 0)
                sonnet_out += m.get("outputTokens", 0)
                cache_create += m.get("cacheCreationInputTokens", 0)
                cache_read += m.get("cacheReadInputTokens", 0)
            elif model_key.startswith("claude-haiku"):
                haiku_in += m.get("inputTokens", 0)
                haiku_out += m.get("outputTokens", 0)

        predictions.append({
            "id": row.get("id"),
            "label": row.get("label"),
            "pred": pred,
            "raw": raw,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"  {i+1}/{len(rows)} | {rate:.2f} req/s | sonnet_in/out={sonnet_in}/{sonnet_out} "
                f"cc/cr={cache_create}/{cache_read} haiku_in/out={haiku_in}/{haiku_out}",
                flush=True,
            )

    y_true = [(p["label"] == "removed") for p in predictions if p.get("pred") is not None]
    y_pred = [p["pred"] for p in predictions if p.get("pred") is not None]
    parse_fails = sum(1 for p in predictions if p.get("pred") is None and not p.get("error"))

    if not y_true:
        accuracy = precision = recall = f1 = kappa = 0.0
    else:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
        tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
        n = len(y_true)
        accuracy = (tp + tn) / n
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        po = accuracy
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else 0.0
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0

    out = {
        "sub": args.sub,
        "template": args.template,
        "thinking": args.thinking,
        "effort": effort,
        "model": args.model,
        "prompt_variant": args.prompt_variant,
        "n_requested": args.n,
        "n_returned": len(predictions),
        "n_used": len(y_true),
        "parse_fails": parse_fails,
        "hard_errors": hard_errors,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa,
        "sonnet_tokens_in": sonnet_in,
        "sonnet_tokens_out": sonnet_out,
        "haiku_tokens_in": haiku_in,
        "haiku_tokens_out": haiku_out,
        "cache_creation_tokens": cache_create,
        "cache_read_tokens": cache_read,
        "elapsed_s": time.time() - t0,
        "predictions": predictions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"WROTE {out_path}: F1={f1:.3f} kappa={kappa:.3f} parse_fails={parse_fails} "
        f"hard_errors={hard_errors}",
        flush=True,
    )
    return 0 if hard_errors == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
