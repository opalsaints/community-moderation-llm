#!/usr/bin/env python3
"""
Plot training and evaluation loss curves from the optimal run metadata.

Usage:
    python plot_training_curves.py \
        --train-meta ~/data/results/optimal/gemma-2-9b-it_changemyview_optimal_train_meta.json \
        --output ~/data/results/optimal/loss_curves.png

    # Also works from the SLURM stderr log (live, before training finishes):
    python plot_training_curves.py \
        --slurm-log ~/capstone/logs/optimal_cmv_21592675.err \
        --output ~/data/results/optimal/loss_curves_live.png
"""

import argparse
import json
import re
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (works on HPC)
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
    sys.exit(1)


def load_from_train_meta(filepath):
    """Extract train/eval loss from the saved training metadata JSON."""
    with open(filepath) as f:
        meta = json.load(f)

    log = meta.get("training_log", [])

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log:
        step = entry.get("step", 0)
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    return train_steps, train_loss, eval_steps, eval_loss


def load_from_slurm_log(filepath):
    """Parse training progress from SLURM stderr (tqdm + HF Trainer logs).

    HuggingFace Trainer logs look like:
      {'loss': 0.6931, 'grad_norm': 1.23, 'learning_rate': 3e-05, 'epoch': 0.02}
      {'eval_loss': 0.5500, 'eval_runtime': 12.3, ...}
    """
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    step_counter = 0

    with open(filepath) as f:
        for line in f:
            line = line.strip()

            # HF Trainer dict-style logs
            if line.startswith("{") and "'loss'" in line:
                try:
                    # Convert single quotes to double quotes for JSON parsing
                    entry = json.loads(line.replace("'", '"'))
                    step_counter += 1
                    if "loss" in entry and "eval_loss" not in entry:
                        step = entry.get("step", step_counter * 10)
                        train_steps.append(step)
                        train_loss.append(entry["loss"])
                    if "eval_loss" in entry:
                        step = entry.get("step", step_counter * 10)
                        eval_steps.append(step)
                        eval_loss.append(entry["eval_loss"])
                except (json.JSONDecodeError, ValueError):
                    pass

    return train_steps, train_loss, eval_steps, eval_loss


def plot_curves(train_steps, train_loss, eval_steps, eval_loss, output_path,
                title="Training Loss Curves"):
    """Create a publication-quality loss curve plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Training loss
    if train_steps:
        ax.plot(train_steps, train_loss, color="#2196F3", alpha=0.4,
                linewidth=0.8, label="Training loss (per step)")

        # Smoothed training loss (rolling average)
        if len(train_loss) > 10:
            window = max(5, len(train_loss) // 20)
            smoothed = []
            for i in range(len(train_loss)):
                start = max(0, i - window)
                smoothed.append(sum(train_loss[start:i+1]) / (i - start + 1))
            ax.plot(train_steps, smoothed, color="#1565C0", linewidth=2,
                    label=f"Training loss (smoothed, window={window})")

    # Eval loss
    if eval_steps:
        ax.plot(eval_steps, eval_loss, color="#E53935", linewidth=2,
                marker="o", markersize=5, label="Validation loss")

        # Mark the best eval checkpoint
        best_idx = eval_loss.index(min(eval_loss))
        ax.axvline(x=eval_steps[best_idx], color="#E53935", linestyle="--",
                   alpha=0.5, linewidth=1)
        ax.annotate(f"Best: {eval_loss[best_idx]:.4f}\n(step {eval_steps[best_idx]})",
                    xy=(eval_steps[best_idx], eval_loss[best_idx]),
                    xytext=(15, 15), textcoords="offset points",
                    fontsize=9, color="#E53935",
                    arrowprops=dict(arrowstyle="->", color="#E53935", alpha=0.7))

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add epoch markers if we know total steps
    if train_steps:
        total = max(train_steps)
        mid = total // 2
        if mid > 0:
            ax.axvline(x=mid, color="gray", linestyle=":", alpha=0.5)
            ax.text(mid, ax.get_ylim()[1], " Epoch 2", fontsize=9,
                    color="gray", va="top")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    # Also print summary stats
    if train_loss:
        print(f"\nTraining loss: {train_loss[0]:.4f} (start) -> {train_loss[-1]:.4f} (end)")
    if eval_loss:
        print(f"Eval loss:     {eval_loss[0]:.4f} (first) -> {min(eval_loss):.4f} (best) -> {eval_loss[-1]:.4f} (last)")
        best_idx = eval_loss.index(min(eval_loss))
        print(f"Best checkpoint: step {eval_steps[best_idx]} (eval_loss={eval_loss[best_idx]:.4f})")
        if len(eval_loss) > 1 and eval_loss[-1] > min(eval_loss) * 1.02:
            print("NOTE: Final eval loss is higher than best -- early stopping likely activated (good).")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--train-meta", help="Training metadata JSON file")
    parser.add_argument("--slurm-log", help="SLURM stderr log file (for live monitoring)")
    parser.add_argument("--output", required=True, help="Output PNG file")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    if not args.train_meta and not args.slurm_log:
        print("ERROR: Provide either --train-meta or --slurm-log")
        sys.exit(1)

    if args.train_meta:
        train_steps, train_loss, eval_steps, eval_loss = load_from_train_meta(args.train_meta)
        source = args.train_meta
    else:
        train_steps, train_loss, eval_steps, eval_loss = load_from_slurm_log(args.slurm_log)
        source = args.slurm_log

    if not train_steps and not eval_steps:
        print(f"No training data found in {source}")
        print("Training may still be in the tokenisation/setup phase.")
        sys.exit(0)

    title = args.title or "Optimal Run: changemyview (DoRA, enriched)"
    plot_curves(train_steps, train_loss, eval_steps, eval_loss, args.output, title)


if __name__ == "__main__":
    main()
