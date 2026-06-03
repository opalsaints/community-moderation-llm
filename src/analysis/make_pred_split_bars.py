"""Generate the Section 4.2 FT-vs-ZS prediction-split horizontal bars (D5).

Tells the "single-class hedge" story by showing, per spot-check subreddit, the
percentage of the balanced test set that each model predicts True vs False.
Balanced test sets are ~50/50 ground truth, so FT should sit near 50% predicted-True
while ZS Qwen 3 14B on r/changemyview hedges to 95%+ predicted-True.

Outputs:
  figures/thesis_2026/pred_split_bars.pdf
  figures/thesis_2026/pred_split_bars.png

Data source: R4_stacked predictions vs zero-shot Qwen 3 14B predictions on the
2000-comment balanced test sets per subreddit (Snellius
~/data/results/finetuned_2026/r4_stacked_*_predictions.jsonl
and ~/data/results/zero_shot_2026/zero_shot_*_predictions.jsonl).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# (sub, condition, pred_true_count, pred_false_count, n)
DATA = [
    ("r/antiai",        "FT",  1027,  973, 2000),
    ("r/antiai",        "ZS",  1699,  301, 2000),
    ("r/AskHistorians", "FT",  1051,  949, 2000),
    ("r/AskHistorians", "ZS",  1334,  666, 2000),
    ("r/changemyview",  "FT",  1058,  942, 2000),
    ("r/changemyview",  "ZS",  1899,  101, 2000),
]

DELTANEG_FILL  = "#E89094"   # deltaneg!50 -- pale red for "predicted True (remove)"
DELTAPOS_FILL  = "#84B898"   # deltapos!50 -- pale green for "predicted False (keep)"
DELTANEG_DARK  = "#B0202E"
DELTAPOS_DARK  = "#0C6E28"

fig, ax = plt.subplots(figsize=(8.8, 4.6))

ypos = np.arange(len(DATA))
# Reverse so r/changemyview (most dramatic) lands at the top of the chart visually
ypos = ypos[::-1]

for idx, (sub, cond, true_n, false_n, n) in enumerate(DATA):
    y = ypos[idx]
    true_frac  = true_n  / n
    false_frac = false_n / n

    # Predicted True (left segment, red)
    ax.barh(y, true_frac, height=0.72, left=0,
            color=DELTANEG_FILL, edgecolor=DELTANEG_DARK, linewidth=0.8)
    # Predicted False (right segment, green)
    ax.barh(y, false_frac, height=0.72, left=true_frac,
            color=DELTAPOS_FILL, edgecolor=DELTAPOS_DARK, linewidth=0.8)

    # In-bar text: percentage in each segment, white if segment is wide enough
    if true_frac >= 0.10:
        ax.text(true_frac / 2, y, f"{true_frac*100:.1f}%",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#3D0510")
    if false_frac >= 0.10:
        ax.text(true_frac + false_frac / 2, y, f"{false_frac*100:.1f}%",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#08381A")

    # Left of the bar: condition label
    ax.text(-0.018, y, cond,
            ha="right", va="center", fontsize=10.5, fontweight="bold")

# Vertical reference line at 0.50 (balanced ground truth prevalence)
ax.axvline(x=0.50, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(0.50, len(DATA) - 0.4, "balanced\nground truth\n(50% True)",
        ha="center", va="bottom", fontsize=8.5, color="black", alpha=0.75,
        style="italic")

# Group brackets: a paired bracket on the left for each subreddit
# DATA is ordered FT, ZS within each sub; pairs are (5,4), (3,2), (1,0) on ypos
subs_unique = []
seen = set()
for sub, _, _, _, _ in DATA:
    if sub not in seen:
        subs_unique.append(sub)
        seen.add(sub)

for sub_idx, sub in enumerate(subs_unique):
    # Find the two y-positions for this sub
    y_top    = ypos[2 * sub_idx]
    y_bottom = ypos[2 * sub_idx + 1]
    y_center = (y_top + y_bottom) / 2
    ax.text(-0.18, y_center, sub,
            ha="right", va="center", fontsize=11, fontweight="bold")
    # Bracket
    ax.annotate("", xy=(-0.05, y_top),    xytext=(-0.05, y_bottom),
                arrowprops=dict(arrowstyle="-", linewidth=1.2, color="black"))

# X-axis
ax.set_xlim(0, 1)
ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9.5)
ax.set_xlabel("Fraction of the balanced test set ($n = 2000$)", fontsize=10.5)

# Y-axis: hide ticks/labels (we placed our own with annotate)
ax.set_yticks([])
ax.set_ylim(-0.7, len(DATA) - 0.3)

# Spines: keep only bottom
for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("black")

# Legend
legend_handles = [
    mpatches.Patch(facecolor=DELTANEG_FILL, edgecolor=DELTANEG_DARK,
                   label='Predicted "True" (remove)'),
    mpatches.Patch(facecolor=DELTAPOS_FILL, edgecolor=DELTAPOS_DARK,
                   label='Predicted "False" (keep)'),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=10)

plt.tight_layout()

OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "thesis_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = OUT_DIR / "pred_split_bars.pdf"
png_path = OUT_DIR / "pred_split_bars.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
