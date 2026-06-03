"""Generate the Section 4.5 cube heatmap (D2a) for thesis.tex.

Rows = 6 matched subreddits.
Cols = 4 (era × recipe) combos: 2017 R4, 2026 R4, 2017 paper, 2026 paper.
Cells = absolute Cohen's kappa, sequential colormap.
Right margin = per-row Δκ for each recipe with direction marker.

Outputs:
  figures/thesis_2026/cube_heatmap.pdf
  figures/thesis_2026/cube_heatmap.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Data from tab:slm_mod_cube_era in thesis.tex L1000-1011.
# Rows: (sub, 2017_R4_kappa, 2026_R4_kappa, 2017_paper_kappa, 2026_paper_kappa,
#         delta_R4, delta_R4_robust, delta_paper, delta_paper_robust)
DATA = [
    ("r/AskHistorians", 0.290, 0.550, 0.270, 0.472, +0.260, True,  +0.203, True),
    ("r/changemyview",  0.760, 0.540, 0.600, 0.503, -0.217, True,  -0.096, False),
    ("r/Games",         0.330, 0.449, 0.210, 0.092, +0.123, False, -0.117, False),
    ("r/politics",      0.480, 0.395, 0.320, 0.142, -0.082, False, -0.176, True),
    ("r/askscience",    0.370, 0.343, 0.010, 0.093, -0.021, False, +0.081, False),
    ("r/science",       0.260, 0.331, 0.000, 0.018, +0.067, False, +0.018, False),
]

COLS = [
    ("2017", "R4"),
    ("2026", "R4"),
    ("2017", "paper"),
    ("2026", "paper"),
]

# Colors matching the thesis palette: deltapos (forest green) and deltaneg (deep red).
DELTAPOS = "#0C6E28"
DELTANEG = "#B0202E"

fig, (ax_main, ax_delta) = plt.subplots(
    1, 2, figsize=(9.6, 4.0),
    gridspec_kw={"width_ratios": [4, 1.6], "wspace": 0.22},
)

# Main heatmap (4 era×recipe columns x 6 subs)
kappa_matrix = np.array([
    [row[1], row[2], row[3], row[4]] for row in DATA
])

# Use sequential blue colormap, matching headerbg/macrobg palette
cmap = mcolors.LinearSegmentedColormap.from_list(
    "thesis_blue",
    ["#F7F9FC", "#D6E2F1", "#88B0DC", "#3A6FB0"],
)
norm = mcolors.Normalize(vmin=0.0, vmax=0.8)

im = ax_main.imshow(kappa_matrix, cmap=cmap, norm=norm, aspect="auto")

# Cell text (kappa values)
for i in range(len(DATA)):
    for j in range(4):
        val = kappa_matrix[i, j]
        # Black text on light cells, white on dark
        txt_color = "white" if val > 0.50 else "black"
        ax_main.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color=txt_color, fontsize=10, fontweight="bold")

# Y-axis: subreddit labels
ax_main.set_yticks(range(len(DATA)))
ax_main.set_yticklabels([row[0] for row in DATA], fontsize=10)

# X-axis: two-level header (era / recipe)
ax_main.set_xticks(range(4))
ax_main.set_xticklabels([f"{era}\n{recipe}" for era, recipe in COLS], fontsize=9.5)
ax_main.tick_params(axis="x", which="both", length=0, pad=4)
ax_main.tick_params(axis="y", which="both", length=0, pad=4)

# Visual separator between recipes (between cols 1 and 2)
ax_main.axvline(x=1.5, color="black", linewidth=1.5)

# Recipe-group labels above the columns
ax_main.text(0.5, -0.85, "R4-style", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", transform=ax_main.transData)
ax_main.text(2.5, -0.85, "paper-exact", ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", transform=ax_main.transData)

# Per-row 2017 / 2026 contrast lines (within each recipe block)
for j in [0.5, 2.5]:
    ax_main.axvline(x=j, color="black", linewidth=0.5, linestyle=":", alpha=0.5)

# Colorbar
cbar = fig.colorbar(im, ax=ax_main, fraction=0.04, pad=0.02)
cbar.set_label("$\\kappa$", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Right panel: Δκ direction badges (one for R4, one for paper, per sub)
ax_delta.set_xlim(-0.5, 1.5)
ax_delta.set_ylim(len(DATA) - 0.5, -0.5)
ax_delta.set_yticks([])
ax_delta.set_xticks([0, 1])
ax_delta.set_xticklabels(["R4", "paper"], fontsize=9.5)
ax_delta.tick_params(axis="x", which="both", length=0, pad=4)
ax_delta.text(0.5, -0.85, "$\\Delta\\kappa_{2026-2017}$", ha="center", va="bottom",
              fontsize=10.5, fontweight="bold", transform=ax_delta.transData)

for i, row in enumerate(DATA):
    for j, (delta, robust) in enumerate([(row[5], row[6]), (row[7], row[8])]):
        if robust:
            color = DELTAPOS if delta > 0 else DELTANEG
            face = color
            txt_color = "white"
            edgecolor = "black"
            linewidth = 1.0
            arrow = "$\\uparrow$" if delta > 0 else "$\\downarrow$"
        else:
            face = "#EAEAEA"
            txt_color = "#666666"
            edgecolor = "#BBBBBB"
            linewidth = 0.5
            arrow = "$\\sim$"
        ax_delta.add_patch(
            plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84,
                          facecolor=face, edgecolor=edgecolor, linewidth=linewidth)
        )
        ax_delta.text(j, i, f"{arrow}\n{delta:+.2f}", ha="center", va="center",
                      color=txt_color, fontsize=9, fontweight="bold",
                      linespacing=0.9)

# Spines
for ax in [ax_main, ax_delta]:
    for spine in ax.spines.values():
        spine.set_visible(False)

# Hide tick lines on ax_delta
ax_delta.tick_params(left=False, bottom=False)

fig.suptitle("",  fontsize=11, y=1.0)

plt.tight_layout()

OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "thesis_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = OUT_DIR / "cube_heatmap.pdf"
png_path = OUT_DIR / "cube_heatmap.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
