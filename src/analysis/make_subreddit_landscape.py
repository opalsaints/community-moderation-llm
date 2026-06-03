"""Generate the Section 5.3 subreddit landscape scatter (D4) for thesis.tex.

Each of the 15 subreddits is a dot in a 2D space of:
  X: natural removal rate (log scale; range 0.5%-28.5%)
  Y: subscriber count (log scale; range 137K - 34M)

Encoded extras:
  - Color: balanced Cohen's kappa (sequential blue-to-green-to-amber)
  - Size: rule count

The point is to visualise where high-kappa subs sit in the feature space, as a
lead-in to the formal correlation analysis in Section 5.3.

Outputs:
  figures/thesis_2026/subreddit_landscape.pdf
  figures/thesis_2026/subreddit_landscape.png

Data sources:
  - tab:subreddits  (subscriber count, removal rate, rule count)
  - tab:per_sub_main (balanced kappa)
Both inlined here to keep the script self-contained.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# (sub, subscribers, removal_rate_pct, rule_count, balanced_kappa)
DATA = [
    ("AskHistorians",     2_700_000,  28.5, 14, 0.589),
    ("askscience",       26_000_000,  10.1,  8, 0.502),
    ("science",          34_000_000,   3.9, 10, 0.547),
    ("legaladvice",       3_400_000,  12.4, 12, 0.477),
    ("personalfinance",  22_000_000,   3.9, 10, 0.511),
    ("relationships",     3_700_000,   3.5,  6, 0.578),
    ("AmItheAsshole",    24_000_000,   1.4, 11, 0.606),
    ("changemyview",      4_200_000,   3.0, 12, 0.622),
    ("explainlikeimfive",24_000_000,   1.7, 11, 0.701),
    ("Games",             3_500_000,   5.1,  8, 0.545),
    ("news",             31_000_000,   2.8, 13, 0.496),
    ("TwoXChromosomes",  14_000_000,   2.4,  5, 0.582),
    ("politics",          9_100_000,   0.9, 14, 0.549),
    ("antiai",              168_000,   1.2,  8, 0.715),
    ("aiwars",              137_000,   0.5, 13, 0.581),
]

fig, ax = plt.subplots(figsize=(9.2, 5.6))

xs       = np.array([d[2] for d in DATA])
ys       = np.array([d[1] for d in DATA])
sizes    = np.array([d[3] for d in DATA])
kappas   = np.array([d[4] for d in DATA])
names    = [d[0] for d in DATA]

# Colour: sequential viridis-like, but matching thesis palette (white -> deltapos green)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "thesis_kappa",
    ["#D6E2F1", "#6D9CC9", "#3A6FB0", "#0C6E28"],
)
norm = mcolors.Normalize(vmin=0.45, vmax=0.75)

# Size: scale rule count to a visually-distinguishable range (50..380)
size_min, size_max = sizes.min(), sizes.max()
sizes_plot = 60 + (sizes - size_min) / (size_max - size_min) * 320

scatter = ax.scatter(
    xs, ys, s=sizes_plot, c=kappas, cmap=cmap, norm=norm,
    edgecolors="black", linewidths=0.8, alpha=0.92,
)

# Per-dot labels. Offset and align tweaks to reduce overlap.
LABEL_OFFSETS = {
    # sub: (dx, dy_factor, ha)
    "AskHistorians":     (-0.40, 1.06, "right"),
    "askscience":        (-0.55, 1.04, "right"),
    "science":           (0.45, 1.06, "left"),
    "legaladvice":       (0.65, 1.04, "left"),
    "personalfinance":   (0.55, 0.94, "left"),
    "relationships":     (-0.40, 1.05, "right"),
    "AmItheAsshole":     (0.07, 0.92, "left"),
    "changemyview":      (0.18, 1.05, "left"),
    "explainlikeimfive": (0.20, 1.05, "left"),
    "Games":             (-0.40, 1.05, "right"),
    "news":              (-0.40, 0.94, "right"),
    "TwoXChromosomes":   (0.20, 1.05, "left"),
    "politics":          (0.05, 1.10, "left"),
    "antiai":            (0.07, 1.08, "left"),
    "aiwars":            (-0.06, 0.92, "right"),
}

for (sub, subs_n, rem, rules, kappa), x, y in zip(DATA, xs, ys):
    dx, dy_factor, ha = LABEL_OFFSETS.get(sub, (0.05, 1.05, "left"))
    ax.text(x + dx, y * dy_factor, f"r/{sub}",
            fontsize=8.5, ha=ha, va="center")

# Reference lines
ax.axvline(x=10, color="black", linestyle=":", linewidth=0.6, alpha=0.4)
ax.text(10, ys.max() * 1.5, "10%\nremoval", ha="center", va="bottom",
        fontsize=8, color="black", alpha=0.6, style="italic")

# Highlight the substantial-agreement subs with a thicker ring
substantial_ring = kappas >= 0.60
for (sub, subs_n, rem, rules, kappa), is_top in zip(DATA, substantial_ring):
    if is_top:
        ax.scatter([rem], [subs_n], s=sizes_plot[names.index(sub)] + 130,
                   facecolors="none", edgecolors="#0C6E28", linewidths=1.6,
                   zorder=2)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.35, 40)
ax.set_ylim(80_000, 80_000_000)
ax.set_xticks([0.5, 1, 2, 5, 10, 20])
ax.set_xticklabels(["0.5%", "1%", "2%", "5%", "10%", "20%"])
ax.set_yticks([100_000, 1_000_000, 10_000_000])
ax.set_yticklabels(["100 K", "1 M", "10 M"])
ax.tick_params(axis="both", which="both", labelsize=9.5)

ax.set_xlabel("Natural removal rate (log scale)", fontsize=10.5)
ax.set_ylabel("Subscriber count (log scale)", fontsize=10.5)

ax.grid(True, which="major", linestyle="-", alpha=0.18)
ax.set_axisbelow(True)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

# Colorbar
cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.025)
cbar.set_label("Balanced Cohen's $\\kappa$", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Size legend
import matplotlib.lines as mlines
size_handles = [
    mlines.Line2D([], [], marker='o', color='w',
                  markerfacecolor="#888", markeredgecolor='black',
                  markersize=np.sqrt(60 + (n - size_min) / (size_max - size_min) * 320),
                  label=f"{n} rules", linewidth=0)
    for n in [5, 10, 14]
]
ring_handle = mlines.Line2D([], [], marker='o', color='w',
                            markerfacecolor="none", markeredgecolor='#0C6E28',
                            markersize=10, markeredgewidth=1.6,
                            label="$\\kappa \\geq 0.60$", linewidth=0)
ax.legend(handles=size_handles + [ring_handle], loc="lower right",
          fontsize=8.5, frameon=True, framealpha=0.85,
          labelspacing=1.2, borderpad=0.8)

plt.tight_layout()

OUT_DIR = Path(__file__).resolve().parent.parent / "figures" / "thesis_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = OUT_DIR / "subreddit_landscape.pdf"
png_path = OUT_DIR / "subreddit_landscape.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")
