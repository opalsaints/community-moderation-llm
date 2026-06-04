"""Regenerate the thesis figures from the released results.

Figures A, B, D, H, I, and J are rebuilt directly from results/main_2026/. The
correlation scatter (E) and the confidence ridgeline (G) depend on per-community
metadata and the per-comment prediction dumps, neither of which is redistributed
(see the Data and ethics section of the README); both are provided pre-rendered
under figures/ and are skipped here. Figures C and F are inline TikZ in thesis.tex.

Outputs under figures/thesis_2026/*.pdf and *.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures" / "thesis_2026"
OUT.mkdir(parents=True, exist_ok=True)

METRICS_DIR = ROOT / "results" / "main_2026"
IMBAL_DIR = ROOT / "results" / "main_2026" / "natural_rate"
PRED_DIR = ROOT / "results" / "main_2026" / "predictions"  # not redistributed; figure G skips if absent
LENGTH_DIR = ROOT / "results" / "main_2026" / "length_baseline"

# Palette matches thesis.tex xcolor definitions.
COLOR_HEADER = "#D6E2F1"
COLOR_MACRO = "#E8F2FE"
COLOR_POS = "#0C6E28"
COLOR_NEG = "#B0202E"
COLOR_BAR = "#4F7FBF"
COLOR_ACCENT = "#2F5D8F"
COLOR_MUTED = "#8B9BB2"

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#3C3C3C",
    "axes.linewidth": 0.7,
    "grid.color": "#CCCCCC",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.6,
    "pdf.fonttype": 42,
})

SUBS_15 = [
    "AskHistorians", "antiai", "aiwars", "changemyview", "science",
    "askscience", "explainlikeimfive", "personalfinance", "TwoXChromosomes",
    "news", "AmItheAsshole", "politics", "relationships", "Games",
    "legaladvice",
]

# Per-community natural removal rates (fraction of comments removed by moderators),
# read from results/main_2026/natural_rate/<sub>.json. Used by figures B and H.
NATURAL_RATE = {
    "AmItheAsshole": 0.014, "AskHistorians": 0.285, "Games": 0.051,
    "TwoXChromosomes": 0.024, "aiwars": 0.005, "antiai": 0.012,
    "askscience": 0.101, "changemyview": 0.030, "explainlikeimfive": 0.017,
    "legaladvice": 0.124, "news": 0.028, "personalfinance": 0.039,
    "politics": 0.009, "relationships": 0.035, "science": 0.039,
}

# Pool κ values from canonical pool_evals on Snellius
# (~/data/results/finetuned_2026/pool_evals/<sub>/pooled_all_metrics.json),
# fetched 2026-05-06 post-pool-retrofit. Matches Table 5 in thesis.tex.
POOL_KAPPA = {
    "antiai": 0.6440, "explainlikeimfive": 0.6880, "changemyview": 0.6340,
    "AmItheAsshole": 0.6700, "AskHistorians": 0.5780,
    "TwoXChromosomes": 0.5620, "aiwars": 0.5899, "relationships": 0.6120,
    "politics": 0.5880, "science": 0.5800, "Games": 0.5020,
    "personalfinance": 0.5340, "askscience": 0.5105, "news": 0.5320,
    "legaladvice": 0.4860,
}

# Per-subreddit fine-tune kappa recomputed on the SAME first-1,000-comment
# deterministic prefix the pooled adapter is evaluated on, for a matched
# comparison against POOL_KAPPA. These are the FT-kappa column of the pooled
# results table; the n=2,000 headline per-sub kappa (macro 0.573) is a
# different, larger test set and must not be mixed with the pooled n=1,000
# values. Macro of this dict is 0.568, so pool - per-sub = +0.013.
PERSUB_KAPPA_N1000 = {
    "antiai": 0.690, "explainlikeimfive": 0.684, "AmItheAsshole": 0.622,
    "changemyview": 0.610, "TwoXChromosomes": 0.576, "aiwars": 0.572,
    "relationships": 0.560, "AskHistorians": 0.552, "personalfinance": 0.552,
    "politics": 0.540, "science": 0.534, "Games": 0.528, "askscience": 0.508,
    "legaladvice": 0.508, "news": 0.478,
}


def load_per_sub_metrics() -> dict[str, dict]:
    out = {}
    for sub in SUBS_15:
        path = METRICS_DIR / f"r4_stacked_{sub}_metrics.json"
        with path.open() as f:
            out[sub] = json.load(f)
    return out


def savefig(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"wrote {OUT / name}.pdf and .png")


# ---------------- Figure A: per-sub κ bar chart ----------------

def figure_a():
    metrics = load_per_sub_metrics()
    subs_sorted = sorted(SUBS_15, key=lambda s: metrics[s]["cohens_kappa"])
    kappa = [metrics[s]["cohens_kappa"] for s in subs_sorted]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    y = np.arange(len(subs_sorted))

    colors = [COLOR_ACCENT if k >= 0.6 else COLOR_BAR for k in kappa]
    ax.barh(y, kappa, color=colors, edgecolor="white", linewidth=0.5, height=0.72)

    ax.axvline(0.6, color=COLOR_NEG, linestyle=(0, (4, 3)), linewidth=0.9,
               zorder=0, label=r"$\kappa = 0.60$ (substantial)")
    ax.axvline(0.4, color=COLOR_MUTED, linestyle=(0, (2, 3)), linewidth=0.8,
               zorder=0, label=r"$\kappa = 0.40$ (moderate)")

    ax.set_yticks(y, labels=[f"r/{s}" for s in subs_sorted])
    for label, k in zip(ax.get_yticklabels(), kappa):
        if k >= 0.6:
            label.set_fontweight("bold")

    ax.set_xlim(0, 0.82)
    ax.set_xlabel(r"Cohen's $\kappa$ (balanced test set)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    for i, k in enumerate(kappa):
        ax.text(k + 0.008, i, f"{k:.3f}", va="center", fontsize=7.5,
                color="#333333")
    macro = float(np.mean(kappa))
    n_substantial = sum(1 for k in kappa if k >= 0.6)
    ax.set_title(f"macro mean $\\kappa = {macro:.3f}$  ({n_substantial} of 15 $\\geq 0.60$)",
                 fontsize=9, loc="right", color=COLOR_ACCENT, style="italic",
                 pad=6)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)
    fig.tight_layout()
    savefig(fig, "per_sub_kappa")
    plt.close(fig)


# ---------------- Figure B: imbalanced-rate degradation curve ----------------

def figure_b():
    data = {}
    for sub in SUBS_15:
        with (IMBAL_DIR / f"{sub}.json").open() as f:
            data[sub] = json.load(f)

    # Macro-averaged curves across the 15 subs: macro_f1, kappa, precision, recall.
    rates = ["0.01", "0.05", "0.10", "natural"]

    def macro_metric(metric: str) -> tuple[np.ndarray, np.ndarray]:
        means, lows, highs = [], [], []
        for r in rates:
            per_sub = [data[s]["rates"][r][metric] for s in SUBS_15]
            vals = np.array([m["mean"] for m in per_sub])
            means.append(vals.mean())
            lows.append(np.array([m["ci_low"] for m in per_sub]).mean())
            highs.append(np.array([m["ci_high"] for m in per_sub]).mean())
        return np.array(means), np.array(lows), np.array(highs)

    rate_positions = np.arange(len(rates))
    avg_natural = float(np.mean([NATURAL_RATE[s] for s in SUBS_15])) * 100
    rate_labels = ["1%", "5%", "10%", f"natural\n(avg {avg_natural:.1f}%)"]

    metrics_spec = [
        ("f1_macro", "macro F1", COLOR_ACCENT, "o"),
        ("kappa", r"Cohen's $\kappa$", COLOR_BAR, "s"),
        ("precision", "precision (removed)", COLOR_NEG, "^"),
        ("recall", "recall (removed)", COLOR_POS, "D"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for metric, label, color, marker in metrics_spec:
        m, lo, hi = macro_metric(metric)
        ax.fill_between(rate_positions, lo, hi, color=color, alpha=0.10,
                        linewidth=0)
        ax.plot(rate_positions, m, color=color, marker=marker, markersize=5,
                linewidth=1.6, label=label)

    ax.set_xticks(rate_positions, labels=rate_labels)
    ax.set_xlabel("simulated prevalence of removed class")
    ax.set_ylabel("macro-averaged metric")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=2, columnspacing=1.2)
    fig.tight_layout()
    savefig(fig, "imbalanced_degradation")
    plt.close(fig)


# ---------------- Figure D: pool-vs-per-sub dumbbell plot ----------------

def figure_d():
    # Matched comparison: both adapters on the first-1,000-comment prefix.
    subs_sorted = sorted(SUBS_15, key=lambda s: PERSUB_KAPPA_N1000[s])
    persub = [PERSUB_KAPPA_N1000[s] for s in subs_sorted]
    pool = [POOL_KAPPA[s] for s in subs_sorted]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    y = np.arange(len(subs_sorted))

    for i, (ps, po) in enumerate(zip(persub, pool)):
        low, high = sorted([ps, po])
        ax.plot([low, high], [i, i], color=COLOR_MUTED, linewidth=1.4, zorder=1)
    ax.scatter(persub, y, color=COLOR_ACCENT, s=32,
               label="per-subreddit adapter (n=1,000)",
               zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(pool, y, color=COLOR_NEG, s=32, marker="D",
               label="pooled adapter (n=1,000)", zorder=3,
               edgecolor="white", linewidth=0.6)

    ax.axvline(0.6, color="#BBBBBB", linestyle=(0, (4, 3)), linewidth=0.8,
               zorder=0)
    ax.set_yticks(y, labels=[f"r/{s}" for s in subs_sorted])
    ax.set_xlim(0.2, 0.82)
    ax.set_xlabel(r"Cohen's $\kappa$")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)

    macro_ps = float(np.mean(persub))
    macro_po = float(np.mean(pool))
    delta_disp = round(macro_po, 3) - round(macro_ps, 3)
    ax.set_title(
        f"per-sub {macro_ps:.3f}  ·  pool {macro_po:.3f}  "
        f"·  $\\Delta\\kappa = {delta_disp:+.3f}$",
        fontsize=9, loc="right", color=COLOR_ACCENT, style="italic", pad=6,
    )
    fig.tight_layout()
    savefig(fig, "pool_vs_persub")
    plt.close(fig)


# ---------------- Figure E: correlation scatter (provided pre-rendered) ----------------
# The 2x2 feature-correlation scatter (kappa vs rule count, subscribers, average
# comment length, and natural removal rate) is built from per-community metadata --
# subscriber counts and average comment length -- that is not part of the released
# data. The rendered figure is provided at figures/correlation_scatter.png.


# ---------------- Figure G: confidence-distribution ridgeline ----------------

def figure_g():
    if not PRED_DIR.exists():
        print("skipping confidence_ridgeline: per-comment prediction dumps are not "
              "part of the released data; see figures/confidence_ridgeline.png")
        return
    metrics = load_per_sub_metrics()
    subs_sorted = sorted(SUBS_15, key=lambda s: metrics[s]["cohens_kappa"],
                         reverse=True)

    data_by_sub = {}
    for sub in subs_sorted:
        path = PRED_DIR / f"r4_stacked_{sub}_predictions.jsonl"
        removed, approved = [], []
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                c = rec.get("confidence_removed")
                if c is None:
                    continue
                if rec["label"] == "removed":
                    removed.append(c)
                else:
                    approved.append(c)
        data_by_sub[sub] = (np.array(approved), np.array(removed))

    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    spacing = 1.0
    bins = np.linspace(0, 1, 41)
    scale = 0.70

    for i, sub in enumerate(subs_sorted):
        y_base = (len(subs_sorted) - 1 - i) * spacing
        appr, rem = data_by_sub[sub]
        h_appr, _ = np.histogram(appr, bins=bins, density=True)
        h_rem, _ = np.histogram(rem, bins=bins, density=True)
        # Normalize so per-row peak is readable.
        peak = max(h_appr.max() if h_appr.size else 1.0,
                   h_rem.max() if h_rem.size else 1.0, 1e-9)
        h_appr = h_appr / peak * scale
        h_rem = h_rem / peak * scale
        centers = (bins[:-1] + bins[1:]) / 2
        ax.fill_between(centers, y_base, y_base + h_appr, color=COLOR_POS,
                        alpha=0.35, linewidth=0)
        ax.fill_between(centers, y_base, y_base + h_rem, color=COLOR_NEG,
                        alpha=0.35, linewidth=0)
        ax.plot(centers, y_base + h_appr, color=COLOR_POS, linewidth=0.7)
        ax.plot(centers, y_base + h_rem, color=COLOR_NEG, linewidth=0.7)
        ax.text(-0.015, y_base + 0.05, f"r/{sub}",
                ha="right", va="bottom", fontsize=8)
        ax.text(1.015, y_base + 0.05,
                f"$\\kappa={metrics[sub]['cohens_kappa']:.2f}$",
                ha="left", va="bottom", fontsize=7, color="#555555")

    ax.axvline(0.5, color="#BBBBBB", linestyle=(0, (3, 3)), linewidth=0.6,
               zorder=0)
    ax.set_xlim(-0.14, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"model confidence $P(\text{removed})$")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COLOR_POS, alpha=0.5, edgecolor=COLOR_POS,
              label="approved (ground truth)"),
        Patch(facecolor=COLOR_NEG, alpha=0.5, edgecolor=COLOR_NEG,
              label="removed (ground truth)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    fig.tight_layout()
    savefig(fig, "confidence_ridgeline")
    plt.close(fig)


# ---------------- Figure H: per-sub natural-rate degradation dumbbell ----------------

def figure_h():
    """Per-sub balanced kappa -> natural-rate kappa, sorted by natural rate."""
    metrics = load_per_sub_metrics()
    imb = {}
    for sub in SUBS_15:
        with (IMBAL_DIR / f"{sub}.json").open() as f:
            imb[sub] = json.load(f)

    subs_sorted = sorted(SUBS_15, key=lambda s: NATURAL_RATE[s], reverse=True)
    kbal = [metrics[s]["cohens_kappa"] for s in subs_sorted]
    knat = [imb[s]["rates"]["natural"]["kappa"]["mean"] for s in subs_sorted]
    rates = [NATURAL_RATE[s] * 100 for s in subs_sorted]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    y = np.arange(len(subs_sorted))

    for i, (b, n) in enumerate(zip(kbal, knat)):
        ax.plot([n, b], [i, i], color=COLOR_NEG, linewidth=1.4,
                alpha=0.55, zorder=1)
    ax.scatter(kbal, y, color=COLOR_ACCENT, s=32,
               label=r"balanced $\kappa$ (50/50 test set)",
               zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter(knat, y, color=COLOR_NEG, s=32, marker="D",
               label=r"natural-rate $\kappa$ (per-sub prevalence)",
               zorder=3, edgecolor="white", linewidth=0.6)

    ax.axvline(0.6, color="#BBBBBB", linestyle=(0, (4, 3)), linewidth=0.8,
               zorder=0)
    ax.axvline(0.4, color="#DDDDDD", linestyle=(0, (2, 3)), linewidth=0.7,
               zorder=0)

    labels = [f"r/{s}  ({rate:.1f}%)"
              for s, rate in zip(subs_sorted, rates)]
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(0.0, 0.82)
    ax.set_xlabel(r"Cohen's $\kappa$")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)

    macro_b = float(np.mean(kbal))
    macro_n = float(np.mean(knat))
    ax.set_title(
        f"macro $\\kappa$: balanced {macro_b:.3f}  $\\rightarrow$  "
        f"natural {macro_n:.3f}  ($\\Delta = {macro_n - macro_b:+.3f}$)",
        fontsize=9, loc="right", color=COLOR_ACCENT, style="italic", pad=6,
    )
    fig.tight_layout()
    savefig(fig, "natrate_per_sub")
    plt.close(fig)


# ---------------- Figure I: length-baseline κ scatter ----------------

def figure_i():
    """Length-only kappa vs R4_stacked kappa as a per-sub gap chart."""
    metrics = load_per_sub_metrics()
    len_kappa = {}
    direction = {}
    for sub in SUBS_15:
        with (LENGTH_DIR / f"{sub}.json").open() as f:
            d = json.load(f)
        len_kappa[sub] = d["test"]["kappa"]
        direction[sub] = d["fit_on_train"]["direction"]

    subs_sorted = sorted(SUBS_15,
                         key=lambda s: metrics[s]["cohens_kappa"],
                         reverse=False)
    klen = [len_kappa[s] for s in subs_sorted]
    kr4 = [metrics[s]["cohens_kappa"] for s in subs_sorted]
    direc = [direction[s] for s in subs_sorted]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    y = np.arange(len(subs_sorted))

    # Length baseline bars (light) -- start from 0
    ax.barh(y, klen, color=COLOR_MUTED, alpha=0.55, edgecolor="white",
            linewidth=0.5, height=0.72, label=r"length-only baseline $\kappa$")
    # R4_stacked bars (saturated) -- overlay
    ax.barh(y, kr4, color=COLOR_ACCENT, alpha=0.9, edgecolor="white",
            linewidth=0.5, height=0.42,
            label=r"fine-tuned adapter $\kappa$")

    # Direction marker on each row.
    for i, d in enumerate(direc):
        sym = r"$\leq$" if d == "short_is_removed" else r"$\geq$"
        col = COLOR_ACCENT if d == "short_is_removed" else COLOR_NEG
        ax.text(-0.018, i, sym, ha="right", va="center", fontsize=8,
                color=col)

    # Δκ annotation on far right.
    for i, (l, r) in enumerate(zip(klen, kr4)):
        ax.text(r + 0.008, i, f"$\\Delta = {r - l:+.2f}$",
                va="center", fontsize=7, color=COLOR_POS)

    ax.axvline(0.6, color=COLOR_NEG, linestyle=(0, (4, 3)), linewidth=0.9,
               zorder=0, alpha=0.7)
    ax.set_yticks(y, labels=[f"r/{s}" for s in subs_sorted])
    ax.set_xlim(-0.05, 0.95)
    ax.set_xlabel(r"Cohen's $\kappa$")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0)

    macro_len = float(np.mean(klen))
    macro_r4 = float(np.mean(kr4))
    ax.set_title(
        f"macro $\\kappa$: length {macro_len:.3f}  $\\rightarrow$  "
        f"fine-tuned {macro_r4:.3f}  "
        f"($\\Delta = {macro_r4 - macro_len:+.3f}$)",
        fontsize=9, loc="right", color=COLOR_ACCENT, style="italic", pad=6,
    )
    fig.tight_layout()
    savefig(fig, "length_baseline_gap")
    plt.close(fig)


# ---------------- Figure J: Section 4.5 era cube forest plot ----------------

# Δκ point estimates and 95% paired-bootstrap CIs for the 12 (sub, recipe)
# pairs in the SLM-Mod 2017 era × recipe cube. Sourced from
# scripts/recompute_era_cube.py output (era_cube_final.json) and matches the
# Δκ column of Table tab:slm_mod_cube_era in thesis.tex.
ERA_CUBE_DELTAS = [
    # (sub, recipe, dk, ci_low, ci_high)
    ("changemyview",  "R4-style",    -0.217, -0.310, -0.116),
    ("changemyview",  "paper-exact", -0.096, -0.209, +0.021),
    ("politics",      "R4-style",    -0.082, -0.209, +0.047),
    ("politics",      "paper-exact", -0.176, -0.290, -0.059),
    ("AskHistorians", "R4-style",    +0.260, +0.124, +0.395),
    ("AskHistorians", "paper-exact", +0.203, +0.063, +0.331),
    ("askscience",    "R4-style",    -0.021, -0.153, +0.125),
    ("askscience",    "paper-exact", +0.081, -0.018, +0.190),
    ("Games",         "R4-style",    +0.123, -0.022, +0.249),
    ("Games",         "paper-exact", -0.117, -0.236, +0.004),
    ("science",       "R4-style",    +0.067, -0.056, +0.199),
    ("science",       "paper-exact", +0.018, -0.001, +0.040),
]


def figure_j():
    rows = sorted(ERA_CUBE_DELTAS, key=lambda r: r[2])
    n = len(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y = np.arange(n)

    for i, (sub, recipe, dk, lo, hi) in enumerate(rows):
        if hi < 0:
            color = COLOR_NEG
        elif lo > 0:
            color = COLOR_ACCENT
        else:
            color = COLOR_MUTED
        ax.errorbar(
            dk, i,
            xerr=[[dk - lo], [hi - dk]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=3.5,
            capthick=1.2,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
        )

    ax.axvline(0.0, color="#3C3C3C", linewidth=0.8, zorder=0)

    ax.set_yticks(
        y,
        labels=[f"r/{sub} – {recipe}" for sub, recipe, *_ in rows],
    )
    for label, (_, _, _, lo, hi) in zip(ax.get_yticklabels(), rows):
        if hi < 0 or lo > 0:
            label.set_fontweight("bold")

    ax.set_xlim(-0.42, 0.42)
    ax.set_xlabel(r"$\Delta\kappa_{2026 - 2017}$ (95% paired bootstrap CI)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    for i, (_, _, dk, lo, hi) in enumerate(rows):
        ax.text(
            hi + 0.012, i, f"{dk:+.3f}",
            va="center", fontsize=7.5, color="#333333",
        )

    n_neg = sum(1 for *_, hi in [(s, r, dk, lo, hi) for s, r, dk, lo, hi in rows] if hi < 0)
    n_pos = sum(1 for s, r, dk, lo, hi in rows if lo > 0)
    n_overlap = n - n_neg - n_pos
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=COLOR_NEG, markersize=6,
                         label=f"2017 easier (robust, $n={n_neg}$)"),
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=COLOR_ACCENT, markersize=6,
                         label=f"2026 easier (robust, $n={n_pos}$)"),
        mpl.lines.Line2D([0], [0], marker="o", color="w",
                         markerfacecolor=COLOR_MUTED, markersize=6,
                         label=f"CI overlaps zero ($n={n_overlap}$)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              frameon=False, handlelength=1.0)

    ax.set_title(
        f"4 of 12 pairs robust; split 2-for-2 by direction",
        fontsize=9, loc="right", color=COLOR_ACCENT, style="italic", pad=6,
    )
    fig.tight_layout()
    savefig(fig, "era_cube_forest")
    plt.close(fig)


if __name__ == "__main__":
    figure_a()
    figure_b()
    figure_d()
    figure_g()  # skips itself if the prediction dumps are absent
    figure_h()
    figure_i()
    figure_j()
