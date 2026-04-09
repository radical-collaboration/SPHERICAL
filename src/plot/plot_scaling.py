#!/usr/bin/env python3
"""
Plot SGDES Dragon scaling results from section 8 of the README.

Hardcoded data from runs on Delta (NCSA), 2026-04-05:
  - 1 node  × 1 GPU: 1 mutation,  443 s wall
  - 2 nodes × 1 GPU: 2 mutations, 484 s wall
  - 4 nodes × 1 GPU: 4 mutations, 517 s wall

Produces two panels:
  Top    — overall scaling: total wall time + per-mutation time vs node count
  Bottom — stacked bar of per-mutation round breakdown (R1/R2/R3 components)
            for each node count

Usage:
    python src/plot/plot_scaling.py --output scaling.png
"""

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Hardcoded data (README §8)
# ---------------------------------------------------------------------------

NODES = [1, 2, 4]

# Overall timing
WALL_TIME = [443, 484, 517]  # seconds, total job wall time
PER_MUT = [443, 242, 129]  # seconds, per-mutation (wall / mutations)

# Round-by-round per-mutation breakdown (seconds)
# Keys: (round, component) -> [1-node, 2-node, 4-node]
ROUND_DATA = {
    # Round 1
    ("R1", "Input embed"): [33, 28, 59],
    ("R1", "DES"): [47, 48, 58],
    ("R1", "Generated embed"): [31, 32, 31],
    ("R1", "Foldseek search"): [40, 44, 50],
    # Round 2
    ("R2", "DES"): [29, 28, 28],
    ("R2", "Generated embed"): [33, 33, 32],
    ("R2", "Foldseek search"): [35, 33, 34],
    # Round 3
    ("R3", "DES"): [29, 28, 28],
    ("R3", "Generated embed"): [32, 37, 32],
    ("R3", "Foldseek search"): [27, 40, 35],
}

# Component colors (consistent across rounds)
COMPONENT_COLORS = {
    "Input embed": "#4C72B0",
    "DES": "#DD8452",
    "Generated embed": "#55A868",
    "Foldseek search": "#C44E52",
}

# Ideal (linear) scaling baseline
IDEAL_PER_MUT = [PER_MUT[0] / n for n in NODES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _efficiency(per_mut):
    """Parallel efficiency relative to 1-node per-mutation time."""
    return [PER_MUT[0] / (n * t) * 100 for n, t in zip(NODES, per_mut)]


def _stacked_round(ax, nodes, round_label, components_vals, bottom_offset):
    """Draw one stacked bar group for a single round."""
    x = np.arange(len(nodes))
    width = 0.22

    # Offset groups: R1 left, R2 centre, R3 right
    offsets = {"R1": -width, "R2": 0.0, "R3": width}
    xoff = x + offsets[round_label]

    bottoms = np.zeros(len(nodes))
    for comp, vals in components_vals:
        color = COMPONENT_COLORS[comp]
        ax.bar(xoff, vals, width, bottom=bottoms, color=color, edgecolor="white", linewidth=0.4)
        bottoms += np.array(vals)

    # Round label above bar
    for i, (xp, tot) in enumerate(zip(xoff, bottoms)):
        ax.text(xp, tot + 3, round_label, ha="center", va="bottom", fontsize=7, color="0.4")

    return bottoms


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------


def plot(output: str):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 9),
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    fig.suptitle(
        "SGDES Dragon Scaling — Delta (NCSA), 2026-04-05\n"
        "Config: des_rounds=3, des_batch=100, n_seq=50, foldtune_rounds=3, fast_folding=True",
        fontsize=10,
        y=0.98,
    )

    # ------------------------------------------------------------------
    # Panel 1: overall scaling
    # ------------------------------------------------------------------
    ax1 = axes[0]
    x = np.array(NODES)

    color_wall = "#4C72B0"
    color_per = "#DD8452"
    color_ideal = "0.6"

    ax1.plot(x, WALL_TIME, "o-", color=color_wall, lw=2, ms=7, label="Wall time (total job)")
    ax1.plot(x, PER_MUT, "s-", color=color_per, lw=2, ms=7, label="Per-mutation time")
    ax1.plot(x, IDEAL_PER_MUT, "--", color=color_ideal, lw=1.4, label="Ideal (linear) per-mutation")

    # Annotate per-mutation efficiency
    efficiencies = _efficiency(PER_MUT)
    for nx, pm, eff in zip(x, PER_MUT, efficiencies):
        ax1.annotate(
            f"{pm} s\n({eff:.0f}%)",
            xy=(nx, pm),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=color_per,
        )
    for nx, wt in zip(x, WALL_TIME):
        ax1.annotate(
            f"{wt} s",
            xy=(nx, wt),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8,
            color=color_wall,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} node{'s' if n > 1 else ''}" for n in NODES])
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Overall Scaling", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------
    # Panel 2: stacked bar round breakdown
    # ------------------------------------------------------------------
    ax2 = axes[1]

    # Gather per-round component lists
    rounds = ["R1", "R2", "R3"]
    for rnd in rounds:
        comps = [(c, v) for (r, c), v in ROUND_DATA.items() if r == rnd]
        _stacked_round(ax2, NODES, rnd, comps, 0)

    # Legend for components
    legend_patches = [
        mpatches.Patch(color=COMPONENT_COLORS[c], label=c)
        for c in ["Input embed", "DES", "Generated embed", "Foldseek search"]
    ]
    ax2.legend(
        handles=legend_patches, fontsize=8, loc="upper right", title="Component", title_fontsize=8
    )

    # Round totals annotation
    for rnd_idx, rnd in enumerate(rounds):
        comps = [(c, v) for (r, c), v in ROUND_DATA.items() if r == rnd]
        offsets = {"R1": -0.22, "R2": 0.0, "R3": 0.22}
        for ni, n in enumerate(NODES):
            total = sum(v[ni] for _, v in comps)
            xpos = ni + offsets[rnd]
            ax2.text(
                xpos,
                total + 6,
                f"{total} s",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="0.2",
                fontweight="bold",
            )

    ax2.set_xticks(np.arange(len(NODES)))
    ax2.set_xticklabels([f"{n} node{'s' if n > 1 else ''}" for n in NODES])
    ax2.set_ylabel("Per-mutation time (s)")
    ax2.set_title("Round-by-Round Breakdown (per mutation)", fontsize=10)
    ax2.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print()
    print("Overall scaling")
    print(
        f"{'Nodes':>6}  {'Mutations':>9}  {'Wall time':>10}  {'Per-mutation':>13}  {'Efficiency':>10}"
    )
    mutations = [1, 2, 4]
    for n, m, wt, pm, eff in zip(NODES, mutations, WALL_TIME, PER_MUT, _efficiency(PER_MUT)):
        print(f"{n:>6}  {m:>9}  {wt:>9} s  {pm:>11} s  {eff:>9.0f}%")

    print()
    print("Round totals (per mutation)")
    header = f"{'Round':<6}  {'1 node':>8}  {'2 nodes':>8}  {'4 nodes':>8}"
    print(header)
    print("-" * len(header))
    for rnd in rounds:
        comps = [(c, v) for (r, c), v in ROUND_DATA.items() if r == rnd]
        totals = [sum(v[i] for _, v in comps) for i in range(len(NODES))]
        print(f"{rnd:<6}  {totals[0]:>7} s  {totals[1]:>7} s  {totals[2]:>7} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot SGDES scaling results (hardcoded from README §8)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scaling.png",
        help="Output PNG path (default: scaling.png)",
    )
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
