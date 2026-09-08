#!/usr/bin/env python3
"""CPU-only vs single-GPU average latency per TPC-H query at SF50.

Redraws 그림 1 keeping only the CPU and GPU x1 series, and only SF50. Values are
read off the original figure, so they approximate that chart rather than being
fresh measurements -- edit DATA to drop in real numbers.

Q2's CPU bar is ~16x the next tallest, so the y-axis is clipped and that bar is
annotated with its true value, as in the original.

Colour note: the original's orange/green pair fails colour-vision separation
(protanopia ΔE 3.2 against a floor of 8 -- red-blind readers see one colour).
The green here is stepped darker to #166534, which keeps the requested hue and
passes every check (CVD ΔE 12.3, normal-vision 30.6, contrast >= 3:1).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

QUERIES = ["Q2", "Q13", "Q15", "Q17", "Q19", "Q20"]

# ms; approximated from the source figure (SF50 panel).
DATA = {
    "CPU":   [214_000, 4_000, 1_400, 7_000, 6_600, 11_400],
    "GPU×1": [  4_100, 1_200,   800, 4_150,   900,  4_000],
}
YLIM = 13_000          # bars above this are clipped and annotated
COLORS = {"CPU": "#eb6834", "GPU×1": "#166534"}
INK, GRID = "#000000", "#d0d0d0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("experiment/figs/cpu_vs_gpu1_latency_sf50"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(QUERIES))
    width = 0.38

    for i, (name, values) in enumerate(DATA.items()):
        offset = (i - 0.5) * width
        heights = [min(v, YLIM) for v in values]
        bars = ax.bar(x + offset, heights, width, label=name,
                      color=COLORS[name], edgecolor="none", zorder=3)
        for bar, raw in zip(bars, values):
            if raw <= YLIM:
                continue
            bx = bar.get_x() + bar.get_width() / 2
            ax.annotate(f"{raw / 1000:.1f}s", (bx, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=9,
                        color=COLORS[name], fontweight="bold", zorder=5)

    ax.set_ylim(0, YLIM)
    ax.set_xticks(x)
    ax.set_xticklabels(QUERIES, rotation=30, ha="right", color=INK)
    ax.set_ylabel("Avg Latency (ms)", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    # Full rectangular frame, as in the original.
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelcolor=INK)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, fontsize=10, labelcolor=INK,
              handlelength=1.2, handleheight=1.2, columnspacing=2.0)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        pth = args.out.with_suffix(f".{ext}")
        fig.savefig(pth, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
