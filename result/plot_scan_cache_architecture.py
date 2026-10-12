"""Draw a presentation schematic verified against the current Parquet scan code.

This reads no data and runs no benchmark. Use:
pixi run -e duckdb-python python result/plot_scan_cache_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(16, 9))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 16), ylim=(0, 9))
ax.axis("off")
ink = "#25343a"
green = "#3c8065"
orange = "#bf6a36"
grey = "#889096"


def box(x, y, w, h, label, fill="#ffffff", size=15, edge=ink, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
        linewidth=1.6, edgecolor=edge, facecolor=fill,
        linestyle="--" if dashed else "-", zorder=3,
    ))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            fontsize=size, color=ink, zorder=4, linespacing=1.3)


def arrow(points, color=ink, dashed=False):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                lw=1.7, linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                 mutation_scale=15, linewidth=1.7, color=color,
                 linestyle="--" if dashed else "-", zorder=2,
                 shrinkA=0, shrinkB=0))


ax.text(.45, 8.60, "Query reordering + decoded GPU page cache",
        fontsize=24, color=ink, weight="bold")
ax.text(.45, 8.22, "Parquet scan path  |  solid: data / execution path    dashed: control / optional cache update",
        fontsize=11, color=grey)

box(.45, 7.2, 2.1, .75, "Query batch")
box(3.2, 7.2, 2.6, .75, "Reorder query sequence\n(benchmark runner)", size=13)
box(6.4, 7.2, 2.4, .75, "Execute selected SQL", size=14)
box(10.2, 7.2, 3.0, .75, "GPU operators\n(e.g., join / aggregate / sort)", size=12)
box(14.0, 7.2, 1.45, .75, "Result")
arrow([(2.55, 7.575), (3.2, 7.575)], dashed=True)
arrow([(5.8, 7.575), (6.4, 7.575)], dashed=True)
arrow([(8.8, 7.575), (10.2, 7.575)], dashed=True)
ax.text(9.5, 7.77, "query plan", fontsize=10, ha="center", color=grey)
arrow([(13.2, 7.575), (14, 7.575)])

ax.add_patch(FancyBboxPatch((.45, 2.6), 15.1, 4.1,
             boxstyle="round,pad=0.02,rounding_size=0.2", facecolor="#f3f7f3",
             edgecolor="none", zorder=0))
ax.text(.7, 6.40, "PARQUET SCAN", fontsize=10, weight="bold", color=green)
box(.8, 5.0, 2.7, .8, "GPU page lookup")
box(4.6, 5.0, 3.1, .8, "Decoded GPU page cache\n16 MiB allocation units", fill="#e1efdf", size=13)
box(8.5, 5.0, 2.9, .8, "Assemble scan table", size=14)
box(12.3, 5.0, 2.9, .8, "Post-filter / projection", size=14)
box(.8, 3.1, 2.7, .9, "Read missing data\nNVMe → pinned HOST", fill="#f8e7db", size=13)
box(4.6, 3.1, 3.1, .9, "H2D + GPU\ndecompress / decode", fill="#f8e7db", size=13)
box(8.5, 3.1, 2.9, .9, "Cache admission / populate\n(skip on full hit)", size=12)

arrow([(7.6, 7.2), (7.6, 6.10), (2.15, 6.10), (2.15, 5.8)], dashed=True)
ax.text(4.9, 6.25, "Scan request: file / columns / row groups", fontsize=10, ha="center", color=ink)
arrow([(3.5, 5.4), (4.6, 5.4)], color=green)
ax.text(4.05, 5.60, "covered", fontsize=10, ha="center", color=green)
arrow([(7.7, 5.4), (8.5, 5.4)], color=green)
ax.text(8.10, 5.72, "GPU copy\n/ concat", fontsize=9, ha="center", color=green)
arrow([(2.15, 5.0), (2.15, 4.0)], color=orange, dashed=True)
ax.text(2.35, 4.50, "miss /\nmissing part", fontsize=10, color=orange, va="center")
arrow([(3.5, 3.55), (4.6, 3.55)], color=orange)
arrow([(7.7, 3.55), (8.0, 3.55), (8.0, 4.65), (9.95, 4.65), (9.95, 5.0)], color=orange)
arrow([(9.95, 5.0), (9.95, 4.0)])
arrow([(11.4, 3.55), (11.85, 3.55), (11.85, 5.4), (12.3, 5.4)])
arrow([(8.5, 3.8), (8.2, 3.8), (8.2, 4.35), (6.15, 4.35), (6.15, 5.0)], color=green, dashed=True)
ax.text(6.15, 4.50, "admitted decoded pages", fontsize=10, ha="center", color=green)
arrow([(13.75, 5.8), (13.75, 6.75), (11.7, 6.75), (11.7, 7.2)])
ax.text(14.0, 6.35, "scan output", fontsize=10, color=ink)

box(.8, .95, 6.9, .85, "Optional HOST compressed-file-byte cache\nOFF in headline experiments", fill="#f5f5f5", edge=grey, dashed=True, size=13)
arrow([(2.15, 1.8), (2.15, 3.1)], color=grey, dashed=True)
ax.text(2.35, 2.17, "alternative byte source", fontsize=10, color=grey)
ax.text(8.5, 1.75, "Eviction discards decoded cache pages.", fontsize=12, color=ink)
ax.text(8.5, 1.30, "HOST backing for these pages: planned.", fontsize=12, color=grey)
ax.text(8.5, .85, "Execution-batch HOST spilling is implemented separately.", fontsize=11, color=grey)
ax.text(.8, .32, "Partial hit: reuse covered pieces and read missing portions. Footer / metadata work and memory reservations are omitted.",
        fontsize=10, color=grey)

for suffix in ("svg", "pdf", "png"):
    fig.savefig(OUT / f"scan_cache_architecture_verified.{suffix}", dpi=160, facecolor="white")
plt.close(fig)
print(OUT / "scan_cache_architecture_verified.svg")
