#!/usr/bin/env python3
"""
ClickBench cache / reorder comparison.

Conditions:
- ClickBench: 41 queries
- GPU page cache: 6 GB
- Prefetch: OFF
- Query order: natural order
- Wall time: executions 2-3 averaged

Measured values:
- Baseline: 38.69 s
- Page cache: 35.83 s
- Page cache + reorder: 34.28 s
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

RUNS = [
    ("Baseline\n(no cache)", 38.69, "#b4b2ab"),
    ("Page cache", 35.83, "#86b6ef"),
    ("Page cache\n+ reorder", 34.28, "#2a78d6"),
]

labels = [name for name, _, _ in RUNS]
vals = [value for _, value, _ in RUNS]
colors = [color for _, _, color in RUNS]

baseline = vals[0]


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(5.2, 4.0),
    dpi=200,
)

background = "#fcfcfb"

fig.patch.set_facecolor(background)
ax.set_facecolor(background)

x = range(len(RUNS))


# ----------------------------------------------------------------------
# Bars
# ----------------------------------------------------------------------

ax.bar(
    x,
    vals,
    width=0.56,
    color=colors,
    edgecolor=background,
    linewidth=2,
    zorder=2,
)


# ----------------------------------------------------------------------
# Value labels
# ----------------------------------------------------------------------

for i, v in enumerate(vals):

    if i == 0:
        text = f"{v:.2f}s"

    else:
        change = (v / baseline - 1) * 100

        text = (
            f"{v:.2f}s\n"
            f"({change:+.1f}%)"
        )

    ax.text(
        i,
        v + 0.55,
        text,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#0b0b0b",
    )


# ----------------------------------------------------------------------
# X axis
# ----------------------------------------------------------------------

ax.set_xticks(
    list(x),
    labels,
)

ax.tick_params(
    axis="x",
    length=0,
    labelsize=10,
    colors="#52514e",
)


# ----------------------------------------------------------------------
# Y axis
# ----------------------------------------------------------------------

ax.set_ylabel(
    "Total wall time (s)",
    fontsize=10,
    color="#52514e",
)

ax.set_ylim(
    0,
    max(vals) * 1.22,
)

ax.tick_params(
    axis="y",
    length=0,
    labelsize=9,
    colors="#898781",
)


# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------

ax.set_title(
    "ClickBench (41 queries), 6 GB GPU page cache",
    fontsize=11,
    color="#0b0b0b",
    loc="left",
    pad=10,
)


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------

ax.grid(
    axis="y",
    color="#e6e5e1",
    linewidth=0.8,
    zorder=0,
)


# ----------------------------------------------------------------------
# Spines
# ----------------------------------------------------------------------

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)

ax.spines["bottom"].set_color("#898781")


# ----------------------------------------------------------------------
# Footnote
# ----------------------------------------------------------------------

fig.text(
    0.01,
    0.01,
    "Lower is better. Prefetch OFF, natural query order, "
    "executions 2-3 averaged.",
    fontsize=7.5,
    color="#898781",
)


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

fig.tight_layout(
    rect=(0, 0.04, 1, 1)
)


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

out = Path(__file__).with_name(
    "clickbench_cache_reorder.png"
)

fig.savefig(
    out,
    facecolor=fig.get_facecolor(),
    bbox_inches="tight",
)

plt.close(fig)


# ----------------------------------------------------------------------
# Print result
# ----------------------------------------------------------------------

print(f"Saved: {out}")

for label, value, _ in RUNS:

    if value == baseline:
        print(
            f"{label.replace(chr(10), ' ')}: "
            f"{value:.2f}s"
        )

    else:
        change = (value / baseline - 1) * 100

        print(
            f"{label.replace(chr(10), ' ')}: "
            f"{value:.2f}s "
            f"({change:+.1f}%)"
        )