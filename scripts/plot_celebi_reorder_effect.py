#!/usr/bin/env python3
"""Does the cache-aware query reorder contribute, or is it all the page cache?

The benchmark runner applies the reorder to every non-baseline condition, so a
cache-vs-baseline bar measures the CELEBI package -- cache AND reorder -- not the
cache alone. These charts run each caching condition twice, once in the queries'
arrival order and once reordered, so the two can be separated.

Hot scan latency only; execution 1 of every run is the cold pass and is dropped.

Usage:  pixi run -e duckdb-python python scripts/plot_celebi_reorder_effect.py
Output: experiment/figs/celebi_reorder_effect.png
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiment" / "figs" / "celebi_reorder_effect.png"
CB, TP = REPO / "experiment/cb_proper", REPO / "experiment/tp_proper"

GROUPS = ["fixed", "fixed + variable"]
# (tag without reorder, tag with reorder)
TAGS = [("nr_fixed", "fixed"), ("nr_fx_var", "fx_var")]
WORKLOADS = [("ClickBench 100M", CB), ("TPC-H SF50", TP)]

INK, MUTED, GRID = "#1a1c23", "#6b6e7d", "#e3e5ea"
NO_REORDER, REORDER = "#9aa0ad", "#0D9488"


def scan_seconds(bucket_csv: Path) -> float:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def series(root: Path, tag: str) -> tuple[float, float]:
    runs = [scan_seconds(root / f"{tag}_{r}" / "bucket.csv")
            for r in (1, 2, 3) if (root / f"{tag}_{r}" / "bucket.csv").exists()]
    if not runs:
        raise SystemExit(f"{root}/{tag}_*: no runs")
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    x = np.arange(len(GROUPS))
    w = 0.34

    for ax, (title, root) in zip(axes, WORKLOADS):
        ax.set_facecolor("white")
        plain = [series(root, nr) for nr, _ in TAGS]
        reord = [series(root, ro) for _, ro in TAGS]

        for offset, stats, color, label in ((-w / 2, plain, NO_REORDER, "arrival order"),
                                            (w / 2, reord, REORDER, "reordered")):
            means = [m for m, _ in stats]
            errs = [e for _, e in stats]
            bars = ax.bar(x + offset, means, w, color=color, zorder=3, label=label,
                          yerr=errs, capsize=3.5,
                          error_kw={"ecolor": INK, "elinewidth": 1.0, "capthick": 1.0, "zorder": 5})
            for bar, mean, err in zip(bars, means, errs):
                ax.text(bar.get_x() + bar.get_width() / 2, mean + err + max(means) * 0.02,
                        f"{mean:.1f}", ha="center", va="bottom",
                        fontsize=9.5, fontweight="600", color=INK, zorder=6)

        # What the reorder alone changed, per group.
        for i, (p, r) in enumerate(zip(plain, reord)):
            delta = (r[0] / p[0] - 1.0) * 100.0
            ax.text(x[i], max(p[0], r[0]) * 1.11, f"{delta:+.1f}%",
                    ha="center", va="bottom", fontsize=11.5, fontweight="700",
                    color=(INK if abs(delta) < 1.0 else ("#16704f" if delta < 0 else "#a83a17")),
                    zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(GROUPS, fontsize=10.5, color=INK)
        ax.set_ylim(0, max(m for m, _ in plain + reord) * 1.28)
        ax.tick_params(axis="y", labelsize=9.5, colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_title(title, fontsize=12.5, fontweight="600", color=INK, pad=12, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=MUTED)

    axes[0].set_ylabel("Scan latency per execution  (s)", fontsize=10.5, color=MUTED, labelpad=10)
    fig.suptitle("Cache-aware query reorder — effect on top of the page cache",
                 fontsize=14.5, fontweight="600", color=INK, x=0.007, ha="left", y=0.995)
    fig.text(0.007, 0.945,
             "Hot scan latency, 3 repeats × 3 executions; execution 1 dropped as cold. "
             "The percentage is what reordering changed.",
             fontsize=9, color=MUTED, ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    for title, root in WORKLOADS:
        print(f"  {title}")
        for (nr, ro), g in zip(TAGS, GROUPS):
            a, _ = series(root, nr)
            b, _ = series(root, ro)
            print(f"    {g:18s} arrival {a:7.2f} s   reordered {b:7.2f} s   {(b/a-1)*100:+5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
