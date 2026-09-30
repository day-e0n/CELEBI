#!/usr/bin/env python3
"""CELEBI page cache on two workloads, hot scan latency.

Three conditions per workload -- no cache, fixed-width page cache + reorder, and
that plus the variable-width (STRING) page cache. Execution 1 of every run is the
cold pass and is dropped; only the `scan` bucket is plotted, since the page cache
can only move that one.

Both workloads run with the cache keyed by (file, filter) rather than by
projection -- see scripts/plot_clickbench_colkey.py for what that key change alone
is worth.

Usage:  pixi run -e duckdb-python python scripts/plot_celebi_two_workloads.py
Output: experiment/figs/celebi_two_workloads.png
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiment" / "figs" / "celebi_two_workloads.png"

CB = REPO / "experiment/clickbench_3cond"
TP = REPO / "experiment/tpch50_colkey"

WORKLOADS = [
    ("ClickBench 100M\n1 table · 105 columns · no joins",
     [("Baseline", [CB / f"base_{r}" for r in (1, 2, 3)]),
      ("Fixed-width cache", [CB / f"f_{r}" for r in (1, 2, 3)]),
      ("+ variable-width", [CB / f"fv_{r}" for r in (1, 2, 3)])]),
    ("TPC-H SF50\n8 tables · joins throughout",
     [("Baseline", [TP / f"base_{r}" for r in (1, 2, 3)]),
      ("Fixed-width cache", [TP / f"f_ck_{r}" for r in (1, 2, 3)]),
      ("+ variable-width", [TP / f"fv_ck_{r}" for r in (1, 2, 3)])]),
]

INK, MUTED, GRID = "#1a1c23", "#6b6e7d", "#e3e5ea"
BARS = ["#9aa0ad", "#0D9488", "#C2410C"]


def scan_seconds(bucket_csv: Path) -> float:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), dpi=200)
    fig.patch.set_facecolor("white")

    for ax, (title, conditions) in zip(axes, WORKLOADS):
        ax.set_facecolor("white")
        means, errs = [], []
        for _, dirs in conditions:
            runs = [scan_seconds(d / "bucket.csv") for d in dirs]
            means.append(statistics.mean(runs))
            errs.append(statistics.stdev(runs) if len(runs) > 1 else 0.0)

        bars = ax.bar(range(len(conditions)), means, width=0.58, color=BARS, zorder=3,
                      yerr=errs, capsize=4,
                      error_kw={"ecolor": INK, "elinewidth": 1.1, "capthick": 1.1, "zorder": 5})
        for i, (bar, mean, err) in enumerate(zip(bars, means, errs)):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + err + max(means) * 0.025,
                    f"{mean:.1f} s", ha="center", va="bottom",
                    fontsize=11.5, fontweight="600", color=INK, zorder=6)
            if i:
                pct = (mean / means[0] - 1.0) * 100.0
                ax.text(bar.get_x() + bar.get_width() / 2, mean / 2, f"{pct:+.1f}%",
                        ha="center", va="center", fontsize=14, fontweight="700",
                        color="white", zorder=6)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([label for label, _ in conditions], fontsize=10, color=INK)
        ax.set_ylim(0, max(means) * 1.24)
        ax.tick_params(axis="y", labelsize=9.5, colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_title(title, fontsize=12, fontweight="600", color=INK, pad=12, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].set_ylabel("Scan latency per execution  (s)", fontsize=10.5, color=MUTED, labelpad=10)
    fig.suptitle("CELEBI page cache — hot scan latency", fontsize=14.5, fontweight="600",
                 color=INK, x=0.008, ha="left", y=0.995)
    fig.text(0.008, 0.945,
             "3 repeats × 3 executions per condition; execution 1 dropped as cold. "
             "Error bars are the spread across repeats.",
             fontsize=9, color=MUTED, ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
