#!/usr/bin/env python3
"""ClickBench 100M: what the page cache is worth, and what re-keying it added.

Three conditions, hot scan latency only (execution 1 of every run is the cold
pass and is dropped). The middle bar is the cache as it was keyed by projection --
one entry per column set. The right bar is the same cache keyed by (file, filter),
so every projection over a table shares one entry.

Usage:  pixi run -e duckdb-python python scripts/plot_clickbench_colkey.py
Output: experiment/figs/clickbench_colkey_scan.png
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
OUT = REPO / "experiment" / "figs" / "clickbench_colkey_scan.png"

# (label, [run directories]) in the order they should appear.
CONDITIONS = [
    ("Baseline\n(no caching)",
     [REPO / "experiment/clickbench_colkey" / f"base_{r}" for r in (1, 2, 3)]),
    ("Page cache\nkeyed by projection",
     [REPO / "experiment/clickbench_colkey" / f"proj_{r}" for r in (1, 2, 3)]),
    ("Page cache\nkeyed by (file, filter)",
     [REPO / "experiment/clickbench_widen" / f"ck6_{r}" for r in (1, 2, 3)]),
]

INK, MUTED, GRID = "#1a1c23", "#6b6e7d", "#e3e5ea"
# Reference bar stays neutral; the two cache conditions take the validated
# categorical pair (CVD dE 13.7 deutan, normal 27.1, contrast >= 3:1).
BARS = ["#9aa0ad", "#0D9488", "#C2410C"]


def scan_seconds(bucket_csv: Path) -> float:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def main() -> int:
    means, errs = [], []
    for _, dirs in CONDITIONS:
        runs = [scan_seconds(d / "bucket.csv") for d in dirs]
        means.append(statistics.mean(runs))
        errs.append(statistics.stdev(runs) if len(runs) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(range(len(CONDITIONS)), means, width=0.56, color=BARS, zorder=3,
                  yerr=errs, capsize=5,
                  error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2, "zorder": 5})

    for i, (bar, mean, err) in enumerate(zip(bars, means, errs)):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + err + 1.2, f"{mean:.1f} s",
                ha="center", va="bottom", fontsize=12.5, fontweight="600", color=INK, zorder=6)
        if i:  # a delta against the reference bar is meaningless on the bar itself
            pct = (mean / means[0] - 1.0) * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, mean / 2, f"{pct:+.1f}%",
                    ha="center", va="center", fontsize=15, fontweight="700",
                    color="white", zorder=6)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([label for label, _ in CONDITIONS], fontsize=10.5, color=INK)
    ax.set_ylabel("Scan latency per execution  (s)", fontsize=11, color=MUTED, labelpad=10)
    ax.set_ylim(0, max(means) * 1.22)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=8)

    ax.set_title("ClickBench 100M — hot scan latency", fontsize=14.5, fontweight="600",
                 color=INK, pad=30, loc="left")
    ax.text(0, 1.035, "37 queries × 3 executions × 3 repeats; execution 1 dropped as cold",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")

    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    for (label, _), m, e in zip(CONDITIONS, means, errs):
        print(f"  {label.replace(chr(10), ' '):34s} {m:7.2f} s  ±{e:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
