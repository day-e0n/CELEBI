#!/usr/bin/env python3
"""ClickBench 100M: baseline vs fixed+reorder vs fixed+variable+reorder.

Hot scan latency only -- execution 1 of every run is the cold pass and is dropped,
the rest are averaged per run, and the three runs give the error bar. The `scan`
bucket alone is plotted: the page cache can only move that one, and folding in
join/aggregate time buries a scan change under unrelated noise.

Usage:  pixi run -e duckdb-python python scripts/plot_clickbench_3cond.py
Output: experiment/figs/clickbench_3cond_scan.png
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
DATA = REPO / "experiment" / "clickbench_100m"
OUT = REPO / "experiment" / "figs" / "clickbench_3cond_scan.png"

# (directory prefix, axis label). Ordered baseline -> treatments.
CONDITIONS = [
    ("baseline", "Baseline\n(no caching)"),
    ("celebi_fixed", "CELEBI\nfixed + reorder"),
    ("celebi_fixed_variable", "CELEBI\nfixed + variable + reorder"),
]
REPEATS = (1, 2, 3)

INK = "#1a1c23"
MUTED = "#6b6e7d"
GRID = "#e3e5ea"
# Baseline is the reference, so it takes a neutral; the two treatments take the
# categorical pair (validated: CVD dE 13.7 deutan, normal 27.1, contrast >= 3:1).
BARS = ["#9aa0ad", "#0D9488", "#C2410C"]


def scan_ms_per_execution(bucket_csv: Path) -> float:
    """Total scan ms for one warm execution of the whole query set."""
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec


def main() -> int:
    means, errs = [], []
    for prefix, _ in CONDITIONS:
        runs = [scan_ms_per_execution(DATA / f"{prefix}_{r}" / "bucket.csv") for r in REPEATS]
        means.append(statistics.mean(runs) / 1000.0)          # -> seconds
        errs.append(statistics.stdev(runs) / 1000.0 if len(runs) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = range(len(CONDITIONS))
    bars = ax.bar(x, means, width=0.56, color=BARS, zorder=3,
                  yerr=errs, capsize=5,
                  error_kw={"ecolor": INK, "elinewidth": 1.2, "capthick": 1.2, "zorder": 5})

    # Direct labels: only 3 bars, so labelling each is selective, not noise.
    for i, (bar, mean, err) in enumerate(zip(bars, means, errs)):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + err + 1.15,
                f"{mean:.1f} s", ha="center", va="bottom",
                fontsize=12.5, fontweight="600", color=INK, zorder=6)
        if i:  # deltas are meaningless on the reference bar
            pct = (mean / means[0] - 1.0) * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, mean / 2,
                    f"{pct:+.1f}%", ha="center", va="center",
                    fontsize=15, fontweight="700", color="white", zorder=6)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in CONDITIONS], fontsize=10.5, color=INK)
    ax.set_ylabel("Scan latency per execution  (s)", fontsize=11, color=MUTED, labelpad=10)
    ax.set_ylim(0, max(means) * 1.22)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=8)

    ax.set_title("ClickBench 100M — hot scan latency", fontsize=14.5,
                 fontweight="600", color=INK, pad=30, loc="left")
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
    for (_, label), m, e in zip(CONDITIONS, means, errs):
        print(f"  {label.replace(chr(10), ' '):34s} {m:7.2f} s  ±{e:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
