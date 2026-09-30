#!/usr/bin/env python3
"""ClickBench under the least favourable arrival order.

Queries arrive in the order that minimises the column overlap between neighbours
(scripts/cache_aware_query_reorder.worst_case_sequence), which is the order a
cache-aware reorder has the most to recover from. Each cache condition is run
twice -- once in that arrival order, once reordered -- so the cache's contribution
and the reorder's are separable.

`fixed` caches only fixed-width columns; STRING columns are left uncached and read
from parquet. `fixed + variable` adds them as variable-width pages.

Hot scan latency only; execution 1 of every run is the cold pass and is dropped.

Usage:  pixi run -e duckdb-python python scripts/plot_clickbench_worst_order.py
Output: experiment/figs/clickbench_worst_order.png
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiment" / "figs" / "clickbench_worst_order.png"
D = REPO / "experiment/cb_worst"

# (bar label, run tag, colour)
SERIES = [
    ("baseline", "base", "#9aa0ad"),
    ("fixed", "nr_fixed", "#5EAAA3"),
    ("fixed\n+ reorder", "fixed", "#0D9488"),
    ("fixed + variable", "nr_fx_var", "#D98055"),
    ("fixed + variable\n+ reorder", "fx_var", "#C2410C"),
]
INK, MUTED, GRID = "#1a1c23", "#6b6e7d", "#e3e5ea"


def scan_seconds(bucket_csv: Path) -> float:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def main() -> int:
    means, errs = [], []
    for _, tag, _ in SERIES:
        runs = [scan_seconds(D / f"{tag}_{r}" / "bucket.csv")
                for r in (1, 2, 3) if (D / f"{tag}_{r}" / "bucket.csv").exists()]
        if not runs:
            raise SystemExit(f"{tag}: no runs")
        means.append(statistics.mean(runs))
        errs.append(statistics.stdev(runs) if len(runs) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(range(len(SERIES)), means, width=0.62,
                  color=[c for _, _, c in SERIES], zorder=3,
                  yerr=errs, capsize=4,
                  error_kw={"ecolor": INK, "elinewidth": 1.1, "capthick": 1.1, "zorder": 5})
    for i, (bar, mean, err) in enumerate(zip(bars, means, errs)):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + err + max(means) * 0.022,
                f"{mean:.1f} s", ha="center", va="bottom",
                fontsize=11, fontweight="600", color=INK, zorder=6)
        if i:
            pct = (mean / means[0] - 1.0) * 100.0
            ax.text(bar.get_x() + bar.get_width() / 2, mean / 2, f"{pct:+.1f}%",
                    ha="center", va="center", fontsize=12.5, fontweight="700",
                    color="white", zorder=6)

    ax.set_xticks(range(len(SERIES)))
    ax.set_xticklabels([label for label, _, _ in SERIES], fontsize=9.5, color=INK)
    ax.set_ylabel("Scan latency per execution  (s)", fontsize=10.5, color=MUTED, labelpad=10)
    ax.set_ylim(0, max(means) * 1.22)
    ax.tick_params(axis="y", labelsize=9.5, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=7)

    ax.set_title("ClickBench 100M — least favourable arrival order", fontsize=14,
                 fontweight="600", color=INK, pad=30, loc="left")
    ax.text(0, 1.035,
            "Adjacent queries share as few columns as possible. "
            "Hot scan latency, 3 repeats × 3 executions; execution 1 dropped as cold.",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")

    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    for (label, _, _), m, e in zip(SERIES, means, errs):
        print(f"  {label.replace(chr(10), ' '):30s} {m:7.2f} s  ±{e:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
