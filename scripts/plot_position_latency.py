#!/usr/bin/env python3
"""Moving a query earlier costs it latency; moving it later pays.

x is how far the reorder shifted a query from its arrival position (positive =
pulled EARLIER), y is what that did to the query's own time. The cache fills as
the workload runs, so an early slot is a cold slot: a query pulled forward gives
up cache it would have found had it stayed put.

Point area is the query's own baseline scan cost, which is why the largest
points dominate the workload total.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

ARRIVAL = [13, 1, 2, 3, 11, 4, 6, 16, 10, 17, 12, 19, 21, 14, 18, 15, 22, 9, 7, 20, 5, 8]
REORDER = [22, 13, 18, 4, 12, 21, 5, 10, 3, 7, 15, 14, 6, 19, 17, 20, 16, 2, 11, 9, 8, 1]

PANELS = [("TPC-H SF50", "m2_tp50"), ("TPC-H SF100", "m2_tp100")]


def per_query(root: str, tag: str, column: str) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for path in glob.glob(str(ROOT / "experiment" / root / f"{tag}_*" / "bucket.csv")):
        for row in csv.DictReader(open(path)):
            if row["execution"] == "1":
                continue
            acc.setdefault(row["query"], []).append(float(row[column]))
    return {q: statistics.mean(v) for q, v in acc.items()}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2))
    for ax, (title, root) in zip(axes, PANELS):
        arrival = per_query(root, "nr_fx_var", "total_ms")
        reorder = per_query(root, "fx_var", "total_ms")
        cost = per_query(root, "base", "scan")
        xs, ys, sizes = [], [], []
        for qnum in range(1, 23):
            key = f"q{qnum}"
            if key not in arrival or key not in reorder:
                continue
            shift = (ARRIVAL.index(qnum) + 1) - (REORDER.index(qnum) + 1)
            xs.append(shift)
            ys.append((reorder[key] - arrival[key]) / 1000.0)
            sizes.append(6 + 44 * cost.get(key, 0.0) / max(cost.values()))
        ax.axhline(0, color="0.55", linewidth=0.6, zorder=2)
        ax.axvline(0, color="0.55", linewidth=0.6, zorder=2)
        ax.scatter(xs, ys, s=sizes, facecolor=ps.DARK["facecolor"],
                   edgecolor=ps.DARK["edgecolor"], linewidth=0.5, zorder=3, alpha=0.9)
        ax.set_xlabel("positions moved earlier  →", fontsize=7.6)
        ax.set_title(title, fontsize=8.5, pad=4)
        ps.finish(ax)
    axes[0].set_ylabel("Δ query time (s)", fontsize=8.5)
    fig.subplots_adjust(wspace=0.26)
    ps.save(fig, "fig_p1_position_latency")
    plt.close(fig)


if __name__ == "__main__":
    main()
