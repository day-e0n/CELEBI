#!/usr/bin/env python3
"""TPC-H SF100, 6GB page cache, prefetch off: per-query wall for baseline / cache / cache+reorder.

Reads experiment/H_{worst,natural}_{base,cache,reorder}/runner.log, averages executions 2-3
(execution 1 is warmup), writes sf100_per_query.csv and sf100_per_query.png next to this file.
H_worst_cache hit one OOM in execution 3 q13 that dropped 4.7GiB of page cache.
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1] / "experiment"
OUT = Path(__file__).resolve().parent
ORDERS = ["worst", "natural"]
CONDS = [("base", "Baseline (no cache)", "#b4b2ab"),
         ("cache", "Page cache", "#86b6ef"),
         ("reorder", "Page cache + reorder", "#2a78d6")]


def per_query(run):
    walls = {}
    for line in open(EXP / run / "runner.log"):
        m = re.search(r"exec=(\d+)/\d+ .* q(\d+) done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            walls.setdefault(int(m[2]), []).append(float(m[3]))
    return {q: sum(v) / len(v) for q, v in walls.items()}


data = {(o, c): per_query(f"H_{o}_{c}") for o in ORDERS for c, *_ in CONDS}
queries = list(range(1, 23))

with open(OUT / "sf100_per_query.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["order", "query", "base_s", "cache_s", "reorder_s",
                "cache_vs_base_pct", "reorder_vs_base_pct", "reorder_vs_cache_pct"])
    for o in ORDERS:
        for q in queries:
            b, c, r = (data[(o, k)][q] for k in ("base", "cache", "reorder"))
            w.writerow([o, f"q{q}", f"{b:.3f}", f"{c:.3f}", f"{r:.3f}",
                        f"{(c / b - 1) * 100:+.1f}", f"{(r / b - 1) * 100:+.1f}",
                        f"{(r / c - 1) * 100:+.1f}"])

fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), dpi=170, sharey=True)
bg = "#fcfcfb"
fig.patch.set_facecolor(bg)
width = 0.27
for ax, o in zip(axes, ORDERS):
    ax.set_facecolor(bg)
    for i, (c, label, color) in enumerate(CONDS):
        xs = [q - 1 + (i - 1) * width for q in queries]
        ax.bar(xs, [data[(o, c)][q] for q in queries], width=width, color=color,
               edgecolor=bg, linewidth=1, zorder=2, label=label)
    tot = {c: sum(data[(o, c)].values()) for c, *_ in CONDS}
    ax.set_title(f"{o} arrival order -- total: baseline {tot['base']:.1f}s, cache {tot['cache']:.1f}s "
                 f"({(tot['cache'] / tot['base'] - 1) * 100:+.1f}%), cache + reorder {tot['reorder']:.1f}s "
                 f"({(tot['reorder'] / tot['base'] - 1) * 100:+.1f}%)",
                 fontsize=10, color="#0b0b0b", loc="left")
    ax.set_xticks(range(22), [f"q{q}" for q in queries], fontsize=9, color="#52514e")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, colors="#898781", labelsize=9)
    ax.set_ylabel("Wall time (s)", fontsize=9, color="#52514e")
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#898781")
    ax.set_xlim(-0.6, 21.6)
axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
fig.suptitle("TPC-H SF100, 6 GB GPU page cache: per-query wall time", fontsize=11, x=0.01, ha="left")
fig.text(0.01, 0.005, "Lower is better. Prefetch OFF, executions 2-3 averaged. "
         "Worst-order cache run had one OOM (execution 3, q13) that dropped the page cache.",
         fontsize=7.5, color="#898781")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(OUT / "sf100_per_query.png", facecolor=bg)
print(OUT / "sf100_per_query.png")
