#!/usr/bin/env python3
"""Cache hit ratio as bars: page caching vs page caching + reordering (CELEBI).

Hit ratio = decoded bytes served from cached pages / decoded bytes the scans handed downstream
(before the filter), over every scan of executions 2-3. Data: result/cache_hit_rate.csv, the
HIT_* runs (2026-10-06, 1 run per condition; engine before the 2026-10-07 changes -- scan
no-history reservation 8x, late materialization off). TPC-H SF50 4 GB worst order;
ClickBench 6 GB natural order, 41 queries. Style of result/plot_query_time_and_cache_hit.py.

  pixi run -e duckdb-python python result/plot_cache_hit_bar.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent
CONDS = [("Baseline+page caching", "paging", "#59a14f"),
         ("Baseline+page caching+reordering (CELEBI)", "paging + reorder", "#f2b134")]
WORKLOADS = ["TPC-H SF50", "ClickBench"]
RC = {"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"}


def main() -> int:
    hit = {(r["workload"], r["condition"]): float(r["byte_hit_pct"])
           for r in csv.DictReader(open(OUT / "cache_hit_rate.csv"))}
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        width = 0.32
        for ci, (label, cond, color) in enumerate(CONDS):
            for wi, workload in enumerate(WORKLOADS):
                value = hit[(workload, cond)]
                x = wi + (ci - 0.5) * width
                ax.bar(x, value, width * 0.9, color=color, alpha=0.9, edgecolor=color,
                       linewidth=0.6, zorder=3, label=label if wi == 0 else None)
                ax.text(x, value + 0.6, f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5,
                        color="#333333")
        ax.set_xticks(range(len(WORKLOADS)), WORKLOADS, fontsize=8.5)
        ax.set_xlim(-0.6, len(WORKLOADS) - 0.4)
        ax.set_ylim(0, 32)
        ax.set_ylabel("Cache hit ratio (% of bytes)", fontsize=9)
        ax.tick_params(axis="both", length=2.5, width=0.5, labelsize=8)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.55)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.55)
        legend = fig.legend(fontsize=6.6, frameon=True, facecolor="white", edgecolor="#999999",
                            fancybox=False, framealpha=1, ncol=1, loc="lower center",
                            bbox_to_anchor=(0.55, 0.97), handlelength=1.2, handleheight=0.6,
                            borderpad=0.4)
        legend.get_frame().set_linewidth(0.55)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"cache_hit_bar.{ext}", dpi=300, facecolor="white",
                        bbox_inches="tight")
    for (w, c), v in sorted(hit.items()):
        if c in ("paging", "paging + reorder"):
            print(f"{w:11} {c:17} {v:.1f}%")
    print(OUT / "cache_hit_bar.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
