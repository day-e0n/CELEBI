#!/usr/bin/env python3
"""Lazy batching: total time vs batch size N (queries collected before a batch runs).

The workload is split at random into batches of about N queries; each batch keeps the arrival
order, is reordered on its own, and the page cache is told only the rest of the current batch
(order-aware eviction). Batches run back to back in one process, so the cache carries over.
Every split has its own baseline run. Prefetch off, O_DIRECT, executions 2-3, end-to-end wall.

  TPC-H SF50 (4 GB, worst arrival order): N=22 experiment/FA_th_* (3 runs),
                                          N=11 / 6 experiment/NS_th_b{2,4}_s{1,2}_*
  ClickBench (6 GB, natural order, 41 q): N=41 experiment/FA_cb_* (2 runs),
                                          N=21 / 10 / 5 experiment/NS_cb_b{2,4,8}_s{1,2}_*,
                                          N=14 experiment/B3_s{1,2,3}_*

  pixi run -e duckdb-python python result/plot_batch_size_sweep.py
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent
# "paging + reorder" is the reordered batch with order-aware eviction.
CONDS = [("baseline", "base", ps.NEUTRAL), ("paging", "lru", ps.DARK),
         ("paging + reorder", "ro_evict", ps.ACCENT)]


def runs(bench: str, batches: int, cond: str) -> list[str]:
    if batches == 1:
        reps = (1, 2, 3) if bench == "th" else (1, 2)
        return [f"FA_{bench}_{cond}_r{r}" for r in reps]
    if bench == "cb" and batches == 3:
        return [f"B3_s{s}_{cond}" for s in (1, 2, 3)]
    return [f"NS_{bench}_b{batches}_s{s}_{cond}" for s in (1, 2)]


def total(run: str) -> float | None:
    path = EXP / run / "runner.log"
    if not path.exists():
        return None
    per_exec: dict[int, float] = {}
    for line in open(path):
        m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
        if m and int(m[1]) >= 2:
            per_exec[int(m[1])] = per_exec.get(int(m[1]), 0.0) + float(m[2])
    return sum(per_exec.values()) / len(per_exec) if per_exec else None


WORKLOADS = [("TPC-H SF50", "th", 22, (1, 2, 4)), ("ClickBench", "cb", 41, (1, 2, 3, 4, 8))]


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [3, 5]})
    width = 0.26
    for ax, (title, bench, n_queries, batch_counts) in zip(axes, WORKLOADS):
        labels = []
        for gi, b in enumerate(batch_counts):
            n = round(n_queries / b)
            labels.append(f"N={n}" + (" (all)" if b == 1 else ""))
            base_vals = [v for v in (total(r) for r in runs(bench, b, "base")) if v]
            base = statistics.mean(base_vals) if base_vals else None
            for ci, (label, cond, style) in enumerate(CONDS):
                vals = [v for v in (total(r) for r in runs(bench, b, cond)) if v]
                if not vals:
                    continue
                mean = statistics.mean(vals)
                err = statistics.stdev(vals) if len(vals) > 1 else 0.0
                x = gi + (ci - 1) * width
                ax.bar(x, mean, width * 0.92, yerr=err, error_kw=ps.ERRBAR, zorder=3,
                       label=label if gi == 0 else None, **style)
                if cond != "base" and base:
                    ax.text(x, mean + err + 0.4, f"{100 * (mean / base - 1):+.0f}%", ha="center",
                            va="bottom", fontsize=5.2, rotation=90)
                print(f"{title:11} batches={b} N={n:2} {cond:9} {mean:6.2f}s "
                      f"({len(vals)} runs){'' if base is None or cond == 'base' else f'  {100 * (mean / base - 1):+.1f}%'}")
        ax.set_xticks(np.arange(len(batch_counts)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 42)
        ax.set_xlabel("batch size (queries per batch)", fontsize=7)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(title, fontsize=8.5, pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"batch_size_sweep.{ext}", dpi=300)
    print(OUT / "batch_size_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
