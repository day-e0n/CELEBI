#!/usr/bin/env python3
"""baseline / paging / paging + reorder on ClickBench and TPC-H SF50 -- headline, v3.

v3: engine changes common to every condition, baseline included -- late materialization on by
default (ClickBench q24 reads ten rows' columns instead of all 105 x 100M), and scan tasks with
no memory history reserve 3x their uncompressed input instead of 8x (the excess had the page
cache declining admission). Runs experiment/A_* on GPU 1 (2026-10-07).

Same look as scripts/plot_paging_and_reorder.py (paper_style), but the metric is end-to-end
wall clock: the sum over queries of con.execute(...).fetchall(), not summed GPU operator time.
Each sample is one execution's total; execution 1 is warmup and dropped. Prefetch is off.

  ClickBench : 41 queries (q28, q29 excluded), 6 GB budget, natural order,
               2 runs x executions 2-3           -> experiment/A_cb_{base,lru,best}_r{1,2}
  TPC-H SF50 : 22 queries, 4 GB budget, worst arrival order,
               3 runs x executions 2-3           -> experiment/A_th_{base,lru,best}_r{1,2,3}
Error bars are the standard deviation over those samples.
"paging + reorder" is the reordered batch with order-aware eviction: the cache is told the
batch's remaining query order and evicts the column needed furthest ahead first.

  pixi run -e duckdb-python python result/plot_headline_cache_reorder_v3.py
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

EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent

WORKLOADS = [
    ("ClickBench", [("baseline", [f"A_cb_base_r{r}" for r in (1, 2)], ps.NEUTRAL),
                    ("paging", [f"A_cb_lru_r{r}" for r in (1, 2)], ps.DARK),
                    ("paging\n+ reorder", [f"A_cb_best_r{r}" for r in (1, 2)], ps.ACCENT)]),
    ("TPC-H SF50", [("baseline", [f"A_th_base_r{r}" for r in (1, 2, 3)], ps.NEUTRAL),
                    ("paging", [f"A_th_lru_r{r}" for r in (1, 2, 3)], ps.DARK),
                    ("paging\n+ reorder", [f"A_th_best_r{r}" for r in (1, 2, 3)], ps.ACCENT)]),
]


def samples(runs: list[str]) -> list[float]:
    """One total wall time per (run, execution >= 2)."""
    out = []
    for run in runs:
        per_exec: dict[int, float] = {}
        for line in open(EXP / run / "runner.log"):
            m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
            if m and int(m[1]) >= 2:
                per_exec[int(m[1])] = per_exec.get(int(m[1]), 0.0) + float(m[2])
        out += per_exec.values()
    return out


def main() -> int:
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(6.2, 2.6))
    for ax, (workload, conditions) in zip(axes, WORKLOADS):
        stats = []
        for label, runs, style in conditions:
            v = samples(runs)
            stats.append((label, statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0,
                          style))
        reference = stats[0][1]
        for x, (label, value, error, style) in enumerate(stats):
            ax.bar(x, value, 0.6, yerr=error, error_kw=ps.ERRBAR, zorder=3, **style)
            caption = f"{value:.1f}s" if x == 0 else \
                f"{value:.1f}s\n{100 * (value - reference) / reference:+.1f}%"
            ax.text(x, value + error + value * 0.015, caption, ha="center", va="bottom",
                    fontsize=6.5)
            print(f"{workload:11} {label.replace(chr(10), ' '):17} {value:6.2f}s +- {error:.2f}")
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels([s[0] for s in stats], fontsize=7)
        ax.set_ylim(0, max(s[1] for s in stats) * 1.30)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(workload, fontsize=8.5, pad=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"headline_cache_reorder_v3.{ext}", dpi=300)
    print(OUT / "headline_cache_reorder_v3.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
