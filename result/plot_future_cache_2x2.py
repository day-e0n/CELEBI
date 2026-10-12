#!/usr/bin/env python3
"""Original order / reorder x LRU / future-aware page cache, on TPC-H SF50 and ClickBench.

Future-aware: before each query the runner tells the page cache the rest of the current
execution's query order (scripts/future_cache_hints.py). "evict" evicts the column needed
furthest ahead first; "admit + evict" also refuses to cache a column no later query reads.

  TPC-H SF50 : 4 GB, worst arrival order, 3 runs x executions 2-3 -> experiment/FA_th_*
  ClickBench : 6 GB, natural order, 41 queries, q1 fixed, 2 runs   -> experiment/FA_cb_*
Prefetch off, O_DIRECT. Dashed line: baseline (page cache off). Error bars: standard deviation
over the samples (one per run and execution).

  pixi run -e duckdb-python python result/plot_future_cache_2x2.py
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
WORKLOADS = [("TPC-H SF50", "th", (1, 2, 3)), ("ClickBench", "cb", (1, 2))]
POLICIES = [("LRU", "", ps.NEUTRAL),
            ("future-aware\nevict", "evict", ps.DARK),
            ("future-aware\nadmit + evict", "admit", ps.ACCENT)]
ORDERS = [("original order", ""), ("reorder", "ro_")]


def samples(bench: str, cond: str, reps) -> list[float]:
    out = []
    for r in reps:
        per_exec: dict[int, float] = {}
        for line in open(EXP / f"FA_{bench}_{cond}_r{r}" / "runner.log"):
            m = re.search(r"exec=(\d+)/\d+ .* done wall=([\d.]+)s", line)
            if m and int(m[1]) >= 2:
                per_exec[int(m[1])] = per_exec.get(int(m[1]), 0.0) + float(m[2])
        out += per_exec.values()
    return out


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    width = 0.26
    for ax, (title, bench, reps) in zip(axes, WORKLOADS):
        base = statistics.mean(samples(bench, "base", reps))
        ax.axhline(base, color="#2f3b3a", linestyle="--", linewidth=0.8, zorder=2)
        ax.text(1.62, base, f"baseline\n{base:.1f}s", fontsize=6, va="center", ha="left")
        top = base
        for gi, (_, prefix) in enumerate(ORDERS):
            for pi, (label, policy, style) in enumerate(POLICIES):
                cond = prefix + (policy or "lru")
                v = samples(bench, cond, reps)
                mean, err = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
                x = gi + (pi - 1) * width
                ax.bar(x, mean, width * 0.92, yerr=err, error_kw=ps.ERRBAR, zorder=3,
                       label=label if gi == 0 else None, **style)
                ax.text(x, mean + err + base * 0.012, f"{100 * (mean / base - 1):+.1f}%",
                        ha="center", va="bottom", fontsize=5.8)
                print(f"{title:11} {cond:9} {mean:6.2f}s +- {err:.2f}  ({100 * (mean / base - 1):+.1f}%)")
                top = max(top, mean + err)
        ax.set_xticks(range(len(ORDERS)))
        ax.set_xticklabels([o[0] for o in ORDERS], fontsize=7)
        ax.set_xlim(-0.55, 1.95)
        ax.set_ylim(0, top * 1.22)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(title, fontsize=8.5, pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6.5, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"future_cache_2x2.{ext}", dpi=300)
    print(OUT / "future_cache_2x2.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
