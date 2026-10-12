#!/usr/bin/env python3
"""baseline / paging / paging + reorder / paging + reorder + future-aware eviction.

Same look as plot_headline_cache_reorder.py, from the 2x2 runs (experiment/FA_*):
  TPC-H SF50 : 22 queries, 4 GB, worst arrival order, 3 runs x executions 2-3
  ClickBench : 41 queries (q1 fixed to read one column), 6 GB, natural order, 2 runs
Future-aware: the runner tells the page cache the rest of the execution's query order and the
cache evicts the column needed furthest ahead first. Prefetch off, O_DIRECT, end-to-end wall.

  pixi run -e duckdb-python python result/plot_headline_future.py
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
FUTURE = dict(facecolor="#7a3a22", edgecolor="#2f3b3a", linewidth=0.6)
WORKLOADS = [("ClickBench", "cb", (1, 2)), ("TPC-H SF50", "th", (1, 2, 3))]
CONDS = [("baseline", "base", ps.NEUTRAL), ("paging", "lru", ps.DARK),
         ("paging\n+ reorder", "ro_lru", ps.ACCENT),
         ("paging + reorder\n+ future-aware", "ro_evict", FUTURE)]


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
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(7.0, 2.7))
    for ax, (title, bench, reps) in zip(axes, WORKLOADS):
        stats = []
        for label, cond, style in CONDS:
            v = samples(bench, cond, reps)
            stats.append((label, statistics.mean(v), statistics.stdev(v), style))
        ref = stats[0][1]
        for x, (label, value, err, style) in enumerate(stats):
            ax.bar(x, value, 0.62, yerr=err, error_kw=ps.ERRBAR, zorder=3, **style)
            caption = f"{value:.1f}s" if x == 0 else f"{value:.1f}s\n{100 * (value / ref - 1):+.1f}%"
            ax.text(x, value + err + value * 0.015, caption, ha="center", va="bottom", fontsize=6.3)
            print(f"{title:11} {label.replace(chr(10), ' '):34} {value:6.2f}s +- {err:.2f}")
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels([s[0] for s in stats], fontsize=6.5)
        ax.set_ylim(0, max(s[1] for s in stats) * 1.30)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(title, fontsize=8.5, pad=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"headline_future.{ext}", dpi=300)
    print(OUT / "headline_future.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
