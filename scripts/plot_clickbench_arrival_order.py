#!/usr/bin/env python3
"""What the page cache and the cache-aware reorder are worth on ClickBench.

One arrival order -- the adversarial order these experiments have used all along,
10.3% adjacent-query column overlap against the benchmark's natural 36.3% -- run
three ways: cache off, page cache on, page cache plus the reorder that groups
queries sharing columns. Every bar is the mean of two repeats, each the mean of
two warm executions (execution 1 discarded).

  pixi run -e duckdb-python python scripts/plot_clickbench_arrival_order.py
"""
from __future__ import annotations

import collections
import csv
import glob
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiment"
NQ = 37


def condition(prefix: str) -> tuple[float, float, int]:
    """(mean, stdev, repeats) seconds of GPU operator time per warm execution."""
    runs = []
    for bucket in sorted(glob.glob(str(ROOT / f"{prefix}_*" / "bucket.csv"))):
        per: dict[str, float] = collections.defaultdict(float)
        seen: dict[str, set] = collections.defaultdict(set)
        for row in csv.DictReader(open(bucket)):
            per[row["execution"]] += float(row["total_ms"])
            seen[row["execution"]].add(row["query"])
        warm = [e for e in per if e != "1" and len(seen[e]) == NQ]
        if warm:
            runs.append(statistics.mean(per[e] for e in warm) / 1000.0)
    if not runs:
        raise SystemExit(f"no runs for {prefix}")
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0), len(runs)


def main() -> int:
    bars = [("baseline\n(no cache)", "cbf_base", ps.NEUTRAL),
            ("page cache", "ro_cb_noreorder", ps.DARK),
            ("page cache\n+ reorder", "ro_cb_reorder", ps.ACCENT)]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    xs = np.arange(len(bars))
    baseline = condition(bars[0][1])[0]
    for x, (label, prefix, style) in zip(xs, bars):
        value, error, n = condition(prefix)
        ax.bar(x, value, 0.56, yerr=error, error_kw=ps.ERRBAR, zorder=3, **style)
        caption = f"{value:.1f}s" if x == 0 else \
            f"{value:.1f}s\n{100 * (value - baseline) / baseline:+.0f}%"
        ax.text(x, value + error + 1.0, caption, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _, _ in bars], fontsize=7.5)
    ax.set_ylim(0, 75)
    ps.finish(ax, "query time (s)")
    ax.set_title("ClickBench, adversarial arrival order", fontsize=8.5, pad=4)
    fig.tight_layout()
    ps.save(fig, "fig_clickbench_arrival_order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
