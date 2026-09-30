#!/usr/bin/env python3
"""Per-query GPU time on ClickBench: baseline, paging, paging + reorder.

Queries on the x axis in their own numbering (q2..q43, minus the six the GPU path
cannot run), not in the order they arrived -- so the three series can be read
against each other query by query. Each bar is the mean of two repeats, each the
mean of that query's two warm executions.

  pixi run -e duckdb-python python scripts/plot_clickbench_per_query.py
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


def per_query(prefix: str) -> dict[str, float]:
    """Mean warm ms per query, averaged over repeats."""
    runs: list[dict[str, float]] = []
    for bucket in sorted(glob.glob(str(ROOT / f"{prefix}_*" / "bucket.csv"))):
        by_query: dict[str, list[float]] = collections.defaultdict(list)
        for row in csv.DictReader(open(bucket)):
            if row["execution"] != "1":
                by_query[row["query"]].append(float(row["total_ms"]))
        if by_query:
            runs.append({q: statistics.mean(v) for q, v in by_query.items()})
    if not runs:
        raise SystemExit(f"no runs for {prefix}")
    queries = set.intersection(*(set(r) for r in runs))
    return {q: statistics.mean(r[q] for r in runs) for q in queries}


def main() -> int:
    series = [("baseline", "cbf_base", ps.NEUTRAL),
              ("paging", "ro_cb_noreorder", ps.DARK),
              ("paging + reorder", "ro_cb_reorder", ps.ACCENT)]
    data = [(label, per_query(prefix), style) for label, prefix, style in series]
    queries = sorted(set.intersection(*(set(d) for _, d, _ in data)),
                     key=lambda q: int(q[1:]))

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    xs = np.arange(len(queries))
    width = 0.27
    for i, (label, values, style) in enumerate(data):
        ax.bar(xs + (i - 1) * width, [values[q] / 1000.0 for q in queries], width,
               label=label, zorder=3, **style)

    ax.set_xticks(xs)
    ax.set_xticklabels(queries, fontsize=6, rotation=90)
    ax.set_xlim(-0.6, len(queries) - 0.4)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ps.finish(ax, "GPU time (s)")
    ax.set_title("ClickBench per-query GPU time", fontsize=8.5, pad=4)
    fig.tight_layout()
    ps.save(fig, "fig_clickbench_per_query")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
