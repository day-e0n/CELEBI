#!/usr/bin/env python3
"""What the GPU page cache and the reorder are worth, on two workloads.

baseline is the same engine with the cache off; paging turns the page cache on;
paging + reorder adds the unfiltered-first reorder, which runs the queries that
scan without a predicate first and discounts a shared column when the query that
would leave it in the cache read it through one. Everything else is identical --
the same adversarial arrival order, 6 GB fixed cache, 6 GB free-memory floor.

Why the reorder weighs predicates rather than column names: names alone treat
`l_shipdate < 1995` and `l_shipdate >= 1998` as a full overlap even though neither
query can use a row the other cached. On TPC-H, where the same table is read
through many different date ranges, a name-only reorder is a net loss (87.3s
against 85.8s with no reorder); weighing the predicate turns it into a gain.

  pixi run -e duckdb-python python scripts/plot_paging_and_reorder.py
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

# (workload label, queries per execution, [(condition label, run prefix, style)])
WORKLOADS = [
    ("ClickBench", 37, [("baseline", "mb_cb", ps.NEUTRAL),
                        ("paging", "fin_cb_paging", ps.DARK),
                        ("paging\n+ reorder", "fin_cb_reorder", ps.ACCENT)]),
    ("TPC-H SF50", 22, [("baseline", "mb_sf50", ps.NEUTRAL),
                        ("paging", "fin_th_paging", ps.DARK),
                        ("paging\n+ reorder", "fin_th_reorder", ps.ACCENT)]),
]


def runs(prefix: str, n_queries: int, metric: str = "total_ms") -> list[dict[str, float]]:
    """Per-query warm ms for each repeat of one condition.

    `metric` picks the bucket column: "total_ms" is every GPU operator, "scan" is
    the read path alone. The cache only ever shortens the scan, so a per-query
    view of total time buries its effect under join and aggregate work that no
    cache can touch.
    """
    out = []
    for bucket in sorted(glob.glob(str(ROOT / f"{prefix}_*" / "bucket.csv"))):
        by_query: dict[str, list[float]] = collections.defaultdict(list)
        executions: dict[str, set] = collections.defaultdict(set)
        for row in csv.DictReader(open(bucket)):
            if row["execution"] == "1":
                continue
            by_query[row["query"]].append(float(row[metric]))
            executions[row["execution"]].add(row["query"])
        if len(by_query) == n_queries and executions:
            out.append({q: statistics.mean(v) for q, v in by_query.items()})
    if not out:
        raise SystemExit(f"no complete runs for {prefix}")
    return out


def total(prefix: str, n_queries: int) -> tuple[float, float, int]:
    totals = [sum(r.values()) / 1000.0 for r in runs(prefix, n_queries)]
    return (statistics.mean(totals),
            statistics.stdev(totals) if len(totals) > 1 else 0.0,
            len(totals))


def per_query(prefix: str, n_queries: int, metric: str = "scan") -> dict[str, float]:
    repeats = runs(prefix, n_queries, metric)
    queries = set.intersection(*(set(r) for r in repeats))
    return {q: statistics.mean(r[q] for r in repeats) for q in queries}


def plot_totals() -> None:
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(6.2, 2.6))
    for ax, (workload, n_queries, conditions) in zip(axes, WORKLOADS):
        xs = np.arange(len(conditions))
        reference = None
        for x, (label, prefix, style) in zip(xs, conditions):
            value, error, n = total(prefix, n_queries)
            if reference is None:
                reference = value
            ax.bar(x, value, 0.6, yerr=error, error_kw=ps.ERRBAR, zorder=3, **style)
            caption = f"{value:.1f}s" if x == 0 else \
                f"{value:.1f}s\n{100 * (value - reference) / reference:+.1f}%"
            ax.text(x, value + error + max(value, 1) * 0.015, caption,
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _, _ in conditions], fontsize=7)
        top = max(total(p, n_queries)[0] for _, p, _ in conditions)
        ax.set_ylim(0, top * 1.30)
        ps.finish(ax, "query time (s)" if ax is axes[0] else None)
        ax.set_title(workload, fontsize=8.5, pad=4)
    fig.tight_layout()
    ps.save(fig, "fig_paging_and_reorder_total")


def plot_per_query() -> None:
    fig, axes = plt.subplots(len(WORKLOADS), 1, figsize=(7.0, 4.6))
    for ax, (workload, n_queries, conditions) in zip(axes, WORKLOADS):
        data = [(label, per_query(prefix, n_queries), style)
                for label, prefix, style in conditions]
        queries = sorted(set.intersection(*(set(d) for _, d, _ in data)),
                         key=lambda q: int(q[1:]))
        xs = np.arange(len(queries))
        width = 0.27
        for i, (label, values, style) in enumerate(data):
            ax.bar(xs + (i - 1) * width, [values[q] / 1000.0 for q in queries], width,
                   label=label, zorder=3, **style)
        ax.set_xticks(xs)
        ax.set_xticklabels(queries, fontsize=6, rotation=90)
        ax.set_xlim(-0.6, len(queries) - 0.4)
        ps.finish(ax, "scan time (s)")
        ax.set_title(workload, fontsize=8.5, pad=4)
        if ax is axes[0]:
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    ps.save(fig, "fig_paging_and_reorder_per_query")


def main() -> int:
    plot_totals()
    plot_per_query()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
