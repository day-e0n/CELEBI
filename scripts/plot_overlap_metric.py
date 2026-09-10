#!/usr/bin/env python3
"""Does weighting the overlap metric by BYTES beat counting columns?

CELEBI's reorder maximises the number of (table, column) pairs two adjacent
queries share. The page cache manages bytes, and on TPC-H SF100 the two are not
close: `nation.n_nationkey` is 100 B and `lineitem.l_orderkey` is 4.47 GB
decoded, a factor of 45,000, and the count-based metric scores them identically.

Measured against an adversarial arrival order -- adjacent overlap minimised and
seeded with the NARROWEST query, which is what makes it unfavourable to a cache
whose entries freeze at a per-entry ceiling once opened.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Same adversarial arrival order for every bar; only the reorder policy differs.
ARRIVAL_SECONDS = 205.2

POLICIES = [
    ("arrival\n(no reorder)", None, ps.NEUTRAL),
    ("overlap\nby column count", "experiment/w2_policy/fov_*/bucket.csv", ps.LIGHT),
    ("overlap\nby bytes", "experiment/w2_policy/bov_*/bucket.csv", ps.DARK),
    ("bytes + LRU\nresidency", "experiment/w2_policy/blru_*/bucket.csv", ps.LIGHT),
]


def totals(pattern: str) -> list[float]:
    out: list[float] = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        per_exec: dict[str, float] = {}
        for row in csv.DictReader(open(path)):
            if row["execution"] == "1":
                continue
            per_exec.setdefault(row["execution"], 0.0)
            per_exec[row["execution"]] += float(row["total_ms"])
        out += [v / 1000.0 for v in per_exec.values()]
    return out


def main() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    labels, means, errs, styles = [], [], [], []
    for label, pattern, style in POLICIES:
        if pattern is None:
            labels.append(label)
            means.append(ARRIVAL_SECONDS)
            errs.append(0.0)
            styles.append(style)
            continue
        v = totals(pattern)
        if not v:
            continue
        labels.append(label)
        means.append(statistics.mean(v))
        errs.append(statistics.stdev(v) if len(v) > 1 else 0.0)
        styles.append(style)

    xs = list(range(len(labels)))
    for x, m, e, style in zip(xs, means, errs, styles):
        ax.bar([x], [m], 0.6, yerr=[e], error_kw=ps.ERRBAR, zorder=3, **style)
    for x, m, e in list(zip(xs, means, errs))[1:]:
        ax.annotate(f"−{(means[0] - m) / means[0] * 100:.1f}%", (x, m + e + 4),
                    ha="center", fontsize=7.2, color="#b4522e")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylim(0, 232)
    ps.finish(ax, "workload GPU time (s)")
    ps.save(fig, "fig_o1_overlap_metric")
    plt.close(fig)


if __name__ == "__main__":
    main()
