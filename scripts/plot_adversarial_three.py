#!/usr/bin/env python3
"""baseline / paging / paging + reorder, on all three benchmarks.

Adversarial arrival order: seeded with the most expensive query (so the costliest
scan meets an empty cache) and then extended by minimising adjacent column
overlap, with cost breaking ties toward the expensive side. Both damages at once
-- nothing to reuse, and the big scans in the cold prefix.

"paging" is both page caches (fixed-width and variable-width). The reorder is
the column-count overlap policy; the byte-weighted variant is a separate figure
because it does not separate from this one outside its own error bars.

Per condition: 3 processes x --executions 3, first pass discarded as warmup.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS = [("TPC-H SF50", "adv50"), ("TPC-H SF100", "adv100"),
              ("ClickBench", "advcb")]
CONDITIONS = [("baseline", "base", ps.NEUTRAL),
              ("paging", "paging", ps.LIGHT),
              ("paging\n+ reorder", "fov", ps.DARK)]


def runs(root: str, tag: str) -> list[float]:
    """One sample per process; execution 1 is warmup."""
    out: list[float] = []
    for path in sorted(glob.glob(str(ROOT / "experiment" / root / f"{tag}_*" / "bucket.csv"))):
        per_exec: dict[str, float] = {}
        for row in csv.DictReader(open(path)):
            if row["execution"] == "1":
                continue
            per_exec.setdefault(row["execution"], 0.0)
            per_exec[row["execution"]] += float(row["total_ms"])
        if per_exec:
            out.append(statistics.mean(per_exec.values()) / 1000.0)
    return out


def main() -> None:
    fig, axes = plt.subplots(1, len(BENCHMARKS), figsize=(5.0, 2.05))
    for ax, (title, root) in zip(axes, BENCHMARKS):
        means, errs, styles = [], [], []
        for _, tag, style in CONDITIONS:
            v = runs(root, tag)
            means.append(statistics.mean(v))
            errs.append(statistics.stdev(v) if len(v) > 1 else 0.0)
            styles.append(style)
        xs = list(range(len(CONDITIONS)))
        for x, m, e, style in zip(xs, means, errs, styles):
            ax.bar([x], [m], 0.6, yerr=[e], error_kw=ps.ERRBAR, zorder=3, **style)
        for x, m, e in list(zip(xs, means, errs))[1:]:
            ax.annotate(f"−{(means[0] - m) / means[0] * 100:.0f}%",
                        (x, m + e + means[0] * 0.035), ha="center",
                        fontsize=7.2, color="#b4522e")
        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _, _ in CONDITIONS], fontsize=7.2)
        ax.set_ylim(0, means[0] * 1.3)
        ax.set_title(title, fontsize=8.5, pad=4)
        ps.finish(ax)
    axes[0].set_ylabel("workload GPU time (s)", fontsize=8.5)
    fig.subplots_adjust(wspace=0.36)
    ps.save(fig, "fig_a1_adversarial")
    plt.close(fig)


if __name__ == "__main__":
    main()
