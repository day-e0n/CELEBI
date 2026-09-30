#!/usr/bin/env python3
"""Caching before the filter vs after it.

An entry built AFTER the reader applied a predicate holds "the rows matching
that predicate", not "rows [a,b) of the table" -- so it carries a filter
signature and only serves a scan with the identical predicate. Caching BEFORE
the filter (SIRIUS_FIXED_PAGE_CACHE_BEFORE_FILTER=1) removes that restriction:
entries hold whole columns and any query can reuse them.

The trade is capacity, and it is decisive here. ClickBench's predicates are
selective, so the pre-filter entries are far larger for the same rows:
admission_stop_widening goes 485 -> 945 per run and hits fall 17 -> 10, which
lands the whole workload BELOW its own no-cache baseline.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# (title, baseline seconds, post-filter dir, pre-filter dir)
PANELS = [("TPC-H SF100", 218.73, "adv100", "pre100"),
          ("ClickBench", 61.30, "advcb", "precb")]
CONDITIONS = [("paging", "paging"), ("paging\n+ reorder", "fov")]


def runs(root: str, tag: str) -> list[float]:
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
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.4, 2.15))
    width = 0.34
    for ax, (title, baseline, post, pre) in zip(axes, PANELS):
        xs = list(range(len(CONDITIONS)))
        for offset, (root, label, style) in enumerate(
                ((post, "after filter", ps.DARK), (pre, "before filter", ps.LIGHT))):
            means, errs = [], []
            for _, tag in CONDITIONS:
                v = runs(root, tag)
                means.append(statistics.mean(v) if v else 0.0)
                errs.append(statistics.stdev(v) if len(v) > 1 else 0.0)
            ax.bar([x + (offset - 0.5) * width for x in xs], means, width,
                   yerr=errs, error_kw=ps.ERRBAR, zorder=3,
                   label=label if ax is axes[0] else None, **style)
        # The no-cache baseline as a rule: a bar above it means caching lost.
        ax.axhline(baseline, color="#b4522e", linewidth=0.8, linestyle=(0, (3, 2)), zorder=4)
        ax.annotate("no cache", (len(xs) - 0.5, baseline), fontsize=6.8,
                    color="#b4522e", va="bottom", ha="right")
        ax.set_xticks(xs)
        ax.set_xticklabels([label for label, _ in CONDITIONS], fontsize=7.4)
        ax.set_ylim(0, baseline * 1.35)
        ax.set_title(title, fontsize=8.5, pad=4)
        ps.finish(ax)
    axes[0].set_ylabel("workload GPU time (s)", fontsize=8.5)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.subplots_adjust(wspace=0.32)
    ps.save(fig, "fig_f1_prefilter")
    plt.close(fig)


if __name__ == "__main__":
    main()
