#!/usr/bin/env python3
"""The page cache's contribution, with and without reordering.

Two figures, each with both benchmarks side by side:

  fig_c1  baseline / fixed / fixed+variable            -- what the cache buys
  fig_c2  baseline / fixed+reorder / fixed+variable+reorder

"fixed" caches only intrinsically fixed-width columns; STRING columns are left
out of it entirely rather than being kept as whole chunks, so the two bars
differ by exactly the variable-width page cache.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS = [("TPC-H SF100", "experiment/tp100_worst"),
              ("ClickBench 100M", "experiment/cb_worst")]


def totals(root: str, tag: str) -> list[float]:
    out: list[float] = []
    for path in sorted(glob.glob(str(ROOT / root / f"{tag}_*" / "bucket.csv"))):
        per_exec: dict[str, float] = {}
        for row in csv.DictReader(open(path)):
            if row["execution"] == "1":       # warmup
                continue
            per_exec.setdefault(row["execution"], 0.0)
            per_exec[row["execution"]] += float(row["total_ms"])
        out += [v / 1000.0 for v in per_exec.values()]
    return out


def stat(root: str, tag: str) -> tuple[float, float]:
    v = totals(root, tag)
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def draw(tags: list[tuple[str, str]], stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95))
    styles = [ps.NEUTRAL, ps.LIGHT, ps.DARK]

    for ax, (bench, root) in zip(axes, BENCHMARKS):
        means, errs = [], []
        for _, tag in tags:
            m, s = stat(root, tag)
            means.append(m)
            errs.append(s)
        xs = list(range(len(tags)))
        for x, m, e, style in zip(xs, means, errs, styles):
            ax.bar([x], [m], 0.6, yerr=[e], error_kw=ps.ERRBAR, zorder=3, **style)
        # Reduction against this benchmark's own no-cache bar.
        for x, m in list(zip(xs, means))[1:]:
            ax.annotate(f"−{(means[0] - m) / means[0] * 100:.1f}%",
                        (x, m + means[0] * 0.035), ha="center", fontsize=7,
                        color="#b4522e")
        ax.set_xticks(xs)
        # The reorder labels are long enough to collide at this width, so they
        # lean rather than wrap into each other.
        long = any(len(label.replace("\n", " ")) > 12 for label, _ in tags)
        ax.set_xticklabels([label for label, _ in tags], fontsize=7.4,
                           rotation=18 if long else 0,
                           ha="right" if long else "center")
        ax.set_ylim(0, means[0] * 1.28)
        ax.set_title(bench, fontsize=8.5, pad=4)
        ps.finish(ax)
    axes[0].set_ylabel("workload GPU time (s)", fontsize=8.5)
    fig.subplots_adjust(wspace=0.34)
    ps.save(fig, stem)
    plt.close(fig)


if __name__ == "__main__":
    draw([("baseline", "base"), ("fixed", "nr_fixed"), ("fixed\n+variable", "nr_fx_var")],
         "fig_c1_cache")
    draw([("baseline", "base"), ("fixed + reorder", "fixed"),
          ("fixed+variable + reorder", "fx_var")],
         "fig_c2_cache_reorder")
