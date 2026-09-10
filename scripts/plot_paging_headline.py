#!/usr/bin/env python3
"""baseline / paging / paging + reorder, from one batch.

TPC-H SF100, adversarial arrival order, 3 processes per condition,
--executions 5 with the first pass discarded as warmup, so each condition has
12 hot samples. Paging means BOTH page caches (fixed-width and variable-width);
the reorder is the byte-weighted overlap policy.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

GROUPS = [("baseline", "base", ps.NEUTRAL),
          ("paging", "nr_fx_var", ps.LIGHT),
          ("paging\n+ reorder", "fx_var", ps.DARK)]


def samples(tag: str) -> list[float]:
    out: list[float] = []
    for path in sorted(glob.glob(str(ROOT / "experiment" / "e5_tp100" / f"{tag}_*" / "bucket.csv"))):
        per_exec: dict[str, float] = {}
        for row in csv.DictReader(open(path)):
            if row["execution"] == "1":       # warmup
                continue
            per_exec.setdefault(row["execution"], 0.0)
            per_exec[row["execution"]] += float(row["total_ms"])
        out += [v / 1000.0 for v in per_exec.values()]
    return out


def main() -> None:
    fig, ax = plt.subplots(figsize=(2.6, 2.15))
    labels, means, errs, styles = [], [], [], []
    for label, tag, style in GROUPS:
        v = samples(tag)
        labels.append(label)
        means.append(statistics.mean(v))
        errs.append(statistics.stdev(v))
        styles.append(style)

    xs = list(range(len(labels)))
    for x, m, e, style in zip(xs, means, errs, styles):
        ax.bar([x], [m], 0.58, yerr=[e], error_kw=ps.ERRBAR, zorder=3, **style)
    for x, m, e in list(zip(xs, means, errs))[1:]:
        ax.annotate(f"−{(means[0] - m) / means[0] * 100:.1f}%", (x, m + e + 5),
                    ha="center", fontsize=7.6, color="#b4522e")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylim(0, 250)
    ps.finish(ax, "workload GPU time (s)")
    ps.save(fig, "fig_h1_paging_sf100")
    plt.close(fig)


if __name__ == "__main__":
    main()
