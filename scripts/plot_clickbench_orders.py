#!/usr/bin/env python3
"""ClickBench under two arrival orders, same five conditions.

Both orders come from the same rule -- minimise adjacent column overlap -- and
differ only in the seed: the query with the MOST columns (q42) or the FEWEST
(q13). That one choice moves the cache's benefit from -12.1% to -48.0%, which is
larger than anything the cache configuration or the reorder does.

The seed change was made because it is adversarial on TPC-H (an entry's column
set is fixed by whoever opens it, so a narrow opener starves later queries). On
ClickBench it is the opposite: one denormalised table means one entry, and a
narrow opener leaves budget for the columns that follow. There is no single
worst-case arrival rule that holds across both.
"""
from __future__ import annotations

import csv
import glob
import statistics
from pathlib import Path

import paper_style as ps
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

ORDERS = [("seed = widest query (q42)", "cb_worst"),
          ("seed = narrowest query (q13)", "w2_cb")]
CONDITIONS = [("baseline", "base", ps.NEUTRAL),
              ("fixed", "nr_fixed", ps.LIGHT),
              ("fixed\n+var", "nr_fx_var", ps.LIGHT),
              ("fixed\n+reo", "fixed", ps.DARK),
              ("fixed+var\n+reo", "fx_var", ps.DARK)]


def runs(root: str, tag: str) -> list[float]:
    """One sample per process; execution 1 is warmup. Skips truncated runs."""
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
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.1), sharey=True)
    for ax, (title, root) in zip(axes, ORDERS):
        labels, means, errs, styles, ns = [], [], [], [], []
        for label, tag, style in CONDITIONS:
            v = runs(root, tag)
            if not v:
                continue
            labels.append(label)
            means.append(statistics.mean(v))
            errs.append(statistics.stdev(v) if len(v) > 1 else 0.0)
            styles.append(style)
            ns.append(len(v))
        xs = list(range(len(labels)))
        for x, m, e, style in zip(xs, means, errs, styles):
            ax.bar([x], [m], 0.62, yerr=[e], error_kw=ps.ERRBAR, zorder=3, **style)
        for x, m, e in list(zip(xs, means, errs))[1:]:
            ax.annotate(f"−{(means[0] - m) / means[0] * 100:.0f}%", (x, m + e + 1.4),
                        ha="center", fontsize=7, color="#b4522e")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 74)
        ps.finish(ax)
    axes[0].set_ylabel("workload GPU time (s)", fontsize=8.5)
    fig.subplots_adjust(wspace=0.08)
    ps.save(fig, "fig_cb_orders")
    plt.close(fig)


if __name__ == "__main__":
    main()
