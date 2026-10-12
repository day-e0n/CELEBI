#!/usr/bin/env python3
"""TPC-H's reorder costs a warm-up pass, not throughput.

The same two query orders -- the runner's own and what small-first-big-last makes
of it -- run five times each, three executions per run. Averaging executions 2 and
3, the standard protocol here, says the reorder costs 5.3%. Splitting by execution
says something else: the reordered order is FASTER cold, slower on execution 2, and
identical by execution 3. It needs one more pass to reach steady state, and the
protocol charges it for that.

  pixi run -e duckdb-python python scripts/plot_tpch_reorder_convergence.py
"""
from __future__ import annotations

import collections
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiment"
REPS = range(1, 6)


def per_execution(run: str) -> dict[int, float]:
    totals: dict[int, float] = collections.defaultdict(float)
    with (ROOT / run / "bucket.csv").open() as handle:
        for row in csv.DictReader(handle):
            totals[int(row["execution"])] += float(row["total_ms"]) / 1000.0
    return totals


def series(tag: str) -> dict[int, list[float]]:
    out: dict[int, list[float]] = collections.defaultdict(list)
    for rep in REPS:
        for execution, value in per_execution(f"X_th_{tag}_{rep}").items():
            out[execution].append(value)
    return out


def main() -> None:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[3, 2], wspace=0.3))

    plain, reord = series("plain"), series("reord")

    # --- per execution -----------------------------------------------------
    x = np.arange(3)
    width = 0.36
    for offset, (data, label, style) in enumerate((
            (plain, "runner's order", ps.LIGHT), (reord, "after reorder", ps.DARK))):
        means = [statistics.mean(data[e]) for e in (1, 2, 3)]
        errs = [statistics.stdev(data[e]) for e in (1, 2, 3)]
        left.bar(x + (offset - 0.5) * width, means, width, yerr=errs, zorder=3,
                 error_kw=ps.ERRBAR, label=label, **style)

    for xi, e in enumerate((1, 2, 3)):
        a, b = statistics.mean(plain[e]), statistics.mean(reord[e])
        left.text(xi, max(a, b) + 2.2, f"{(b - a) / a * 100:+.0f}%", ha="center",
                  va="bottom", fontsize=8,
                  color="#b4522e" if b > a else "#2f3b3a")

    left.set_xticks(x)
    left.set_xticklabels(["execution 1\n(cold)", "execution 2", "execution 3"])
    left.set_ylim(0, 56)
    left.legend(loc="upper right", fontsize=7.5, handlelength=1.2)
    ps.finish(left, "GPU operator time (s)")

    # --- the protocol's own number ----------------------------------------
    warm = {tag: [statistics.mean([per_execution(f"X_th_{tag}_{r}")[e] for e in (2, 3)])
                  for r in REPS] for tag in ("plain", "reord")}
    for xi, (tag, label, style) in enumerate((
            ("plain", "runner's\norder", ps.LIGHT), ("reord", "after\nreorder", ps.DARK))):
        values = warm[tag]
        right.bar(xi, statistics.mean(values), 0.5, yerr=statistics.stdev(values),
                  zorder=3, error_kw=ps.ERRBAR, **style)
        right.scatter([xi] * len(values), values, s=7, color="#2f3b3a", zorder=4,
                      linewidths=0)
        right.text(xi, max(values) + 0.5, f"{statistics.mean(values):.2f}",
                   ha="center", va="bottom", fontsize=7.5)

    right.set_xticks([0, 1])
    right.set_xticklabels(["runner's\norder", "after\nreorder"], fontsize=7.5)
    right.set_xlim(-0.6, 1.6)
    right.set_ylim(26, 34)
    right.set_xlabel("mean of executions 2–3, 5 repeats", fontsize=8, labelpad=3)
    ps.finish(right, None)

    ps.save(fig, "tpch_reorder_convergence")


if __name__ == "__main__":
    main()
