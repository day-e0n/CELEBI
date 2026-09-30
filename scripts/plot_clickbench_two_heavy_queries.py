#!/usr/bin/env python3
"""Two ClickBench queries carry the whole arrival-order effect.

q28 reads URL (9.3GB) and q29 reads Referer (6.8GB); both columns are larger than
the 6GB page budget, so neither can stay resident and each run either serves them
from the cache or does not. Stacking those two against the other 41 shows where the
spread lives: the 41 sit within a few seconds of each other in every run, while the
pair moves by a factor of two.

Every bar is the mean of executions 2 and 3; execution 1 is the cold pass.

  pixi run -e duckdb-python python scripts/plot_clickbench_two_heavy_queries.py
"""
from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "experiment"
HEAVY = {"q28", "q29"}

ORDERS = [("natural", "F_cb_cache", "F_cb_reorder"),
          ("worst_case", "S_cb_worst_c", "S_cb_worst_r"),
          ("random_a", "T_cb_randa_c", "T_cb_randa_r"),
          ("random_b", "T_cb_randb_c", "T_cb_randb_r"),
          ("random_c", "T_cb_randc_c", "T_cb_randc_r")]
BASE = "F_cb_base"


def split(run: str) -> tuple[float, float]:
    """(q28+q29, the other 41) seconds for one warm execution."""
    per: dict[int, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    with (ROOT / run / "bucket.csv").open() as handle:
        for row in csv.DictReader(handle):
            execution = int(row["execution"])
            if execution == 1:
                continue
            per[execution][0 if row["query"] in HEAVY else 1] += float(row["total_ms"]) / 1000.0
    n = len(per)
    return (sum(v[0] for v in per.values()) / n, sum(v[1] for v in per.values()) / n)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))

    labels, heavy, rest, positions = [], [], [], []
    base_heavy, base_rest = split(BASE)
    labels.append("no cache\n(natural)")
    heavy.append(base_heavy)
    rest.append(base_rest)
    positions.append(0.0)

    x = 1.1
    centres = []
    for name, plain_run, reorder_run in ORDERS:
        centres.append((x + 0.45, name))
        for run, tag in ((plain_run, "cache"), (reorder_run, "+reorder")):
            h, r = split(run)
            labels.append(tag)
            heavy.append(h)
            rest.append(r)
            positions.append(x)
            x += 0.9
        x += 0.5

    ax.bar(positions, heavy, 0.8, label="q28 (URL) + q29 (Referer)", zorder=3, **ps.ACCENT)
    ax.bar(positions, rest, 0.8, bottom=heavy, label="the other 41 queries",
           zorder=3, **ps.LIGHT)

    for pos, h, r in zip(positions, heavy, rest):
        ax.text(pos, h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7,
                color="white")
        ax.text(pos, h + r + 2.5, f"{h + r:.0f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5)
    # The arrival order names sit under the pair they label; the no-cache bar is its
    # own reference and carries its order in the tick label instead.
    for centre, name in centres:
        ax.text(centre, -36, name, ha="center", va="top", fontsize=8.5)

    ax.set_ylim(0, 248)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2, handlelength=1.2,
              bbox_to_anchor=(0.0, 1.02))
    ps.finish(ax, "GPU operator time (s)")
    ax.set_xlabel("ClickBench 43 queries — arrival order", fontsize=8.5, labelpad=20)

    ps.save(fig, "clickbench_two_heavy_queries")


if __name__ == "__main__":
    main()
