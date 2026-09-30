#!/usr/bin/env python3
"""What cache-aware reordering is worth on top of the flat page store.

The two benchmarks disagree, and the disagreement is the point. Reordering puts
queries that share columns next to each other so the second one finds the first
one's pages resident. ClickBench has 105 columns and every query reads a different
handful, so ordering decides almost everything: -34.2% becomes -64.8%. TPC-H's 22
queries keep reading the same few lineitem columns, so they already overlap in any
order -- there is nothing for the reorder to arrange, and it costs 3.0s to find
that out.

Floors differ by benchmark on purpose: reordering raises the hit rate, which keeps
pages resident longer, which raises memory pressure. SF50 ran out of device memory
twice at a 4 GB floor (same query both times) and needed 6 GB; both SF50 bars use
that 6 GB floor so the comparison holds.

  pixi run -e duckdb-python python scripts/plot_reorder_effect.py
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


def condition(prefix: str, mode: str, nq: int) -> tuple[float, float] | None:
    runs = []
    for bucket in sorted(glob.glob(str(ROOT / f"{prefix}_{mode}_*" / "bucket.csv"))):
        per: dict[str, float] = collections.defaultdict(float)
        seen: dict[str, set] = collections.defaultdict(set)
        for row in csv.DictReader(open(bucket)):
            per[row["execution"]] += float(row["total_ms"])
            seen[row["execution"]].add(row["query"])
        warm = [e for e in per if e != "1" and len(seen[e]) == nq]
        if warm:
            runs.append(statistics.mean(per[e] for e in warm) / 1000.0)
    if not runs:
        return None
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def main() -> int:
    benches = [
        ("ClickBench", "ro_cb", 37, 61.27),
        ("TPC-H SF50", "ro2_sf50", 22, 108.29),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    width = 0.26
    xs = np.arange(len(benches))

    groups = [
        ("baseline (no cache)", ps.NEUTRAL, lambda _p, _n, base: (base, 0.0)),
        ("page store", ps.DARK, lambda p, n, _b: condition(p, "noreorder", n)),
        ("page store + reorder", ps.ACCENT, lambda p, n, _b: condition(p, "reorder", n)),
    ]
    for i, (label, style, get) in enumerate(groups):
        values, errors = [], []
        for _, prefix, nq, base in benches:
            got = get(prefix, nq, base)
            values.append(got[0] if got else 0.0)
            errors.append(got[1] if got else 0.0)
        offset = (i - 1) * width
        ax.bar(xs + offset, values, width, yerr=errors, label=label,
               error_kw=ps.ERRBAR, zorder=3, **style)
        for x, value, (_, _, _, base) in zip(xs + offset, values, benches):
            if not value:
                continue
            tag = f"{value:.1f}" if i == 0 else f"{value:.1f}\n{100 * (value - base) / base:+.0f}%"
            ax.text(x, value, tag, ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _, _, _ in benches], fontsize=8)
    ax.set_ylim(0, 128)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.18), columnspacing=1.2, handlelength=1.4)
    ps.finish(ax, "query time (s)")
    fig.tight_layout()
    ps.save(fig, "fig_reorder_effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
