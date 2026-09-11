#!/usr/bin/env python3
"""The flat page store on ClickBench, and why a static budget was the wrong knob.

ClickBench is 61.5 GB decoded against a device that cannot spare more than a few
GB for cache, and a static 6 GB budget does not survive it: the store fills to
~4.5 GB and the query runs out of device memory. That reads like a hardware wall,
but it is not -- the same 6 GB with a free-memory floor holds itself at 3.3 GB,
finishes, and beats every static budget including the 3 GB one that also fits.
Spending more when there is room and giving it back when there is not is worth
more than picking a number that always fits.

  pixi run -e duckdb-python python scripts/plot_page_store_clickbench.py
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

ROOT = Path(__file__).resolve().parents[1] / "experiment"
NQ = 37


def condition(tag: str) -> tuple[float, float] | None:
    runs = []
    for bucket in sorted(glob.glob(str(ROOT / f"{tag}_*" / "bucket.csv"))):
        per: dict[str, float] = collections.defaultdict(float)
        seen: dict[str, set] = collections.defaultdict(set)
        for row in csv.DictReader(open(bucket)):
            per[row["execution"]] += float(row["total_ms"])
            seen[row["execution"]].add(row["query"])
        warm = [e for e in per if e != "1" and len(seen[e]) == NQ]
        if warm:
            runs.append(statistics.mean(per[e] for e in warm) / 1000.0)
    if not runs:
        return None
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def main() -> int:
    bars = [
        ("baseline\n(no cache)", "cbf_base", ps.NEUTRAL),
        ("static\n1 GB", "cbf_page1GB", ps.LIGHT),
        ("static\n2 GB", "cbf_page2GB", ps.LIGHT),
        ("static\n3 GB", "cbf_page3GB", ps.DARK),
        ("soft budget\n6 GB, 4 GB floor", "cbf_soft", ps.ACCENT),
    ]
    measured = [(lab, condition(tag), sty) for lab, tag, sty in bars]
    for lab, value, _ in measured:
        if value is None:
            print(f"missing: {lab.replace(chr(10), ' ')}")
    measured = [(lab, v, sty) for lab, v, sty in measured if v]
    if not measured:
        raise SystemExit("no runs under experiment/cbf_*")

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xs = range(len(measured))
    ax.bar(xs, [v[0] for _, v, _ in measured], yerr=[v[1] for _, v, _ in measured],
           facecolor=[s["facecolor"] for _, _, s in measured],
           edgecolor=ps.NEUTRAL["edgecolor"], linewidth=ps.NEUTRAL["linewidth"],
           error_kw=ps.ERRBAR, zorder=3)
    base = measured[0][1][0]
    for x, (_, (mean, _), _) in zip(xs, measured):
        delta = "" if x == 0 else f"\n{100 * (mean - base) / base:+.1f}%"
        ax.text(x, mean, f"{mean:.1f}s{delta}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lab for lab, _, _ in measured], fontsize=7)
    ax.set_ylim(0, max(v[0] for _, v, _ in measured) * 1.3)
    ps.finish(ax, "query time (s)")
    ax.set_title("ClickBench, adversarial order", fontsize=8.5, pad=4)
    fig.tight_layout()
    ps.save(fig, "fig_page_store_clickbench")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
