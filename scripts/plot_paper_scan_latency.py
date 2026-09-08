#!/usr/bin/env python3
"""Publication figure: page-cache scan latency, ClickBench and TPC-H.

Normalised to the uncached baseline so the two workloads share one axis and the
reader compares the thing that matters -- how much of a scan the cache removes.
Absolute seconds are printed under each group for reference.

Each workload has two cache configurations. `fixed` caches only fixed-width
columns; STRING columns are read from parquet. `fixed+var` adds them as
variable-width pages. Both are run in the queries' arrival order and again after
the cache-aware reorder, so the cache's contribution and the reorder's are
separable. Arrival order is the least favourable one -- adjacent queries share as
few columns as possible.

Hot scan latency; execution 1 of every run is the cold pass and is dropped.

Usage:  pixi run -e duckdb-python python scripts/plot_paper_scan_latency.py
Output: experiment/figs/paper_scan_latency.{png,pdf}
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiment" / "figs" / "paper_scan_latency"

WORKLOADS = [
    ("ClickBench", REPO / "experiment/cb_worst"),
    ("TPC-H SF100", REPO / "experiment/tp100_worst"),
]
GROUPS = [("fixed", "nr_fixed", "fixed"), ("fixed+var", "nr_fx_var", "fx_var")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ARRIVAL = dict(facecolor="#c9d6d4", edgecolor="#2f3b3a", linewidth=0.6)
REORDER = dict(facecolor="#3d6b66", edgecolor="#2f3b3a", linewidth=0.6)


def scan_seconds(bucket_csv: Path) -> float:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def series(root: Path, tag: str) -> tuple[float, float] | None:
    runs = [scan_seconds(root / f"{tag}_{r}" / "bucket.csv")
            for r in (1, 2, 3) if (root / f"{tag}_{r}" / "bucket.csv").exists()]
    if not runs:
        return None
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def main() -> int:
    ready = [(t, r) for t, r in WORKLOADS if series(r, "base")]
    if not ready:
        raise SystemExit("no data")

    # One x position per (workload, cache configuration).
    positions, labels, wl_spans, base_secs = [], [], [], []
    pos = 0.0
    for title, root in ready:
        start = pos
        for label, _, _ in GROUPS:
            positions.append(pos)
            labels.append(label)
            pos += 1.0
        wl_spans.append((title, start, pos - 1.0))
        base_secs.append(series(root, "base")[0])
        pos += 0.8  # gap between workloads

    fig, ax = plt.subplots(figsize=(5.6, 2.55), dpi=300)
    w = 0.36
    k = 0
    for (title, root), base in zip(ready, base_secs):
        for label, plain_tag, reord_tag in GROUPS:
            for off, tag, style in ((-w / 2, plain_tag, ARRIVAL), (w / 2, reord_tag, REORDER)):
                s = series(root, tag)
                if s is None:
                    k += 0
                    continue
                pct = (1.0 - s[0] / base) * 100.0
                err = s[1] / base * 100.0
                ax.bar(positions[k] + off, pct, w, yerr=err, capsize=1.8,
                       error_kw=dict(elinewidth=0.6, capthick=0.6), zorder=3, **style)
                ax.text(positions[k] + off, pct + err + 1.1, f"{pct:.0f}",
                        ha="center", va="bottom", fontsize=7)
            k += 1

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Scan latency reduction (%)", fontsize=8.5)
    ax.set_ylim(0, 50)
    ax.tick_params(length=2.2, pad=2)
    ax.yaxis.grid(True, color="0.88", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # Workload band under the axis, with the uncached baseline it is relative to.
    for (title, lo, hi), base in zip(wl_spans, base_secs):
        ax.text((lo + hi) / 2, -8.6, f"{title}\n(baseline {base:.0f} s)",
                ha="center", va="top", fontsize=8.5, linespacing=1.35)
    ax.set_xlim(positions[0] - 0.75, positions[-1] + 0.75)

    handles = [plt.Rectangle((0, 0), 1, 1, **ARRIVAL), plt.Rectangle((0, 0), 1, 1, **REORDER)]
    ax.legend(handles, ["arrival order", "+ cache-aware reorder"],
              loc="upper left", fontsize=8, handlelength=1.5, handleheight=0.95,
              borderaxespad=0.2, labelspacing=0.35)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
    print(f"wrote {OUT}.png / .pdf")
    for (title, root), base in zip(ready, base_secs):
        print(f"  {title}  baseline {base:.2f} s")
        for label, p, r in GROUPS:
            a, b = series(root, p), series(root, r)
            if not a:
                continue
            line = f"    {label:10s} arrival {a[0]:7.2f}s ({(1-a[0]/base)*100:4.1f}%)"
            if b:
                line += f"   reordered {b[0]:7.2f}s ({(1-b[0]/base)*100:4.1f}%)"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
