#!/usr/bin/env python3
"""CELEBI page cache: baseline, fixed-width columns, and adding variable-width.

`fixed` caches ONLY fixed-width columns; STRING columns are left out and read from
parquet. This needs saying because it is not what the condition used to do -- with
the variable-width cache off, a STRING column was still stored as a whole chunk, so
a run labelled "fixed" was really fixed-width-paged plus string-whole-chunk, and
comparing it against `fixed + variable` measured a change of storage format for
data that was cached either way rather than the value of caching strings at all.

`fixed + variable` adds the STRING columns as variable-width pages.

Hot scan latency only; execution 1 of every run is the cold pass and is dropped.
Note that both cache conditions also run the cache-aware query reorder, which the
baseline does not -- the runner applies it to every non-baseline condition -- so
these bars measure the CELEBI package, not caching in isolation.

Usage:  pixi run -e duckdb-python python scripts/plot_celebi_fixed_variable.py
Output: experiment/figs/celebi_fixed_variable.png
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiment" / "figs" / "celebi_fixed_variable.png"
CB, TP = REPO / "experiment/cb_proper", REPO / "experiment/tp_proper"

LABELS = ["baseline", "fixed", "fixed + variable"]
TAGS = ["base", "fixed", "fx_var"]
WORKLOADS = [("ClickBench 100M\n1 table · 105 columns · no joins", CB),
             ("TPC-H SF50\n8 tables · joins throughout", TP)]

INK, MUTED, GRID = "#1a1c23", "#6b6e7d", "#e3e5ea"
BARS = ["#9aa0ad", "#0D9488", "#C2410C"]


def scan_seconds(bucket_csv: Path) -> float:
    rows = [r for r in csv.DictReader(bucket_csv.open()) if int(r["execution"]) > 1]
    if not rows:
        raise SystemExit(f"{bucket_csv}: no warm executions")
    n_exec = len({int(r["execution"]) for r in rows})
    return sum(float(r["scan"]) for r in rows) / n_exec / 1000.0


def series(root: Path, tag: str) -> tuple[float, float]:
    runs = [scan_seconds(root / f"{tag}_{r}" / "bucket.csv")
            for r in (1, 2, 3) if (root / f"{tag}_{r}" / "bucket.csv").exists()]
    if not runs:
        raise SystemExit(f"{root}/{tag}_*: no runs")
    return statistics.mean(runs), (statistics.stdev(runs) if len(runs) > 1 else 0.0)


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), dpi=200)
    fig.patch.set_facecolor("white")

    for ax, (title, root) in zip(axes, WORKLOADS):
        ax.set_facecolor("white")
        stats = [series(root, tag) for tag in TAGS]
        means = [m for m, _ in stats]
        errs = [e for _, e in stats]

        bars = ax.bar(range(len(TAGS)), means, width=0.62, color=BARS, zorder=3,
                      yerr=errs, capsize=4,
                      error_kw={"ecolor": INK, "elinewidth": 1.1, "capthick": 1.1, "zorder": 5})
        for i, (bar, mean, err) in enumerate(zip(bars, means, errs)):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + err + max(means) * 0.025,
                    f"{mean:.1f} s", ha="center", va="bottom",
                    fontsize=11, fontweight="600", color=INK, zorder=6)
            if i:
                pct = (mean / means[0] - 1.0) * 100.0
                ax.text(bar.get_x() + bar.get_width() / 2, mean / 2, f"{pct:+.1f}%",
                        ha="center", va="center", fontsize=13, fontweight="700",
                        color="white", zorder=6)

        ax.set_xticks(range(len(TAGS)))
        ax.set_xticklabels(LABELS, fontsize=9.5, color=INK)
        ax.set_ylim(0, max(means) * 1.24)
        ax.tick_params(axis="y", labelsize=9.5, colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.set_title(title, fontsize=12, fontweight="600", color=INK, pad=12, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].set_ylabel("Scan latency per execution  (s)", fontsize=10.5, color=MUTED, labelpad=10)
    fig.suptitle("CELEBI page cache — hot scan latency",
                 fontsize=14.5, fontweight="600", color=INK, x=0.007, ha="left", y=0.995)
    fig.text(0.007, 0.945,
             "Hot scan latency, 3 repeats × 3 executions per condition; execution 1 dropped as cold.",
             fontsize=9, color=MUTED, ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    for title, root in WORKLOADS:
        print(f"  {title.splitlines()[0]}")
        for tag, label in zip(TAGS, LABELS):
            m, e = series(root, tag)
            print(f"    {label.replace(chr(10), ' '):28s} {m:7.2f} s  ±{e:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
