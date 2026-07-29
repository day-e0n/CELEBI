#!/usr/bin/env python3
"""Experiment B final figure: reorder latency and quality (overlap_ratio) vs
batch size N, baseline (no keep-first) vs --reorder-keep-first, log-scale time
axis. Uses the validated categorical palette (dataviz skill references/palette.md).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--keepfirst-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, default=None)
    args = parser.parse_args()

    baseline = read_rows(args.baseline_csv)
    keepfirst = read_rows(args.keepfirst_csv)

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "sans-serif"

    color_baseline = "#2a78d6"   # slot 1 blue
    color_keepfirst = "#eb6834"  # slot 2 orange
    ink_primary = "#0b0b0b"

    n_base = [int(r["n"]) for r in baseline]
    t_base = [float(r["mean_ms"]) for r in baseline]
    q_base = [float(r["overlap_ratio"]) for r in baseline]
    n_kf = [int(r["n"]) for r in keepfirst]
    t_kf = [float(r["mean_ms"]) for r in keepfirst]
    q_kf = [float(r["overlap_ratio"]) for r in keepfirst]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot(n_base, t_base, color=color_baseline, linewidth=2.4, marker="o", markersize=7, label="exhaustive")
    ax1.plot(n_kf, t_kf, color=color_keepfirst, linewidth=2.4, marker="o", markersize=7, label="fixed-start")
    ax1.set_yscale("log")
    ax1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.set_ylabel("reorder latency (ms)", fontsize=18, color=ink_primary)
    ax1.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax1.set_ylim(top=max(t_base + t_kf) * 6)
    last_n = n_base[-1]
    for x, y in zip(n_base, t_base):
        if x == last_n:
            ax1.annotate(f"{y/1000:.1f}s" if y >= 1000 else f"{y:.0f}ms", (x, y), textcoords="offset points",
                         xytext=(-10, 8), ha="right", fontsize=14, color=ink_primary)
    for x, y in zip(n_kf, t_kf):
        if x == last_n:
            ax1.annotate(f"{y:.0f}ms" if y >= 1 else f"{y:.2f}ms", (x, y), textcoords="offset points",
                         xytext=(2, -30), ha="right", fontsize=14, color=ink_primary)

    ax2.plot(n_base, q_base, color=color_baseline, linewidth=2.4, marker="o", markersize=7, label="exhaustive")
    ax2.plot(n_kf, q_kf, color=color_keepfirst, linewidth=2.4, marker="o", markersize=7, label="fixed-start")
    ax2.set_ylabel("overlap_ratio", fontsize=18, color=ink_primary)
    ax2.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    q_min = min(q_base + q_kf)
    q_max = max(q_base + q_kf)
    pad = max(0.05, (q_max - q_min) * 0.15)
    ax2.set_ylim(max(0.0, q_min - pad), min(1.0, q_max + pad))

    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.28)
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    center_x = (pos1.x0 + pos2.x1) / 2

    fig.supxlabel("batch size N", x=center_x, y=0.01, fontsize=18, color=ink_primary)

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(center_x, 0.97),
        ncols=2,
        frameon=True,
        fontsize=19,
        labelcolor=ink_primary,
        handlelength=2.6,
        handleheight=0.7,
        columnspacing=1.4,
        edgecolor="black",
        fancybox=False,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(1.2)

    for out_path in (args.out_png, args.out_pdf):
        if out_path is None:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200 if out_path.suffix == ".png" else None, bbox_inches="tight")
        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
