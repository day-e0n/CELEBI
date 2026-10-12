#!/usr/bin/env python3
"""result/plot_latency_and_hit_bars.py split into two figures.

  1. ClickBench (top) and TPC-H SF50 (bottom) cumulative latency, stacked vertically
  2. byte cache hit ratio bars on their own

Same data, colours and styling as plot_latency_and_hit_bars.py; --hits as there.

  pixi run -e duckdb-python python result/plot_latency_stacked_and_hit_bars.py [--hits same]
writes result/query_latency_stacked{,_same}.{png,pdf} and result/cache_hit_bars{,_same}.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_latency_and_hit_bars import CONDITIONS, FIG_DIR, finish, hits

OUT = Path(__file__).resolve().parent
RC = {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
      "pdf.fonttype": 42, "savefig.bbox": None}


def legend(fig, handles, labels, ncol, y, fontsize=12):
    lg = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
                    frameon=True, edgecolor="#777777", facecolor="white", fontsize=fontsize,
                    handlelength=2.5, columnspacing=2, borderpad=0.6)
    lg.get_frame().set_linewidth(0.7)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    print(OUT / f"{stem}.png")


def latency_figure(means, suffix):
    width, height = 9.4, 11.75
    panel_width, panel_height = 7.2, 3.5
    fig = plt.figure(figsize=(width, height))
    # top panel: ClickBench, bottom panel: TPC-H SF50
    axes = [fig.add_axes([1.15 / width, y / height, panel_width / width, panel_height / height])
            for y in (6.35, 1.0)]
    max_latency = max(r["cumulative_query_wall_s"] for r in means)
    tick_step = 10 if max_latency > 40 else 5
    ymax = math.ceil((max_latency * 1.08) / tick_step) * tick_step
    for ax, title in zip(axes, ("ClickBench", "TPC-H SF50")):
        ends = []
        for condition, label, color in CONDITIONS:
            rows = [r for r in means if r["workload"] == title and r["condition"] == condition]
            n = len(rows)
            x = list(range(1, n + 1))
            y = [r["cumulative_query_wall_s"] for r in rows]
            ax.plot(x, y, color=color, marker="o", markersize=3.3, linewidth=1.6, label=label)
            ends.append((y[-1], color))
        ax.set_xticks(x, [r["query"] for r in rows], rotation=90, fontsize=10)
        ax.tick_params(axis="x", pad=5)
        ax.tick_params(axis="y", labelsize=13)
        ax.set_xlim(0.4, n + 0.6)
        ax.set_ylim(0, ymax)
        ax.set_yticks(range(0, ymax + 1, tick_step))
        ax.set_xlabel(f"{'ClickBench' if title == 'ClickBench' else 'TPC-H'} query", fontsize=15,
                      labelpad=12)
        ax.set_ylabel("Cumulative query latency (s)", fontsize=15)
        ax.set_title(title, fontsize=18, pad=12)
        finish(ax)
        upper = float("inf")
        for value, _ in sorted(ends, reverse=True):
            position = min(value, upper - 0.05 * ymax)
            upper = position
            ax.annotate(f"{value:.1f}s", xy=(n, value), xytext=(1.025, position),
                        textcoords=ax.get_yaxis_transform(), fontsize=14, va="center",
                        annotation_clip=False)
    handles, labels = axes[0].get_legend_handles_labels()
    legend(fig, handles, labels, ncol=1, y=0.995, fontsize=14)
    save(fig, "query_latency_stacked" + suffix)


def hit_figure(hit, suffix):
    width, height = 6.6, 4.95
    panel_width, panel_height = 5.3, 3.5
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.95 / width, 0.6 / height, panel_width / width, panel_height / height])
    bar_w = 0.34
    for ci, (condition, label, color) in enumerate(CONDITIONS[1:]):
        for wi, workload in enumerate(("TPC-H SF50", "ClickBench")):
            value = hit[(workload, condition)]
            xpos = wi + (ci - 0.5) * bar_w
            ax.bar(xpos, value, bar_w * 0.9, color=color, edgecolor="#222222", linewidth=0.6,
                   zorder=3, label=label if wi == 0 else None)
            ax.text(xpos, value + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=12)
    top = max(hit.values())
    hit_ymax = max(30, math.ceil((top + 8) / 10) * 10)
    ax.set_xticks([0, 1], ["TPC-H SF50", "ClickBench"], fontsize=12)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, hit_ymax)
    ax.set_yticks(range(0, hit_ymax + 1, 10))
    ax.set_ylabel("Cache hit ratio (%)", fontsize=13)
    finish(ax)
    handles, labels = ax.get_legend_handles_labels()
    legend(fig, handles, labels, ncol=1, y=0.995)
    save(fig, "cache_hit_bars" + suffix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", choices=("legacy", "same"), default="legacy")
    args = parser.parse_args()
    means = list(csv.DictReader(open(FIG_DIR / "query_latency_means.csv")))
    for r in means:
        r["cumulative_query_wall_s"] = float(r["cumulative_query_wall_s"])
    hit = hits(args.hits)
    suffix = "_same" if args.hits == "same" else ""
    with plt.rc_context(RC):
        latency_figure(means, suffix)
        hit_figure(hit, suffix)
    print({k: round(v, 1) for k, v in hit.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
