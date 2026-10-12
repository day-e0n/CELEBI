#!/usr/bin/env python3
"""Render audited SF50 operator-work means as paired stacked bars."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

BUCKETS = ("scan", "join", "aggregate", "filter", "sort", "other")
COLORS = {
    "scan": "#2a78d6",
    "join": "#eb6834",
    "aggregate": "#1baf7a",
    "filter": "#eda100",
    "sort": "#e87ba4",
    "other": "#008300",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    directory = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=directory / "tpch_sf50_operator_breakdown.csv")
    parser.add_argument("--output-prefix", type=Path, default=directory / "tpch_sf50_operator_breakdown")
    parser.add_argument("--proposed-condition", default="best")
    parser.add_argument("--proposed-label", default="Page caching + reordering + future eviction (CELEBI)")
    parser.add_argument("--expected-samples", type=int, default=6)
    parser.add_argument("--no-caption", action="store_true",
                        help="Leave out the in-figure caption line (the paper sets its own) and crop.")
    args = parser.parse_args()
    with args.input.open(newline="") as source:
        rows = list(csv.DictReader(source))
    by_condition = {condition: {} for condition in ("baseline", args.proposed_condition)}
    for row in rows:
        condition, query = row["condition"], row["query"]
        if condition not in by_condition or query in by_condition[condition]:
            raise ValueError(f"Unexpected/duplicate condition/query: {condition}, {query}")
        if int(row["samples"]) != args.expected_samples:
            raise ValueError(f"Expected {args.expected_samples} completed samples for {condition} {query}")
        values = {bucket: float(row[bucket]) for bucket in BUCKETS}
        if any(value < 0 or not np.isfinite(value) for value in values.values()):
            raise ValueError(f"Invalid measured durations: {row}")
        if not np.isclose(sum(values.values()), float(row["total_ms"]), rtol=1e-8, atol=1e-5):
            raise ValueError(f"Bucket sum differs from total: {row}")
        by_condition[condition][query] = values
    queries = [f"q{q}" for q in range(1, 23)]
    if any(set(data) != set(queries) for data in by_condition.values()):
        raise ValueError("Each condition must contain q1 through q22")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "hatch.linewidth": 0.6, "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(16.8, 6.0))
    fig.subplots_adjust(left=0.075, right=0.989, bottom=0.19, top=0.80)
    positions = np.arange(22)
    width = 0.34
    for offset, condition, hatch in ((-1, "baseline", None), (1, args.proposed_condition, "////")):
        bottom = np.zeros(22)
        for bucket in BUCKETS:
            values = np.array([by_condition[condition][query][bucket] for query in queries])
            ax.bar(positions + offset * width / 2, values, width=width,
                   bottom=bottom, color=COLORS[bucket], edgecolor="black",
                   linewidth=0.55, hatch=hatch, zorder=3)
            bottom += values

    ax.set_xticks(positions, queries)
    ax.set_xlabel("TPC-H query", fontsize=15)
    ax.set_ylabel("GPU operator work time (ms)", fontsize=15)
    ax.set_xlim(-0.7, 21.7)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.tick_params(axis="both", labelsize=12, length=3)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")

    color_legend = fig.legend(
        handles=[Patch(facecolor=COLORS[bucket], edgecolor="black", linewidth=0.5,
                       label=bucket) for bucket in BUCKETS],
        loc="upper center", bbox_to_anchor=(0.54, 0.98), ncol=6,
        fontsize=12, handlelength=2.9, columnspacing=1.6,
        fancybox=False, frameon=True, edgecolor="#444444", borderpad=0.8,
    )
    color_legend.get_frame().set_linewidth(0.7)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="black", linewidth=0.6, label="Baseline"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.6, hatch="////",
              label=args.proposed_label),
    ], loc="upper left", fontsize=10, handlelength=3.0, borderpad=0.65,
       labelspacing=0.5, fancybox=False, frameon=True, edgecolor="#444444")
    if not args.no_caption:
        fig.text(0.54, 0.058, "TPC-H SF50: operator work breakdown, Baseline vs. CELEBI.",
                 ha="center", va="center", fontfamily="DejaVu Serif", fontsize=15,
                 fontweight="bold")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        path = args.output_prefix.with_suffix(f".{extension}")
        fig.savefig(path, dpi=200, facecolor="white",
                    bbox_inches="tight" if args.no_caption else None, pad_inches=0.05)
        print(path)
    plt.close(fig)
    for condition, data in by_condition.items():
        print(condition, {bucket: round(sum(row[bucket] for row in data.values()), 3)
                          for bucket in BUCKETS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
