#!/usr/bin/env python3
"""The AVG-fix latency figure with its right panel replaced by cache-hit bars.

Left and middle panels are drawn exactly as result/plot_avg_fix_query_latency_and_hits.py draws
them, from its output experiment/AVG_FIX_CB_20261007/figures/query_latency_means.csv.
Right panel: byte cache hit ratio as bars, page caching vs CELEBI.

  --hits legacy   result/cache_hit_rate.csv (HIT_* runs, 2026-10-06; engine before the
                  2026-10-07 changes, ClickBench 41 queries, metric v1)          [default]
  --hits same     the same figure's own hit summary (AVG_FIX_CB_20261007, value_bytes_v2,
                  ClickBench 43 queries)

  pixi run -e duckdb-python python result/plot_latency_and_hit_bars.py [--hits same]
writes result/query_latency_and_cache_hit_bars{,_same}.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "experiment" / "AVG_FIX_CB_20261007" / "figures"
CONDITIONS = (
    ("baseline", "Baseline", "#4c7caf"),
    ("paging_lru", "Baseline+page caching", "#54a34b"),
    ("reorder_future", "Baseline+page caching+reordering (CELEBI)", "#f5b12b"),
)


def finish(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.tick_params(length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.7)


def hits(source: str) -> dict[tuple[str, str], float]:
    """(workload title, condition key) -> byte hit %."""
    if source == "legacy":
        names = {"paging": "paging_lru", "paging + reorder": "reorder_future"}
        return {(r["workload"], names[r["condition"]]): float(r["byte_hit_pct"])
                for r in csv.DictReader(open(OUT / "cache_hit_rate.csv"))
                if r["condition"] in names}
    titles = {"th": "TPC-H SF50", "cb": "ClickBench"}
    return {(titles[r["workload"]], r["condition"]): float(r["byte_hit_pct"])
            for r in csv.DictReader(open(FIG_DIR / "cache_hit_summary.csv"))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hits", choices=("legacy", "same"), default="legacy")
    args = parser.parse_args()
    means = list(csv.DictReader(open(FIG_DIR / "query_latency_means.csv")))
    for r in means:
        r["cumulative_query_wall_s"] = float(r["cumulative_query_wall_s"])
    hit = hits(args.hits)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "savefig.bbox": None}):
        width, height = 20.5, 5.5
        panel_width, panel_height = 5.3, 3.5
        fig = plt.figure(figsize=(width, height))
        axes = [fig.add_axes([x / width, 1.05 / height, panel_width / width, panel_height / height])
                for x in (0.82, 7.5, 14.18)]
        max_latency = max(r["cumulative_query_wall_s"] for r in means)
        tick_step = 10 if max_latency > 40 else 5
        ymax = math.ceil((max_latency * 1.08) / tick_step) * tick_step
        for ax, title in zip(axes[:2], ("ClickBench", "TPC-H SF50")):
            ends = []
            for condition, label, color in CONDITIONS:
                rows = [r for r in means if r["workload"] == title and r["condition"] == condition]
                n = len(rows)
                x = list(range(1, n + 1))
                y = [r["cumulative_query_wall_s"] for r in rows]
                ax.plot(x, y, color=color, marker="o", markersize=3.3, linewidth=1.6, label=label)
                ends.append((y[-1], color))
            ax.set_xticks(x, [r["query"] for r in rows], rotation=90, fontsize=8)
            ax.set_xlim(0.4, n + 0.6)
            ax.set_ylim(0, ymax)
            ax.set_yticks(range(0, ymax + 1, tick_step))
            ax.set_xlabel(f"{'ClickBench' if title == 'ClickBench' else 'TPC-H'} query", fontsize=13)
            ax.set_title(title, fontsize=16, pad=12)
            if title == "ClickBench":
                ax.set_ylabel("Cumulative query latency (s)", fontsize=13)
            finish(ax)
            upper = float("inf")
            for value, _ in sorted(ends, reverse=True):
                position = min(value, upper - 0.05 * ymax)
                upper = position
                ax.annotate(f"{value:.1f}s", xy=(n, value), xytext=(1.025, position),
                            textcoords=ax.get_yaxis_transform(), fontsize=12, va="center",
                            annotation_clip=False)

        # Right panel: bars, colours of the two cached conditions above.
        ax = axes[2]
        bar_w = 0.34
        for ci, (condition, label, color) in enumerate(CONDITIONS[1:]):
            for wi, workload in enumerate(("TPC-H SF50", "ClickBench")):
                value = hit[(workload, condition)]
                xpos = wi + (ci - 0.5) * bar_w
                ax.bar(xpos, value, bar_w * 0.9, color=color, edgecolor="#222222", linewidth=0.6,
                       zorder=3)
                ax.text(xpos, value + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=12)
        top = max(hit.values())
        hit_ymax = max(30, math.ceil((top + 8) / 10) * 10)
        ax.set_xticks([0, 1], ["TPC-H SF50", "ClickBench"], fontsize=12)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, hit_ymax)
        ax.set_yticks(range(0, hit_ymax + 1, 10))
        ax.set_ylabel("Byte cache hit ratio (%)", fontsize=13)
        ax.set_title("Cache hit ratio", fontsize=16, pad=12)
        finish(ax)

        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
                            frameon=True, edgecolor="#777777", facecolor="white", fontsize=12,
                            handlelength=2.5, columnspacing=2, borderpad=0.6)
        legend.get_frame().set_linewidth(0.7)
        stem = "query_latency_and_cache_hit_bars" + ("_same" if args.hits == "same" else "")
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=220, facecolor="white")
        plt.close(fig)
    print({k: round(v, 1) for k, v in hit.items()})
    print(OUT / f"{stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
