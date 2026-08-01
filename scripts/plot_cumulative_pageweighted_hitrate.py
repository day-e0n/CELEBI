#!/usr/bin/env python3
"""Single-panel figure: cumulative PAGE-weighted cache hit ratio (16MB
fixed-width pages) over the FIXED q1..q22 order. Same data/definition as the
left panel of plot_hitrate_and_scanwork_combined.py, standalone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import query_order_from_summary, stage_page_series  # noqa: E402
from plot_hitrate_and_scanwork_combined import cumulative_by_qnum  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=4,
                         metavar=("LABEL", "LOG_DIR", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
                         required=True)
    parser.add_argument("--expected-iterations", type=int, default=None,
                         help="Pass --executions if a stage's log_dir may have been appended to "
                              "by more than one run invocation on the same day.")
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    colors = ["#4c78a8", "#59a14f", "#f2b134", "#e45756"]
    ink_primary = "#0b0b0b"

    x_labels = [f"q{i}" for i in range(1, 23)]

    stage_data = []
    for label, log_dir, query_summary, series_name in args.stage:
        query_order = query_order_from_summary(Path(query_summary), series_name)
        series = stage_page_series(Path(log_dir), query_order, expected_iterations=args.expected_iterations)
        hit_rate, _scan_work = cumulative_by_qnum(series, Path(query_summary), series_name, x_labels)
        stage_data.append((label, hit_rate))

    n = len(x_labels)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(9, 6.5))

    for idx, (label, hit_rate) in enumerate(stage_data):
        ax1.plot(x, hit_rate, color=colors[idx % len(colors)], linewidth=2.4,
                  marker="o", markersize=5, label=label)
        ax1.annotate(f"{hit_rate[-1]:.1f}%", (x[-1], hit_rate[-1]), textcoords="offset points",
                     xytext=(8, 0), ha="left", va="center", fontsize=12, color=ink_primary)
    ax1.set_xlabel("TPC-H query", fontsize=14, color=ink_primary)
    ax1.set_ylabel("Cumulative cache hit ratio\n(16MB page-weighted, %)", fontsize=15, color=ink_primary)
    ax1.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, n + 3)

    handles, labels = ax1.get_legend_handles_labels()
    ncol = 2 if len(stage_data) == 4 else len(stage_data)
    fig.legend(handles, labels, frameon=True, edgecolor=ink_primary, fontsize=12,
               loc="upper center", bbox_to_anchor=(0.5, 1.14 if ncol == 2 else 1.1),
               ncol=ncol, labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
