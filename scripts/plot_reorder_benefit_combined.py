#!/usr/bin/env python3
"""Combined figure: cumulative cache hit rate (top) and cumulative
scan/materialize work (bottom) across the same three pipeline stages
(baseline, paging-no-reorder, paging+reorder), sharing one x-axis and one
legend so the two panels read as a single comparison rather than two
unrelated charts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import (  # noqa: E402
    cumulative as cumulative_hitrate,
    query_order_from_summary,
    stage_eligible_series,
)
from plot_scan_work_stages import cumulative as cumulative_scanwork, series_scan_work  # noqa: E402

# unified 3-stage color scheme, shared by both panels
COLORS = ["#4c78a8", "#59a14f", "#f2b134"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hitrate-stage",
        action="append",
        nargs=4,
        metavar=("LABEL", "LOG_DIR", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
        required=True,
        help="Repeatable, top panel. Same 4-tuple plot_cache_hit_rate_eligible.py takes.",
    )
    parser.add_argument(
        "--scanwork-stage",
        action="append",
        nargs=3,
        metavar=("LABEL", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
        required=True,
        help="Repeatable, bottom panel. Same 3-tuple plot_scan_work_stages.py takes.",
    )
    parser.add_argument("--scanwork-metric", default="scan_materialize_work_ms")
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-pdf", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    hit_stages = []
    for label, log_dir, query_summary, series_name in args.hitrate_stage:
        query_order = query_order_from_summary(Path(query_summary), series_name)
        series = stage_eligible_series(Path(log_dir), query_order)
        hit_stages.append((label, cumulative_hitrate(series)))

    work_stages = []
    for label, query_summary, series_name in args.scanwork_stage:
        series = series_scan_work(Path(query_summary), series_name, args.scanwork_metric)
        cum = (np.array(cumulative_scanwork(series)) / 1000.0).tolist()
        work_stages.append((label, cum))

    n = len(hit_stages[0][1])
    if len({len(s) for _, s in hit_stages} | {len(s) for _, s in work_stages}) != 1:
        raise ValueError("all stages (both panels) must have the same number of query positions")
    x = np.arange(1, n + 1)

    ink_primary = "#0b0b0b"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(15, n * 0.6), 6.4))

    def declutter(values: list[float], min_gap: float) -> list[float]:
        """Nudge a set of label y-positions apart (in data units) so nearly-equal
        end-of-line values don't render as overlapping text."""
        order = sorted(range(len(values)), key=lambda i: values[i])
        adjusted = list(values)
        for j in range(1, len(order)):
            prev_i, cur_i = order[j - 1], order[j]
            if adjusted[cur_i] - adjusted[prev_i] < min_gap:
                adjusted[cur_i] = adjusted[prev_i] + min_gap
        return adjusted

    for idx, (label, rate) in enumerate(hit_stages):
        ax1.plot(x, rate, color=COLORS[idx % len(COLORS)], linewidth=2.6, marker="o", markersize=6, label=label)
    hit_final = [rate[-1] for _, rate in hit_stages]
    hit_label_y = declutter(hit_final, max(hit_final) * 0.09)
    for idx, ((label, rate), y) in enumerate(zip(hit_stages, hit_label_y)):
        ax1.annotate(f"{rate[-1]:.1f}%", xy=(1.0, y), xycoords=ax1.get_yaxis_transform(),
                     textcoords="offset points", xytext=(8, 0), annotation_clip=False,
                     fontsize=14, color=ink_primary, va="center")
    ax1.set_ylabel("Cumulative cache hit rate\namong cacheable scans (%)", fontsize=18, color=ink_primary)
    ax1.set_ylim(0, max(v for _, rate in hit_stages for v in rate) * 1.30)
    ax1.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax1.grid(False)

    baseline_final = work_stages[0][1][-1]
    for idx, (label, cum) in enumerate(work_stages):
        ax2.plot(x, cum, color=COLORS[idx % len(COLORS)], linewidth=2.6, marker="o", markersize=6, label=label)
    work_final = [cum[-1] for _, cum in work_stages]
    work_label_y = declutter(work_final, max(work_final) * 0.07)
    for idx, ((label, cum), y) in enumerate(zip(work_stages, work_label_y)):
        note = "" if idx == 0 else f" (-{(baseline_final - cum[-1]) / baseline_final * 100.0:.1f}%)"
        ax2.annotate(f"{cum[-1]:.1f}s{note}", xy=(1.0, y), xycoords=ax2.get_yaxis_transform(),
                     textcoords="offset points", xytext=(8, 0), annotation_clip=False,
                     fontsize=14, color=ink_primary, va="center")
    ax2.set_ylabel("Cumulative scan/materialize work (s)", fontsize=18, color=ink_primary)
    ax2.tick_params(colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax2.grid(False)

    fig.subplots_adjust(top=0.80, bottom=0.16, wspace=0.32)
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    center_x = (pos1.x0 + pos2.x1) / 2

    fig.supxlabel("Query position in workload (order differs by stage)", x=center_x, y=0.01,
                  fontsize=21, color=ink_primary)

    handles, labels = ax1.get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(center_x, 0.97),
        ncols=3,
        frameon=True,
        fontsize=17,
        labelcolor=ink_primary,
        handlelength=2.6,
        handleheight=0.7,
        columnspacing=1.4,
        edgecolor="black",
        fancybox=False,
        borderpad=0.4,
    )
    legend.get_frame().set_linewidth(1.2)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    if args.out_pdf:
        fig.savefig(args.out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
