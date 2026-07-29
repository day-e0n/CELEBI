#!/usr/bin/env python3
"""Plot total pipeline latency before vs. after reordering, as two vertical bars.

Left bar: no-reorder total workload latency (paging series, original order) --
this is what you pay if you never reorder at all.

Right bar: reorder total pipeline latency = the reordered workload's total
latency PLUS the one-time reorder_elapsed_ms cost of computing the order --
i.e. the full, honest end-to-end cost of "compute the order, then run it."
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def workload_total_ms(workload_summary: Path, series_name: str) -> float:
    with workload_summary.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("series") == series_name]
    return to_float(rows[0]["total_ms"]) if rows else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reorder-run-dir", type=Path, required=True, help="Run dir containing metadata.json")
    parser.add_argument("--noreorder-workload-summary", type=Path, required=True)
    parser.add_argument("--reorder-workload-summary", type=Path, required=True)
    parser.add_argument("--series-name", default="paging")
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.reorder_run_dir / "metadata.json").read_text())
    reorder_ms = to_float(metadata.get("reorder_elapsed_ms"))

    before_ms = workload_total_ms(args.noreorder_workload_summary, args.series_name)
    after_ms = workload_total_ms(args.reorder_workload_summary, args.series_name) + reorder_ms

    print(f"before (no reorder) total_ms = {before_ms:.1f} ms")
    print(f"after (reorder cost + reordered run) total_ms = {after_ms:.1f} ms")
    print(f"net savings = {before_ms - after_ms:.1f} ms ({(before_ms - after_ms) / before_ms * 100.0:.2f}%)")

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]

    labels = ["Before reorder", "After reorder"]
    values = [before_ms, after_ms]
    colors = ["#9AA1A9", "#3B5E93"]
    pct = (before_ms - after_ms) / before_ms * 100.0

    # Truncated y-axis (starts above 0) so the ~4% drop reads visually, with an
    # explicit axis-break mark so the truncation is disclosed, not hidden.
    y_bottom = 16000.0
    y_top = before_ms * 1.06

    fig, ax = plt.subplots(figsize=(2.6, 2.5), constrained_layout=True)
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Total pipeline latency (ms)", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylim(bottom=y_bottom, top=y_top)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)

    # Axis-break marks (standard "//" convention for a non-zero baseline).
    for xpos in (-0.06, 0.0):
        ax.plot(
            [xpos - 0.015, xpos + 0.015], [-0.02, 0.02],
            transform=ax.transAxes, color="black", linewidth=1.0, clip_on=False,
        )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + (y_top - y_bottom) * 0.015,
            f"{value:,.0f} ms", ha="center", va="bottom", fontsize=8,
        )

    # Diagonal arrow in the gap between the bars, pointing down-right.
    arrow = FancyArrowPatch(
        (0.30, before_ms - (before_ms - after_ms) * 0.15),
        (0.68, after_ms + (before_ms - after_ms) * 0.85),
        connectionstyle="arc3,rad=-0.25",
        arrowstyle="-|>", mutation_scale=14,
        color="#D62728", linewidth=2.4,
    )
    ax.add_patch(arrow)
    ax.text(
        0.5, (before_ms + after_ms) / 2 + (before_ms - after_ms) * 0.55,
        f"−{pct:.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color="#D62728",
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
