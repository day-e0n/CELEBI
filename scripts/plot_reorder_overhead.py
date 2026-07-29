#!/usr/bin/env python3
"""Plot the one-time query-reorder computation cost against what it buys.

Reads reorder_elapsed_ms from a reorder run's metadata.json (the one-time cost
of computing the reordered query sequence, paid once per workload, not per
query) and compares it against other wall-clock (total_ms) quantities only --
scan_materialize_work_ms is a work-sum metric (can exceed wall-clock time under
parallel execution) and is deliberately NOT mixed into this chart:
  - the wall-clock workload latency saved by reorder (paging+reorder total_ms
    vs paging-no-reorder total_ms, both from workload_summary_breakdown.csv)
  - the total single-pass workload latency, for scale.
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

    noreorder_total_ms = workload_total_ms(args.noreorder_workload_summary, args.series_name)
    reorder_total_ms = workload_total_ms(args.reorder_workload_summary, args.series_name)
    saved_ms = noreorder_total_ms - reorder_total_ms

    print(f"reorder_elapsed_ms = {reorder_ms:.2f} ms")
    print(f"paging (no reorder) total_ms = {noreorder_total_ms:.1f} ms")
    print(f"paging + reorder total_ms = {reorder_total_ms:.1f} ms")
    print(f"wall-clock latency saved by reorder = {saved_ms:.1f} ms ({saved_ms / reorder_ms:.0f}x reorder cost)")
    print(f"single-pass workload total_ms (with reorder) = {reorder_total_ms:.1f} ms ({reorder_total_ms / reorder_ms:.0f}x reorder cost)")
    print(f"reorder cost as %% of workload total_ms = {reorder_ms / reorder_total_ms * 100.0:.3f}%%")
    workload_ms = reorder_total_ms

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]

    roi = saved_ms / reorder_ms

    labels = ["Reorder cost", "Latency saved"]
    values = [reorder_ms, saved_ms]
    accent_light = "#7FA0D4"
    accent_dark = "#3B5E93"
    colors = [accent_light, accent_dark]

    fig, ax = plt.subplots(figsize=(3.6, 2.0), constrained_layout=True)
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Milliseconds (log scale)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=9.5)
    ax.invert_yaxis()

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() * 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.1f} ms",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlim(left=reorder_ms * 0.6, right=saved_ms * 55)

    # Hero number in the open space to the right of the bars, clear of any text.
    ax.text(
        0.99, 0.5, f"{roi:.0f}×",
        transform=ax.transAxes, ha="right", va="center",
        fontsize=30, fontweight="bold", color=accent_dark,
    )
    ax.text(
        0.99, 0.14, "return", transform=ax.transAxes, ha="right", va="center",
        fontsize=9, color="#555555",
    )

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
