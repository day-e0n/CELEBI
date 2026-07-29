#!/usr/bin/env python3
"""Plot cumulative scan/materialize work across multiple pipeline stages
(e.g. baseline, paging, naive reorder, budget-aware reorder) for one workload.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def series_scan_work(query_summary: Path, series_name: str, metric: str) -> list[tuple[str, float]]:
    rows = [r for r in read_rows(query_summary) if r.get("series") == series_name]
    rows.sort(key=lambda r: int(r["position"]))
    return [(r["query"], to_float(r[metric])) for r in rows]


def cumulative(series: list[tuple[str, float]]) -> list[float]:
    out = []
    total = 0.0
    for _, ms in series:
        total += ms
        out.append(total)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=3,
        metavar=("LABEL", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
        required=True,
        help="Repeatable. One cumulative-scan-work line per stage.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--title", default="Cumulative scan/materialize work by stage")
    parser.add_argument("--metric", default="scan_materialize_work_ms", help="query_summary_breakdown.csv column to plot")
    parser.add_argument("--ylabel", default="Cumulative scan/materialize work (s)")
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    colors = ["#4c78a8", "#f2b134", "#e45756", "#59a14f", "#9d6bb0"]

    stages = []
    for label, csv_path, series_name in args.stage:
        series = series_scan_work(Path(csv_path), series_name, args.metric)
        stages.append((label, series))

    n = len(stages[0][1])
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(max(11, n * 0.45), 7.2), constrained_layout=True)
    baseline_final = None
    for idx, (label, series) in enumerate(stages):
        cum = np.array(cumulative(series)) / 1000.0
        if baseline_final is None:
            baseline_final = cum[-1]
            pct_label = label
        else:
            pct = (baseline_final - cum[-1]) / baseline_final * 100.0 if baseline_final else 0.0
            pct_label = f"{label} ({pct:.1f}% lower)"
        ax.plot(x, cum, color=colors[idx % len(colors)], linewidth=3.0, marker="o", markersize=5, label=pct_label)

    ax.set_ylabel(args.ylabel, fontsize=26)
    ax.set_xlabel("Query position in workload (order differs by stage)", fontsize=22)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=21)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=21, frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=190, bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
