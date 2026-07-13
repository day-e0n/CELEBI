#!/usr/bin/env python3
"""Plot cumulative scan/materialize work for fixed-page workload runs."""

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


def to_int(value: object) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except Exception:
        return 0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pick_baseline(rows: list[dict[str, str]], requested: str) -> str:
    series = {row.get("series", "") for row in rows}
    if requested != "auto":
        return requested
    if "baseline" in series:
        return "baseline"
    if "hot" in series:
        return "hot"
    if "cold" in series:
        return "cold"
    raise SystemExit("could not infer baseline series")


def build_cumulative_rows(
    workload: str,
    rows: list[dict[str, str]],
    baseline_series: str,
    paging_series: str,
) -> list[dict[str, object]]:
    by_key = {(row.get("series", ""), to_int(row.get("position"))): row for row in rows}
    positions = sorted(
        {to_int(row.get("position")) for row in rows if row.get("series") == baseline_series}
    )
    out: list[dict[str, object]] = []
    baseline_cum = 0.0
    paging_cum = 0.0
    for position in positions:
        baseline = by_key.get((baseline_series, position))
        paging = by_key.get((paging_series, position))
        if not baseline or not paging:
            continue
        baseline_scan = to_float(baseline.get("scan_materialize_work_ms"))
        paging_scan = to_float(paging.get("scan_materialize_work_ms"))
        baseline_cum += baseline_scan
        paging_cum += paging_scan
        saved = baseline_scan - paging_scan
        cum_saved = baseline_cum - paging_cum
        out.append(
            {
                "workload": workload,
                "position": position,
                "query": baseline.get("query", f"q{position}"),
                "baseline_series": baseline_series,
                "paging_series": paging_series,
                "baseline_scan_materialize_work_ms": baseline_scan,
                "paging_scan_materialize_work_ms": paging_scan,
                "scan_materialize_work_saved_ms": saved,
                "scan_materialize_work_reduction_ratio": saved / baseline_scan
                if baseline_scan
                else 0.0,
                "cumulative_baseline_scan_materialize_work_ms": baseline_cum,
                "cumulative_paging_scan_materialize_work_ms": paging_cum,
                "cumulative_saved_ms": cum_saved,
                "cumulative_reduction_ratio": cum_saved / baseline_cum if baseline_cum else 0.0,
            }
        )
    return out


def plot_cumulative(
    grouped_rows: list[tuple[str, list[dict[str, object]]]],
    out: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig_width = max(16.0, 8.0 * len(grouped_rows))
    fig, axes = plt.subplots(
        1,
        len(grouped_rows),
        figsize=(fig_width, 7.2),
        sharey=False,
        constrained_layout=True,
    )
    if len(grouped_rows) == 1:
        axes = [axes]

    for ax, (workload, rows) in zip(axes, grouped_rows):
        labels = [str(row["query"]) for row in rows]
        x = np.arange(1, len(rows) + 1)
        baseline = np.array(
            [to_float(row["cumulative_baseline_scan_materialize_work_ms"]) / 1000.0 for row in rows]
        )
        paging = np.array(
            [to_float(row["cumulative_paging_scan_materialize_work_ms"]) / 1000.0 for row in rows]
        )
        ax.plot(x, baseline, color="#4c78a8", linewidth=3.0, linestyle="--", label="baseline")
        ax.plot(x, paging, color="#59a14f", linewidth=3.0, marker="o", markersize=4.5, label="paging")
        ax.set_title(workload, fontsize=22, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_xlabel("Query order", fontsize=16, labelpad=10)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.08)

        if len(rows) > 0:
            final = rows[-1]
            pct = to_float(final["cumulative_reduction_ratio"]) * 100.0
            end_x = x[-1]
            y_mid = (baseline[-1] + paging[-1]) / 2.0
            label = f"{abs(pct):.1f}% lower" if pct >= 0 else f"{abs(pct):.1f}% higher"
            ax.annotate(
                label,
                xy=(end_x, y_mid),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=15,
                fontweight="bold",
                color="#2f2f2f",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
            )

    axes[0].set_ylabel("Cumulative scan/materialize work (s)", fontsize=17, labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncols=2,
        frameon=False,
        fontsize=17,
    )
    fig.suptitle(title, fontsize=24, y=1.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        action="append",
        nargs=2,
        metavar=("NAME", "QUERY_SUMMARY_CSV"),
        required=True,
        help="Workload display name and query_summary_breakdown.csv path. Can be repeated.",
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--baseline-series", default="auto")
    parser.add_argument("--paging-series", default="paging")
    parser.add_argument("--title", default="Cumulative scan/materialize work")
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    grouped: list[tuple[str, list[dict[str, object]]]] = []
    for workload_name, query_summary_path in args.workload:
        query_rows = read_rows(Path(query_summary_path))
        baseline = pick_baseline(query_rows, args.baseline_series)
        cumulative = build_cumulative_rows(workload_name, query_rows, baseline, args.paging_series)
        if not cumulative:
            raise SystemExit(f"no cumulative rows for {workload_name}")
        grouped.append((workload_name, cumulative))
        all_rows.extend(cumulative)

    write_csv(args.out_csv, all_rows)
    plot_cumulative(grouped, args.out_png, args.title)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
