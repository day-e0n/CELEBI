#!/usr/bin/env python3
"""Plot cumulative random-workload query cost: SiriusDB baseline vs SiriusDB+paging+reorder vs DuckDB CPU.

Reads:
- baseline: series="baseline" total_ms from a non-reorder fixed-page run's
  summary/query_summary_breakdown.csv (original random query order)
- paging+reorder: series="paging" total_ms from the matching *_reorder_* run's
  summary/query_summary_breakdown.csv, plus that run's one-time
  reorder_elapsed_ms from metadata.json
- duckdb cpu: csv/runtimes.csv from a performance_test.py --engine cpu run
  executed with the same query order as the SiriusDB baseline
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_metadata(query_summary_path: Path) -> dict[str, object]:
    metadata_path = query_summary_path.parent.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def baseline_series(query_summary: Path, series_name: str = "baseline") -> list[tuple[str, float]]:
    rows = [r for r in read_rows(query_summary) if r.get("series") == series_name]
    rows.sort(key=lambda r: int(r["position"]))
    return [(r["query"], to_float(r["total_ms"])) for r in rows]


def reorder_paging_series(query_summary: Path) -> tuple[list[tuple[str, float]], float]:
    rows = [r for r in read_rows(query_summary) if r.get("series") == "paging"]
    rows.sort(key=lambda r: int(r["position"]))
    metadata = read_metadata(query_summary)
    reorder_ms = to_float(metadata.get("reorder_elapsed_ms"))
    return [(r["query"], to_float(r["total_ms"])) for r in rows], reorder_ms


def duckdb_cpu_series(runtimes_csv: Path, skip_first_iteration: bool = False) -> list[tuple[str, float]]:
    """Average runtime_s across iterations per query, in first-seen (run) order.

    Mirrors the SiriusDB side: with skip_first_iteration, drops iteration 0
    (cold) and averages the rest, matching the "hot" series definition
    (avg of executions 2..N) used by run_fixed_page_workload_sequence.py's
    cold_hot summary mode.
    """
    rows = [r for r in read_rows(runtimes_csv) if r.get("engine") == "duckdb"]
    order: list[str] = []
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in rows:
        if skip_first_iteration and to_float(r.get("iteration")) == 0:
            continue
        query = r["query"]
        if query not in totals:
            order.append(query)
            totals[query] = 0.0
            counts[query] = 0
        totals[query] += to_float(r["runtime_s"]) * 1000.0
        counts[query] += 1
    return [(query, totals[query] / counts[query]) for query in order]


def cumulative(series: list[tuple[str, float]], head_offset_ms: float = 0.0) -> list[float]:
    out = []
    total = head_offset_ms
    for _, ms in series:
        total += ms
        out.append(total)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        action="append",
        nargs=4,
        metavar=("LABEL", "BASELINE_QUERY_SUMMARY", "REORDER_QUERY_SUMMARY", "DUCKDB_RUNTIMES_CSV"),
        required=True,
    )
    parser.add_argument(
        "--no-reorder-query-summary",
        type=Path,
        default=None,
        help="Optional: query_summary_breakdown.csv with a 'paging' series (no reorder) to plot as a 4th line.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--title", default="Random workload cumulative query latency")
    parser.add_argument("--baseline-series", default="baseline")
    parser.add_argument("--duckdb-skip-cold", action="store_true")
    parser.add_argument("--third-label", default="SiriusDB + paging + reorder")
    parser.add_argument("--fourth-label", default="SiriusDB + paging (no reorder)")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    panels = []
    for label, baseline_path, reorder_path, duckdb_path in args.workload:
        baseline = baseline_series(Path(baseline_path), args.baseline_series)
        reorder, reorder_ms = reorder_paging_series(Path(reorder_path))
        duckdb = duckdb_cpu_series(Path(duckdb_path), args.duckdb_skip_cold)
        no_reorder = None
        if args.no_reorder_query_summary:
            no_reorder, _ = reorder_paging_series(args.no_reorder_query_summary)
        panels.append((label, baseline, reorder, reorder_ms, duckdb, no_reorder))

    fig, axes = plt.subplots(1, len(panels), figsize=(max(16.0, 7.0 * len(panels)), 6.6), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (label, baseline, reorder, reorder_ms, duckdb, no_reorder) in zip(axes, panels):
        n = len(baseline)
        x = np.arange(1, n + 1)
        baseline_cum = np.array(cumulative(baseline)) / 1000.0
        reorder_cum = np.array(cumulative(reorder, head_offset_ms=reorder_ms)) / 1000.0
        duckdb_cum = np.array(cumulative(duckdb)) / 1000.0

        ax.plot(x, duckdb_cum, color="#e45756", linewidth=2.6, linestyle=":", marker="s", markersize=4, label="DuckDB CPU")
        ax.plot(x, baseline_cum, color="#4c78a8", linewidth=2.6, marker="o", markersize=4, label="SiriusDB")
        if no_reorder is not None:
            no_reorder_cum = np.array(cumulative(no_reorder)) / 1000.0
            ax.plot(x, no_reorder_cum, color="#f2b134", linewidth=2.6, marker="o", markersize=4, label=args.fourth_label)
        ax.plot(x, reorder_cum, color="#59a14f", linewidth=2.6, marker="o", markersize=4, label=args.third_label)

        ax.set_title(label, fontsize=20, pad=12)
        ax.set_xlabel("Query position in workload", fontsize=15)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.25)

        pct_vs_baseline = (baseline_cum[-1] - reorder_cum[-1]) / baseline_cum[-1] * 100.0 if baseline_cum[-1] else 0.0
        reorder_line = f"reorder: {reorder_ms:.2f} ms\n" if reorder_ms > 1.0 else ""
        ax.text(
            0.02,
            0.96,
            f"{reorder_line}{pct_vs_baseline:.1f}% lower vs SiriusDB",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="#2f2f2f",
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
        )

    axes[0].set_ylabel("Cumulative query latency (s)", fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncols=3, frameon=False, fontsize=15)
    fig.suptitle(args.title, fontsize=22, y=1.18)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
