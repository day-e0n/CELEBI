#!/usr/bin/env python3
"""Combined 2-panel figure: cumulative eligible cache hit rate (left) and
cumulative scan/materialize work (right), by query position, for 3 stages
(baseline / paging / CELEBI). Matches the reference two-panel layout."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import (  # noqa: E402
    query_order_from_summary,
    stage_page_series,
)


def qnum_key(q: str) -> int:
    return int(q.lstrip("qQ"))


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def scan_ms_by_qnum(query_summary: Path, series_name: str) -> dict[str, float]:
    with query_summary.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("series") == series_name]
    return {r["query"]: to_float(r.get("scan_materialize_work_ms")) for r in rows}


def cumulative_by_qnum(series: list[tuple[str, int, int, int]],
                        query_summary: Path, series_name: str,
                        x_labels: list[str]) -> tuple[list[float], list[float]]:
    """Accumulate PAGE-weighted hit rate (16MB fixed-width pages, not split-count
    events -- see stage_page_series) and scan work over the FIXED q1..q22 order
    (each stage's own value for a given query, looked up regardless of where
    that query actually sat in that stage's execution order)."""
    hit_by_q = {q: (backed, miss, excluded) for q, backed, miss, excluded in series}
    scan_by_q = scan_ms_by_qnum(query_summary, series_name)

    hit_rate, scan_work = [], []
    cbacked = ctotal = cscan = 0.0
    for q in x_labels:
        b, m, e = hit_by_q.get(q, (0, 0, 0))
        cbacked += b
        ctotal += b + m + e
        hit_rate.append(cbacked / ctotal * 100.0 if ctotal else 0.0)
        cscan += scan_by_q.get(q, 0.0) / 1000.0
        scan_work.append(cscan)
    return hit_rate, scan_work


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", nargs=4,
                         metavar=("LABEL", "LOG_DIR", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
                         required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, default=None,
                         help="Pass --executions if a stage's log_dir may have been appended to "
                              "by more than one run invocation on the same day.")
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
        hit_rate, scan_work = cumulative_by_qnum(series, Path(query_summary), series_name, x_labels)
        stage_data.append((label, hit_rate, scan_work))

    n = len(x_labels)
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

    for idx, (label, hit_rate, _) in enumerate(stage_data):
        ax1.plot(x, hit_rate, color=colors[idx % len(colors)], linewidth=2.4,
                  marker="o", markersize=5, label=label)
        ax1.annotate(f"{hit_rate[-1]:.1f}%", (x[-1], hit_rate[-1]), textcoords="offset points",
                     xytext=(8, 0), ha="left", va="center", fontsize=12, color=ink_primary)
    ax1.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    ax1.set_ylabel("Cumulative cache hit ratio (%)", fontsize=18, color=ink_primary)
    ax1.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax1.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    ax1.set_ylim(0, 10)
    ax1.set_xlim(0, n + 3)

    baseline_final = stage_data[0][2][-1]
    for idx, (label, _, scan_work) in enumerate(stage_data):
        ax2.plot(x, scan_work, color=colors[idx % len(colors)], linewidth=2.4,
                  marker="o", markersize=5, label=label)
        final = scan_work[-1]
        pct = (baseline_final - final) / baseline_final * 100.0 if baseline_final else 0.0
        text = f"{final:.1f}s" if idx == 0 else f"{final:.1f}s ({pct:+.1f}%)"
        ax2.annotate(text, (x[-1], final), textcoords="offset points",
                     xytext=(8, 0), ha="left", va="center", fontsize=12, color=ink_primary)
    ax2.set_xlabel("TPC-H query", fontsize=18, color=ink_primary)
    ax2.set_ylabel("Cumulative scan work (s)", fontsize=18, color=ink_primary)
    ax2.tick_params(axis="y", colors=ink_primary, labelcolor=ink_primary, labelsize=15)
    ax2.tick_params(axis="x", colors=ink_primary, labelcolor=ink_primary)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=10, color=ink_primary, rotation=90)
    ax2.set_xlim(0, n + 3)

    handles, labels = ax1.get_legend_handles_labels()
    ncol = 2 if len(stage_data) == 4 else len(stage_data)
    fig.legend(handles, labels, frameon=True, edgecolor=ink_primary, fontsize=17,
               loc="upper center", bbox_to_anchor=(0.5, 1.1 if ncol == 2 else 1.06),
               ncol=ncol, labelcolor=ink_primary)

    fig.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
