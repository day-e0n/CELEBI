#!/usr/bin/env python3
"""Per-query scan/materialize time as grouped bars + PAGE-weighted cache hit
ratio (16MB fixed-width pages, dynamic-filter scans counted as misses) as
lines, dual y-axis, x-axis = FIXED canonical query identity Q1..Q22 (not
position -- each stage's own value for a given qnum is looked up regardless of
where that query sat in that stage's execution order). Hot executions only,
matching stage_page_series's warmup-exclusion convention.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import query_order_from_summary, stage_page_series  # noqa: E402


def to_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def qnum_key(q: str) -> int:
    return int(q.lstrip("qQ"))


def per_query_scan_ms_by_qnum(query_summary: Path, series_name: str) -> dict[str, float]:
    with query_summary.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("series") == series_name]
    return {r["query"]: to_float(r.get("scan_materialize_work_ms")) for r in rows}


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

    ink_primary = "#0b0b0b"
    colors = ["#4c78a8", "#59a14f", "#f2b134", "#e45756"]

    # canonical, fixed x-axis: q1..q22 regardless of any stage's own order
    all_qnums = sorted({qnum_key(f"q{i}") for i in range(1, 23)})
    x_labels = [f"q{i}" for i in all_qnums]

    stages = []
    for label, log_dir, qsummary, series in args.stage:
        order = query_order_from_summary(Path(qsummary), series)
        hit_series = stage_page_series(Path(log_dir), order, expected_iterations=args.expected_iterations)
        hit_by_q: dict[str, float] = {}
        for q, backed, miss, excluded in hit_series:
            total = backed + miss + excluded
            hit_by_q[q] = backed / total * 100.0 if total else 0.0
        scan_by_q = per_query_scan_ms_by_qnum(Path(qsummary), series)

        scan_s = [scan_by_q.get(ql, 0.0) / 1000.0 for ql in x_labels]
        hit_rates = [hit_by_q.get(ql, 0.0) for ql in x_labels]
        stages.append((label, scan_s, hit_rates))

    n = len(x_labels)
    x = np.arange(1, n + 1)
    width = 0.8 / len(stages)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.08})

    for idx, (label, scan_s, _) in enumerate(stages):
        offset = (idx - (len(stages) - 1) / 2) * width
        ax1.bar(x + offset, scan_s, width=width, color=colors[idx % len(colors)],
                edgecolor=ink_primary, linewidth=0.4, alpha=0.85, label=label)
    ax1.set_ylabel("Scan time (s)", fontsize=20, color=ink_primary)
    ax1.tick_params(axis="y", labelcolor=ink_primary, labelsize=16)

    for idx, (label, _, hit_rates) in enumerate(stages):
        ax2.plot(x, hit_rates, color=colors[idx % len(colors)], linewidth=2.2,
                 marker="o", markersize=5, linestyle="--", label=label)
    ax2.set_ylabel("Cache hit ratio\n(16MB page-weighted, %)", fontsize=18, color=ink_primary)
    ax2.tick_params(axis="y", labelcolor=ink_primary, labelsize=16)
    ax2.set_ylim(0, 105)

    ax2.set_xlabel("TPC-H query", fontsize=20, color=ink_primary, labelpad=10)
    ax2.tick_params(axis="x", labelsize=19, colors=ink_primary)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=19, color=ink_primary)
    ax2.set_xlim(0.3, n + 0.7)

    fig.tight_layout()
    fig.subplots_adjust(top=0.80)

    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, frameon=True, edgecolor=ink_primary, fontsize=20,
               loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=2, labelcolor=ink_primary)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
