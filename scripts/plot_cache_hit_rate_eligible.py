#!/usr/bin/env python3
"""Plot cache hit rate among cache-*eligible* scans only, per query position.

Naive hit rate (plot_cache_hit_rate_timeline.py) divides backed_provider_count
by (backed_provider_count + auto_populate_count + auto_skip_count). But most
auto_cache_skip events are reason=dynamic_filter_scan: scans with a runtime
(join-pushdown) filter attached, which the fixed-page auto-cache refuses to
even attempt to cache (see parquet_gpu_ingestible.cpp:759) -- this is a
structural exclusion unrelated to cache budget or query order, and dilutes the
naive hit rate regardless of how good reordering is.

This script re-segments the raw Sirius log per query (same QueryBegin-based
segmentation as run_fixed_page_workload_sequence.py, so positions line up 1:1
with query_summary_breakdown.csv) and computes:

    eligible_hit_rate = backed_provider_count
                         / (backed_provider_count + auto_populate_count
                            + auto_cache_skip[admission_entry_bytes]
                            + auto_cache_skip[admission_previously_rejected])

excluding auto_cache_skip[dynamic_filter_scan] from the denominator entirely.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_fixed_page_workload_sequence import is_timed_query_begin  # noqa: E402

SKIP_REASON_RE = re.compile(r"auto_cache_skip reason=(\w+)")


def query_order_from_summary(query_summary: Path, series_name: str) -> list[str]:
    with query_summary.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("series") == series_name]
    rows.sort(key=lambda r: int(r["position"]))
    return [r["query"] for r in rows]


def segment_counts(lines: list[str]) -> tuple[int, int, int]:
    """Returns (backed, eligible_denom_extra, excluded_dynamic_filter) for one query segment."""
    backed = 0
    eligible_denom_extra = 0
    excluded_dynamic_filter = 0
    for line in lines:
        if "using page-backed cached provider" in line or "using hybrid page/chunk cached provider" in line:
            backed += 1
        elif "auto_cache_populate" in line:
            eligible_denom_extra += 1
        elif "auto_cache_skip" in line:
            m = SKIP_REASON_RE.search(line)
            reason = m.group(1) if m else ""
            if reason == "dynamic_filter_scan":
                excluded_dynamic_filter += 1
            else:
                eligible_denom_extra += 1
    return backed, eligible_denom_extra, excluded_dynamic_filter


def all_query_segments(log_dir: Path) -> list[list[str]]:
    """Segments the raw Sirius log per timed query, across the *entire* log file
    (all repeated executions), unlike run_fixed_page_workload_sequence.log_segments
    which only keeps the first `expected_count` segments (i.e. just the warmup pass
    for the paging/paging_warmup condition).
    """
    logs = sorted(log_dir.glob("*.log"))
    lines = logs[-1].read_text(errors="replace").splitlines()
    begins = [idx for idx, line in enumerate(lines) if is_timed_query_begin(line)]
    segments = []
    for pos, start in enumerate(begins):
        end = begins[pos + 1] if pos + 1 < len(begins) else len(lines)
        segments.append(lines[start:end])
    return segments


def stage_eligible_series(log_dir: Path, query_order: list[str]) -> list[tuple[str, float, int, int, int]]:
    """Sums counts over the executions that query_summary_breakdown.csv's series
    actually includes: the first execution of the workload is a warmup pass
    excluded from both the 'hot'/'cold' and 'paging'/'paging_warmup' series
    (see run_fixed_page_workload_sequence.series_for) -- only executions 2..N
    are included, so we skip the first `len(query_order)` segments here too.
    """
    n = len(query_order)
    segments = all_query_segments(log_dir)
    if len(segments) % n != 0:
        raise ValueError(f"{log_dir}: {len(segments)} query segments not divisible by {n} queries")
    num_iterations = len(segments) // n
    if num_iterations < 2:
        raise ValueError(f"{log_dir}: only {num_iterations} iteration(s) found; need >=2 (1 warmup + >=1 counted)")

    out = []
    for p, query in enumerate(query_order):
        backed = eligible_denom_extra = excluded = 0
        for it in range(1, num_iterations):  # skip iteration 0 (warmup)
            b, e, x = segment_counts(segments[it * n + p])
            backed += b
            eligible_denom_extra += e
            excluded += x
        denom = backed + eligible_denom_extra
        rate = backed / denom * 100.0 if denom else 0.0
        out.append((query, rate, backed, denom, excluded))
    return out


def cumulative(series: list[tuple[str, float, int, int, int]]) -> list[float]:
    out = []
    cbacked = cdenom = 0
    for _, _, backed, denom, _ in series:
        cbacked += backed
        cdenom += denom
        out.append(cbacked / cdenom * 100.0 if cdenom else 0.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=4,
        metavar=("LABEL", "LOG_DIR", "QUERY_SUMMARY_CSV", "SERIES_NAME"),
        required=True,
        help="Repeatable. LOG_DIR is the run's log_dir (e.g. .../paging/workload/log_dir); "
        "QUERY_SUMMARY_CSV/SERIES_NAME give the true position->query order to zip against log segments.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    colors = ["#4c78a8", "#59a14f", "#f2b134", "#e45756", "#9d6bb0"]

    stages = []
    for label, log_dir, query_summary, series_name in args.stage:
        query_order = query_order_from_summary(Path(query_summary), series_name)
        series = stage_eligible_series(Path(log_dir), query_order)
        stages.append((label, series))
        final_backed = sum(s[2] for s in series)
        final_denom = sum(s[3] for s in series)
        final_excluded = sum(s[4] for s in series)
        print(
            f"{label}: eligible_hit_rate={final_backed / final_denom * 100.0:.1f}% "
            f"(backed={final_backed} eligible_denom={final_denom} excluded_dynamic_filter={final_excluded})"
        )

    n = len(stages[0][1])
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(max(11, n * 0.45), 7.2), constrained_layout=True)
    all_rates = []
    for idx, (label, series) in enumerate(stages):
        rate = cumulative(series)
        all_rates.append(rate)
        ax.plot(x, rate, color=colors[idx % len(colors)], linewidth=3.0, marker="o", markersize=5, label=f"{label} ({rate[-1]:.1f}%)")

    ax.set_ylabel("Cumulative cache hit rate\namong cacheable scans (%)", fontsize=25)
    ax.set_xlabel("Query position in workload (order differs by stage)", fontsize=22)
    ax.set_ylim(0, max(v for rate in all_rates for v in rate) * 1.35)
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
