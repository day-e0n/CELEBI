#!/usr/bin/env python3
"""Per-run (not cumulative-by-position) cache hit rate for a
run_fixed_page_workload_sequence.py output directory.

Unlike plot_cache_hit_rate_eligible.py (which sums counts across queries at a
fixed *position*, skipping execution 1 as warmup, to build a
cumulative-by-position series), this pools counts across all 22 queries
*within* each execution/run and does NOT skip execution 1 -- the point is to
show the cold-run -> hot-run progression itself (run 1..N on the x-axis), so
the "cold" run has to stay in the series.

eligible_hit_rate definition matches plot_cache_hit_rate_eligible.py:
    backed / (backed + auto_cache_populate + auto_cache_skip[not dynamic_filter_scan])
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from plot_cache_hit_rate_eligible import all_query_segments, segment_counts  # noqa: E402


def per_run_hit_rate(log_dir: Path, n_queries: int) -> list[tuple[int, float, int, int]]:
    """Returns [(run_index (1-based), hit_rate_pct, backed, denom), ...] for every
    execution found in log_dir, in order. run_index 1 is the cold/warmup run."""
    segments = all_query_segments(log_dir)
    if len(segments) % n_queries != 0:
        raise ValueError(f"{log_dir}: {len(segments)} query segments not divisible by {n_queries}")
    num_runs = len(segments) // n_queries

    out = []
    for run in range(num_runs):
        backed = denom = 0
        for p in range(n_queries):
            b, e, _x = segment_counts(segments[run * n_queries + p])
            backed += b
            denom += b + e
        rate = backed / denom * 100.0 if denom else 0.0
        out.append((run + 1, rate, backed, denom))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True, help=".../<condition>/workload/log_dir")
    parser.add_argument("--n-queries", type=int, default=22)
    parser.add_argument("--label", required=True, help="series label written into the output CSV")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = per_run_hit_rate(args.log_dir, args.n_queries)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "run", "hit_rate_pct", "backed", "denom"])
        for run, rate, backed, denom in rows:
            writer.writerow([args.label, run, f"{rate:.4f}", backed, denom])

    for run, rate, backed, denom in rows:
        print(f"[{args.label}] run={run} hit_rate={rate:.1f}% (backed={backed} denom={denom})")
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
