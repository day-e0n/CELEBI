#!/usr/bin/env python3
"""Experiment B (part 1): does the greedy cache-aware reorder's own latency stay
negligible as batch size N grows past the 22-query TPC-H default?

reorder_query_sequence() (cache_aware_query_reorder.py) tries every start
position (O(N) when --reorder-window is unset, matching the real paper run),
and each greedy path does O(N) steps, each evaluating O(N) candidates, each
candidate summing overlap against O(N) remaining queries -- a naive O(N^4)
nesting. This measures wall-clock reorder time at increasing N (built by
tiling the 22 canonical TPC-H queries) to find the real empirical exponent,
independent of any GPU execution.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from cache_aware_query_reorder import ReorderConfig, reorder_query_sequence  # noqa: E402

BASE_ORDER = [3, 10, 7, 5, 8, 9, 20, 11, 2, 16, 19, 17, 14, 1, 6, 15, 21, 12, 4, 18, 13, 22]


def tiled_batch(n: int) -> list[int]:
    """Tile BASE_ORDER (repeating the same 22-query mix) out to length n."""
    out: list[int] = []
    i = 0
    while len(out) < n:
        out.append(BASE_ORDER[i % len(BASE_ORDER)])
        i += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="22,44,66,100,150,220")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--window", type=int, default=0, help="cache_aware_query_reorder --reorder-window (0 = unbounded, matches the real paper run)")
    parser.add_argument("--keep-first", action="store_true", help="cache_aware_query_reorder --reorder-keep-first")
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    cfg = ReorderConfig(policy="fixed-overlap", scope="fixed_width", window=args.window, keep_first=args.keep_first, resident_column_budget=0)

    rows = []
    print(f"{'N':>6}{'mean_ms':>12}{'std_ms':>10}{'min_ms':>10}{'max_ms':>10}{'overlap_ratio':>15}")
    for n in sizes:
        batch = tiled_batch(n)
        wall_times_ms = []
        internal_times_ms = []
        overlap_ratios = []
        for _ in range(args.trials):
            t0 = time.perf_counter()
            result = reorder_query_sequence(batch, cfg)
            t1 = time.perf_counter()
            wall_times_ms.append((t1 - t0) * 1000.0)
            internal_times_ms.append(result.elapsed_ms)
            overlap_ratios.append(result.after.overlap_ratio)
        mean_ms = statistics.mean(wall_times_ms)
        std_ms = statistics.stdev(wall_times_ms) if len(wall_times_ms) >= 2 else 0.0
        mean_ratio = statistics.mean(overlap_ratios)
        print(f"{n:>6}{mean_ms:>12.3f}{std_ms:>10.3f}{min(wall_times_ms):>10.3f}{max(wall_times_ms):>10.3f}{mean_ratio:>15.4f}")
        rows.append({
            "n": n,
            "window": args.window,
            "keep_first": args.keep_first,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "min_ms": min(wall_times_ms),
            "max_ms": max(wall_times_ms),
            "mean_internal_elapsed_ms": statistics.mean(internal_times_ms),
            "overlap_ratio": mean_ratio,
            "trials": args.trials,
        })

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out_csv}")

    # crude log-log slope estimate between consecutive points (empirical exponent)
    print()
    print("empirical local exponent (slope of log(time) vs log(N)) between consecutive sizes:")
    for (r1, r2) in zip(rows, rows[1:]):
        import math
        if r1["mean_ms"] <= 0 or r2["mean_ms"] <= 0:
            continue
        exponent = math.log(r2["mean_ms"] / r1["mean_ms"]) / math.log(r2["n"] / r1["n"])
        print(f"  N={r1['n']:>4} -> N={r2['n']:>4}: exponent ~= {exponent:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
