#!/usr/bin/env python3
"""Measure the reorder ALGORITHM's own CPU-side compute cost (not query
execution time) vs batch size N, for the SF50 scrambled-order experiment's
exact input construction (worst_case_order applied to the tiled qnum
sequence, same as run_scaled_real_execution.py --scramble-order).

Reports both --reorder-exhaustive (all N starting positions) and the
default fixed-start (keep_first) heuristic, since both appear in the paper."""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from cache_aware_query_reorder import ReorderConfig, reorder_query_sequence, worst_case_order  # noqa: E402

NS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TRIALS = 5


def tiled_qnum_sequence(n: int) -> list[int]:
    return [(i % 22) + 1 for i in range(n)]


def main() -> int:
    rows = []
    for n in NS:
        scrambled = list(worst_case_order(tiled_qnum_sequence(n)))

        cfg_ex = ReorderConfig(policy="fixed-overlap", scope="fixed_width", window=0,
                                keep_first=False, resident_column_budget=0)
        cfg_fs = ReorderConfig(policy="fixed-overlap", scope="fixed_width", window=0,
                                keep_first=True, resident_column_budget=0)

        ex_times = [reorder_query_sequence(scrambled, cfg_ex).elapsed_ms for _ in range(TRIALS)]
        fs_times = [reorder_query_sequence(scrambled, cfg_fs).elapsed_ms for _ in range(TRIALS)]

        rows.append({
            "n": n,
            "exhaustive_mean_ms": statistics.mean(ex_times),
            "exhaustive_std_ms": statistics.stdev(ex_times),
            "fixedstart_mean_ms": statistics.mean(fs_times),
            "fixedstart_std_ms": statistics.stdev(fs_times),
        })

    out_csv = REPO_ROOT / "experiment" / "graph" / "expB_sf50_scrambled_reorder_cost_vs_n.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    for r in rows:
        print(f"N={r['n']:3d}  exhaustive={r['exhaustive_mean_ms']:8.2f}ms (+-{r['exhaustive_std_ms']:.2f})  "
              f"fixed-start={r['fixedstart_mean_ms']:6.2f}ms (+-{r['fixedstart_std_ms']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
