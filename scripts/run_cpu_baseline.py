#!/usr/bin/env python3
"""DuckDB on the CPU (no Sirius) over the same data, query text and order as the GPU runners.

Prints the same `exec=E/N pos=P/N qX done wall=Ys` lines as run_random22_breakdown_var.py and
run_clickbench_breakdown.py, so the same wall-time parsing applies to both sides.

  pixi run -e duckdb-python python scripts/run_cpu_baseline.py --benchmark tpch \
      --order 8,1,4,... --threads 10 --executions 3
  pixi run -e duckdb-python python scripts/run_cpu_baseline.py --benchmark clickbench --threads 40

DuckDB reads parquet through the kernel page cache, so from execution 2 on the files come from
host RAM; the GPU runners read with O_DIRECT unless BENCH_USE_ODIRECT=0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "test" / "tpch_performance"))
from performance_test import load_query_set, open_connection  # noqa: E402
from queries import QUERIES  # noqa: E402

import duckdb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("tpch", "clickbench"), required=True)
    parser.add_argument("--input", help="TPC-H parquet dir or ClickBench parquet file")
    parser.add_argument("--order", help="Comma-separated query numbers (default: benchmark order)")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--executions", type=int, default=3)
    args = parser.parse_args()

    if args.benchmark == "tpch":
        source = args.input or "/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized"
        con = open_connection(source, gpu_execution=False)
        default_order = list(range(1, 23))
    else:
        source = args.input or "/mnt/nvme/clickbench/hits_v2.parquet"
        print(f"query set: {load_query_set('clickbench_queries')}", flush=True)
        con = duckdb.connect(":memory:")
        con.execute(f"CREATE OR REPLACE VIEW hits AS SELECT * FROM read_parquet('{source}')")
        default_order = [q for q in range(1, 44) if q not in (28, 29)]
    con.execute(f"SET threads={args.threads}")
    order = [int(x) for x in args.order.split(",")] if args.order else default_order
    print(f"cpu threads={args.threads} source={source} order={','.join(f'q{q}' for q in order)}",
          flush=True)

    for execution in range(1, args.executions + 1):
        for position, qnum in enumerate(order, start=1):
            t0 = time.perf_counter()
            con.execute(QUERIES[f"q{qnum}"]).fetchall()
            wall = time.perf_counter() - t0
            print(f"[cpu] exec={execution}/{args.executions} pos={position}/{len(order)} "
                  f"q{qnum} done wall={wall:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
