#!/usr/bin/env python3
"""Run DuckDB CPU over a fixed query order with a capped thread count.

Used for cost-normalized comparisons against a GPU whose cloud rental price
maps to a specific vCPU count on a comparable CPU instance (e.g. AWS c8i) —
the physical CPU here is left untouched; only DuckDB's own `SET threads=N`
is capped, approximating what that vCPU-limited instance would deliver.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(TPCH_DIR))

from performance_test import open_connection  # noqa: E402
from queries import QUERIES  # noqa: E402


def parse_query_list(value: str) -> list[int]:
    return [int(q.strip()) for q in value.split(",") if q.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parquet directory")
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--queries", required=True, help="e.g. 3,10,7,5,8,...")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    order = parse_query_list(args.queries)
    con = open_connection(args.input, gpu_execution=False, data_source="parquet")
    con.execute(f"SET threads={args.threads};")

    rows = []
    for it in range(args.iterations):
        for q in order:
            sql = QUERIES[f"q{q}"]
            t0 = time.perf_counter()
            con.execute(sql).fetchall()
            t1 = time.perf_counter()
            rows.append({"engine": "duckdb", "query": f"q{q}", "iteration": it, "runtime_s": t1 - t0})
            print(f"[threads={args.threads}] q{q} iter{it}: {t1 - t0:.4f}s", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["engine", "query", "iteration", "runtime_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_csv}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
