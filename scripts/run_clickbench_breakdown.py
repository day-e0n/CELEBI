#!/usr/bin/env python3
"""ClickBench 43-query latency breakdown, same 4 conditions as
run_random22_breakdown_var.py:

  baseline              -- no caching at all
  celebi_fixed          -- fixed-width page cache + reorder
  celebi_variable       -- variable-width (STRING) page cache + reorder
  celebi_fixed_variable -- both page caches + reorder

Why ClickBench rather than TPC-H/JCC-H for the page-cache work: it is a single
denormalized table, so no scan ever carries a join's dynamic filter -- the gate
that excluded 675 of 795 lineitem scans from the cache on SF50 simply has
nothing to fire on. And its queries are narrow (median 2 columns of 105), so a
typical cache entry is 0.2-1.6 GB at 100M rows instead of the 3.4-14.8 GiB
lineitem projections that every one of TPC-H's 18 distinct keys exceeded the
per-entry admission cap with.

The three caching conditions reuse the SAME reordered sequence (computed once
from the fixed-width-scope reorder) so only the cache configuration differs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TPCH_DIR))
from performance_test import load_query_set  # noqa: E402
from queries import QUERIES  # noqa: E402
import cache_aware_query_reorder as reorder  # noqa: E402
from run_random22_breakdown import write_config  # noqa: E402
from run_random22_breakdown_var import CONDITIONS, make_env  # noqa: E402

import duckdb  # noqa: E402

# Arrival order = the benchmark's own query order, so a reorder's benefit is
# measured against the sequence a user would actually submit.
ARRIVAL_ORDER = list(range(1, 44))


def open_clickbench_connection(parquet_path: str, extension_path: str):
    """In-memory DuckDB with `hits` as a view over the single parquet file.

    performance_test.open_connection() registers the 8 TPC-H tables from a
    directory; ClickBench is one table from one file, so it gets its own opener
    rather than another special case threaded through that one.
    """
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(
        f"CREATE OR REPLACE VIEW hits AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    threads = os.environ.get("BENCH_DUCKDB_THREADS")
    if threads:
        con.execute(f"SET threads={int(threads)}")
    con.execute(f"LOAD '{extension_path}'")
    return con


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/clickbench/hits_v2.parquet")
    parser.add_argument("--executions", type=int, default=3)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--fixed-cache-budget", default="6GB")
    parser.add_argument("--variable-cache-budget", default="4GB")
    parser.add_argument("--variable-page-bytes", default="16777216")
    # Defaults to 0, unlike the TPC-H runner's 4096MB: the memory-pressure sweep
    # that setting arms was measured to free 0 bytes in 29 of 29 events on SF50
    # while being the entire run-to-run variance (+-2756ms -> +-649ms with it off).
    parser.add_argument("--min-free-bytes-per-gpu", default="0")
    parser.add_argument("--gpu-usage-limit", default="20GB")
    parser.add_argument("--skip-queries", default="",
                        help="Comma-separated query numbers to drop, e.g. '9,28'. One "
                             "query that aborts the engine would otherwise take the "
                             "whole multi-execution run with it.")
    args = parser.parse_args()

    print(f"query set: {load_query_set('clickbench_queries')}", flush=True)
    print(f"column set: {reorder.load_column_set('clickbench_pin_columns')}", flush=True)

    skip = {int(x) for x in args.skip_queries.split(",") if x.strip()}
    qnums = [q for q in ARRIVAL_ORDER if q not in skip]
    if skip:
        print(f"skipping q{', q'.join(str(q) for q in sorted(skip))} "
              f"({len(qnums)} queries per execution)", flush=True)
    if args.condition != "baseline":
        cfg = reorder.ReorderConfig(policy="fixed-overlap", scope="fixed_width",
                                    window=0, keep_first=False,
                                    resident_column_budget=0)
        qnums = list(reorder.reorder_query_sequence(qnums, cfg).reordered_queries)
        print(f"reordered: {','.join(f'q{q}' for q in qnums)}", flush=True)

    output = args.output.resolve()
    case_dir = output / args.condition
    log_dir = case_dir / "log_dir"
    config_path = case_dir / "sirius.yaml"
    write_config(config_path, case_dir / "telemetry_data", args.gpu_usage_limit)
    env = make_env(args.condition, args.devices, log_dir, config_path,
                   args.fixed_cache_budget, args.variable_cache_budget,
                   args.variable_page_bytes, args.min_free_bytes_per_gpu)
    os.environ.update(env)

    # build_operator_breakdown_comparison.py's steady_state_series() only knows
    # the literal labels "cold_hot" and "paging" for "execution 1 is warmup" --
    # reuse them rather than teaching that shared script new condition names.
    series_label = "cold_hot" if args.condition == "baseline" else "paging"

    from performance_test import EXTENSION_PATH
    con = open_clickbench_connection(args.input, EXTENSION_PATH)
    try:
        for execution in range(1, args.executions + 1):
            for position, qnum in enumerate(qnums, start=1):
                label = f"{series_label}_q{qnum}_exec{execution}"
                con.execute(f"CALL sirius_set_query_label('{label}')")
                con.execute("SET gpu_execution = true;")
                con.execute(QUERIES[f"q{qnum}"]).fetchall()
                print(f"[{args.condition}] exec={execution}/{args.executions} "
                      f"pos={position}/{len(qnums)} q{qnum} done", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
