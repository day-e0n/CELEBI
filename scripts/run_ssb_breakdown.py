#!/usr/bin/env python3
"""SSB (Star Schema Benchmark) runner, mirroring run_random22_breakdown_var.py.

Exists because TPC-H gives variable-width page caching almost nothing to reuse:
its string columns are read by at most two of the 22 queries. SSB's dimension
strings (s_region, c_nation, p_brand1, c_city, s_city, ...) are each read by
three to six of the 13 queries, so this is the workload where a cached string
page can actually find a second reader.

Conditions match the TPC-H runner so the two can be compared directly:
  baseline              -- no page cache
  celebi_fixed          -- fixed-width page cache
  celebi_fixed_variable -- fixed + variable-width page cache
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test" / "tpch_performance"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test" / "ssb_performance"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from performance_test import EXTENSION_PATH, log  # noqa: E402
from queries import QUERIES  # noqa: E402
from run_random22_breakdown import write_config  # noqa: E402

import cache_aware_query_reorder as reorder_mod  # noqa: E402
from cache_aware_query_reorder import (ReorderConfig, load_query_table_filters,  # noqa: E402
                                       reorder_query_sequence, worst_case_sequence)
from ssb_pin_columns import FIXED_WIDTH_COLUMNS as SSB_FIXED, QUERY_COLUMNS as SSB_COLUMNS  # noqa: E402

# The reorder policy scores adjacent-query overlap over a (query -> table -> columns)
# map, which the module imports for TPC-H at load time. Point it at SSB's map
# instead; the policy itself is schema-agnostic.
reorder_mod.QUERY_COLUMNS = SSB_COLUMNS
reorder_mod.FIXED_WIDTH_COLUMNS = SSB_FIXED

import duckdb  # noqa: E402

TABLES = ["customer", "part", "supplier", "date", "lineorder"]
# A fixed arrival order over the 13 queries. Grouped so that queries sharing a
# dimension column are NOT adjacent -- otherwise reuse would be trivially high
# and would not say anything about the cache holding data across a working set.
ARRIVAL_ORDER = ["q4.3", "q1.1", "q3.2", "q2.1", "q4.1", "q1.3", "q3.4",
                 "q2.3", "q3.1", "q1.2", "q4.2", "q2.2", "q3.3"]
CONDITIONS = ["baseline", "celebi_fixed", "celebi_fixed_variable"]


def make_env(condition: str, devices: str, log_dir: Path, config_path: Path,
             fixed_cache_budget: str, variable_cache_budget: str,
             min_free_bytes_per_gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = "info"
    if condition == "baseline":
        for k in ("SIRIUS_ENABLE_FIXED_PAGE_REUSE", "SIRIUS_FIXED_PAGE_AUTO_CACHE",
                  "SIRIUS_FIXED_PAGE_OWNED_PAGES", "SIRIUS_FIXED_PAGE_DEMAND_LOAD",
                  "SIRIUS_FIXED_PAGE_PRUNING"):
            env[k] = "0"
        return env
    if condition not in CONDITIONS:
        raise ValueError(condition)

    for k, v in {
        "SIRIUS_ENABLE_FIXED_PAGE_REUSE": "1",
        "SIRIUS_FIXED_PAGE_AUTO_CACHE": "1",
        "SIRIUS_FIXED_PAGE_AUTO_CACHE_ROUND_ROBIN_CHUNKS": "1",
        "SIRIUS_FIXED_PAGE_VIEW_ALIGNED_SPLITS": "1",
        "SIRIUS_FIXED_PAGE_PRUNING": "1",
        "SIRIUS_FIXED_PAGE_OWNED_PAGES": "1",
        "SIRIUS_FIXED_PAGE_DEMAND_LOAD": "1",
        "SIRIUS_FIXED_PAGE_BACKED_PROVIDER": "1",
        # needed for any non-page-cached column to be servable at all
        "SIRIUS_FIXED_PAGE_HYBRID_PROVIDER": "1",
        "SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU": fixed_cache_budget,
    }.items():
        env[k] = v
    if min_free_bytes_per_gpu:
        env["SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU"] = min_free_bytes_per_gpu
    if condition == "celebi_fixed":
        env["SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"] = "0"
    else:
        env["SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"] = "1"
        env["SIRIUS_VARIABLE_PAGE_CACHE_BYTES_PER_GPU"] = variable_cache_budget
    return env


def open_ssb_connection(source: str, gpu_execution: bool = True):
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{source}/{t}.parquet')")
    log(f"Registered {len(TABLES)} SSB parquet views from {source}")
    if gpu_execution:
        log(f"Loading Sirius extension from {EXTENSION_PATH}")
        con.execute(f"LOAD '{EXTENSION_PATH}'")
        log("Sirius extension loaded")
    return con


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="/mnt/nvme/sirius_ssb/ssb_parquet_sf10")
    ap.add_argument("--executions", type=int, default=10)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--fixed-cache-budget", default="6GB")
    ap.add_argument("--variable-cache-budget", default="4GB")
    ap.add_argument("--min-free-bytes-per-gpu", default="4096MB")
    ap.add_argument("--gpu-usage-limit", default="20GB")
    ap.add_argument("--arrival", choices=("natural", "worst"), default="natural",
                    help="'worst' replaces the built-in arrival order with the one that "
                         "minimises adjacent-query column overlap, so a reorder has "
                         "something to recover -- the same treatment the TPC-H and "
                         "ClickBench runners give.")
    ap.add_argument("--reorder-policy", choices=("fixed-overlap", "unfiltered-overlap"),
                    default="fixed-overlap")
    ap.add_argument("--query-table-filters",
                    help="JSON from scripts/extract_query_filters.py --workload ssb, "
                         "required by unfiltered-overlap.")
    ap.add_argument("--no-reorder", action="store_true",
                    help="Run the arrival order even for a caching condition, so a "
                         "cache-vs-baseline comparison measures the cache alone.")
    args = ap.parse_args()

    case_dir = args.output.resolve() / args.condition
    log_dir = case_dir / "log_dir"
    config_path = case_dir / "sirius.yaml"
    write_config(config_path, case_dir / "telemetry_data", args.gpu_usage_limit)
    os.environ.update(make_env(args.condition, args.devices, log_dir, config_path,
                               args.fixed_cache_budget, args.variable_cache_budget,
                               args.min_free_bytes_per_gpu))

    # Reuse the labels build_operator_breakdown_comparison.py already understands:
    # it drops execution 1 as warmup only for these two prefixes.
    series_label = "cold_hot" if args.condition == "baseline" else "paging"

    # Same treatment as the TPC-H runner: baseline runs the arrival order, the
    # cached conditions run the CELEBI-reordered one, so the two benchmarks'
    # condition definitions line up.
    order = list(ARRIVAL_ORDER)
    by_id = {int(q[1:].replace(".", "")): q for q in ARRIVAL_ORDER}
    if args.arrival == "worst":
        order = [by_id[i] for i in worst_case_sequence(list(by_id), "fixed_width")]
        print(f"arrival(worst): {','.join(order)}", flush=True)
    if args.condition != "baseline" and not args.no_reorder:
        if args.reorder_policy == "unfiltered-overlap":
            if not args.query_table_filters:
                ap.error("--reorder-policy unfiltered-overlap requires --query-table-filters")
            print(f"query filters: {load_query_table_filters(args.query_table_filters)} queries",
                  flush=True)
        ids = [int(q[1:].replace(".", "")) for q in order]
        result = reorder_query_sequence(ids, ReorderConfig(
            policy=args.reorder_policy, scope="fixed_width", window=0,
            keep_first=False, resident_column_budget=0))
        order = [by_id[i] for i in result.reordered_queries]
        print(f"reordered: {','.join(order)}", flush=True)

    con = open_ssb_connection(args.input, gpu_execution=True)
    try:
        for execution in range(1, args.executions + 1):
            for position, q in enumerate(order, start=1):
                # parse_quent_operator_breakdown.py's LABEL_RE demands
                # "_q<digits>_exec<digits>", so "q4.3" becomes "q43". The mapping
                # stays unambiguous: SSB group and index are both single digits.
                label_q = q.replace(".", "")
                con.execute(
                    f"CALL sirius_set_query_label('{series_label}_{label_q}_exec{execution}')")
                con.execute("SET gpu_execution = true;")
                con.execute(QUERIES[q]).fetchall()
                print(f"[{args.condition}] exec={execution}/{args.executions} "
                      f"pos={position}/{len(order)} {q} done", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
