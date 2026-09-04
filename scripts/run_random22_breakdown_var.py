#!/usr/bin/env python3
"""22-query random-order workload latency breakdown, extended to 4 conditions
for the variable-width (STRING) page-cache prototype (see
src/scan_manager/variable_width_page_index.{hpp,cpp}):

  baseline           -- no caching at all (matches run_random22_breakdown.py cold_hot)
  celebi_fixed       -- existing CELEBI: fixed-width page cache + reorder
  celebi_variable    -- variable-width page cache (STRING columns only) + reorder,
                        fixed-width columns fall back to the existing whole-chunk
                        hybrid path (SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED=0)
  celebi_fixed_variable -- both page caches on together + reorder

All three non-baseline conditions reuse the SAME reordered query sequence
(computed once from the fixed-width-scope reorder), so the only thing that
differs between them is the caching configuration, not the order."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TPCH_DIR))
from performance_test import load_query_set, open_connection  # noqa: E402
from queries import QUERIES  # noqa: E402
from cache_aware_query_reorder import ReorderConfig, reorder_query_sequence  # noqa: E402
from run_random22_breakdown import ARRIVAL_ORDER, write_config  # noqa: E402

CONDITIONS = ["baseline", "celebi_fixed", "celebi_variable", "celebi_fixed_variable"]


def make_env(condition: str, devices: str, log_dir: Path, config_path: Path,
             fixed_cache_budget: str, variable_cache_budget: str,
             variable_page_bytes: str, min_free_bytes_per_gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = "info"

    if condition == "baseline":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "0"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "0"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "0"
        return env

    if condition not in CONDITIONS:
        raise ValueError(condition)

    # Shared base for all three caching conditions: the existing auto-cache
    # machinery must be on for either page cache (fixed or variable) to be
    # reached at all -- see fixed_page_auto_cache_enabled() in
    # parquet_gpu_ingestible.cpp, which gates insert_fixed_page_entry_from_view
    # (variable_width_page_index's only call site) same as it always has for
    # fixed-width.
    env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "1"
    env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "1"
    env["SIRIUS_FIXED_PAGE_AUTO_CACHE_ROUND_ROBIN_CHUNKS"] = "1"
    env["SIRIUS_FIXED_PAGE_VIEW_ALIGNED_SPLITS"] = "1"
    env["SIRIUS_FIXED_PAGE_PRUNING"] = "1"
    env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "1"
    env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "1"
    env["SIRIUS_FIXED_PAGE_BACKED_PROVIDER"] = "1"
    env["SIRIUS_FIXED_PAGE_HYBRID_PROVIDER"] = "1"  # needed for any non-page-cached column to be servable at all
    if min_free_bytes_per_gpu:
        env["SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU"] = min_free_bytes_per_gpu

    if condition == "celebi_fixed":
        env["SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"] = fixed_cache_budget
        env["SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"] = "0"
    elif condition == "celebi_variable":
        env["SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED"] = "0"  # force fixed-width columns to the whole-chunk fallback
        env["SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"] = "1"
        env["SIRIUS_VARIABLE_WIDTH_PAGE_BYTES"] = variable_page_bytes
        env["SIRIUS_VARIABLE_PAGE_CACHE_BYTES_PER_GPU"] = variable_cache_budget
    elif condition == "celebi_fixed_variable":
        env["SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"] = fixed_cache_budget
        env["SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"] = "1"
        env["SIRIUS_VARIABLE_WIDTH_PAGE_BYTES"] = variable_page_bytes
        env["SIRIUS_VARIABLE_PAGE_CACHE_BYTES_PER_GPU"] = variable_cache_budget
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized")
    parser.add_argument("--executions", type=int, default=10)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--fixed-cache-budget", default="6GB")
    parser.add_argument("--variable-cache-budget", default="2GB")
    parser.add_argument("--variable-page-bytes", default="16777216")  # 16MB, matches fixed default
    parser.add_argument("--min-free-bytes-per-gpu", default="4096MB")
    parser.add_argument("--gpu-usage-limit", default="20GB")
    parser.add_argument("--query-set", default=None,
                        help="Alternate QUERIES module/.py (e.g. "
                             "test/tpch_performance/jcch_queries.py for JCC-H's skewed "
                             "constants). Defaults to the stock TPC-H text.")
    parser.add_argument("--skip-queries", default="",
                        help="Comma-separated query numbers to drop from the arrival "
                             "order, e.g. '21'. One query that aborts the engine would "
                             "otherwise take the whole multi-execution run with it.")
    args = parser.parse_args()

    if args.query_set:
        print(f"query set: {load_query_set(args.query_set)}", flush=True)

    skip = {int(x) for x in args.skip_queries.split(",") if x.strip()}
    qnums = [q for q in ARRIVAL_ORDER if q not in skip]
    if skip:
        print(f"skipping q{', q'.join(str(q) for q in sorted(skip))} "
              f"({len(qnums)} queries per execution)", flush=True)
    if args.condition != "baseline":
        cfg = ReorderConfig(policy="fixed-overlap", scope="fixed_width", window=0,
                             keep_first=False, resident_column_budget=0)
        result = reorder_query_sequence(qnums, cfg)
        qnums = list(result.reordered_queries)
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

    # build_operator_breakdown_comparison.py's steady_state_series() only
    # recognizes the literal condition strings "cold_hot" and "paging" to know
    # execution 1 is warmup and should be dropped -- reuse those exact label
    # prefixes (directories still stay separated by args.condition/args.output)
    # rather than teaching that shared script 4 new condition names.
    series_label = "cold_hot" if args.condition == "baseline" else "paging"

    con = open_connection(args.input, gpu_execution=True)
    try:
        for execution in range(1, args.executions + 1):
            for position, qnum in enumerate(qnums, start=1):
                label = f"{series_label}_q{qnum}_exec{execution}"
                con.execute(f"CALL sirius_set_query_label('{label}')")
                con.execute("SET gpu_execution = true;")
                con.execute(QUERIES[f"q{qnum}"]).fetchall()
                print(f"[{args.condition}] exec={execution}/{args.executions} pos={position}/{len(qnums)} q{qnum} done", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
