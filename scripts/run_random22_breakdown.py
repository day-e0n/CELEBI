#!/usr/bin/env python3
"""22-query random-order workload latency breakdown (with Quent telemetry).

Runs the SAME fixed arrival order used by final_sf50_6gb_v4_*/final_sf50_6gb_worst_*
(q8,q1,q16,q3,q14,q4,q12,q19,q20,q15,q9,q17,q6,q11,q2,q22,q7,q13,q10,q18,q5,q21),
8 executions (1 cold + 7 hot) on one connection, for Baseline (cold_hot, caching
off) and CELEBI (paging, caching on + exhaustive reorder of this arrival order),
with Quent telemetry enabled so parse_quent_operator_breakdown.py +
build_operator_breakdown_comparison.py can produce the scan/join/aggregate/
filter/sort/other breakdown -- the SAME categorization as the per-query
(Q1-Q22 repeated) breakdown figure, but for this continuous random-order batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TPCH_DIR))
from performance_test import open_connection  # noqa: E402
from queries import QUERIES  # noqa: E402
from cache_aware_query_reorder import ReorderConfig, reorder_query_sequence  # noqa: E402

# Same arrival order as final_sf50_6gb_v4_*/final_sf50_6gb_worst_*.
ARRIVAL_ORDER = [8, 1, 16, 3, 14, 4, 12, 19, 20, 15, 9, 17, 6, 11, 2, 22, 7, 13, 10, 18, 5, 21]


def write_config(path: Path, telemetry_dir: Path, gpu_usage_limit: str = "20GB") -> None:
    # BENCH_USE_ODIRECT=0 routes reads through the kernel page cache instead of
    # O_DIRECT, so Linux keeps the compressed parquet bytes in host RAM. That is
    # the cheapest way to ask whether the NVMe read is what a scan waits on: the
    # decode still happens either way, only the disk trip disappears.
    # BENCH_PREFETCH_CACHE=1 turns on the scan manager's own prefetching cache,
    # which is off by default: with it off, sirius_datasource::fadvise returns on
    # its first line and nothing is ever read ahead. O_DIRECT means the kernel
    # does not read ahead either, so a scan is a pure request-wait loop.
    odirect = os.environ.get("BENCH_USE_ODIRECT")
    prefetch = os.environ.get("BENCH_PREFETCH_CACHE")
    scan_keys = ""
    if prefetch is not None:
        scan_keys += f"      enable_prefetch_cache: {'true' if prefetch == '1' else 'false'}\n"
    if odirect is not None:
        scan_keys += ("      local:\n"
                      f"        use_odirect: {'true' if odirect == '1' else 'false'}\n")
    scan_block = ("" if not scan_keys else
                  "  executor:\n    scan_manager:\n" + scan_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    disk_dir = Path("/mnt/nvme/sirius_scratch/random22_disk_downgrade") / path.parent.name
    disk_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "sirius:\n"
        "  topology:\n"
        "    num_gpus: 1\n"
        "  memory:\n"
        "    gpu:\n"
        f"      usage_limit_bytes: {gpu_usage_limit}\n"
        "      reservation_limit_fraction: 0.85\n"
        "      downgrade_trigger_fraction: 0.8\n"
        "      downgrade_stop_fraction: 0.7\n"
        "    host:\n"
        "      capacity_bytes: 32GB\n"
        "    disk:\n"
        "      disk_id: 0\n"
        "      capacity_bytes: 100GB\n"
        f"      downgrade_root_dirs: \"{disk_dir}\"\n"
        + scan_block +
        "  telemetry:\n"
        "    enable_quent: true\n"
        f"    output_directory: {telemetry_dir}\n"
        "    engine_name: siriusDB\n"
    )


def make_env(condition: str, devices: str, log_dir: Path, config_path: Path,
             cache_budget: str, min_free_bytes_per_gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = "info"
    if condition == "cold_hot":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "0"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "0"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "0"
    elif condition == "paging":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "1"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "1"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE_ROUND_ROBIN_CHUNKS"] = "1"
        env["SIRIUS_FIXED_PAGE_VIEW_ALIGNED_SPLITS"] = "1"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "1"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "1"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "1"
        env["SIRIUS_FIXED_PAGE_BACKED_PROVIDER"] = "1"
        env["SIRIUS_FIXED_PAGE_HYBRID_PROVIDER"] = "1"
        env["SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"] = cache_budget
        if min_free_bytes_per_gpu:
            env["SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU"] = min_free_bytes_per_gpu
    else:
        raise ValueError(condition)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized")
    parser.add_argument("--executions", type=int, default=8)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["cold_hot", "paging"], required=True)
    parser.add_argument("--reorder", action="store_true", help="CELEBI: exhaustive reorder of ARRIVAL_ORDER")
    parser.add_argument("--cache-budget", default="6GB")
    parser.add_argument("--min-free-bytes-per-gpu", default="4096MB")
    parser.add_argument("--gpu-usage-limit", default="20GB")
    args = parser.parse_args()

    qnums = list(ARRIVAL_ORDER)
    if args.reorder:
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
                    args.cache_budget, args.min_free_bytes_per_gpu)
    os.environ.update(env)

    con = open_connection(args.input, gpu_execution=True)
    try:
        for execution in range(1, args.executions + 1):
            for position, qnum in enumerate(qnums, start=1):
                label = f"{args.condition}_q{qnum}_exec{execution}"
                con.execute(f"CALL sirius_set_query_label('{label}')")
                con.execute("SET gpu_execution = true;")
                con.execute(QUERIES[f"q{qnum}"]).fetchall()
                print(f"[{args.condition}] exec={execution}/{args.executions} pos={position}/{len(qnums)} q{qnum} done", flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
