#!/usr/bin/env python3
"""Per-query (Q1-Q22) repeated-execution latency breakdown.

For each canonical TPC-H query, opens a FRESH connection (VRAM/cache cleared),
runs the SAME query `--executions` times (default 10: 1 cold + 9 hot), for both
Baseline (cold_hot, caching off) and CELEBI (paging, caching on) conditions.
Quent telemetry is written per condition to a shared telemetry_data root so
parse_quent_operator_breakdown.py + build_operator_breakdown_comparison.py can
aggregate across all 22 queries into the paper's per-query operator-kind
breakdown figure (steady_state_series already excludes execution 1 / cold and
averages the rest -- 9 hot runs for --executions 10).
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
from performance_test import open_connection  # noqa: E402
from queries import QUERIES  # noqa: E402


def write_config(path: Path, telemetry_dir: Path, gpu_usage_limit: str = "20GB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    disk_dir = Path("/mnt/nvme/sirius_scratch/perquery_disk_downgrade") / path.parent.name
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


def run_one_query(input_dir: str, condition: str, qnum: int, executions: int,
                   devices: str, output: Path, cache_budget: str,
                   min_free_bytes_per_gpu: str, gpu_usage_limit: str) -> None:
    case_dir = output / condition
    log_dir = case_dir / "log_dir" / f"q{qnum}"
    config_path = case_dir / "configs" / f"q{qnum}.yaml"
    write_config(config_path, case_dir / "telemetry_data", gpu_usage_limit)
    env = make_env(condition, devices, log_dir, config_path, cache_budget, min_free_bytes_per_gpu)
    os.environ.update(env)

    con = open_connection(input_dir, gpu_execution=True)
    try:
        sql = QUERIES[f"q{qnum}"]
        for execution in range(1, executions + 1):
            label = f"{condition}_q{qnum}_exec{execution}"
            con.execute(f"CALL sirius_set_query_label('{label}')")
            con.execute("SET gpu_execution = true;")
            con.execute(sql).fetchall()
            print(f"[{condition}] q{qnum} exec={execution}/{executions} done", flush=True)
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized")
    parser.add_argument("--queries", default="1-22")
    parser.add_argument("--executions", type=int, default=10)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["cold_hot", "paging"], required=True)
    parser.add_argument("--cache-budget", default="6GB")
    parser.add_argument("--min-free-bytes-per-gpu", default="4096MB")
    parser.add_argument("--gpu-usage-limit", default="20GB")
    parser.add_argument("--qnum", type=int, required=True, help="single query number for this process invocation")
    args = parser.parse_args()

    run_one_query(args.input, args.condition, args.qnum, args.executions, args.devices,
                  args.output.resolve(), args.cache_budget, args.min_free_bytes_per_gpu,
                  args.gpu_usage_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
