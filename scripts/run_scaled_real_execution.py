#!/usr/bin/env python3
"""Experiment B follow-up: does the REAL scan-time benefit from paging+reorder
(the -19.5% headline measured at N=22) hold up at larger, realistic batch
sizes -- built from real qgen-varied query instances (not repeated identical
queries)?

For a target batch size N:
  - Tile the natural 1..22 query order out to length N. Each repeat of a
    given qnum draws a genuinely different parameterization from a distinct
    qgen stream (stream 1, 2, 3, ... up to 10) -- see parse_qgen_streams.py.
  - baseline condition: this natural tiled order, paging OFF (cold_hot env).
  - proposed condition: the SAME qnum multiset reordered via
    cache_aware_query_reorder (fixed-overlap, --reorder-keep-first, the
    validated fast+high-quality config), paging ON. Streams are reassigned to
    the reordered qnum sequence FIFO-per-qnum so each occurrence still gets a
    distinct real instance.
  - 3 executions per condition, Quent telemetry + per-query labeling so
    parse_quent_operator_breakdown.py can compute scan time same as before.

Q15 is special: qgen emits it as three statements (create view / select /
drop view) -- these are split and only the middle SELECT is timed/labeled;
the view create/drop run untimed immediately around it.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(TPCH_DIR))
from performance_test import open_connection  # noqa: E402
from parse_qgen_streams import load_all_streams  # noqa: E402
from cache_aware_query_reorder import ReorderConfig, reorder_query_sequence, worst_case_order  # noqa: E402

NUM_STREAMS = 10


def tiled_qnum_sequence(n: int) -> list[int]:
    out = []
    i = 0
    while len(out) < n:
        out.append((i % 22) + 1)
        i += 1
    return out


def assign_streams(qnum_sequence: list[int]) -> list[tuple[int, int]]:
    """Assign a distinct stream (1..10, cycling) to each qnum occurrence, in
    the order the qnum first appears in qnum_sequence (FIFO per qnum)."""
    counters: dict[int, int] = defaultdict(int)
    out = []
    for q in qnum_sequence:
        counters[q] += 1
        stream = ((counters[q] - 1) % NUM_STREAMS) + 1
        out.append((q, stream))
    return out


def reorder_with_streams(base_pairs: list[tuple[int, int]], keep_first: bool = True) -> list[tuple[int, int]]:
    """Reorder by qnum (cache_aware_query_reorder, keep-first by default, or
    exhaustive/all-starting-positions when keep_first=False), then reassign
    streams FIFO-per-qnum over the new order so repeats still draw distinct
    real instances (Sig() doesn't see streams at all, matching the
    coarse-proxy property we already established)."""
    qnums = [q for q, _ in base_pairs]
    cfg = ReorderConfig(policy="fixed-overlap", scope="fixed_width", window=0, keep_first=keep_first, resident_column_budget=0)
    result = reorder_query_sequence(qnums, cfg)
    reordered_qnums = list(result.reordered_queries)

    # FIFO per-qnum stream queues, in ORIGINAL encounter order.
    queues: dict[int, deque[int]] = defaultdict(deque)
    for q, s in base_pairs:
        queues[q].append(s)
    return [(q, queues[q].popleft()) for q in reordered_qnums]


def make_env(condition: str, devices: str, log_dir: Path, config_path: Path,
             cache_budget: str = "3GB", min_free_bytes_per_gpu: str = "4096MB") -> dict[str, str]:
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


def write_config(path: Path, telemetry_dir: Path, gpu_usage_limit: str = "20GB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    # root filesystem is nearly full (17GB free) -- downgrade scratch space
    # must live on the large NVMe mount, not under the repo checkout.
    run_id = f"{path.parent.parent.name}_{path.parent.name}"
    disk_dir = Path("/mnt/nvme/sirius_scratch/expB_disk_downgrade") / run_id
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


def run_query(con, label: str, sql: str) -> None:
    con.execute(f"CALL sirius_set_query_label('{label}')")
    con.execute("SET gpu_execution = true;")
    if "create view" in sql.lower():
        # Q15: create view ...; select ...; drop view ...  -- only the
        # middle SELECT is the timed/labeled query; split and run the rest
        # untimed immediately around it.
        stmts = [s.strip() for s in sql.split(";") if s.strip()]
        con.execute(stmts[0])  # create view (untimed)
        con.execute(f"CALL sirius_set_query_label('{label}')")
        con.execute("SET gpu_execution = true;")
        con.execute(stmts[1]).fetchall()  # the real query (timed/labeled)
        con.execute(stmts[2])  # drop view (untimed)
    else:
        con.execute(sql).fetchall()


def run_condition(input_dir: str, condition: str, pairs: list[tuple[int, int]],
                   streams: dict[tuple[int, int], str], executions: int,
                   devices: str, output: Path, execution_start: int = 1,
                   cache_budget: str = "3GB", min_free_bytes_per_gpu: str = "4096MB",
                   gpu_usage_limit: str = "20GB") -> None:
    case_dir = output / condition
    log_dir = case_dir / "log_dir"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = case_dir / "sirius.yaml"
    write_config(config_path, case_dir / "telemetry_data", gpu_usage_limit)
    env = make_env(condition, devices, log_dir, config_path, cache_budget, min_free_bytes_per_gpu)
    os.environ.update(env)

    con = open_connection(input_dir, gpu_execution=True)
    try:
        for execution in range(execution_start, execution_start + executions):
            for position, (qnum, stream) in enumerate(pairs, start=1):
                label = f"{condition}_q{qnum}_s{stream}_exec{execution}"
                sql = streams[(stream, qnum)]
                run_query(con, label, sql)
                print(f"[{condition}] exec={execution} pos={position}/{len(pairs)} "
                      f"q{qnum} stream{stream}", flush=True)
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/mnt/nvme/sirius_tpch/tpch_parquet_sf30_optimized")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--executions", type=int, default=3)
    parser.add_argument("--execution-start", type=int, default=1,
                        help="label executions starting from this number (for running each "
                        "execution as a separate process to avoid cross-execution GPU memory buildup)")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["cold_hot", "paging"], required=True)
    parser.add_argument("--reorder-exhaustive", action="store_true",
                        help="Use the exhaustive (all-starting-positions) reorder search instead "
                        "of the default fixed-start (keep_first) heuristic.")
    parser.add_argument("--no-reorder", action="store_true",
                        help="For --condition paging: skip reordering entirely, keep the natural "
                        "tiled query order (caching on, arrival order) -- 'baseline+paging'.")
    parser.add_argument("--scramble-order", action="store_true",
                        help="Replace the natural tiled arrival order with a deliberately "
                        "worst-case (minimum adjacent overlap) order before any condition-specific "
                        "logic runs -- gives CELEBI's reorder something real to fix instead of the "
                        "already-benign natural cyclic tiling.")
    parser.add_argument("--cache-budget", default="3GB",
                        help="SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU for the paging condition "
                        "(ignored for cold_hot).")
    parser.add_argument("--gpu-usage-limit", default="20GB",
                        help="sirius.yaml gpu.usage_limit_bytes. Physical card is 24GB; raise "
                        "this (e.g. 22GB) when the GPU is exclusively available to give query "
                        "execution more headroom above the fixed-page cache's resident footprint.")
    parser.add_argument("--min-free-bytes-per-gpu", default="4096MB",
                        help="SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU for the paging "
                        "condition -- without this, memory-pressure eviction is disabled "
                        "(min_free==0) and the fixed-page cache only evicts once ITS OWN budget "
                        "is exceeded, never in response to overall GPU memory pressure from query "
                        "execution buffers. Matches run_fixed_page_workload_sequence.py's default.")
    args = parser.parse_args()

    streams = load_all_streams()
    qnum_seq = tiled_qnum_sequence(args.n)
    if args.scramble_order:
        qnum_seq = list(worst_case_order(qnum_seq))
    base_pairs = assign_streams(qnum_seq)

    if args.condition == "cold_hot" or args.no_reorder:
        pairs = base_pairs
    else:
        pairs = reorder_with_streams(base_pairs, keep_first=not args.reorder_exhaustive)

    print(f"N={args.n} condition={args.condition} pairs[:10]={pairs[:10]}", flush=True)
    run_condition(args.input, args.condition, pairs, streams, args.executions, args.devices,
                  args.output.resolve(), args.execution_start, args.cache_budget,
                  args.min_free_bytes_per_gpu, args.gpu_usage_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
