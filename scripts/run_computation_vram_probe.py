#!/usr/bin/env python3
# wdy start
"""Measure baseline Sirius computation VRAM peaks for TPC-H queries.

This script intentionally does not exercise the fixed-page cache. It answers:

  "How much GPU memory does baseline query execution reserve/use?"

Each query runs in a fresh child process so RMM/cuda context state from one
query does not carry into the next measurement. The parent polls nvidia-smi for
the child PID and records peak process VRAM.

Outputs stay compact:

  experiment/memory_budget_runs/<run-name>/
    README.md
    config.yml
    metadata.json
    query_vram.csv
    summary.csv
    cases/q<N>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiment" / "memory_budget_runs"
DEFAULT_QUERIES = "1-22"


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_query_list(value: str) -> list[int]:
    queries: list[int] = []
    for item in parse_csv_list(value):
        item = item.lower().removeprefix("q")
        if "-" in item:
            lo, hi = item.split("-", 1)
            lo = lo.lower().removeprefix("q")
            hi = hi.lower().removeprefix("q")
            queries.extend(range(int(lo), int(hi) + 1))
        else:
            queries.append(int(item))
    return queries


def size_to_bytes(value: str) -> int:
    raw = value.strip()
    units = [
        ("tib", 1024**4),
        ("tb", 1024**4),
        ("gib", 1024**3),
        ("gb", 1024**3),
        ("mib", 1024**2),
        ("mb", 1024**2),
        ("kib", 1024),
        ("kb", 1024),
        ("b", 1),
    ]
    lower = raw.lower().replace(" ", "")
    for suffix, multiplier in units:
        if lower.endswith(suffix):
            return int(float(lower[: -len(suffix)]) * multiplier)
    return int(float(lower))


def bytes_to_gib(value: float) -> float:
    return value / float(1024**3)


def write_config(path: Path, num_gpus: int, gpu_limit: str, reservation: str, host_capacity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "sirius:",
                "  topology:",
                f"    num_gpus: {num_gpus}",
                "  memory:",
                "    gpu:",
                f"      usage_limit_bytes: {gpu_limit}",
                f"      reservation_limit_fraction: {reservation}",
                "    host:",
                f"      capacity_bytes: {host_capacity}",
                "",
            ]
        )
    )


def run_capture(cmd: list[str]) -> str:
    return subprocess.run(cmd, text=True, capture_output=True, check=False).stdout.strip()


def gpu_inventory() -> list[dict[str, str]]:
    out = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": parts[3],
                "memory_free_mib": parts[4],
                "memory_used_mib": parts[5],
            }
        )
    return rows


def query_compute_apps() -> list[dict[str, str]]:
    out = run_capture(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append({"gpu_uuid": parts[0], "pid": parts[1], "used_memory_mib": parts[2]})
    return rows


def memory_for_pid(pid: int, uuid_to_index: dict[str, str]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for row in query_compute_apps():
        try:
            row_pid = int(row["pid"])
        except ValueError:
            continue
        if row_pid != pid:
            continue
        gpu_key = uuid_to_index.get(row["gpu_uuid"], row["gpu_uuid"])
        try:
            mib = int(float(row["used_memory_mib"]))
        except ValueError:
            continue
        usage[gpu_key] = max(usage.get(gpu_key, 0), mib)
    return usage


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def child_main(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(TPCH_DIR))
    result = {
        "query": f"q{args.case_query}",
        "status": "error",
        "runtime_s": "",
        "row_count": "",
        "error": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        from performance_test import open_connection, time_query  # noqa: PLC0415

        con = open_connection(str(args.input), gpu_execution=True)
        try:
            elapsed, rows = time_query(con, args.case_query, use_gpu=True)
        finally:
            con.close()
        result.update(
            {
                "status": "ok",
                "runtime_s": elapsed,
                "row_count": len(rows),
                "error": "",
            }
        )
        Path(args.case_output).write_text(json.dumps(result, indent=2) + "\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        Path(args.case_output).write_text(json.dumps(result, indent=2) + "\n")
        return 1


def run_one_query(args: argparse.Namespace,
                  run_root: Path,
                  config_path: Path,
                  qnum: int,
                  uuid_to_index: dict[str, str]) -> dict[str, object]:
    case_path = run_root / "cases" / f"q{qnum}.json"
    child_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--input",
        str(args.input),
        "--case-query",
        str(qnum),
        "--case-output",
        str(case_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_LEVEL"] = args.log_level
    env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
    env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "0"
    env.pop("SIRIUS_PIN_TIER", None)

    stdout_target = subprocess.DEVNULL
    log_file = None
    if args.keep_child_logs:
        log_dir = run_root / "child_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = (log_dir / f"q{qnum}.log").open("w")
        stdout_target = log_file

    print(f"[RUN] q{qnum}", flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen(
        child_cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=stdout_target,
        stderr=subprocess.STDOUT,
        text=True,
    )
    samples = 0
    peak_by_gpu: dict[str, int] = {}
    total_samples: list[int] = []
    deadline = start + args.timeout
    try:
        while True:
            usage = memory_for_pid(proc.pid, uuid_to_index)
            if usage:
                samples += 1
                total = sum(usage.values())
                total_samples.append(total)
                for gpu, mib in usage.items():
                    peak_by_gpu[gpu] = max(peak_by_gpu.get(gpu, 0), mib)
            ret = proc.poll()
            if ret is not None:
                break
            if time.perf_counter() > deadline:
                proc.kill()
                ret = proc.wait(timeout=10)
                break
            time.sleep(args.sample_interval_ms / 1000.0)
        # One final poll catches memory held just before process teardown when possible.
        usage = memory_for_pid(proc.pid, uuid_to_index)
        if usage:
            samples += 1
            total = sum(usage.values())
            total_samples.append(total)
            for gpu, mib in usage.items():
                peak_by_gpu[gpu] = max(peak_by_gpu.get(gpu, 0), mib)
    finally:
        if log_file is not None:
            log_file.close()

    wall_s = time.perf_counter() - start
    if not case_path.exists():
        case = {
            "query": f"q{qnum}",
            "status": "error",
            "runtime_s": "",
            "row_count": "",
            "error": "child did not write case json",
        }
    else:
        case = json.loads(case_path.read_text())

    timeout_hit = wall_s >= args.timeout and proc.returncode != 0
    status = str(case.get("status", "error"))
    if timeout_hit:
        status = "timeout"
        case["status"] = "timeout"
        case["error"] = f"timed out after {args.timeout}s"
        case_path.write_text(json.dumps(case, indent=2) + "\n")

    peak_total = sum(peak_by_gpu.values())
    row = {
        "query": f"q{qnum}",
        "status": status,
        "returncode": proc.returncode,
        "runtime_s": case.get("runtime_s", ""),
        "row_count": case.get("row_count", ""),
        "wall_s": wall_s,
        "samples": samples,
        "peak_total_mib": peak_total,
        "peak_total_gib": peak_total / 1024.0,
        "peak_max_gpu_mib": max(peak_by_gpu.values()) if peak_by_gpu else 0,
        "peak_by_gpu_mib": json.dumps(peak_by_gpu, sort_keys=True),
        "avg_sample_total_mib": statistics.mean(total_samples) if total_samples else 0,
        "error": str(case.get("error", "")).splitlines()[0] if case.get("error") else "",
    }
    print(
        f"[DONE] q{qnum} status={status} runtime={row['runtime_s']} "
        f"peak_total_gib={row['peak_total_gib']:.2f}",
        flush=True,
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(run_root: Path,
                  rows: list[dict[str, object]],
                  gpu_limit_bytes: int,
                  num_gpus: int) -> list[dict[str, object]]:
    ok_rows = [row for row in rows if row["status"] == "ok" and float(row["peak_total_mib"]) > 0]
    peaks_gib = [float(row["peak_total_mib"]) / 1024.0 for row in ok_rows]
    runtimes = [float(row["runtime_s"]) for row in ok_rows if row["runtime_s"] != ""]
    configured_total_gib = bytes_to_gib(gpu_limit_bytes * num_gpus)
    summary = [
        {"metric": "queries_total", "value": len(rows), "unit": "count"},
        {"metric": "queries_ok", "value": len(ok_rows), "unit": "count"},
        {"metric": "queries_failed_or_timeout", "value": len(rows) - len(ok_rows), "unit": "count"},
        {"metric": "configured_gpu_limit_per_gpu", "value": bytes_to_gib(gpu_limit_bytes), "unit": "GiB"},
        {"metric": "configured_gpu_limit_total", "value": configured_total_gib, "unit": "GiB"},
        {"metric": "peak_total_mean", "value": statistics.mean(peaks_gib) if peaks_gib else "", "unit": "GiB"},
        {"metric": "peak_total_p50", "value": percentile(peaks_gib, 0.50) if peaks_gib else "", "unit": "GiB"},
        {"metric": "peak_total_p95", "value": percentile(peaks_gib, 0.95) if peaks_gib else "", "unit": "GiB"},
        {"metric": "peak_total_max", "value": max(peaks_gib) if peaks_gib else "", "unit": "GiB"},
        {
            "metric": "headroom_mean_vs_configured_total",
            "value": configured_total_gib - statistics.mean(peaks_gib) if peaks_gib else "",
            "unit": "GiB",
        },
        {
            "metric": "headroom_p95_vs_configured_total",
            "value": configured_total_gib - percentile(peaks_gib, 0.95) if peaks_gib else "",
            "unit": "GiB",
        },
        {"metric": "runtime_mean", "value": statistics.mean(runtimes) if runtimes else "", "unit": "s"},
        {"metric": "runtime_p50", "value": percentile(runtimes, 0.50) if runtimes else "", "unit": "s"},
        {"metric": "runtime_p95", "value": percentile(runtimes, 0.95) if runtimes else "", "unit": "s"},
    ]
    write_csv(run_root / "summary.csv", summary, ["metric", "value", "unit"])
    return summary


def parent_main(args: argparse.Namespace) -> int:
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")

    queries = parse_query_list(args.queries)
    devices = parse_csv_list(args.devices)
    run_name = args.run_name or datetime.now().strftime("computation_vram_%Y%m%d_%H%M%S")
    run_root = (args.output_root / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "cases").mkdir(exist_ok=True)

    config_path = run_root / "config.yml"
    write_config(
        config_path,
        num_gpus=len(devices),
        gpu_limit=args.gpu_usage_limit,
        reservation=args.reservation_limit_fraction,
        host_capacity=args.host_capacity,
    )

    inventory = gpu_inventory()
    uuid_to_index = {row["uuid"]: row["index"] for row in inventory}
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input.resolve()),
        "queries": [f"q{q}" for q in queries],
        "devices": args.devices,
        "num_gpus": len(devices),
        "gpu_usage_limit": args.gpu_usage_limit,
        "reservation_limit_fraction": args.reservation_limit_fraction,
        "host_capacity": args.host_capacity,
        "sample_interval_ms": args.sample_interval_ms,
        "timeout": args.timeout,
        "cache_disabled_env": {
            "SIRIUS_ENABLE_FIXED_PAGE_REUSE": "0",
            "SIRIUS_PIN_ROUND_ROBIN_CHUNKS": "0",
        },
        "gpu_inventory": inventory,
    }
    (run_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    (run_root / "README.md").write_text(
        f"""# Computation VRAM Probe

Purpose: baseline Sirius GPU query execution peak VRAM, with fixed-page reuse disabled.

Input: `{metadata['input']}`
Queries: `{','.join(metadata['queries'])}`
Devices: `{args.devices}`
GPU usage limit per GPU: `{args.gpu_usage_limit}`
Sampling interval: `{args.sample_interval_ms} ms`

Files:
- `query_vram.csv`: per-query runtime/status/peak process VRAM
- `summary.csv`: mean/p50/p95/max peak VRAM and estimated headroom
- `cases/q<N>.json`: compact child status and error details
"""
    )

    fields = [
        "query",
        "status",
        "returncode",
        "runtime_s",
        "row_count",
        "wall_s",
        "samples",
        "peak_total_mib",
        "peak_total_gib",
        "peak_max_gpu_mib",
        "peak_by_gpu_mib",
        "avg_sample_total_mib",
        "error",
    ]
    rows: list[dict[str, object]] = []
    for qnum in queries:
        row = run_one_query(args, run_root, config_path, qnum, uuid_to_index)
        rows.append(row)
        write_csv(run_root / "query_vram.csv", rows, fields)

    write_summary(run_root, rows, size_to_bytes(args.gpu_usage_limit), len(devices))
    print(f"==> wrote {run_root / 'query_vram.csv'}", flush=True)
    print(f"==> wrote {run_root / 'summary.csv'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--gpu-usage-limit", default="12GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--log-level", default="warn")
    parser.add_argument("--keep-child-logs", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-query", type=int, default=0)
    parser.add_argument("--case-output", type=Path, default=Path(""))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
