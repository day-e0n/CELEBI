#!/usr/bin/env python3
"""Record and plot process-level VRAM timelines for hot vs fixed-page paging."""

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
DEFAULT_OUTPUT_PREFIX = REPO_ROOT / "experiment" / "graph" / "hot_paging_vram_timeline_q6_q15_q20"


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_query_list(value: str) -> list[int]:
    out: list[int] = []
    for item in parse_csv_list(value):
        item = item.lower().removeprefix("q")
        if "-" in item:
            lo, hi = item.split("-", 1)
            out.extend(range(int(lo.lower().removeprefix("q")), int(hi.lower().removeprefix("q")) + 1))
        else:
            out.append(int(item))
    return out


def query_num(query: str) -> int:
    return int(query[1:]) if query.startswith("q") else 10_000


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
            mib = int(float(row["used_memory_mib"]))
        except ValueError:
            continue
        if row_pid != pid:
            continue
        gpu = uuid_to_index.get(row["gpu_uuid"], row["gpu_uuid"])
        usage[gpu] = max(usage.get(gpu, 0), mib)
    return usage


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


def condition_env(args: argparse.Namespace, condition: str, config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_LEVEL"] = args.log_level
    env["SIRIUS_LOG_DIR"] = str(Path("/tmp") / "sirius_hot_paging_vram_timeline")
    if condition == "hot":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "0"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "0"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE_SINGLE_MASK"] = "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "0"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "0"
        env.pop("SIRIUS_PIN_TIER", None)
    elif condition == "paging":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "1"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "1"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE_ROUND_ROBIN_CHUNKS"] = "1"
        env["SIRIUS_FIXED_PAGE_VIEW_ALIGNED_SPLITS"] = "1"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE"] = "1"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE_SINGLE_MASK"] = "1"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "1"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "1"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "1"
        env["SIRIUS_FIXED_PAGE_BACKED_PROVIDER"] = "1"
        env["SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"] = args.page_cache_bytes_per_gpu
        env["SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU"] = (
            args.page_cache_workspace_reserve_bytes_per_gpu
        )
    else:
        raise ValueError(condition)
    return env


def child_main(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(TPCH_DIR))
    result = {
        "query": f"q{args.case_query}",
        "condition": args.case_condition,
        "status": "error",
        "executions": [],
        "error": "",
    }
    try:
        from performance_test import open_connection, time_query  # noqa: PLC0415

        con = open_connection(str(args.input), gpu_execution=True)
        try:
            for execution in range(1, args.executions + 1):
                start = time.perf_counter()
                elapsed, rows = time_query(con, args.case_query, use_gpu=True)
                end = time.perf_counter()
                result["executions"].append(
                    {
                        "execution": execution,
                        "runtime_s": elapsed,
                        "wall_start_s": start,
                        "wall_end_s": end,
                        "row_count": len(rows),
                    }
                )
        finally:
            con.close()
        result["status"] = "ok"
        Path(args.case_output).write_text(json.dumps(result, indent=2) + "\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        Path(args.case_output).write_text(json.dumps(result, indent=2) + "\n")
        return 1


def run_case(
    args: argparse.Namespace,
    qnum: int,
    condition: str,
    config_path: Path,
    uuid_to_index: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    case_output = args.output_prefix.with_name(f"{args.output_prefix.name}_{condition}_q{qnum}_case.json")
    child_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--input",
        str(args.input),
        "--case-query",
        str(qnum),
        "--case-condition",
        condition,
        "--case-output",
        str(case_output),
        "--executions",
        str(args.executions),
    ]
    env = condition_env(args, condition, config_path)
    print(f"[RUN] q{qnum} {condition}", flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen(
        child_cmd,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    samples: list[dict[str, object]] = []
    deadline = start + args.timeout_s
    while True:
        usage = memory_for_pid(proc.pid, uuid_to_index)
        now = time.perf_counter()
        if usage:
            total_mib = sum(usage.values())
            samples.append(
                {
                    "query": f"q{qnum}",
                    "condition": condition,
                    "relative_s": now - start,
                    "total_mib": total_mib,
                    "total_gib": total_mib / 1024.0,
                    "max_gpu_mib": max(usage.values()) if usage else 0,
                    "per_gpu_mib": json.dumps(usage, sort_keys=True),
                }
            )
        ret = proc.poll()
        if ret is not None:
            break
        if now > deadline:
            proc.kill()
            proc.wait(timeout=10)
            break
        time.sleep(args.sample_interval_ms / 1000.0)

    usage = memory_for_pid(proc.pid, uuid_to_index)
    if usage:
        total_mib = sum(usage.values())
        samples.append(
            {
                "query": f"q{qnum}",
                "condition": condition,
                "relative_s": time.perf_counter() - start,
                "total_mib": total_mib,
                "total_gib": total_mib / 1024.0,
                "max_gpu_mib": max(usage.values()) if usage else 0,
                "per_gpu_mib": json.dumps(usage, sort_keys=True),
            }
        )

    if case_output.exists():
        case = json.loads(case_output.read_text())
    else:
        case = {"status": "error", "executions": [], "error": "child did not write case json"}

    run_rows: list[dict[str, object]] = []
    status = case.get("status", "error")
    for item in case.get("executions", []):
        run_rows.append(
            {
                "query": f"q{qnum}",
                "condition": condition,
                "status": status,
                "execution": item.get("execution"),
                "runtime_s": item.get("runtime_s"),
                "total_ms": float(item.get("runtime_s", 0.0)) * 1000.0,
                "row_count": item.get("row_count"),
            }
        )
    if not run_rows:
        run_rows.append(
            {
                "query": f"q{qnum}",
                "condition": condition,
                "status": status,
                "execution": "",
                "runtime_s": "",
                "total_ms": "",
                "row_count": "",
            }
        )
    peak = max((float(row["total_gib"]) for row in samples), default=0.0)
    warm = [float(row["runtime_s"]) for row in run_rows if row["runtime_s"] != "" and int(row["execution"]) >= 2]
    print(
        f"[DONE] q{qnum} {condition} status={status} peak={peak:.2f}GiB "
        f"warm_avg={(statistics.mean(warm) if warm else 0.0):.3f}s",
        flush=True,
    )
    return samples, run_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(samples: list[dict[str, object]], runs: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    queries = sorted({str(row["query"]) for row in samples}, key=query_num)
    fig, axes = plt.subplots(len(queries), 1, figsize=(13.5, 3.9 * len(queries)), sharex=False)
    if len(queries) == 1:
        axes = [axes]

    colors = {"hot": "#4c78a8", "paging": "#f58518"}
    labels = {"hot": "baseline hot", "paging": "fixed-page paging"}
    by_query_condition: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in samples:
        by_query_condition.setdefault((str(row["query"]), str(row["condition"])), []).append(row)

    run_key = {(str(row["query"]), str(row["condition"])): [] for row in runs}
    for row in runs:
        if row.get("runtime_s") != "":
            run_key.setdefault((str(row["query"]), str(row["condition"])), []).append(float(row["runtime_s"]))

    for ax, query in zip(axes, queries):
        for condition in ["hot", "paging"]:
            rows = by_query_condition.get((query, condition), [])
            rows.sort(key=lambda item: float(item["relative_s"]))
            if not rows:
                continue
            ax.plot(
                [float(row["relative_s"]) for row in rows],
                [float(row["total_gib"]) for row in rows],
                color=colors[condition],
                linewidth=2.4,
                label=labels[condition],
            )
        hot_warm = run_key.get((query, "hot"), [])[1:]
        paging_warm = run_key.get((query, "paging"), [])[1:]
        subtitle = query
        if hot_warm and paging_warm:
            subtitle += f"  warm avg: hot {statistics.mean(hot_warm):.2f}s, paging {statistics.mean(paging_warm):.2f}s"
        ax.set_title(subtitle, loc="left")
        ax.set_ylabel("VRAM usage (GiB)")
        ax.grid(axis="both", alpha=0.24)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time since process start (s)")
    fig.suptitle("Process-level VRAM timeline: baseline hot vs fixed-page paging", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parent_main(args: argparse.Namespace) -> int:
    queries = parse_query_list(args.queries)
    config_path = args.output_prefix.with_name(args.output_prefix.name + "_config.yml")
    write_config(
        config_path,
        num_gpus=len(parse_csv_list(args.devices)),
        gpu_limit=args.gpu_usage_limit,
        reservation=args.reservation_limit_fraction,
        host_capacity=args.host_capacity,
    )
    inventory = gpu_inventory()
    uuid_to_index = {row["uuid"]: row["index"] for row in inventory}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input.resolve()),
        "queries": [f"q{q}" for q in queries],
        "executions": args.executions,
        "devices": args.devices,
        "gpu_usage_limit": args.gpu_usage_limit,
        "page_cache_bytes_per_gpu": args.page_cache_bytes_per_gpu,
        "sample_interval_ms": args.sample_interval_ms,
        "gpu_inventory": inventory,
    }
    args.output_prefix.with_name(args.output_prefix.name + "_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    samples: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for qnum in queries:
        for condition in ["hot", "paging"]:
            case_samples, case_runs = run_case(args, qnum, condition, config_path, uuid_to_index)
            samples.extend(case_samples)
            runs.extend(case_runs)
            write_csv(args.output_prefix.with_name(args.output_prefix.name + "_samples.csv"), samples)
            write_csv(args.output_prefix.with_name(args.output_prefix.name + "_runs.csv"), runs)

    plot(samples, runs, args.output_prefix.with_name(args.output_prefix.name + ".png"))
    print(args.output_prefix.with_name(args.output_prefix.name + "_samples.csv"))
    print(args.output_prefix.with_name(args.output_prefix.name + "_runs.csv"))
    print(args.output_prefix.with_name(args.output_prefix.name + ".png"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--queries", default="6,15,20")
    parser.add_argument("--executions", type=int, default=5)
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--gpu-usage-limit", default="7GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument("--page-cache-bytes-per-gpu", default="7GB")
    parser.add_argument("--page-cache-workspace-reserve-bytes-per-gpu", default="3072MB")
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--timeout-s", type=int, default=420)
    parser.add_argument("--log-level", default="warn")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-query", type=int, default=0)
    parser.add_argument("--case-condition", default="")
    parser.add_argument("--case-output", type=Path, default=Path(""))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
