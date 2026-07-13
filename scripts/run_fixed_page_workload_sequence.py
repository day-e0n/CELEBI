#!/usr/bin/env python3
"""Run a fixed TPC-H workload sequence 3x for Sirius cold/hot vs fixed-page paging.

One condition process runs the whole workload repeatedly in one DuckDB/Sirius
session, so baseline execution 1 is cold and executions 2..N are hot. Paging
uses the same rule: execution 1 populates cache, executions 2..N are measured.

With --repeat-layout=query, each query is repeated N times before moving to the
next query. This is useful for per-query hot-vs-paging experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "experiment" / "fixed_page_runs" / "fixedcol_seq22_3x"
DEFAULT_GRAPH_DIR = REPO_ROOT / "experiment" / "graph"
DEFAULT_QUERIES = "3,10,7,5,8,9,20,11,2,16,19,17,14,1,6,15,21,12,4,18,13,22"
DEFAULT_CONDITIONS = "cold_hot,paging"

sys.path.insert(0, str(TPCH_DIR))
from performance_test import open_connection, time_query  # noqa: E402

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")

STAGE_ORDER = [
    "SCAN",
    "FILTER",
    "JOIN",
    "AGGREGATE",
    "PROJECTION",
    "PARTITION",
    "CONCAT",
    "SORT",
    "RESULT",
    "OTHER",
]
CACHE_STAGE_TO_FIELD = {
    "cache_column_materialize": "cache_column_materialize_ms",
    "cache_column_view": "cache_column_view_ms",
    "splice_materialize": "splice_materialize_ms",
    "splice_view_build": "splice_view_build_ms",
    "filtered_splice_materialize": "filtered_splice_materialize_ms",
    "filtered_splice_materialize_single_mask": "filtered_splice_materialize_single_mask_ms",
    "partial_filter_cached_mask": "partial_filter_cached_mask_ms",
    "partial_filter_mask_splice": "partial_filter_mask_splice_ms",
    "single_mask_filter_apply": "single_mask_filter_apply_ms",
    "cached_filter_select": "cached_filter_select_ms",
    "post_filter_select": "post_filter_select_ms",
    "inline_assembly": "inline_assembly_ms",
    "post_filter_project_assembly": "post_filter_project_assembly_ms",
}
CACHE_STAGE_FIELDS = list(CACHE_STAGE_TO_FIELD.values())
EXTRA_CACHE_STAGE_FIELDS = [
    "cache_column_materialize_ms",
    "cache_column_view_ms",
    "splice_materialize_ms",
    "splice_view_build_ms",
    "filtered_splice_materialize_ms",
    "filtered_splice_materialize_single_mask_ms",
    "partial_filter_cached_mask_ms",
    "partial_filter_mask_splice_ms",
    "single_mask_filter_apply_ms",
    "cached_filter_select_ms",
]
FIXED_COUNTER_FIELDS = [
    "fixed_page_cached_gb",
    "fixed_page_reuse_split_count",
    "fixed_page_materialize_count",
    "fixed_page_materialize_view_count",
    "fixed_page_fallback_count",
    "fixed_page_auto_populate_count",
    "fixed_page_auto_skip_count",
    "fixed_page_backed_provider_count",
    "page_budget_evicted_pages",
    "page_budget_evicted_gb",
    "page_pressure_evicted_pages",
    "page_pressure_evicted_gb",
    "page_directory_resident_gb_max",
    "page_directory_entries_max",
    "page_directory_eviction_count_max",
    "demand_load_pages",
    "demand_load_gb",
    "demand_load_ms",
    "demand_load_read_calls",
]


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_query_list(value: str) -> list[int]:
    out: list[int] = []
    for token in parse_csv_list(value):
        token = token.lower().removeprefix("q")
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return out


def parse_kv(line: str) -> dict[str, str]:
    return dict(KV_RE.findall(line))


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def stdev(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) >= 2 else ""


def parse_log_timestamp_ms(line: str) -> float | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000.0
    except ValueError:
        return None


def union_interval_ms(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += max(cur_end - cur_start, 0.0)
            cur_start, cur_end = start, end
    total += max(cur_end - cur_start, 0.0)
    return total


def column_bytes_uncompressed_total(value: str | None) -> int:
    if not value:
        return 0
    total = 0
    for item in value.split(","):
        parts = item.rsplit(":", 2)
        if len(parts) != 3:
            continue
        total += to_int(parts[2])
    return total


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_config(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        "sirius:",
        "  topology:",
        f"    num_gpus: {args.num_gpus}",
        "  memory:",
        "    gpu:",
        f"      usage_limit_bytes: {args.gpu_usage_limit}",
        f"      reservation_limit_fraction: {args.reservation_limit_fraction}",
        "    host:",
        f"      capacity_bytes: {args.host_capacity}",
        "",
    ]))


def make_env(args: argparse.Namespace, condition: str, log_dir: Path, config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = args.log_level
    if args.fixed_width_page_bytes:
        env["SIRIUS_FIXED_WIDTH_PAGE_BYTES"] = args.fixed_width_page_bytes
    if condition == "cold_hot":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "0"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "0"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE_SINGLE_MASK"] = "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "0"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "0"
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
        env["SIRIUS_FIXED_PAGE_HYBRID_PROVIDER"] = args.fixed_page_hybrid_provider
        if args.provider_coalesce_pages:
            env["SIRIUS_FIXED_PAGE_PROVIDER_COALESCE_PAGES"] = args.provider_coalesce_pages
        if args.page_cache_bytes_per_gpu:
            env["SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"] = args.page_cache_bytes_per_gpu
        if args.page_cache_workspace_reserve_bytes_per_gpu:
            env["SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU"] = (
                args.page_cache_workspace_reserve_bytes_per_gpu
            )
        if args.page_cache_min_free_bytes_per_gpu:
            env["SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU"] = (
                args.page_cache_min_free_bytes_per_gpu
            )
        if args.demand_max_bytes_per_split:
            env["SIRIUS_FIXED_PAGE_DEMAND_MAX_BYTES_PER_SPLIT"] = args.demand_max_bytes_per_split
    else:
        raise ValueError(f"unknown condition: {condition}")
    return env


def is_timed_query_begin(line: str) -> bool:
    marker = "QueryBegin: SQL:"
    if marker not in line:
        return False
    sql = line.split(marker, 1)[1].strip().lower()
    return not (
        sql.startswith("set ")
        or sql.startswith("create view")
        or sql.startswith("call pin_table")
        or sql.startswith("call unpin_table")
    )


def log_segments(log_dir: Path, expected_count: int) -> list[list[str]]:
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return [[] for _ in range(expected_count)]
    lines = logs[-1].read_text(errors="replace").splitlines()
    begins = [idx for idx, line in enumerate(lines) if is_timed_query_begin(line)]
    segments: list[list[str]] = []
    for pos, start in enumerate(begins[:expected_count]):
        end = begins[pos + 1] if pos + 1 < len(begins) else len(lines)
        segments.append(lines[start:end])
    while len(segments) < expected_count:
        segments.append([])
    return segments


def canonical_stage(raw: str) -> str:
    stage = raw if raw in STAGE_ORDER else "OTHER"
    return stage


def summarize_segment(lines: list[str], total_ms: float) -> dict[str, object]:
    scan_work_ms = 0.0
    scan_uncompressed_bytes = 0
    scan_intervals: list[tuple[float, float]] = []
    stage_work = defaultdict(float)
    stage_input_bytes = defaultdict(int)
    stage_output_bytes = defaultdict(int)
    stage_tasks = defaultdict(int)
    stage_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    all_stage_intervals: list[tuple[float, float]] = []
    cache_stage = defaultdict(float)
    fixed_cached_bytes = 0
    fixed_reuse_split_count = 0
    fixed_materialize_count = 0
    fixed_materialize_view_count = 0
    fixed_fallback_count = 0
    fixed_auto_populate_count = 0
    fixed_auto_skip_count = 0
    fixed_backed_provider_count = 0
    evicted_pages = 0
    evicted_bytes = 0
    pressure_evicted_pages = 0
    pressure_evicted_bytes = 0
    demand_pages = 0
    demand_bytes = 0
    demand_ms = 0.0
    demand_read_calls = 0
    max_resident_bytes = 0
    max_directory_entries = 0
    max_eviction_count = 0

    for line in lines:
        if "[scan-audit] parquet_materialize" in line:
            fields = parse_kv(line)
            duration_ms = to_int(fields.get("duration_us")) / 1000.0
            scan_work_ms += duration_ms
            scan_uncompressed_bytes += column_bytes_uncompressed_total(fields.get("column_bytes")) or to_int(fields.get("uncompressed_bytes"))
            end_ms = parse_log_timestamp_ms(line)
            if end_ms is not None:
                scan_intervals.append((end_ms - duration_ms, end_ms))
        elif "[stage-audit]" in line:
            fields = parse_kv(line)
            stage = canonical_stage(fields.get("stage_kind", "OTHER"))
            duration_ms = to_int(fields.get("duration_us")) / 1000.0
            stage_work[stage] += duration_ms
            stage_input_bytes[stage] += to_int(fields.get("input_bytes"))
            stage_output_bytes[stage] += to_int(fields.get("output_bytes"))
            stage_tasks[stage] += 1
            end_ms = parse_log_timestamp_ms(line)
            if end_ms is not None:
                interval = (end_ms - duration_ms, end_ms)
                stage_intervals[stage].append(interval)
                all_stage_intervals.append(interval)
        elif "[fixed-page-cache] hybrid_reuse_split" in line:
            fields = parse_kv(line)
            fixed_cached_bytes += to_int(fields.get("useful_bytes") or fields.get("cached_bytes"))
            fixed_reuse_split_count += 1
        elif "[fixed-page-cache] hybrid_reuse_materialize_view" in line:
            fixed_materialize_view_count += 1
        elif "[fixed-page-cache] hybrid_reuse_materialize " in line:
            fixed_materialize_count += 1
        elif "[fixed-page-cache] hybrid_reuse_row_group_fallback" in line:
            fixed_fallback_count += 1
        elif "[fixed-page-cache] auto_cache_populate" in line:
            fixed_auto_populate_count += 1
        elif "[fixed-page-cache] auto_cache_skip" in line:
            fixed_auto_skip_count += 1
        elif ("[fixed-page-cache] using page-backed cached provider" in line or
              "[fixed-page-cache] using hybrid page/chunk cached provider" in line):
            fixed_backed_provider_count += 1
        elif "[fixed-page-cache] stage_timing" in line:
            fields = parse_kv(line)
            field = CACHE_STAGE_TO_FIELD.get(fields.get("stage", ""))
            if field:
                cache_stage[field] += to_int(fields.get("duration_us")) / 1000.0
        elif "[fixed-page-cache] page_budget applied" in line:
            fields = parse_kv(line)
            evicted_pages += to_int(fields.get("evicted_pages"))
            evicted_bytes += to_int(fields.get("evicted_bytes"))
        elif "[fixed-page-cache] memory_pressure applied" in line:
            fields = parse_kv(line)
            pressure_evicted_pages += to_int(fields.get("evicted_pages"))
            pressure_evicted_bytes += to_int(fields.get("evicted_bytes"))
        elif "[fixed-page-cache] demand_load loaded_pages" in line:
            fields = parse_kv(line)
            demand_pages += to_int(fields.get("loaded_pages"))
            demand_bytes += to_int(fields.get("loaded_bytes"))
            demand_ms += to_int(fields.get("duration_us")) / 1000.0
            demand_read_calls += to_int(fields.get("read_calls"))
        elif "[fixed-page-cache] page_directory" in line:
            fields = parse_kv(line)
            max_resident_bytes = max(max_resident_bytes, to_int(fields.get("resident_bytes")))
            max_directory_entries = max(max_directory_entries, to_int(fields.get("directory_entries")))
            max_eviction_count = max(max_eviction_count, to_int(fields.get("eviction_count")))

    load_wall_ms = min(union_interval_ms(scan_intervals), total_ms)
    cache_management_ms = sum(cache_stage.values()) + demand_ms
    out: dict[str, object] = {
        "load_ms": load_wall_ms,
        "scan_materialize_work_ms": scan_work_ms,
        "scan_uncompressed_gb": scan_uncompressed_bytes / 1e9,
        "computation_ms": max(total_ms - load_wall_ms, 0.0),
        "stage_total_work_ms": sum(stage_work.values()),
        "stage_total_wall_ms": union_interval_ms(all_stage_intervals),
        "cache_management_ms": cache_management_ms,
        "fixed_page_cached_gb": fixed_cached_bytes / 1e9,
        "fixed_page_reuse_split_count": fixed_reuse_split_count,
        "fixed_page_materialize_count": fixed_materialize_count,
        "fixed_page_materialize_view_count": fixed_materialize_view_count,
        "fixed_page_fallback_count": fixed_fallback_count,
        "fixed_page_auto_populate_count": fixed_auto_populate_count,
        "fixed_page_auto_skip_count": fixed_auto_skip_count,
        "fixed_page_backed_provider_count": fixed_backed_provider_count,
        "page_budget_evicted_pages": evicted_pages,
        "page_budget_evicted_gb": evicted_bytes / 1e9,
        "page_pressure_evicted_pages": pressure_evicted_pages,
        "page_pressure_evicted_gb": pressure_evicted_bytes / 1e9,
        "page_directory_resident_gb_max": max_resident_bytes / 1e9,
        "page_directory_entries_max": max_directory_entries,
        "page_directory_eviction_count_max": max_eviction_count,
        "demand_load_pages": demand_pages,
        "demand_load_gb": demand_bytes / 1e9,
        "demand_load_ms": demand_ms,
        "demand_load_read_calls": demand_read_calls,
    }
    for stage in STAGE_ORDER:
        lower = stage.lower()
        out[f"{lower}_work_ms"] = stage_work[stage]
        out[f"{lower}_wall_ms"] = union_interval_ms(stage_intervals[stage])
        out[f"{lower}_input_gb"] = stage_input_bytes[stage] / 1e9
        out[f"{lower}_output_gb"] = stage_output_bytes[stage] / 1e9
        out[f"{lower}_task_count"] = stage_tasks[stage]
    for field in CACHE_STAGE_FIELDS:
        out[field] = cache_stage[field]
    return out


def series_for(condition: str, execution: int, summary_mode: str = "cold_hot") -> tuple[str, int]:
    if summary_mode == "avg_all":
        if condition == "cold_hot":
            return "baseline", 1
        if condition == "paging":
            return "paging", 1
    if condition == "cold_hot":
        return ("cold", 1) if execution == 1 else ("hot", 1)
    if condition == "paging":
        return ("paging_warmup", 0) if execution == 1 else ("paging", 1)
    return condition, 1


def run_condition_child(args: argparse.Namespace) -> int:
    condition = args.case_condition
    output = args.output.resolve()
    case_dir = output / condition / "workload"
    csv_dir = case_dir / "csv"
    log_dir = case_dir / "log_dir"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    queries = parse_query_list(args.queries)
    config_path = Path(args.config).resolve()
    env = make_env(args, condition, log_dir.resolve(), config_path)
    os.environ.update(env)

    metadata = {
        "condition": condition,
        "queries": queries,
        "executions": args.executions,
        "repeat_layout": args.repeat_layout,
        "input": str(args.input),
        "devices": args.devices,
        "num_gpus": args.num_gpus,
        "gpu_usage_limit": args.gpu_usage_limit,
        "page_cache_bytes_per_gpu": args.page_cache_bytes_per_gpu,
        "fixed_width_page_bytes": args.fixed_width_page_bytes,
        "summary_mode": args.summary_mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows: list[dict[str, object]] = []
    print(f"[CASE] condition={condition} executions={args.executions} queries={','.join('q'+str(q) for q in queries)}", flush=True)
    con = open_connection(str(args.input), gpu_execution=True)
    try:
        if args.repeat_layout == "query":
            schedule = [
                (execution, position, qnum)
                for position, qnum in enumerate(queries, start=1)
                for execution in range(1, args.executions + 1)
            ]
        else:
            schedule = [
                (execution, position, qnum)
                for execution in range(1, args.executions + 1)
                for position, qnum in enumerate(queries, start=1)
            ]
        for execution, position, qnum in schedule:
            elapsed, result_rows = time_query(con, qnum, use_gpu=True)
            rows.append({
                    "condition": condition,
                    "execution": execution,
                    "position": position,
                    "query": f"q{qnum}",
                    "runtime_s": elapsed,
                    "total_ms": elapsed * 1000.0,
                    "result_rows": len(result_rows),
            })
            write_csv(csv_dir / "runtimes.csv", rows, RUNTIME_FIELDS)
            print(f"[CASE] {condition} exec={execution} pos={position} q{qnum} {elapsed:.4f}s rows={len(result_rows)}", flush=True)
    finally:
        con.close()
    return 0


RUNTIME_FIELDS = ["condition", "execution", "position", "query", "runtime_s", "total_ms", "result_rows"]
STAGE_FIELDS_FLAT: list[str] = []
for _stage in STAGE_ORDER:
    _lower = _stage.lower()
    STAGE_FIELDS_FLAT.extend([
        f"{_lower}_work_ms",
        f"{_lower}_wall_ms",
        f"{_lower}_input_gb",
        f"{_lower}_output_gb",
        f"{_lower}_task_count",
    ])
EXECUTION_FIELDS = [
    "condition",
    "series",
    "included_in_summary",
    "execution",
    "position",
    "query",
    "runtime_s",
    "total_ms",
    "result_rows",
    "load_ms",
    "scan_materialize_work_ms",
    "scan_uncompressed_gb",
    "computation_ms",
    "stage_total_work_ms",
    "stage_total_wall_ms",
    *STAGE_FIELDS_FLAT,
    "cache_management_ms",
    *FIXED_COUNTER_FIELDS,
    *CACHE_STAGE_FIELDS,
    "benchmark_dir",
]
SUMMARY_NUMERIC_FIELDS = [
    "total_ms",
    "load_ms",
    "scan_materialize_work_ms",
    "scan_uncompressed_gb",
    "computation_ms",
    "stage_total_work_ms",
    "stage_total_wall_ms",
    *[f"{s.lower()}_work_ms" for s in STAGE_ORDER],
    *[f"{s.lower()}_wall_ms" for s in STAGE_ORDER],
    "cache_management_ms",
    *FIXED_COUNTER_FIELDS,
    *CACHE_STAGE_FIELDS,
]
WORKLOAD_MAX_FIELDS = {
    "page_directory_resident_gb_max",
    "page_directory_entries_max",
    "page_directory_eviction_count_max",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_condition(output: Path, condition: str, summary_mode: str = "cold_hot") -> list[dict[str, object]]:
    case_dir = output / condition / "workload"
    runtime_rows = read_csv(case_dir / "csv" / "runtimes.csv")
    segments = log_segments(case_dir / "log_dir", len(runtime_rows))
    rows: list[dict[str, object]] = []
    for row, lines in zip(runtime_rows, segments):
        total_ms = to_float(row.get("total_ms"))
        execution = to_int(row.get("execution"))
        series, included = series_for(condition, execution, summary_mode)
        rows.append({
            "condition": condition,
            "series": series,
            "included_in_summary": included,
            "execution": execution,
            "position": to_int(row.get("position")),
            "query": row.get("query", ""),
            "runtime_s": to_float(row.get("runtime_s")),
            "total_ms": total_ms,
            "result_rows": to_int(row.get("result_rows")),
            **summarize_segment(lines, total_ms),
            "benchmark_dir": str(case_dir),
        })
    return rows


def aggregate_query_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not to_int(row.get("included_in_summary")):
            continue
        groups[(str(row["series"]), to_int(row["position"]), str(row["query"]))].append(row)
    out: list[dict[str, object]] = []
    for (series, position, query), group in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        item: dict[str, object] = {"series": series, "position": position, "query": query, "samples": len(group)}
        for field in SUMMARY_NUMERIC_FIELDS:
            values = [to_float(row.get(field)) for row in group]
            item[field] = mean(values)
            item[f"{field}_std"] = stdev(values)
        out.append(item)
    return out


def aggregate_workload_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_execution: list[dict[str, object]] = []
    exec_groups: dict[tuple[str, int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        series = str(row["series"])
        if series == "paging_warmup":
            include = 0
        else:
            include = to_int(row.get("included_in_summary"))
        exec_groups[(str(row["condition"]), to_int(row["execution"]), series, include)].append(row)
    for (condition, execution, series, include), group in sorted(exec_groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        item: dict[str, object] = {
            "condition": condition,
            "execution": execution,
            "series": series,
            "included_in_summary": include,
            "query_count": len(group),
        }
        for field in SUMMARY_NUMERIC_FIELDS:
            values = [to_float(row.get(field)) for row in group]
            item[field] = max(values) if field in WORKLOAD_MAX_FIELDS else sum(values)
        per_execution.append(item)

    series_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_execution:
        if to_int(row.get("included_in_summary")):
            series_groups[str(row["series"])].append(row)
    summary: list[dict[str, object]] = []
    for series, group in sorted(series_groups.items()):
        item: dict[str, object] = {"series": series, "samples": len(group), "query_count": group[0].get("query_count", "")}
        for field in SUMMARY_NUMERIC_FIELDS:
            values = [to_float(row.get(field)) for row in group]
            item[field] = mean(values)
            item[f"{field}_std"] = stdev(values)
        summary.append(item)
    return per_execution, summary


def write_comparison(graph_dir: Path, prefix: str, workload_summary: list[dict[str, object]]) -> None:
    by_series = {str(row["series"]): row for row in workload_summary}
    rows: list[dict[str, object]] = []
    baseline = by_series.get("baseline")
    hot = by_series.get("hot")
    paging = by_series.get("paging")
    cold = by_series.get("cold")
    if baseline and paging:
        rows.append({
            "comparison": "baseline_vs_paging_avg_all",
            "baseline_series": "baseline",
            "target_series": "paging",
            "baseline_total_ms": baseline.get("total_ms"),
            "target_total_ms": paging.get("total_ms"),
            "speedup": to_float(baseline.get("total_ms")) / to_float(paging.get("total_ms")) if to_float(paging.get("total_ms")) else "",
            "baseline_load_ms": baseline.get("load_ms"),
            "target_load_ms": paging.get("load_ms"),
            "load_reduction_ratio": (to_float(baseline.get("load_ms")) - to_float(paging.get("load_ms"))) / to_float(baseline.get("load_ms")) if to_float(baseline.get("load_ms")) else "",
            "paging_fixed_page_cached_gb": paging.get("fixed_page_cached_gb"),
            "paging_cache_management_ms": paging.get("cache_management_ms"),
            "paging_demand_load_ms": paging.get("demand_load_ms"),
            "paging_page_backed_provider_count": paging.get("fixed_page_backed_provider_count"),
            "paging_page_budget_evicted_pages": paging.get("page_budget_evicted_pages"),
            "paging_page_directory_resident_gb_max": paging.get("page_directory_resident_gb_max"),
        })
    if cold and hot:
        rows.append({
            "comparison": "cold_vs_hot",
            "baseline_series": "cold",
            "target_series": "hot",
            "baseline_total_ms": cold.get("total_ms"),
            "target_total_ms": hot.get("total_ms"),
            "speedup": to_float(cold.get("total_ms")) / to_float(hot.get("total_ms")) if to_float(hot.get("total_ms")) else "",
            "baseline_load_ms": cold.get("load_ms"),
            "target_load_ms": hot.get("load_ms"),
            "load_reduction_ratio": (to_float(cold.get("load_ms")) - to_float(hot.get("load_ms"))) / to_float(cold.get("load_ms")) if to_float(cold.get("load_ms")) else "",
        })
    if hot and paging:
        rows.append({
            "comparison": "hot_vs_paging",
            "baseline_series": "hot",
            "target_series": "paging",
            "baseline_total_ms": hot.get("total_ms"),
            "target_total_ms": paging.get("total_ms"),
            "speedup": to_float(hot.get("total_ms")) / to_float(paging.get("total_ms")) if to_float(paging.get("total_ms")) else "",
            "baseline_load_ms": hot.get("load_ms"),
            "target_load_ms": paging.get("load_ms"),
            "load_reduction_ratio": (to_float(hot.get("load_ms")) - to_float(paging.get("load_ms"))) / to_float(hot.get("load_ms")) if to_float(hot.get("load_ms")) else "",
            "paging_fixed_page_cached_gb": paging.get("fixed_page_cached_gb"),
            "paging_cache_management_ms": paging.get("cache_management_ms"),
            "paging_demand_load_ms": paging.get("demand_load_ms"),
            "paging_page_backed_provider_count": paging.get("fixed_page_backed_provider_count"),
            "paging_page_budget_evicted_pages": paging.get("page_budget_evicted_pages"),
            "paging_page_directory_resident_gb_max": paging.get("page_directory_resident_gb_max"),
        })
    fields = [
        "comparison",
        "baseline_series",
        "target_series",
        "baseline_total_ms",
        "target_total_ms",
        "speedup",
        "baseline_load_ms",
        "target_load_ms",
        "load_reduction_ratio",
        "paging_fixed_page_cached_gb",
        "paging_cache_management_ms",
        "paging_demand_load_ms",
        "paging_page_backed_provider_count",
        "paging_page_budget_evicted_pages",
        "paging_page_directory_resident_gb_max",
    ]
    write_csv(graph_dir / f"{prefix}_comparison.csv", rows, fields)


def write_query_comparison(graph_dir: Path, prefix: str, query_summary: list[dict[str, object]]) -> None:
    by_key = {(str(row["series"]), to_int(row["position"])): row for row in query_summary}
    series_names = {str(row["series"]) for row in query_summary}
    base_series = "baseline" if "baseline" in series_names else "hot"
    positions = sorted({to_int(row["position"]) for row in query_summary})
    rows: list[dict[str, object]] = []
    for position in positions:
        base = by_key.get((base_series, position))
        paging = by_key.get(("paging", position))
        if not base or not paging:
            continue
        base_total = to_float(base.get("total_ms"))
        paging_total = to_float(paging.get("total_ms"))
        base_scan = to_float(base.get("scan_materialize_work_ms"))
        paging_scan = to_float(paging.get("scan_materialize_work_ms"))
        base_load = to_float(base.get("load_ms"))
        paging_load = to_float(paging.get("load_ms"))
        rows.append({
            "position": position,
            "query": base.get("query", ""),
            "baseline_series": base_series,
            "baseline_total_ms": base_total,
            "paging_total_ms": paging_total,
            "latency_saved_ms": base_total - paging_total,
            "baseline_load_ms": base_load,
            "paging_load_ms": paging_load,
            "load_saved_ms": base_load - paging_load,
            "baseline_scan_materialize_work_ms": base_scan,
            "paging_scan_materialize_work_ms": paging_scan,
            "scan_work_saved_ms": base_scan - paging_scan,
            "baseline_join_work_ms": base.get("join_work_ms"),
            "paging_join_work_ms": paging.get("join_work_ms"),
            "baseline_filter_work_ms": base.get("filter_work_ms"),
            "paging_filter_work_ms": paging.get("filter_work_ms"),
            "paging_page_backed_provider_count": paging.get("fixed_page_backed_provider_count"),
            "paging_page_budget_evicted_pages": paging.get("page_budget_evicted_pages"),
            "paging_page_directory_resident_gb_max": paging.get("page_directory_resident_gb_max"),
        })
    fields = [
        "position", "query", "baseline_series", "baseline_total_ms", "paging_total_ms", "latency_saved_ms",
        "baseline_load_ms", "paging_load_ms", "load_saved_ms",
        "baseline_scan_materialize_work_ms", "paging_scan_materialize_work_ms", "scan_work_saved_ms",
        "baseline_join_work_ms", "paging_join_work_ms", "baseline_filter_work_ms", "paging_filter_work_ms",
        "paging_page_backed_provider_count", "paging_page_budget_evicted_pages",
        "paging_page_directory_resident_gb_max",
    ]
    write_csv(graph_dir / f"{prefix}_{base_series}_vs_paging_query_delta.csv", rows, fields)


def plot_outputs(graph_dir: Path, prefix: str, query_summary: list[dict[str, object]], workload_summary: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] plotting skipped: {exc}", flush=True)
        return

    graph_dir.mkdir(parents=True, exist_ok=True)
    available_series = {str(row["series"]) for row in workload_summary} | {str(row["series"]) for row in query_summary}
    series_order = [s for s in ["baseline", "cold", "hot", "paging"] if s in available_series]
    series_colors = {"baseline": "#4c78a8", "cold": "#9ecae1", "hot": "#4c78a8", "paging": "#59a14f"}
    stage_groups = [
        ("SCAN", "scan_work_ms", "#4c78a8"),
        ("FILTER", "filter_work_ms", "#f58518"),
        ("JOIN", "join_work_ms", "#54a24b"),
        ("AGG", "aggregate_work_ms", "#e45756"),
        ("PROJ", "projection_work_ms", "#72b7b2"),
        ("PART/CONCAT", "partition_concat_work_ms", "#b279a2"),
        ("SORT/RESULT", "sort_result_work_ms", "#ff9da6"),
        ("CACHE MGMT", "cache_management_ms", "#8d6e63"),
        ("OTHER", "other_work_ms", "#bab0ac"),
    ]

    # Materialize derived fields for plots without mutating CSV schema.
    def row_value(row: dict[str, object], field: str) -> float:
        if field == "partition_concat_work_ms":
            return to_float(row.get("partition_work_ms")) + to_float(row.get("concat_work_ms"))
        if field == "sort_result_work_ms":
            return to_float(row.get("sort_work_ms")) + to_float(row.get("result_work_ms"))
        return to_float(row.get(field))

    by_workload = {str(row["series"]): row for row in workload_summary}
    labels = [s for s in series_order if s in by_workload]
    if labels:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        vals = [to_float(by_workload[s].get("total_ms")) for s in labels]
        ax.bar(labels, vals, color=[series_colors[s] for s in labels], width=0.62)
        ax.set_ylabel("Total workload latency (ms)", fontsize=12)
        ax.set_title("Workload latency", fontsize=15)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(graph_dir / f"{prefix}_workload_latency.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        for label, field, color in stage_groups:
            vals = np.array([row_value(by_workload[s], field) for s in labels])
            ax.bar(x, vals, bottom=bottoms, label=label, color=color, width=0.64)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Stage work / cache work (ms)", fontsize=12)
        ax.set_title("Workload stage-level breakdown", fontsize=15)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
        ax.grid(axis="y", alpha=0.22)
        fig.savefig(graph_dir / f"{prefix}_workload_stage_breakdown.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
        load_vals = np.array([to_float(by_workload[s].get("load_ms")) for s in labels])
        comp_vals = np.array([to_float(by_workload[s].get("computation_ms")) for s in labels])
        ax.bar(labels, load_vals, label="load/materialize wall", color="#4c78a8", width=0.62)
        ax.bar(labels, comp_vals, bottom=load_vals, label="non-load latency", color="#bab0ac", width=0.62)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title("Load vs non-load latency", fontsize=15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(graph_dir / f"{prefix}_workload_load_vs_compute.png", dpi=180)
        plt.close(fig)

    positions = sorted({to_int(row["position"]) for row in query_summary})
    query_labels = []
    for pos in positions:
        sample = next((row for row in query_summary if to_int(row["position"]) == pos), None)
        query_labels.append(str(sample["query"]) if sample else str(pos))
    by_key = {(str(row["series"]), to_int(row["position"])): row for row in query_summary}
    if positions:
        fig, ax = plt.subplots(figsize=(max(13, len(positions) * 0.62), 5.8), constrained_layout=True)
        x = np.arange(len(positions))
        width = 0.25
        offset_center = (len(series_order) - 1) / 2.0
        for idx, series in enumerate(series_order):
            vals = [to_float(by_key.get((series, pos), {}).get("total_ms")) for pos in positions]
            ax.bar(x + (idx - offset_center) * width, vals, width=width, label=series, color=series_colors[series])
        ax.set_xticks(x)
        ax.set_xticklabels(query_labels, rotation=35, ha="right", fontsize=11)
        ax.set_ylabel("Query latency (ms)", fontsize=12)
        ax.set_title("Per-query latency in workload order", fontsize=15)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(graph_dir / f"{prefix}_per_query_latency.png", dpi=180)
        plt.close(fig)

        breakdown_series_order = [s for s in ["hot", "baseline", "paging"] if s in available_series]
        if "hot" in breakdown_series_order and "baseline" in breakdown_series_order:
            breakdown_series_order.remove("baseline")
        if len(breakdown_series_order) >= 2:
            fig, ax = plt.subplots(figsize=(max(15, len(positions) * 0.92), 8.2), constrained_layout=True)
            width = min(0.34, 0.74 / max(len(breakdown_series_order), 1))
            offset_center = (len(breakdown_series_order) - 1) / 2.0
            for idx, series in enumerate(breakdown_series_order):
                offset = (idx - offset_center) * width
                bottoms = np.zeros(len(positions))
                for label, field, color in stage_groups:
                    vals = np.array([row_value(by_key.get((series, pos), {}), field) for pos in positions])
                    ax.bar(
                        x + offset,
                        vals,
                        bottom=bottoms,
                        width=width,
                        label=label if idx == 0 else None,
                        color=color,
                        edgecolor="white",
                        linewidth=0.28,
                    )
                    bottoms += vals
            ax.set_xticks(x)
            ax.set_xticklabels(query_labels, rotation=35, ha="right", fontsize=15)
            ax.set_ylabel("Stage/cache work (ms)", fontsize=17)
            ax.set_title("Per-query stage breakdown", fontsize=21, pad=14)
            ax.tick_params(axis="y", labelsize=15)
            ax.grid(axis="y", alpha=0.22)
            ax.text(
                0.5,
                -0.18,
                "For each query: bars are " + " / ".join(breakdown_series_order) + " in order.",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=14,
            )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=5, frameon=False, fontsize=13)
            fig.savefig(graph_dir / f"{prefix}_query_stage_breakdown.png", dpi=180)
            plt.close(fig)

        # Paging cache-management diagnostics by position.
        paging_rows = [by_key.get(("paging", pos), {}) for pos in positions]
        fig, ax1 = plt.subplots(figsize=(max(13, len(positions) * 0.62), 5.8), constrained_layout=True)
        cached = [to_float(row.get("fixed_page_cached_gb")) for row in paging_rows]
        cache_ms = [to_float(row.get("cache_management_ms")) for row in paging_rows]
        ax1.bar(x, cached, color="#59a14f", width=0.62, label="cached useful GB")
        ax1.set_xticks(x)
        ax1.set_xticklabels(query_labels, rotation=35, ha="right", fontsize=11)
        ax1.set_ylabel("Paging cache hit/useful GB", fontsize=12)
        ax2 = ax1.twinx()
        ax2.plot(x, cache_ms, color="#8d6e63", marker="o", linewidth=1.8, label="cache mgmt ms")
        ax2.set_ylabel("Cache management work (ms)", fontsize=12)
        ax1.set_title("Paging cache reuse and cache-management cost", fontsize=15)
        ax1.grid(axis="y", alpha=0.25)
        lines, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels1 + labels2, loc="upper left", fontsize=10)
        fig.savefig(graph_dir / f"{prefix}_paging_cache_management.png", dpi=180)
        plt.close(fig)


def summarize_run(args: argparse.Namespace, conditions: list[str]) -> None:
    all_rows: list[dict[str, object]] = []
    for condition in conditions:
        all_rows.extend(collect_condition(args.output.resolve(), condition, args.summary_mode))
    summary_dir = args.output / "summary"
    graph_dir = args.graph_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    query_summary = aggregate_query_rows(all_rows)
    workload_execution, workload_summary = aggregate_workload_rows(all_rows)
    write_csv(summary_dir / "query_execution_breakdown.csv", all_rows, EXECUTION_FIELDS)

    query_summary_fields: list[str] = ["series", "position", "query", "samples"]
    for field in SUMMARY_NUMERIC_FIELDS:
        query_summary_fields.extend([field, f"{field}_std"])
    write_csv(summary_dir / "query_summary_breakdown.csv", query_summary, query_summary_fields)

    workload_exec_fields = ["condition", "execution", "series", "included_in_summary", "query_count", *SUMMARY_NUMERIC_FIELDS]
    write_csv(summary_dir / "workload_execution_breakdown.csv", workload_execution, workload_exec_fields)
    workload_summary_fields: list[str] = ["series", "samples", "query_count"]
    for field in SUMMARY_NUMERIC_FIELDS:
        workload_summary_fields.extend([field, f"{field}_std"])
    write_csv(summary_dir / "workload_summary_breakdown.csv", workload_summary, workload_summary_fields)
    write_comparison(graph_dir, args.graph_prefix, workload_summary)
    write_query_comparison(graph_dir, args.graph_prefix, query_summary)
    # Also mirror compact summary CSVs into experiment/graph for quick plotting/notebook access.
    write_csv(graph_dir / f"{args.graph_prefix}_query_summary_breakdown.csv", query_summary, query_summary_fields)
    write_csv(graph_dir / f"{args.graph_prefix}_workload_summary_breakdown.csv", workload_summary, workload_summary_fields)
    plot_outputs(graph_dir, args.graph_prefix, query_summary, workload_summary)


def run_parent(args: argparse.Namespace) -> int:
    conditions = parse_csv_list(args.conditions)
    bad = [c for c in conditions if c not in {"cold_hot", "paging"}]
    if bad:
        raise SystemExit(f"bad condition(s): {','.join(bad)}")
    args.output = args.output.resolve()
    args.graph_dir = args.graph_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary").mkdir(parents=True, exist_ok=True)
    config_path = args.output / "configs" / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config_path, args)
    metadata = {
        "queries": parse_query_list(args.queries),
        "query_labels": [f"q{q}" for q in parse_query_list(args.queries)],
        "conditions": conditions,
        "executions": args.executions,
        "repeat_layout": args.repeat_layout,
        "input": str(args.input),
        "devices": args.devices,
        "num_gpus": args.num_gpus,
        "gpu_usage_limit": args.gpu_usage_limit,
        "page_cache_bytes_per_gpu": args.page_cache_bytes_per_gpu,
        "fixed_width_page_bytes": args.fixed_width_page_bytes,
        "summary_mode": args.summary_mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    series_rule = (
        "Series rule: baseline=cold_hot executions 1-3, paging=paging executions 1-3.\n"
        if args.summary_mode == "avg_all"
        else "Series rule: cold=cold_hot execution 1, hot=cold_hot executions 2-3, "
             "paging=paging executions 2-3.\n"
    )
    (args.output / "README.md").write_text(
        "# Fixed-column sequence workload 3x\n\n"
        f"Queries: `{','.join(metadata['query_labels'])}`\n\n"
        f"Conditions: `{','.join(conditions)}`\n\n"
        f"Repeat layout: `{args.repeat_layout}`\n\n"
        f"Fixed-width page bytes: `{args.fixed_width_page_bytes or 'default'}`\n\n"
        + series_rule
    )

    if not args.skip_build:
        print("[BUILD] pixi run make", flush=True)
        code = subprocess.run(["pixi", "run", "make"], cwd=REPO_ROOT).returncode
        if code != 0:
            return code

    failures: list[dict[str, object]] = []
    for condition in conditions:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case-condition", condition,
            "--input", str(args.input),
            "--output", str(args.output),
            "--graph-dir", str(args.graph_dir),
            "--graph-prefix", args.graph_prefix,
            "--queries", args.queries,
            "--executions", str(args.executions),
            "--devices", args.devices,
            "--num-gpus", str(args.num_gpus),
            "--gpu-usage-limit", args.gpu_usage_limit,
            "--host-capacity", args.host_capacity,
            "--reservation-limit-fraction", args.reservation_limit_fraction,
            "--page-cache-bytes-per-gpu", args.page_cache_bytes_per_gpu,
            "--page-cache-workspace-reserve-bytes-per-gpu", args.page_cache_workspace_reserve_bytes_per_gpu,
            "--page-cache-min-free-bytes-per-gpu", args.page_cache_min_free_bytes_per_gpu,
            "--demand-max-bytes-per-split", args.demand_max_bytes_per_split,
            "--fixed-width-page-bytes", args.fixed_width_page_bytes,
            "--provider-coalesce-pages", args.provider_coalesce_pages,
            "--fixed-page-hybrid-provider", args.fixed_page_hybrid_provider,
            "--repeat-layout", args.repeat_layout,
            "--summary-mode", args.summary_mode,
            "--log-level", args.log_level,
            "--config", str(config_path),
        ]
        print(f"[RUN] {condition}", flush=True)
        try:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, timeout=args.condition_timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
        if code != 0:
            print(f"[WARN] condition {condition} exited with {code}; parsing partial results", flush=True)
            failures.append({"condition": condition, "returncode": code})
        summarize_run(args, conditions)
    write_csv(args.output / "summary" / "failed_conditions.csv", failures, ["condition", "returncode"])
    summarize_run(args, conditions)
    print(f"==> run root: {args.output}", flush=True)
    print(f"==> summary:  {args.output / 'summary' / 'workload_summary_breakdown.csv'}", flush=True)
    print(f"==> graph:    {args.graph_dir / (args.graph_prefix + '_workload_stage_breakdown.png')}", flush=True)
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--graph-prefix", default="fixedcol_seq22_3x")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--executions", type=int, default=3)
    parser.add_argument("--devices", default="2,3")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-usage-limit", default="7GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument("--page-cache-bytes-per-gpu", default="7GB")
    parser.add_argument("--page-cache-workspace-reserve-bytes-per-gpu", default="")
    parser.add_argument("--page-cache-min-free-bytes-per-gpu", default="4096MB")
    parser.add_argument("--demand-max-bytes-per-split", default="")
    parser.add_argument("--fixed-width-page-bytes", default="")
    parser.add_argument("--provider-coalesce-pages", default="")
    parser.add_argument("--fixed-page-hybrid-provider", choices=["0", "1"], default="1")
    parser.add_argument("--repeat-layout", choices=["workload", "query"], default="workload")
    parser.add_argument("--condition-timeout-s", type=int, default=2400)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--summary-mode", choices=["cold_hot", "avg_all"], default="cold_hot")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--config", default="")
    parser.add_argument("--case-condition", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.case_condition:
        return run_condition_child(args)
    if args.summarize_only:
        args.output = args.output.resolve()
        args.graph_dir = args.graph_dir.resolve()
        summarize_run(args, parse_csv_list(args.conditions))
        print(f"==> summary:  {args.output / 'summary' / 'workload_summary_breakdown.csv'}", flush=True)
        print(f"==> graph:    {args.graph_dir / (args.graph_prefix + '_workload_stage_breakdown.png')}", flush=True)
        return 0
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
