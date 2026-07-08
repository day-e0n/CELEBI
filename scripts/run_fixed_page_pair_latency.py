#!/usr/bin/env python3
# wdy start
"""Run fixed-page paging latency experiments for ordered TPC-H query pairs.

The experiment isolates every (condition, Qi -> Qj, repeat) in a separate
process so the measured reuse is pair-local:

- baseline: normal Sirius execution, no fixed-page reuse.
- paging_filter_aware: no-pin automatic fixed-page cache population on scan miss,
  then reuse on later scans in the same process.
- pinned_*: manual pin_table upper-bound variants, kept separate so they are not
  confused with automatic paging.

Outputs:
- summary/query_latency.csv
- summary/pair_latency.csv
- summary/pair_latency_summary.csv
- summary/baseline_vs_paging_second_query.csv
- summary/*_matrix.csv and PNG heatmaps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "experiment" / "fixed_page_pair_latency"
DEFAULT_QUERIES = "3,5,7,8,9,10,18,21"
DEFAULT_CONDITIONS = "baseline,paging_filter_aware"

STAGE_TIMING_FIELDS = [
    "cache_column_view_ms",
    "cache_column_materialize_ms",
    "splice_view_build_ms",
    "splice_materialize_ms",
    "filtered_splice_materialize_ms",
    "filtered_splice_materialize_single_mask_ms",
    "partial_filter_cached_mask_ms",
    "partial_filter_mask_splice_ms",
    "single_mask_filter_apply_ms",
    "cached_filter_select_ms",
    "post_filter_select_ms",
    "inline_assembly_ms",
    "post_filter_project_assembly_ms",
    "assembly_ms",
    "fixed_page_extra_stage_ms",
]

# wdy start
PAGE_PRUNING_FIELDS = [
    "page_pruning_matched_cols",
    "page_pruning_pages",
    "page_pruning_all_fail_pages",
    "page_pruning_all_pass_pages",
    "page_pruning_partial_pages",
    "page_pruning_unknown_pages",
    "page_pruning_skip_reuse_count",
    "page_pruning_keep_reuse_count",
]
# wdy end

FIXED_PAGE_EXTRA_STAGES = {
    "splice_view_build": "splice_view_build_ms",
    "splice_materialize": "splice_materialize_ms",
    "filtered_splice_materialize": "filtered_splice_materialize_ms",
    "filtered_splice_materialize_single_mask": "filtered_splice_materialize_single_mask_ms",
    "partial_filter_cached_mask": "partial_filter_cached_mask_ms",
    "partial_filter_mask_splice": "partial_filter_mask_splice_ms",
}

FIXED_PAGE_SUB_STAGES = {
    "single_mask_filter_apply": "single_mask_filter_apply_ms",
    "cache_column_view": "cache_column_view_ms",
    "cache_column_materialize": "cache_column_materialize_ms",
}

OTHER_STAGE_FIELDS = {
    "cached_filter_select": "cached_filter_select_ms",
    "post_filter_select": "post_filter_select_ms",
    "inline_assembly": "inline_assembly_ms",
    "post_filter_project_assembly": "post_filter_project_assembly_ms",
}


sys.path.insert(0, str(TPCH_DIR))
from performance_test import _execute_multi, open_connection, time_query  # noqa: E402
from tpch_pin_columns import QUERY_COLUMNS, detect_pin_glob  # noqa: E402


KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
PAIR_NAME_RE = re.compile(r"q(\d+)_then_q(\d+)_iter(\d+)")
LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")

FIXED_WIDTH_COLUMNS: dict[str, set[str]] = {
    "customer": {"c_custkey", "c_nationkey", "c_acctbal"},
    "lineitem": {
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
    },
    "nation": {"n_nationkey", "n_regionkey"},
    "orders": {"o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"},
    "part": {"p_partkey", "p_size", "p_retailprice"},
    "partsupp": {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"},
    "region": {"r_regionkey"},
    "supplier": {"s_suppkey", "s_nationkey", "s_acctbal"},
}

# Keep the optional budget policy conservative: use normalized join-key style
# fixed-width columns, but avoid extra fact-table measure/date columns and large
# partsupp keys by default. The stable baseline+paging comparison should use
# paging_key_only.
BUDGETED_FIXED_WIDTH_COLUMNS: dict[str, set[str]] = {
    "customer": {"c_custkey", "c_nationkey"},
    "lineitem": {"l_orderkey"},
    "nation": {"n_nationkey", "n_regionkey"},
    "orders": {"o_orderkey", "o_custkey"},
    "part": {"p_partkey"},
    "region": {"r_regionkey"},
    "supplier": {"s_suppkey", "s_nationkey"},
}

# wdy start
# Filter-aware paging keeps the order-key reuse columns, then adds common
# fixed-width range-filter columns so SIRIUS_FIXED_PAGE_PRUNING has page stats
# for predicates such as o_orderdate/l_shipdate ranges without pinning every
# fixed-width column in the workload.
FILTER_AWARE_FIXED_WIDTH_COLUMNS: dict[str, set[str]] = {
    "lineitem": {
        "l_orderkey",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_quantity",
        "l_discount",
    },
    "orders": {"o_orderkey", "o_orderdate"},
    "part": {"p_size"},
}
# wdy end


@dataclass(frozen=True)
class RunSpec:
    condition: str
    qi: int
    qj: int
    repeat: int

    @property
    def name(self) -> str:
        return f"q{self.qi}_then_q{self.qj}_iter{self.repeat}"


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_query_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def parse_pairs(raw: str, queries: list[int], include_diagonal: bool) -> list[tuple[int, int]]:
    query_set = set(queries)
    if not raw:
        return [(qi, qj) for qi in queries for qj in queries if include_diagonal or qi != qj]
    pairs: list[tuple[int, int]] = []
    for item in parse_csv_list(raw):
        left, right = item.split(":", 1)
        qi, qj = int(left), int(right)
        if qi not in query_set or qj not in query_set:
            raise SystemExit(f"bad pair {item}: pair must use --queries set")
        if qi == qj and not include_diagonal:
            raise SystemExit(f"bad pair {item}: use --include-diagonal to allow it")
        pairs.append((qi, qj))
    return pairs


def write_config(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
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
            ]
        )
    )


def union_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for qnum in queries:
        for table, cols in QUERY_COLUMNS[qnum].items():
            out.setdefault(table, set()).update(cols)
    return {table: sorted(cols) for table, cols in sorted(out.items())}


def fixed_width_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for table, cols in union_columns_for_queries(queries).items():
        fixed = FIXED_WIDTH_COLUMNS.get(table, set())
        selected = sorted(col for col in cols if col in fixed)
        if selected:
            out[table] = selected
    return out


def budgeted_fixed_width_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for table, cols in union_columns_for_queries(queries).items():
        fixed = FIXED_WIDTH_COLUMNS.get(table, set())
        allowed = BUDGETED_FIXED_WIDTH_COLUMNS.get(table, fixed)
        selected = sorted(col for col in cols if col in fixed and col in allowed)
        if selected:
            out[table] = selected
    return out


def key_only_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    union = union_columns_for_queries(queries)
    out: dict[str, list[str]] = {}
    if "l_orderkey" in union.get("lineitem", []):
        out["lineitem"] = ["l_orderkey"]
    if "o_orderkey" in union.get("orders", []):
        out["orders"] = ["o_orderkey"]
    return out


# wdy start
def filter_aware_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for table, cols in union_columns_for_queries(queries).items():
        allowed = FILTER_AWARE_FIXED_WIDTH_COLUMNS.get(table, set())
        selected = sorted(col for col in cols if col in allowed)
        if selected:
            out[table] = selected
    return out


# wdy end
def pin_sql_for_condition(
    condition: str,
    parquet_dir: str,
    queries: list[int],
    pin_rows: int | None,
) -> tuple[str, list[str]]:
    if condition == "baseline" or condition.startswith("paging_"):
        return "", []
    if condition == "pinned_key_only":
        cols_by_table = key_only_columns_for_queries(queries)
    elif condition == "pinned_budget":
        cols_by_table = budgeted_fixed_width_columns_for_queries(queries)
    elif condition == "pinned_filter_aware":
        cols_by_table = filter_aware_columns_for_queries(queries)
    elif condition == "pinned_full_fixed":
        cols_by_table = fixed_width_columns_for_queries(queries)
    else:
        raise ValueError(f"unknown condition: {condition}")

    n_rows_clause = f", n_rows={pin_rows}" if pin_rows is not None else ""
    lines: list[str] = []
    tables: list[str] = []
    for table, cols in cols_by_table.items():
        if not cols:
            continue
        path = detect_pin_glob(parquet_dir, table)
        col_literals = ",".join(f"'{col}'" for col in cols)
        lines.append(
            f"CALL pin_table('{path}', tier='gpu', name='{table}', "
            f"cols=[{col_literals}]{n_rows_clause});"
        )
        tables.append(table)
    return "\n".join(lines) + ("\n" if lines else ""), tables


def unpin_sql(tables: list[str]) -> str:
    return "\n".join(f"CALL unpin_table('{table}');" for table in tables) + ("\n" if tables else "")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def runtime_csv_complete(path: Path) -> bool:
    if not path.exists():
        return False
    return len(read_csv(path)) >= 2


def column_bytes_uncompressed_total(value: str | None) -> int:
    if not value:
        return 0
    total = 0
    for item in value.split(','):
        parts = item.rsplit(':', 2)
        if len(parts) != 3:
            continue
        try:
            total += int(parts[2])
        except ValueError:
            continue
    return total


def parse_kv(line: str) -> dict[str, str]:
    return {k: v for k, v in KV_RE.findall(line)}


def parse_log_timestamp_ms(line: str) -> float | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000.0
    except ValueError:
        return None


def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def stdev(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) >= 2 else ""


def union_interval_ms(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(current_end - current_start, 0.0)
            current_start, current_end = start, end
    total += max(current_end - current_start, 0.0)
    return total


def timed_query_segments(bench: Path) -> list[tuple[str, list[str]]]:
    runtimes = [row.get("query", "") for row in read_csv(bench / "csv" / "runtimes.csv")]
    logs = sorted((bench / "log_dir").glob("*.log"))
    if not logs:
        return []
    lines = logs[-1].read_text(errors="replace").splitlines()
    begins: list[int] = []
    for idx, line in enumerate(lines):
        marker = "QueryBegin: SQL:"
        if marker not in line:
            continue
        sql = line.split(marker, 1)[1].strip().lower()
        if sql.startswith("set ") or sql.startswith("create view"):
            continue
        if sql.startswith("call pin_table") or sql.startswith("call unpin_table"):
            continue
        begins.append(idx)
    segments: list[tuple[str, list[str]]] = []
    for pos, start in enumerate(begins[: len(runtimes)]):
        end = begins[pos + 1] if pos + 1 < len(begins) else len(lines)
        query = runtimes[pos] if pos < len(runtimes) else "unknown"
        segments.append((query, lines[start:end]))
    return segments


def summarize_one_case(condition: str, bench: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    match = PAIR_NAME_RE.match(bench.name)
    if not match:
        return [], {}
    qi, qj, repeat = map(int, match.groups())
    runtime_rows = [row for row in read_csv(bench / "csv" / "runtimes.csv") if row.get("engine") == "sirius"]
    segments = timed_query_segments(bench)

    scan_work_by_query: dict[str, float] = defaultdict(float)
    scan_intervals_by_query: dict[str, list[tuple[float, float]]] = defaultdict(list)
    scan_uncompressed_by_query: dict[str, int] = defaultdict(int)
    fixed_cached_by_query: dict[str, int] = defaultdict(int)
    fixed_split_count_by_query: dict[str, int] = defaultdict(int)
    fixed_materialize_count_by_query: dict[str, int] = defaultdict(int)
    fixed_materialize_view_count_by_query: dict[str, int] = defaultdict(int)
    fixed_fallback_count_by_query: dict[str, int] = defaultdict(int)
    stage_ms_by_query: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # wdy start
    page_pruning_by_query: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # wdy end

    for query, lines in segments:
        for line in lines:
            if "[scan-audit] parquet_materialize" in line:
                fields = parse_kv(line)
                duration_ms = to_int(fields.get("duration_us")) / 1000.0
                scan_work_by_query[query] += duration_ms
                scan_uncompressed_by_query[query] += column_bytes_uncompressed_total(fields.get("column_bytes")) or to_int(fields.get("uncompressed_bytes"))
                end_ms = parse_log_timestamp_ms(line)
                if end_ms is not None:
                    scan_intervals_by_query[query].append((end_ms - duration_ms, end_ms))
            elif "[fixed-page-cache] hybrid_reuse_split" in line:
                fixed_cached_by_query[query] += to_int((lambda fields: fields.get("useful_bytes") or fields.get("cached_bytes"))(parse_kv(line)))
                fixed_split_count_by_query[query] += 1
            elif "[fixed-page-cache] hybrid_reuse_materialize_view" in line:
                fixed_materialize_view_count_by_query[query] += 1
            elif "[fixed-page-cache] hybrid_reuse_materialize_filtered" in line:
                fixed_materialize_count_by_query[query] += 1
            elif "[fixed-page-cache] hybrid_reuse_materialize " in line:
                fixed_materialize_count_by_query[query] += 1
            elif "[fixed-page-cache] hybrid_reuse_row_group_fallback" in line:
                fixed_fallback_count_by_query[query] += 1
            # wdy start
            elif "[fixed-page-cache] page_pruning_decision" in line:
                fields = parse_kv(line)
                pruning = page_pruning_by_query[query]
                pruning["page_pruning_matched_cols"] += to_int(fields.get("matched_cols"))
                pruning["page_pruning_pages"] += to_int(fields.get("pages"))
                pruning["page_pruning_all_fail_pages"] += to_int(fields.get("all_fail"))
                pruning["page_pruning_all_pass_pages"] += to_int(fields.get("all_pass"))
                pruning["page_pruning_partial_pages"] += to_int(fields.get("partial"))
                pruning["page_pruning_unknown_pages"] += to_int(fields.get("unknown"))
                action = fields.get("action", "")
                if action == "skip_reuse_preserve_pushdown":
                    pruning["page_pruning_skip_reuse_count"] += 1
                elif action == "keep_reuse":
                    pruning["page_pruning_keep_reuse_count"] += 1
            # wdy end
            elif "[fixed-page-cache] stage_timing" in line:
                fields = parse_kv(line)
                stage = fields.get("stage", "")
                field = FIXED_PAGE_EXTRA_STAGES.get(stage) or FIXED_PAGE_SUB_STAGES.get(stage) or OTHER_STAGE_FIELDS.get(stage)
                if field:
                    stage_ms_by_query[query][field] += to_int(fields.get("duration_us")) / 1000.0

    query_rows: list[dict[str, object]] = []
    for position, row in enumerate(runtime_rows[:2], 1):
        query = row.get("query", "")
        total_ms = to_float(row.get("runtime_s")) * 1000.0
        scan_work_ms = scan_work_by_query.get(query, 0.0)
        scan_wall_ms = union_interval_ms(scan_intervals_by_query.get(query, []))
        load_ms = min(scan_wall_ms, total_ms)
        stage_values = {field: stage_ms_by_query[query].get(field, 0.0) for field in STAGE_TIMING_FIELDS}
        stage_values["assembly_ms"] = (
            stage_values["inline_assembly_ms"] + stage_values["post_filter_project_assembly_ms"]
        )
        stage_values["fixed_page_extra_stage_ms"] = sum(
            stage_values[field] for field in FIXED_PAGE_EXTRA_STAGES.values()
        )
        # wdy start
        pruning_values = {field: page_pruning_by_query[query].get(field, 0) for field in PAGE_PRUNING_FIELDS}
        # wdy end
        query_rows.append(
            {
                "condition": condition,
                "previous_query": f"q{qi}",
                "second_query": f"q{qj}",
                "repeat": repeat,
                "query_position": position,
                "query": query,
                "total_ms": total_ms,
                "load_ms": load_ms,
                "scan_materialize_wall_ms": scan_wall_ms,
                "scan_materialize_work_ms": scan_work_ms,
                "scan_uncompressed_gb": scan_uncompressed_by_query.get(query, 0) / 1e9,
                "computation_ms": max(total_ms - load_ms, 0.0),
                "fixed_page_cached_gb": fixed_cached_by_query.get(query, 0) / 1e9,
                "fixed_page_reuse_split_count": fixed_split_count_by_query.get(query, 0),
                "fixed_page_materialize_count": fixed_materialize_count_by_query.get(query, 0),
                "fixed_page_materialize_view_count": fixed_materialize_view_count_by_query.get(query, 0),
                "fixed_page_fallback_count": fixed_fallback_count_by_query.get(query, 0),
                **pruning_values,
                **stage_values,
                "benchmark_dir": str(bench),
            }
        )

    first = query_rows[0] if len(query_rows) > 0 else {}
    second = query_rows[1] if len(query_rows) > 1 else {}
    pair_row = {
        "condition": condition,
        "previous_query": f"q{qi}",
        "second_query": f"q{qj}",
        "repeat": repeat,
        "first_total_ms": first.get("total_ms", ""),
        "first_load_ms": first.get("load_ms", ""),
        "first_scan_materialize_wall_ms": first.get("scan_materialize_wall_ms", ""),
        "first_scan_materialize_work_ms": first.get("scan_materialize_work_ms", ""),
        "first_scan_uncompressed_gb": first.get("scan_uncompressed_gb", ""),
        "first_computation_ms": first.get("computation_ms", ""),
        "first_fixed_page_cached_gb": first.get("fixed_page_cached_gb", ""),
        **{f"first_{field}": first.get(field, "") for field in PAGE_PRUNING_FIELDS},
        **{f"first_{field}": first.get(field, "") for field in STAGE_TIMING_FIELDS},
        "second_total_ms": second.get("total_ms", ""),
        "second_load_ms": second.get("load_ms", ""),
        "second_scan_materialize_wall_ms": second.get("scan_materialize_wall_ms", ""),
        "second_scan_materialize_work_ms": second.get("scan_materialize_work_ms", ""),
        "second_scan_uncompressed_gb": second.get("scan_uncompressed_gb", ""),
        "second_computation_ms": second.get("computation_ms", ""),
        "second_fixed_page_cached_gb": second.get("fixed_page_cached_gb", ""),
        "second_fixed_page_reuse_split_count": second.get("fixed_page_reuse_split_count", ""),
        "second_fixed_page_materialize_count": second.get("fixed_page_materialize_count", ""),
        "second_fixed_page_materialize_view_count": second.get("fixed_page_materialize_view_count", ""),
        "second_fixed_page_fallback_count": second.get("fixed_page_fallback_count", ""),
        **{f"second_{field}": second.get(field, "") for field in PAGE_PRUNING_FIELDS},
        **{f"second_{field}": second.get(field, "") for field in STAGE_TIMING_FIELDS},
        "pair_total_ms": to_float(first.get("total_ms", "")) + to_float(second.get("total_ms", "")),
        "pair_load_ms": to_float(first.get("load_ms", "")) + to_float(second.get("load_ms", "")),
        "pair_computation_ms": to_float(first.get("computation_ms", "")) + to_float(second.get("computation_ms", "")),
        **{f"pair_{field}": to_float(first.get(field, "")) + to_float(second.get(field, "")) for field in PAGE_PRUNING_FIELDS},
        **{f"pair_{field}": to_float(first.get(field, "")) + to_float(second.get(field, "")) for field in STAGE_TIMING_FIELDS},
        "benchmark_dir": str(bench),
    }
    return query_rows, pair_row


QUERY_FIELDS = [
    "condition",
    "previous_query",
    "second_query",
    "repeat",
    "query_position",
    "query",
    "total_ms",
    "load_ms",
    "scan_materialize_wall_ms",
    "scan_materialize_work_ms",
    "scan_uncompressed_gb",
    "computation_ms",
    "fixed_page_cached_gb",
    "fixed_page_reuse_split_count",
    "fixed_page_materialize_count",
    "fixed_page_materialize_view_count",
    "fixed_page_fallback_count",
    *PAGE_PRUNING_FIELDS,
    *STAGE_TIMING_FIELDS,
    "benchmark_dir",
]

PAIR_FIELDS = [
    "condition",
    "previous_query",
    "second_query",
    "repeat",
    "first_total_ms",
    "first_load_ms",
    "first_scan_materialize_wall_ms",
    "first_scan_materialize_work_ms",
    "first_scan_uncompressed_gb",
    "first_computation_ms",
    "first_fixed_page_cached_gb",
    *[f"first_{field}" for field in PAGE_PRUNING_FIELDS],
    *[f"first_{field}" for field in STAGE_TIMING_FIELDS],
    "second_total_ms",
    "second_load_ms",
    "second_scan_materialize_wall_ms",
    "second_scan_materialize_work_ms",
    "second_scan_uncompressed_gb",
    "second_computation_ms",
    "second_fixed_page_cached_gb",
    "second_fixed_page_reuse_split_count",
    "second_fixed_page_materialize_count",
    "second_fixed_page_materialize_view_count",
    "second_fixed_page_fallback_count",
    *[f"second_{field}" for field in PAGE_PRUNING_FIELDS],
    *[f"second_{field}" for field in STAGE_TIMING_FIELDS],
    "pair_total_ms",
    "pair_load_ms",
    "pair_computation_ms",
    *[f"pair_{field}" for field in PAGE_PRUNING_FIELDS],
    *[f"pair_{field}" for field in STAGE_TIMING_FIELDS],
    "benchmark_dir",
]


def summarize_results(run_root: Path, conditions: list[str], queries: list[int], no_plots: bool) -> None:
    all_query_rows: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    for condition in conditions:
        for runtime_csv in sorted((run_root / condition / "pairs").glob("q*_then_q*_iter*/csv/runtimes.csv")):
            bench = runtime_csv.parents[1]
            if not runtime_csv_complete(runtime_csv):
                continue
            query_rows, pair_row = summarize_one_case(condition, bench)
            all_query_rows.extend(query_rows)
            if pair_row:
                all_pair_rows.append(pair_row)

    summary_dir = run_root / "summary"
    write_csv(summary_dir / "query_latency.csv", all_query_rows, QUERY_FIELDS)
    write_csv(summary_dir / "pair_latency.csv", all_pair_rows, PAIR_FIELDS)
    write_pair_latency_summary(summary_dir / "pair_latency_summary.csv", all_pair_rows)
    write_baseline_vs_paging(summary_dir, all_pair_rows, queries)
    if not no_plots:
        plot_outputs(summary_dir)


def vals(group: list[dict[str, object]], field: str) -> list[float]:
    return [to_float(row.get(field, "")) for row in group if row.get(field, "") != ""]


def write_pair_latency_summary(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["previous_query"]), str(row["second_query"]))].append(row)
    fields = [
        "condition",
        "previous_query",
        "second_query",
        "repeats_completed",
        "second_total_ms_mean",
        "second_total_ms_std",
        "second_load_ms_mean",
        "second_load_ms_std",
        "second_scan_materialize_wall_ms_mean",
        "second_scan_materialize_work_ms_mean",
        "second_scan_uncompressed_gb_mean",
        "second_computation_ms_mean",
        "second_computation_ms_std",
        "second_fixed_page_cached_gb_mean",
        "second_fixed_page_reuse_split_count_mean",
        "second_fixed_page_materialize_view_count_mean",
        "second_fixed_page_fallback_count_mean",
        *[f"second_{field}_mean" for field in PAGE_PRUNING_FIELDS],
        *[f"second_{field}_mean" for field in STAGE_TIMING_FIELDS],
        "pair_total_ms_mean",
        "pair_load_ms_mean",
        "pair_computation_ms_mean",
    ]
    out: list[dict[str, object]] = []
    for (condition, qi, qj), group in sorted(groups.items()):
        out.append(
            {
                "condition": condition,
                "previous_query": qi,
                "second_query": qj,
                "repeats_completed": len(group),
                "second_total_ms_mean": mean(vals(group, "second_total_ms")),
                "second_total_ms_std": stdev(vals(group, "second_total_ms")),
                "second_load_ms_mean": mean(vals(group, "second_load_ms")),
                "second_load_ms_std": stdev(vals(group, "second_load_ms")),
                "second_scan_materialize_wall_ms_mean": mean(vals(group, "second_scan_materialize_wall_ms")),
                "second_scan_materialize_work_ms_mean": mean(vals(group, "second_scan_materialize_work_ms")),
                "second_scan_uncompressed_gb_mean": mean(vals(group, "second_scan_uncompressed_gb")),
                "second_computation_ms_mean": mean(vals(group, "second_computation_ms")),
                "second_computation_ms_std": stdev(vals(group, "second_computation_ms")),
                "second_fixed_page_cached_gb_mean": mean(vals(group, "second_fixed_page_cached_gb")),
                "second_fixed_page_reuse_split_count_mean": mean(vals(group, "second_fixed_page_reuse_split_count")),
                "second_fixed_page_materialize_view_count_mean": mean(vals(group, "second_fixed_page_materialize_view_count")),
                "second_fixed_page_fallback_count_mean": mean(vals(group, "second_fixed_page_fallback_count")),
                **{f"second_{field}_mean": mean(vals(group, f"second_{field}")) for field in PAGE_PRUNING_FIELDS},
                **{f"second_{field}_mean": mean(vals(group, f"second_{field}")) for field in STAGE_TIMING_FIELDS},
                "pair_total_ms_mean": mean(vals(group, "pair_total_ms")),
                "pair_load_ms_mean": mean(vals(group, "pair_load_ms")),
                "pair_computation_ms_mean": mean(vals(group, "pair_computation_ms")),
            }
        )
    write_csv(path, out, fields)


def avg(groups: dict[tuple[str, str, str], list[dict[str, object]]], condition: str, qi: str, qj: str, field: str) -> float | str:
    group = groups.get((condition, qi, qj), [])
    values = vals(group, field)
    return mean(values)


def write_matrix(path: Path, rows: list[dict[str, object]], column: str, labels: list[str]) -> None:
    lookup = {(row["previous_query"], row["second_query"]): row.get(column, "") for row in rows}
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + labels)
        for qi in labels:
            writer.writerow([qi] + [lookup.get((qi, qj), "") for qj in labels])


def write_baseline_vs_paging(summary_dir: Path, rows: list[dict[str, object]], queries: list[int]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["previous_query"]), str(row["second_query"]))].append(row)

    paging_conditions = sorted({str(row["condition"]) for row in rows if str(row["condition"]) != "baseline"})
    labels = [f"q{q}" for q in queries]
    fields = [
        "paging_condition",
        "previous_query",
        "second_query",
        "baseline_repeats",
        "paging_repeats",
        "baseline_second_total_ms",
        "paging_second_total_ms",
        "total_ms_delta",
        "total_ms_speedup",
        "baseline_second_load_ms",
        "paging_second_load_ms",
        "load_ms_delta",
        "load_ms_reduction_ratio",
        "baseline_second_scan_work_ms",
        "paging_second_scan_work_ms",
        "scan_work_ms_delta",
        "scan_work_reduction_ratio",
        "baseline_second_computation_ms",
        "paging_second_computation_ms",
        "computation_ms_delta",
        "baseline_second_fixed_page_extra_stage_ms",
        "paging_second_fixed_page_extra_stage_ms",
        "fixed_page_extra_stage_ms_delta",
        "baseline_second_post_filter_select_ms",
        "paging_second_post_filter_select_ms",
        "post_filter_select_ms_delta",
        "baseline_second_assembly_ms",
        "paging_second_assembly_ms",
        "assembly_ms_delta",
        "paging_fixed_page_cached_gb",
        "paging_fixed_page_reuse_split_count",
        "paging_fixed_page_materialize_view_count",
        "paging_fixed_page_fallback_count",
        "paging_page_pruning_pages",
        "paging_page_pruning_all_fail_pages",
        "paging_page_pruning_partial_pages",
        "paging_page_pruning_skip_reuse_count",
        "paging_page_pruning_keep_reuse_count",
    ]
    all_out: list[dict[str, object]] = []
    for paging in paging_conditions:
        out: list[dict[str, object]] = []
        for qi in labels:
            for qj in labels:
                b_total = avg(groups, "baseline", qi, qj, "second_total_ms")
                p_total = avg(groups, paging, qi, qj, "second_total_ms")
                b_load = avg(groups, "baseline", qi, qj, "second_load_ms")
                p_load = avg(groups, paging, qi, qj, "second_load_ms")
                b_scan = avg(groups, "baseline", qi, qj, "second_scan_materialize_work_ms")
                p_scan = avg(groups, paging, qi, qj, "second_scan_materialize_work_ms")
                b_comp = avg(groups, "baseline", qi, qj, "second_computation_ms")
                p_comp = avg(groups, paging, qi, qj, "second_computation_ms")
                b_extra = avg(groups, "baseline", qi, qj, "second_fixed_page_extra_stage_ms")
                p_extra = avg(groups, paging, qi, qj, "second_fixed_page_extra_stage_ms")
                b_filter = avg(groups, "baseline", qi, qj, "second_post_filter_select_ms")
                p_filter = avg(groups, paging, qi, qj, "second_post_filter_select_ms")
                b_assembly = avg(groups, "baseline", qi, qj, "second_assembly_ms")
                p_assembly = avg(groups, paging, qi, qj, "second_assembly_ms")
                row = {
                    "paging_condition": paging,
                    "previous_query": qi,
                    "second_query": qj,
                    "baseline_repeats": len(groups.get(("baseline", qi, qj), [])),
                    "paging_repeats": len(groups.get((paging, qi, qj), [])),
                    "baseline_second_total_ms": b_total,
                    "paging_second_total_ms": p_total,
                    "total_ms_delta": float(p_total) - float(b_total) if b_total != "" and p_total != "" else "",
                    "total_ms_speedup": float(b_total) / float(p_total) if b_total != "" and p_total != "" and float(p_total) > 0 else "",
                    "baseline_second_load_ms": b_load,
                    "paging_second_load_ms": p_load,
                    "load_ms_delta": float(p_load) - float(b_load) if b_load != "" and p_load != "" else "",
                    "load_ms_reduction_ratio": (float(b_load) - float(p_load)) / float(b_load) if b_load != "" and p_load != "" and float(b_load) > 0 else "",
                    "baseline_second_scan_work_ms": b_scan,
                    "paging_second_scan_work_ms": p_scan,
                    "scan_work_ms_delta": float(p_scan) - float(b_scan) if b_scan != "" and p_scan != "" else "",
                    "scan_work_reduction_ratio": (float(b_scan) - float(p_scan)) / float(b_scan) if b_scan != "" and p_scan != "" and float(b_scan) > 0 else "",
                    "baseline_second_computation_ms": b_comp,
                    "paging_second_computation_ms": p_comp,
                    "computation_ms_delta": float(p_comp) - float(b_comp) if b_comp != "" and p_comp != "" else "",
                    "baseline_second_fixed_page_extra_stage_ms": b_extra,
                    "paging_second_fixed_page_extra_stage_ms": p_extra,
                    "fixed_page_extra_stage_ms_delta": float(p_extra) - float(b_extra) if b_extra != "" and p_extra != "" else "",
                    "baseline_second_post_filter_select_ms": b_filter,
                    "paging_second_post_filter_select_ms": p_filter,
                    "post_filter_select_ms_delta": float(p_filter) - float(b_filter) if b_filter != "" and p_filter != "" else "",
                    "baseline_second_assembly_ms": b_assembly,
                    "paging_second_assembly_ms": p_assembly,
                    "assembly_ms_delta": float(p_assembly) - float(b_assembly) if b_assembly != "" and p_assembly != "" else "",
                    "paging_fixed_page_cached_gb": avg(groups, paging, qi, qj, "second_fixed_page_cached_gb"),
                    "paging_fixed_page_reuse_split_count": avg(groups, paging, qi, qj, "second_fixed_page_reuse_split_count"),
                    "paging_fixed_page_materialize_view_count": avg(groups, paging, qi, qj, "second_fixed_page_materialize_view_count"),
                    "paging_fixed_page_fallback_count": avg(groups, paging, qi, qj, "second_fixed_page_fallback_count"),
                    "paging_page_pruning_pages": avg(groups, paging, qi, qj, "second_page_pruning_pages"),
                    "paging_page_pruning_all_fail_pages": avg(groups, paging, qi, qj, "second_page_pruning_all_fail_pages"),
                    "paging_page_pruning_partial_pages": avg(groups, paging, qi, qj, "second_page_pruning_partial_pages"),
                    "paging_page_pruning_skip_reuse_count": avg(groups, paging, qi, qj, "second_page_pruning_skip_reuse_count"),
                    "paging_page_pruning_keep_reuse_count": avg(groups, paging, qi, qj, "second_page_pruning_keep_reuse_count"),
                }
                out.append(row)
                all_out.append(row)
        write_matrix(summary_dir / f"{paging}_total_speedup_matrix.csv", out, "total_ms_speedup", labels)
        write_matrix(summary_dir / f"{paging}_load_reduction_ratio_matrix.csv", out, "load_ms_reduction_ratio", labels)
        write_matrix(summary_dir / f"{paging}_scan_work_reduction_ratio_matrix.csv", out, "scan_work_reduction_ratio", labels)
        write_matrix(summary_dir / f"{paging}_fixed_page_cached_gb_matrix.csv", out, "paging_fixed_page_cached_gb", labels)
    write_csv(summary_dir / "baseline_vs_paging_second_query.csv", all_out, fields)


def plot_heatmap(csv_path: Path, png_path: Path, title: str, cbar_label: str, cmap: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] plotting skipped: {exc}", flush=True)
        return
    matrix = read_csv(csv_path)
    if not matrix:
        return
    labels = [key for key in matrix[0].keys() if key != "previous_query"]
    values = []
    for row in matrix:
        values.append([math.nan if row.get(label, "") == "" else float(row[label]) for label in labels])
    arr = np.array(values, dtype=float)
    masked = np.ma.masked_invalid(arr)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.75), max(6, len(matrix) * 0.65)))
    im = ax.imshow(masked, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(matrix)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels([row["previous_query"] for row in matrix])
    ax.set_xlabel("Second query")
    ax.set_ylabel("Previous query")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def plot_latency_breakdown(summary_dir: Path, paging: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] latency breakdown plot skipped: {exc}", flush=True)
        return
    path = summary_dir / "baseline_vs_paging_second_query.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["paging_condition"] == paging].copy()
    if df.empty:
        return
    df["speedup_num"] = pd.to_numeric(df["total_ms_speedup"], errors="coerce")
    df = df.dropna(subset=["speedup_num"]).sort_values("speedup_num", ascending=False).head(12)
    if df.empty:
        return
    labels = [f"{a}->{b}" for a, b in zip(df["previous_query"], df["second_query"], strict=False)]
    x = range(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar([i - width / 2 for i in x], df["baseline_second_load_ms"], width, label="baseline load", color="#4c78a8")
    ax.bar(
        [i - width / 2 for i in x],
        df["baseline_second_computation_ms"],
        width,
        bottom=df["baseline_second_load_ms"],
        label="baseline compute",
        color="#9ecae9",
    )
    ax.bar([i + width / 2 for i in x], df["paging_second_load_ms"], width, label=f"{paging} load", color="#f58518")
    ax.bar(
        [i + width / 2 for i in x],
        df["paging_second_computation_ms"],
        width,
        bottom=df["paging_second_load_ms"],
        label=f"{paging} compute",
        color="#ffbf79",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Second-query latency (ms)")
    ax.set_title(f"Top second-query speedups: baseline vs {paging}")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncols=2)
    fig.savefig(summary_dir / f"{paging}_top_speedup_latency_breakdown.png", dpi=180)
    plt.close(fig)


def plot_outputs(summary_dir: Path) -> None:
    for matrix in sorted(summary_dir.glob("*_total_speedup_matrix.csv")):
        paging = matrix.name.removesuffix("_total_speedup_matrix.csv")
        plot_heatmap(matrix, summary_dir / f"{paging}_total_speedup_heatmap.png", f"{paging}: second-query speedup", "baseline / paging", "RdYlGn")
        plot_heatmap(summary_dir / f"{paging}_load_reduction_ratio_matrix.csv", summary_dir / f"{paging}_load_reduction_heatmap.png", f"{paging}: load reduction ratio", "(baseline - paging) / baseline", "YlGn")
        plot_heatmap(summary_dir / f"{paging}_fixed_page_cached_gb_matrix.csv", summary_dir / f"{paging}_fixed_page_cached_gb_heatmap.png", f"{paging}: fixed-page cached GB", "GB", "YlOrRd")
        plot_latency_breakdown(summary_dir, paging)


def make_case_env(args: argparse.Namespace, config_path: Path, condition: str, log_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = args.log_level
    filter_aware = condition in {"paging_filter_aware", "pinned_filter_aware"}
    if condition == "baseline":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "0"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE_SINGLE_MASK"] = "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "0"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "0"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "0"
    else:
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "1"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "1" if condition.startswith("paging_") else "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE_ROUND_ROBIN_CHUNKS"] = "1"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "1"
        env["SIRIUS_PIN_TIER"] = "gpu"
        env["SIRIUS_FIXED_PAGE_VIEW_ALIGNED_SPLITS"] = "1"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE"] = "1" if filter_aware else "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE_SINGLE_MASK"] = "1" if filter_aware else "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "1" if filter_aware else "0"
        env.setdefault("SIRIUS_FIXED_PAGE_OWNED_PAGES", "1")
        env.setdefault("SIRIUS_FIXED_PAGE_DEMAND_LOAD", "1")
    return env


def run_case(args: argparse.Namespace) -> int:
    run_root = Path(args.output).resolve()
    qi, qj = (int(x) for x in args.case_pair.split(":", 1))
    spec = RunSpec(args.case_condition, qi, qj, args.case_repeat)
    bench = run_root / args.case_condition / "pairs" / spec.name
    csv_dir = bench / "csv"
    log_dir = bench / "log_dir"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_case_env(args, Path(args.config).resolve(), args.case_condition, log_dir.resolve())
    os.environ.update(env)
    metadata = {
        "condition": args.case_condition,
        "previous_query": f"q{qi}",
        "second_query": f"q{qj}",
        "repeat": args.case_repeat,
        "input": str(args.input),
        "config": str(args.config),
        "devices": args.devices,
        "pin_rows": args.pin_rows,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (bench / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[CASE] {args.case_condition} {spec.name}", flush=True)
    con = open_connection(str(args.input), gpu_execution=True)
    pinned_tables: list[str] = []
    try:
        pin_sql, pinned_tables = pin_sql_for_condition(args.case_condition, str(args.input), [qi, qj], args.pin_rows)
        if pin_sql:
            (bench / "pin.sql").write_text(pin_sql)
            print(f"[CASE] pinning tables: {','.join(pinned_tables)}", flush=True)
            _execute_multi(con, pin_sql)
        with (csv_dir / "runtimes.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["engine", "query", "iteration", "runtime_s"])
            for qnum in (qi, qj):
                elapsed, _rows = time_query(con, qnum, use_gpu=True)
                writer.writerow(["sirius", f"q{qnum}", 0, f"{elapsed:.6f}"])
                f.flush()
                print(f"[CASE] q{qnum} runtime={elapsed:.4f}s", flush=True)
    finally:
        if pinned_tables:
            try:
                print("[CASE] unpinning", flush=True)
                _execute_multi(con, unpin_sql(pinned_tables))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] unpin failed: {exc}", flush=True)
        con.close()
    return 0



def run_cmd(cmd: list[str], env: dict[str, str], timeout_s: int | None, dry_run: bool) -> int:
    printable = " ".join(cmd)
    print(f"==> {printable}", flush=True)
    if dry_run:
        return 0
    full_cmd = cmd
    if timeout_s is not None:
        full_cmd = ["timeout", "--kill-after=30s", str(timeout_s)] + cmd
    return subprocess.run(full_cmd, cwd=REPO_ROOT, env=env).returncode


def run_orchestrator(args: argparse.Namespace) -> int:
    queries = list(range(1, 23)) if args.all_tpch else parse_query_list(args.queries)
    pairs = parse_pairs(args.pairs, queries, args.include_diagonal)
    conditions = parse_csv_list(args.conditions)
    valid_conditions = {
        "baseline",
        "paging_key_only",
        "paging_budget",
        "paging_filter_aware",
        "paging_full_fixed",
        "pinned_key_only",
        "pinned_budget",
        "pinned_filter_aware",
        "pinned_full_fixed",
    }
    bad = [condition for condition in conditions if condition not in valid_conditions]
    if bad:
        raise SystemExit(f"bad condition(s): {', '.join(bad)}")

    run_root = args.output.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "configs" / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config_path, args)
    (run_root / "README.md").write_text(
        f"""# Fixed-page pair latency experiment

Created: {datetime.now().isoformat(timespec='seconds')}
Input: `{args.input}`
Queries: `{','.join(f'q{q}' for q in queries)}`
Pairs: `{len(pairs)}`
Repeats: `{args.repeats}`
Conditions: `{','.join(conditions)}`
Devices: `{args.devices}`

Each condition/query-pair/repeat is isolated in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_second_query.csv`.
"""
    )

    if not args.skip_build:
        code = run_cmd(["pixi", "run", "make", "-j4"], os.environ.copy(), None, args.dry_run)
        if code != 0:
            return code

    failed_fields = ["condition", "previous_query", "second_query", "repeat", "returncode"]
    failed_path = run_root / "failed_cases.csv"
    write_csv(failed_path, [], failed_fields)
    total = len(conditions) * len(pairs) * args.repeats
    current = 0
    for condition in conditions:
        for qi, qj in pairs:
            for repeat in range(args.repeats):
                current += 1
                spec = RunSpec(condition, qi, qj, repeat)
                bench = run_root / condition / "pairs" / spec.name
                csv_path = bench / "csv" / "runtimes.csv"
                if runtime_csv_complete(csv_path):
                    print(f"[SKIP] {condition} {spec.name} ({current}/{total})", flush=True)
                    continue
                print(f"[RUN] {condition} {spec.name} ({current}/{total})", flush=True)
                log_dir = bench / "log_dir"
                env = make_case_env(args, config_path.resolve(), condition, log_dir.resolve())
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case-condition",
                    condition,
                    "--case-pair",
                    f"{qi}:{qj}",
                    "--case-repeat",
                    str(repeat),
                    "--input",
                    str(args.input.resolve()),
                    "--output",
                    str(run_root),
                    "--config",
                    str(config_path.resolve()),
                    "--devices",
                    args.devices,
                    "--num-gpus",
                    str(args.num_gpus),
                    "--gpu-usage-limit",
                    args.gpu_usage_limit,
                    "--host-capacity",
                    args.host_capacity,
                    "--reservation-limit-fraction",
                    args.reservation_limit_fraction,
                    "--log-level",
                    args.log_level,
                ]
                if args.pin_rows is not None:
                    cmd.extend(["--pin-rows", str(args.pin_rows)])
                code = run_cmd(cmd, env, args.pair_timeout, args.dry_run)
                if code != 0 or not runtime_csv_complete(csv_path):
                    with failed_path.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=failed_fields)
                        writer.writerow(
                            {
                                "condition": condition,
                                "previous_query": f"q{qi}",
                                "second_query": f"q{qj}",
                                "repeat": repeat,
                                "returncode": code,
                            }
                        )
                    print(f"[FAILED] {condition} {spec.name} returncode={code}", flush=True)
                summarize_results(run_root, conditions, queries, args.no_plots)

    summarize_results(run_root, conditions, queries, args.no_plots)
    print(f"==> done: {run_root}", flush=True)
    print(f"==> comparison: {run_root / 'summary' / 'baseline_vs_paging_second_query.csv'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--all-tpch", action="store_true")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--include-diagonal", action="store_true")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-usage-limit", default="12GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument("--pin-rows", type=int, default=None)
    parser.add_argument("--pair-timeout", type=int, default=900)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--config", default="")
    parser.add_argument("--case-condition", default="")
    parser.add_argument("--case-pair", default="")
    parser.add_argument("--case-repeat", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.case_condition:
        return run_case(args)
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
