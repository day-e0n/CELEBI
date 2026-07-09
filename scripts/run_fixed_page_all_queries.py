#!/usr/bin/env python3
"""Run all TPC-H queries under cold/hot/pin_table/paging fixed-page conditions.

For each query and condition, this runs a single process with repeated executions:
- cold/hot: baseline Sirius, first execution is cold, executions 2..N are hot.
- pin_table: pin query columns first, execution 1 is discarded, executions 2..N are averaged.
- paging: automatic fixed-page cache populates on execution 1, executions 2..N are averaged.

Outputs stay compact:
- <output>/summary/query_runs.csv
- <output>/summary/query_summary.csv
- experiment/graph/allq_cold_hot_pin_paging_breakdown.{csv,png}
- experiment/graph/allq_cold_hot_pin_paging_speedup.csv
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

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "experiment" / "fixed_page_runs" / "allq_cold_hot_pin_paging"
DEFAULT_GRAPH_DIR = REPO_ROOT / "experiment" / "graph"
DEFAULT_QUERIES = ",".join(str(i) for i in range(1, 23))
DEFAULT_CONDITIONS = "cold_hot,pin_table,paging"

sys.path.insert(0, str(TPCH_DIR))
from performance_test import QUERIES, _execute_multi, open_connection, time_query  # noqa: E402
from tpch_pin_columns import QUERY_COLUMNS, detect_pin_glob  # noqa: E402

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")

STAGE_FIELDS = [
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
    "post_filter_select_ms",
    "inline_assembly_ms",
    "post_filter_project_assembly_ms",
]

FIXED_STAGE_TO_FIELD = {
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

BREAKDOWN_FIELDS = [
    "total_ms",
    "load_ms",
    "scan_materialize_work_ms",
    "scan_uncompressed_gb",
    "compute_ms",
    "fixed_page_cached_gb",
    "fixed_page_reuse_split_count",
    "fixed_page_materialize_count",
    "fixed_page_fallback_count",
    "fixed_page_extra_stage_ms",
    "page_budget_evicted_pages",
    "page_budget_evicted_gb",
    "demand_load_pages",
    "demand_load_gb",
    "demand_load_ms",
    "demand_load_read_calls",
]


def parse_csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_query_list(value: str) -> list[int]:
    return [int(x) for x in parse_csv_list(value)]


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
    return dict(KV_RE.findall(line))


def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def parse_log_timestamp_ms(line: str) -> float | None:
    m = LOG_TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000.0
    except ValueError:
        return None


def union_interval_ms(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += max(ce - cs, 0.0)
            cs, ce = s, e
    total += max(ce - cs, 0.0)
    return total


def mean(vals: list[float]) -> float | str:
    return statistics.mean(vals) if vals else ""


def stdev(vals: list[float]) -> float | str:
    return statistics.stdev(vals) if len(vals) >= 2 else ""


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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


def pin_sql_for_query(parquet_dir: str, qnum: int) -> tuple[str, list[str]]:
    lines: list[str] = []
    tables: list[str] = []
    for table, cols in sorted(QUERY_COLUMNS[qnum].items()):
        if not cols:
            continue
        path = detect_pin_glob(parquet_dir, table)
        col_literals = ",".join(f"'{c}'" for c in sorted(cols))
        lines.append(f"CALL pin_table('{path}', tier='gpu', name='{table}', cols=[{col_literals}]);")
        tables.append(table)
    return "\n".join(lines) + ("\n" if lines else ""), tables


def unpin_sql(tables: list[str]) -> str:
    return "\n".join(f"CALL unpin_table('{t}');" for t in tables) + ("\n" if tables else "")


def make_env(args: argparse.Namespace, condition: str, log_dir: Path, config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = args.log_level
    if condition in {"cold_hot", "pin_table"}:
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_AUTO_CACHE"] = "0"
        env["SIRIUS_FIXED_PAGE_OWNED_PAGES"] = "0"
        env["SIRIUS_FIXED_PAGE_DEMAND_LOAD"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE"] = "0"
        env["SIRIUS_FIXED_PAGE_FILTERED_REUSE_SINGLE_MASK"] = "0"
        env["SIRIUS_FIXED_PAGE_PRUNING"] = "0"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "1" if condition == "pin_table" else "0"
        env["SIRIUS_PIN_TIER"] = "gpu"
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
        if args.page_cache_bytes_per_gpu:
            env["SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU"] = args.page_cache_bytes_per_gpu
        if args.page_cache_workspace_reserve_bytes_per_gpu:
            env["SIRIUS_FIXED_PAGE_CACHE_WORKSPACE_RESERVE_BYTES_PER_GPU"] = (
                args.page_cache_workspace_reserve_bytes_per_gpu
            )
        if args.demand_max_bytes_per_split:
            env["SIRIUS_FIXED_PAGE_DEMAND_MAX_BYTES_PER_SPLIT"] = args.demand_max_bytes_per_split
    else:
        raise ValueError(condition)
    return env


def timed_segments(log_dir: Path, execution_count: int) -> list[list[str]]:
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return [[] for _ in range(execution_count)]
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
    out: list[list[str]] = []
    for pos, start in enumerate(begins[:execution_count]):
        end = begins[pos + 1] if pos + 1 < len(begins) else len(lines)
        out.append(lines[start:end])
    while len(out) < execution_count:
        out.append([])
    return out


def summarize_execution(lines: list[str], total_ms: float) -> dict[str, object]:
    scan_work = 0.0
    scan_unc = 0
    intervals: list[tuple[float, float]] = []
    fixed_cached = 0
    split_count = 0
    mat_count = 0
    fallback_count = 0
    stage = defaultdict(float)
    evicted_pages = 0
    evicted_bytes = 0
    demand_pages = 0
    demand_bytes = 0
    demand_ms = 0.0
    demand_read_calls = 0
    for line in lines:
        if "[scan-audit] parquet_materialize" in line:
            f = parse_kv(line)
            dur = to_int(f.get("duration_us")) / 1000.0
            scan_work += dur
            scan_unc += column_bytes_uncompressed_total(f.get("column_bytes")) or to_int(f.get("uncompressed_bytes"))
            end = parse_log_timestamp_ms(line)
            if end is not None:
                intervals.append((end - dur, end))
        elif "[fixed-page-cache] hybrid_reuse_split" in line:
            f = parse_kv(line)
            fixed_cached += to_int(f.get("useful_bytes") or f.get("cached_bytes"))
            split_count += 1
        elif "[fixed-page-cache] hybrid_reuse_materialize" in line:
            mat_count += 1
        elif "[fixed-page-cache] hybrid_reuse_row_group_fallback" in line:
            fallback_count += 1
        elif "[fixed-page-cache] stage_timing" in line:
            f = parse_kv(line)
            field = FIXED_STAGE_TO_FIELD.get(f.get("stage", ""))
            if field:
                stage[field] += to_int(f.get("duration_us")) / 1000.0
        elif "[fixed-page-cache] page_budget applied" in line:
            f = parse_kv(line)
            evicted_pages += to_int(f.get("evicted_pages"))
            evicted_bytes += to_int(f.get("evicted_bytes"))
        elif "[fixed-page-cache] demand_load loaded_pages" in line:
            f = parse_kv(line)
            demand_pages += to_int(f.get("loaded_pages"))
            demand_bytes += to_int(f.get("loaded_bytes"))
            demand_ms += to_int(f.get("duration_us")) / 1000.0
            demand_read_calls += to_int(f.get("read_calls"))
    fixed_extra = sum(stage[f] for f in [
        "cache_column_materialize_ms",
        "cache_column_view_ms",
        "splice_materialize_ms",
        "splice_view_build_ms",
        "filtered_splice_materialize_ms",
        "filtered_splice_materialize_single_mask_ms",
        "partial_filter_cached_mask_ms",
        "partial_filter_mask_splice_ms",
    ])
    load_ms = min(union_interval_ms(intervals), total_ms)
    return {
        "total_ms": total_ms,
        "load_ms": load_ms,
        "scan_materialize_work_ms": scan_work,
        "scan_uncompressed_gb": scan_unc / 1e9,
        "compute_ms": max(total_ms - load_ms, 0.0),
        "fixed_page_cached_gb": fixed_cached / 1e9,
        "fixed_page_reuse_split_count": split_count,
        "fixed_page_materialize_count": mat_count,
        "fixed_page_fallback_count": fallback_count,
        "fixed_page_extra_stage_ms": fixed_extra,
        "page_budget_evicted_pages": evicted_pages,
        "page_budget_evicted_gb": evicted_bytes / 1e9,
        "demand_load_pages": demand_pages,
        "demand_load_gb": demand_bytes / 1e9,
        "demand_load_ms": demand_ms,
        "demand_load_read_calls": demand_read_calls,
        **{f: stage[f] for f in STAGE_FIELDS},
    }


def run_case(args: argparse.Namespace, qnum: int, condition: str, config_path: Path) -> dict[str, object]:
    case_dir = args.output / condition / f"q{qnum}"
    csv_dir = case_dir / "csv"
    log_dir = case_dir / "log_dir"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args, condition, log_dir.resolve(), config_path.resolve())
    os.environ.update(env)
    metadata = {
        "query": qnum,
        "condition": condition,
        "executions": args.executions,
        "input": str(args.input),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    run_rows: list[dict[str, object]] = []
    con = open_connection(str(args.input), gpu_execution=True)
    pinned: list[str] = []
    try:
        if condition == "pin_table":
            sql, pinned = pin_sql_for_query(str(args.input), qnum)
            if sql:
                _execute_multi(con, sql)
        for execution in range(1, args.executions + 1):
            elapsed, rows = time_query(con, qnum, use_gpu=True)
            out_dir = case_dir / "sirius" / f"exec{execution}" / f"q{qnum}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.txt").write_text("".join(repr(r) + "\n" for r in rows))
            run_rows.append({
                "query": f"q{qnum}",
                "condition": condition,
                "execution": execution,
                "runtime_s": elapsed,
                "total_ms": elapsed * 1000.0,
            })
    finally:
        if pinned:
            try:
                _execute_multi(con, unpin_sql(pinned))
            except Exception:
                pass
        con.close()
    write_csv(csv_dir / "runtimes.csv", run_rows, ["query", "condition", "execution", "runtime_s", "total_ms"])
    segments = timed_segments(log_dir, len(run_rows))
    detailed: list[dict[str, object]] = []
    for row, lines in zip(run_rows, segments):
        detailed.append({**row, **summarize_execution(lines, float(row["total_ms"])), "benchmark_dir": str(case_dir)})
    return {"status": "ok", "rows": detailed}


def run_case_subprocess(args: argparse.Namespace, qnum: int, condition: str, config_path: Path) -> dict[str, object]:
    case_dir = args.output / condition / f"q{qnum}"
    result_path = case_dir / "case_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input", str(args.input),
        "--output", str(args.output),
        "--graph-dir", str(args.graph_dir),
        "--graph-prefix", args.graph_prefix,
        "--queries", str(qnum),
        "--conditions", condition,
        "--executions", str(args.executions),
        "--devices", args.devices,
        "--num-gpus", str(args.num_gpus),
        "--gpu-usage-limit", args.gpu_usage_limit,
        "--host-capacity", args.host_capacity,
        "--reservation-limit-fraction", args.reservation_limit_fraction,
        "--demand-max-bytes-per-split", args.demand_max_bytes_per_split,
        "--log-level", args.log_level,
        "--case-timeout-s", str(args.case_timeout_s),
        "--single-query", str(qnum),
        "--single-condition", condition,
        "--single-result", str(result_path),
    ]
    if args.page_cache_bytes_per_gpu:
        cmd.extend(["--page-cache-bytes-per-gpu", args.page_cache_bytes_per_gpu])
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ.copy(), text=True, timeout=args.case_timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"case_timeout_{args.case_timeout_s}s"}
    if result_path.exists():
        try:
            return json.loads(result_path.read_text())
        except Exception as exc:
            return {"status": "failed", "error": f"failed_to_read_case_result: {exc!r}"}
    return {"status": "failed", "error": f"case_process_exit_{proc.returncode}"}


def aggregate(rows: list[dict[str, object]], executions: int) -> list[dict[str, object]]:
    by = defaultdict(list)
    for row in rows:
        by[(row["query"], row["condition"])].append(row)
    out: list[dict[str, object]] = []
    for (query, condition), vals in sorted(by.items(), key=lambda x: (int(str(x[0][0])[1:]), x[0][1])):
        vals = sorted(vals, key=lambda r: int(r["execution"]))
        if condition == "cold_hot":
            groups = [("cold", vals[:1]), ("hot", vals[1:])]
        elif condition == "pin_table":
            groups = [("pin_table", vals[1:])]
        elif condition == "paging":
            groups = [("paging", vals[1:])]
        else:
            groups = [(condition, vals)]
        for label, group in groups:
            if not group:
                continue
            item = {"query": query, "series": label, "condition": condition, "samples": len(group)}
            for field in BREAKDOWN_FIELDS:
                item[field] = mean([to_float(r.get(field)) for r in group])
                item[field + "_std"] = stdev([to_float(r.get(field)) for r in group])
            out.append(item)
    return out


def write_plots(summary_rows: list[dict[str, object]], graph_dir: Path, prefix: str) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    fields = ["query", "series", "condition", "samples"]
    for f in BREAKDOWN_FIELDS:
        fields.extend([f, f + "_std"])
    write_csv(graph_dir / f"{prefix}_breakdown.csv", summary_rows, fields)

    speed_rows: list[dict[str, object]] = []
    byq = defaultdict(dict)
    for row in summary_rows:
        byq[row["query"]][row["series"]] = row
    for query, series in sorted(byq.items(), key=lambda x: int(str(x[0])[1:])):
        hot = series.get("hot")
        if not hot:
            continue
        for target in ["pin_table", "paging"]:
            row = series.get(target)
            if row and to_float(row.get("total_ms")) > 0:
                speed_rows.append({
                    "query": query,
                    "series": target,
                    "hot_total_ms": hot.get("total_ms"),
                    "target_total_ms": row.get("total_ms"),
                    "total_speedup": to_float(hot.get("total_ms")) / to_float(row.get("total_ms")),
                    "hot_scan_work_ms": hot.get("scan_materialize_work_ms"),
                    "target_scan_work_ms": row.get("scan_materialize_work_ms"),
                    "scan_work_reduction_ratio": (to_float(hot.get("scan_materialize_work_ms")) - to_float(row.get("scan_materialize_work_ms"))) / to_float(hot.get("scan_materialize_work_ms")) if to_float(hot.get("scan_materialize_work_ms")) > 0 else "",
                    "target_fixed_page_extra_stage_ms": row.get("fixed_page_extra_stage_ms"),
                    "target_fixed_page_cached_gb": row.get("fixed_page_cached_gb"),
                })
    write_csv(graph_dir / f"{prefix}_speedup.csv", speed_rows, [
        "query", "series", "hot_total_ms", "target_total_ms", "total_speedup",
        "hot_scan_work_ms", "target_scan_work_ms", "scan_work_reduction_ratio",
        "target_fixed_page_extra_stage_ms", "target_fixed_page_cached_gb",
    ])

    try:
        import matplotlib.pyplot as plt
        labels = sorted({r["query"] for r in summary_rows}, key=lambda q: int(str(q)[1:]))
        series_order = ["cold", "hot", "pin_table", "paging"]
        colors = {"cold": "#9ecae1", "hot": "#4c78a8", "pin_table": "#54a24b", "paging": "#f58518"}
        by_key = {(r["query"], r["series"]): r for r in summary_rows}
        x = range(len(labels))
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5.5))
        for idx, s in enumerate(series_order):
            xs = [i + (idx - 1.5) * width for i in x]
            vals = [to_float(by_key.get((q, s), {}).get("total_ms")) for q in labels]
            ax.bar(xs, vals, width, label=s, color=colors[s])
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("TPC-H all queries: cold/hot/pin_table/paging")
        ax.legend(ncols=4)
        fig.tight_layout()
        fig.savefig(graph_dir / f"{prefix}_latency.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 5.5))
        for idx, s in enumerate(series_order):
            xs = [i + (idx - 1.5) * width for i in x]
            load = [to_float(by_key.get((q, s), {}).get("load_ms")) for q in labels]
            comp = [max(to_float(by_key.get((q, s), {}).get("total_ms")) - l, 0.0) for q, l in zip(labels, load)]
            ax.bar(xs, load, width, color=colors[s], alpha=0.95, label=s if idx == 0 else None)
            ax.bar(xs, comp, width, bottom=load, color=colors[s], alpha=0.35)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Latency breakdown (ms): solid=scan/load, pale=other")
        ax.set_title("TPC-H all queries: scan/load breakdown")
        fig.tight_layout()
        fig.savefig(graph_dir / f"{prefix}_breakdown.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] plot skipped: {exc}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    ap.add_argument("--graph-prefix", default="allq_cold_hot_pin_paging")
    ap.add_argument("--queries", default=DEFAULT_QUERIES)
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--executions", type=int, default=3)
    ap.add_argument("--devices", default="2,3")
    ap.add_argument("--num-gpus", type=int, default=2)
    ap.add_argument("--gpu-usage-limit", default="7GB")
    ap.add_argument("--host-capacity", default="32GB")
    ap.add_argument("--reservation-limit-fraction", default="0.85")
    ap.add_argument("--page-cache-bytes-per-gpu", default="")
    ap.add_argument("--page-cache-workspace-reserve-bytes-per-gpu", default="3072MB")
    ap.add_argument("--demand-max-bytes-per-split", default="3221225472")
    ap.add_argument("--log-level", default="info")
    ap.add_argument("--case-timeout-s", type=int, default=300)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--single-query", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--single-condition", default="", help=argparse.SUPPRESS)
    ap.add_argument("--single-result", type=Path, default=None, help=argparse.SUPPRESS)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    args.output = args.output.resolve()
    config = args.output / "configs" / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config, args)
    if args.single_query and args.single_condition:
        result_path = args.single_result or (args.output / args.single_condition / f"q{args.single_query}" / "case_result.json")
        try:
            result = run_case(args, args.single_query, args.single_condition, config)
        except Exception as exc:
            result = {"status": "failed", "error": repr(exc)}
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    queries = parse_query_list(args.queries)
    conditions = parse_csv_list(args.conditions)
    all_rows: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    total = len(queries) * len(conditions)
    n = 0
    for q in queries:
        if f"q{q}" not in QUERIES:
            failed.append({"query": f"q{q}", "condition": "all", "error": "query_not_found"})
            continue
        for cond in conditions:
            n += 1
            print(f"[RUN] q{q} {cond} ({n}/{total})", flush=True)
            try:
                case_dir = args.output / cond / f"q{q}"
                summary_path = case_dir / "csv" / "runtimes.csv"
                if args.skip_existing and summary_path.exists():
                    print(f"[SKIP] q{q} {cond}", flush=True)
                    continue
                result = run_case_subprocess(args, q, cond, config)
                if result.get("status") == "ok":
                    all_rows.extend(result["rows"])
                else:
                    failed.append({"query": f"q{q}", "condition": cond, "error": result.get("error", "unknown_error")})
            except Exception as exc:
                print(f"[FAILED] q{q} {cond}: {exc}", flush=True)
                failed.append({"query": f"q{q}", "condition": cond, "error": repr(exc)})
    summary_rows = aggregate(all_rows, args.executions)
    run_fields = ["query", "condition", "execution", "runtime_s", "total_ms"] + BREAKDOWN_FIELDS + STAGE_FIELDS + ["benchmark_dir"]
    summary_fields = ["query", "series", "condition", "samples"]
    for f in BREAKDOWN_FIELDS:
        summary_fields.extend([f, f + "_std"])
    write_csv(args.output / "summary" / "query_runs.csv", all_rows, run_fields)
    write_csv(args.output / "summary" / "query_summary.csv", summary_rows, summary_fields)
    write_csv(args.output / "summary" / "failed_runs.csv", failed, ["query", "condition", "error"])
    write_plots(summary_rows, args.graph_dir.resolve(), args.graph_prefix)
    print(f"==> summary: {args.output / 'summary' / 'query_summary.csv'}", flush=True)
    print(f"==> graph: {args.graph_dir.resolve() / (args.graph_prefix + '_breakdown.png')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
