#!/usr/bin/env python3
# wdy start
"""Run fixed-page paging experiments on generated TPC-H query workloads.

This complements run_fixed_page_pair_latency.py:

- pair latency answers: "does Qj benefit after Qi?"
- workload latency answers: "does a sequence benefit as cached fixed-width
  pages accumulate across several queries?"

Each (condition, workload, repeat) runs in a fresh process. Pinning happens
before the timed query sequence, so the recorded latency measures query
execution after the cache/memory-manager state is prepared.

Outputs:
- summary/workloads.csv
- summary/query_latency.csv
- summary/workload_latency.csv
- summary/workload_latency_summary.csv
- summary/baseline_vs_paging_workload.csv
- summary/baseline_vs_paging_position.csv
- summary/*.png unless --no-plots is used
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "experiment" / "fixed_page_random_workloads"
DEFAULT_QUERIES = "3,5,7,8,9,10,18,21"
DEFAULT_CONDITIONS = "baseline,paging_key_only"

sys.path.insert(0, str(TPCH_DIR))
from performance_test import _execute_multi, open_connection, time_query  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_fixed_page_pair_latency import (  # noqa: E402
    LOG_TS_RE,
    mean,
    parse_csv_list,
    parse_kv,
    pin_sql_for_condition,
    stdev,
    to_float,
    to_int,
    union_interval_ms,
    unpin_sql,
    write_config,
    write_csv,
)


@dataclass(frozen=True)
class Workload:
    workload_id: str
    queries: tuple[int, ...]

    @property
    def query_string(self) -> str:
        return ",".join(f"q{q}" for q in self.queries)


@dataclass(frozen=True)
class CaseSpec:
    condition: str
    workload_id: str
    queries: tuple[int, ...]
    repeat: int

    @property
    def name(self) -> str:
        return f"{self.workload_id}_iter{self.repeat}"


def parse_query_list(value: str) -> list[int]:
    nums: list[int] = []
    for token in parse_csv_list(value):
        token = token.lower().removeprefix("q")
        if "-" in token:
            lo, hi = token.split("-", 1)
            lo = lo.lower().removeprefix("q")
            hi = hi.lower().removeprefix("q")
            nums.extend(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(token))
    return nums


def parse_workload_specs(raw: str) -> list[Workload]:
    workloads: list[Workload] = []
    if not raw:
        return workloads
    for idx, item in enumerate(raw.split(";"), 1):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, query_spec = item.split(":", 1)
            workload_id = name.strip()
        else:
            workload_id = f"w{idx:03d}"
            query_spec = item
        queries = tuple(parse_query_list(query_spec))
        if not queries:
            raise SystemExit(f"empty workload spec: {item}")
        workloads.append(Workload(workload_id, queries))
    return workloads


def load_workloads(path: Path) -> list[Workload]:
    if not path.exists():
        raise SystemExit(f"workloads file does not exist: {path}")
    workloads: list[Workload] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            raise SystemExit("JSON workloads file must be a list")
        for idx, item in enumerate(raw, 1):
            if isinstance(item, dict):
                workload_id = str(item.get("workload_id") or item.get("id") or f"w{idx:03d}")
                queries_raw = item.get("queries")
            else:
                workload_id = f"w{idx:03d}"
                queries_raw = item
            if not isinstance(queries_raw, list):
                raise SystemExit(f"bad workload entry {idx}: queries must be a list")
            workloads.append(Workload(workload_id, tuple(int(q) for q in queries_raw)))
        return workloads

    with path.open(newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            for idx, row in enumerate(csv.DictReader(f), 1):
                workload_id = row.get("workload_id") or row.get("id") or f"w{idx:03d}"
                query_spec = row.get("queries") or row.get("query_sequence") or ""
                workloads.append(Workload(workload_id, tuple(parse_query_list(query_spec))))
        else:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                workloads.append(Workload(f"w{idx:03d}", tuple(parse_query_list(line))))
    return workloads


def generate_workloads(args: argparse.Namespace, queries: list[int]) -> list[Workload]:
    rng = random.Random(args.seed)
    workloads: list[Workload] = []
    for idx in range(1, args.workload_count + 1):
        if args.no_replacement:
            if args.workload_length > len(queries):
                raise SystemExit("--workload-length cannot exceed query count with --no-replacement")
            seq = tuple(rng.sample(queries, args.workload_length))
        else:
            seq = tuple(rng.choice(queries) for _ in range(args.workload_length))
        workloads.append(Workload(f"w{idx:03d}", seq))
    return workloads


def write_workload_manifest(path: Path, workloads: list[Workload]) -> None:
    rows = [
        {
            "workload_id": workload.workload_id,
            "query_count": len(workload.queries),
            "queries": ",".join(str(q) for q in workload.queries),
            "query_labels": workload.query_string,
        }
        for workload in workloads
    ]
    write_csv(path, rows, ["workload_id", "query_count", "queries", "query_labels"])


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def runtime_csv_complete(path: Path, query_count: int) -> bool:
    rows = [row for row in read_csv_rows(path) if row.get("engine") == "sirius"]
    return len(rows) >= query_count


def make_case_env(args: argparse.Namespace, config_path: Path, condition: str, log_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_DIR"] = str(log_dir)
    env["SIRIUS_LOG_LEVEL"] = args.log_level
    if condition == "baseline":
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "0"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "0"
    else:
        env["SIRIUS_ENABLE_FIXED_PAGE_REUSE"] = "1"
        env["SIRIUS_PIN_ROUND_ROBIN_CHUNKS"] = "1"
        env["SIRIUS_PIN_TIER"] = "gpu"
    return env


def run_case(args: argparse.Namespace) -> int:
    run_root = Path(args.output).resolve()
    queries = tuple(parse_query_list(args.case_queries))
    spec = CaseSpec(args.case_condition, args.case_workload_id, queries, args.case_repeat)
    bench = run_root / args.case_condition / "workloads" / spec.name
    csv_dir = bench / "csv"
    log_dir = bench / "log_dir"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_case_env(args, Path(args.config).resolve(), args.case_condition, log_dir.resolve())
    os.environ.update(env)

    metadata = {
        "condition": args.case_condition,
        "workload_id": args.case_workload_id,
        "repeat": args.case_repeat,
        "queries": list(queries),
        "input": str(args.input),
        "config": str(args.config),
        "devices": args.devices,
        "pin_rows": args.pin_rows,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (bench / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"[CASE] {args.case_condition} {spec.name} queries={','.join(f'q{q}' for q in queries)}", flush=True)
    con = open_connection(str(args.input), gpu_execution=True)
    pinned_tables: list[str] = []
    try:
        pin_sql, pinned_tables = pin_sql_for_condition(
            args.case_condition,
            str(args.input),
            list(queries),
            args.pin_rows,
        )
        if pin_sql:
            (bench / "pin.sql").write_text(pin_sql)
            print(f"[CASE] pinning tables: {','.join(pinned_tables)}", flush=True)
            _execute_multi(con, pin_sql)

        with (csv_dir / "runtimes.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["engine", "query", "iteration", "runtime_s", "workload_position"])
            for position, qnum in enumerate(queries, 1):
                elapsed, _rows = time_query(con, qnum, use_gpu=True)
                writer.writerow(["sirius", f"q{qnum}", 0, f"{elapsed:.6f}", position])
                f.flush()
                print(f"[CASE] pos={position} q{qnum} runtime={elapsed:.4f}s", flush=True)
    finally:
        if pinned_tables:
            try:
                print("[CASE] unpinning", flush=True)
                _execute_multi(con, unpin_sql(pinned_tables))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] unpin failed: {exc}", flush=True)
        con.close()
    return 0


def parse_log_timestamp_ms(line: str) -> float | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000.0
    except ValueError:
        return None


def timed_query_segments(bench: Path) -> list[tuple[int, str, list[str]]]:
    runtime_rows = [row for row in read_csv_rows(bench / "csv" / "runtimes.csv") if row.get("engine") == "sirius"]
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

    segments: list[tuple[int, str, list[str]]] = []
    for pos, start in enumerate(begins[: len(runtime_rows)]):
        end = begins[pos + 1] if pos + 1 < len(begins) else len(lines)
        row = runtime_rows[pos]
        position = to_int(row.get("workload_position"), pos + 1)
        query = row.get("query", "unknown")
        segments.append((position, query, lines[start:end]))
    return segments


def load_case_metadata(bench: Path) -> dict[str, object]:
    path = bench / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def summarize_one_case(condition: str, bench: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata = load_case_metadata(bench)
    workload_id = str(metadata.get("workload_id") or bench.name.split("_iter", 1)[0])
    repeat = to_int(metadata.get("repeat"))
    query_sequence = [int(q) for q in metadata.get("queries", [])]
    runtime_rows = [row for row in read_csv_rows(bench / "csv" / "runtimes.csv") if row.get("engine") == "sirius"]
    segments = timed_query_segments(bench)

    by_position: dict[int, dict[str, object]] = {}
    for row_idx, row in enumerate(runtime_rows, 1):
        position = to_int(row.get("workload_position"), row_idx)
        by_position[position] = {
            "query": row.get("query", ""),
            "total_ms": to_float(row.get("runtime_s")) * 1000.0,
        }

    scan_work_by_pos: dict[int, float] = defaultdict(float)
    scan_intervals_by_pos: dict[int, list[tuple[float, float]]] = defaultdict(list)
    scan_uncompressed_by_pos: dict[int, int] = defaultdict(int)
    fixed_cached_by_pos: dict[int, int] = defaultdict(int)
    fixed_split_count_by_pos: dict[int, int] = defaultdict(int)
    fixed_materialize_count_by_pos: dict[int, int] = defaultdict(int)
    fixed_materialize_view_count_by_pos: dict[int, int] = defaultdict(int)
    fixed_fallback_count_by_pos: dict[int, int] = defaultdict(int)

    for position, _query, lines in segments:
        for line in lines:
            if "[scan-audit] parquet_materialize" in line:
                fields = parse_kv(line)
                duration_ms = to_int(fields.get("duration_us")) / 1000.0
                scan_work_by_pos[position] += duration_ms
                scan_uncompressed_by_pos[position] += to_int(fields.get("uncompressed_bytes"))
                end_ms = parse_log_timestamp_ms(line)
                if end_ms is not None:
                    scan_intervals_by_pos[position].append((end_ms - duration_ms, end_ms))
            elif "[fixed-page-cache] hybrid_reuse_split" in line:
                fixed_cached_by_pos[position] += to_int(parse_kv(line).get("cached_bytes"))
                fixed_split_count_by_pos[position] += 1
            elif "[fixed-page-cache] hybrid_reuse_materialize_view" in line:
                fixed_materialize_view_count_by_pos[position] += 1
            elif "[fixed-page-cache] hybrid_reuse_materialize " in line:
                fixed_materialize_count_by_pos[position] += 1
            elif "[fixed-page-cache] hybrid_reuse_row_group_fallback" in line:
                fixed_fallback_count_by_pos[position] += 1

    query_rows: list[dict[str, object]] = []
    cumulative_ms = 0.0
    for position in sorted(by_position):
        base = by_position[position]
        total_ms = to_float(base.get("total_ms"))
        cumulative_ms += total_ms
        scan_wall_ms = union_interval_ms(scan_intervals_by_pos.get(position, []))
        load_ms = min(scan_wall_ms, total_ms)
        query_rows.append(
            {
                "condition": condition,
                "workload_id": workload_id,
                "repeat": repeat,
                "position": position,
                "query": base.get("query", ""),
                "query_sequence": ",".join(f"q{q}" for q in query_sequence),
                "total_ms": total_ms,
                "cumulative_total_ms": cumulative_ms,
                "load_ms": load_ms,
                "scan_materialize_wall_ms": scan_wall_ms,
                "scan_materialize_work_ms": scan_work_by_pos.get(position, 0.0),
                "scan_uncompressed_gb": scan_uncompressed_by_pos.get(position, 0) / 1e9,
                "computation_ms": max(total_ms - load_ms, 0.0),
                "fixed_page_cached_gb": fixed_cached_by_pos.get(position, 0) / 1e9,
                "fixed_page_reuse_split_count": fixed_split_count_by_pos.get(position, 0),
                "fixed_page_materialize_count": fixed_materialize_count_by_pos.get(position, 0),
                "fixed_page_materialize_view_count": fixed_materialize_view_count_by_pos.get(position, 0),
                "fixed_page_fallback_count": fixed_fallback_count_by_pos.get(position, 0),
                "benchmark_dir": str(bench),
            }
        )

    workload_row = {
        "condition": condition,
        "workload_id": workload_id,
        "repeat": repeat,
        "query_count": len(query_rows),
        "query_sequence": ",".join(f"q{q}" for q in query_sequence),
        "total_ms": sum(to_float(row.get("total_ms")) for row in query_rows),
        "load_ms": sum(to_float(row.get("load_ms")) for row in query_rows),
        "scan_materialize_wall_ms": sum(to_float(row.get("scan_materialize_wall_ms")) for row in query_rows),
        "scan_materialize_work_ms": sum(to_float(row.get("scan_materialize_work_ms")) for row in query_rows),
        "scan_uncompressed_gb": sum(to_float(row.get("scan_uncompressed_gb")) for row in query_rows),
        "computation_ms": sum(to_float(row.get("computation_ms")) for row in query_rows),
        "fixed_page_cached_gb": sum(to_float(row.get("fixed_page_cached_gb")) for row in query_rows),
        "fixed_page_reuse_split_count": sum(to_int(row.get("fixed_page_reuse_split_count")) for row in query_rows),
        "fixed_page_materialize_view_count": sum(to_int(row.get("fixed_page_materialize_view_count")) for row in query_rows),
        "fixed_page_fallback_count": sum(to_int(row.get("fixed_page_fallback_count")) for row in query_rows),
        "benchmark_dir": str(bench),
    }
    return query_rows, workload_row


QUERY_FIELDS = [
    "condition",
    "workload_id",
    "repeat",
    "position",
    "query",
    "query_sequence",
    "total_ms",
    "cumulative_total_ms",
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
    "benchmark_dir",
]

WORKLOAD_FIELDS = [
    "condition",
    "workload_id",
    "repeat",
    "query_count",
    "query_sequence",
    "total_ms",
    "load_ms",
    "scan_materialize_wall_ms",
    "scan_materialize_work_ms",
    "scan_uncompressed_gb",
    "computation_ms",
    "fixed_page_cached_gb",
    "fixed_page_reuse_split_count",
    "fixed_page_materialize_view_count",
    "fixed_page_fallback_count",
    "benchmark_dir",
]


def vals(group: list[dict[str, object]], field: str) -> list[float]:
    return [to_float(row.get(field, "")) for row in group if row.get(field, "") != ""]


def write_workload_latency_summary(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["workload_id"]))].append(row)
    fields = [
        "condition",
        "workload_id",
        "query_sequence",
        "repeats_completed",
        "total_ms_mean",
        "total_ms_std",
        "load_ms_mean",
        "load_ms_std",
        "scan_materialize_work_ms_mean",
        "scan_uncompressed_gb_mean",
        "computation_ms_mean",
        "fixed_page_cached_gb_mean",
        "fixed_page_reuse_split_count_mean",
        "fixed_page_materialize_view_count_mean",
        "fixed_page_fallback_count_mean",
    ]
    out: list[dict[str, object]] = []
    for (condition, workload_id), group in sorted(groups.items()):
        out.append(
            {
                "condition": condition,
                "workload_id": workload_id,
                "query_sequence": group[0].get("query_sequence", ""),
                "repeats_completed": len(group),
                "total_ms_mean": mean(vals(group, "total_ms")),
                "total_ms_std": stdev(vals(group, "total_ms")),
                "load_ms_mean": mean(vals(group, "load_ms")),
                "load_ms_std": stdev(vals(group, "load_ms")),
                "scan_materialize_work_ms_mean": mean(vals(group, "scan_materialize_work_ms")),
                "scan_uncompressed_gb_mean": mean(vals(group, "scan_uncompressed_gb")),
                "computation_ms_mean": mean(vals(group, "computation_ms")),
                "fixed_page_cached_gb_mean": mean(vals(group, "fixed_page_cached_gb")),
                "fixed_page_reuse_split_count_mean": mean(vals(group, "fixed_page_reuse_split_count")),
                "fixed_page_materialize_view_count_mean": mean(vals(group, "fixed_page_materialize_view_count")),
                "fixed_page_fallback_count_mean": mean(vals(group, "fixed_page_fallback_count")),
            }
        )
    write_csv(path, out, fields)


def avg(groups: dict[tuple[str, str], list[dict[str, object]]], condition: str, workload_id: str, field: str) -> float | str:
    values = vals(groups.get((condition, workload_id), []), field)
    return mean(values)


def write_baseline_vs_paging(summary_dir: Path, query_rows: list[dict[str, object]], workload_rows: list[dict[str, object]]) -> None:
    workload_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in workload_rows:
        workload_groups[(str(row["condition"]), str(row["workload_id"]))].append(row)

    query_groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in query_rows:
        query_groups[(str(row["condition"]), str(row["workload_id"]), to_int(row["position"]))].append(row)

    paging_conditions = sorted({str(row["condition"]) for row in workload_rows if str(row["condition"]) != "baseline"})
    workload_ids = sorted({str(row["workload_id"]) for row in workload_rows})

    workload_fields = [
        "paging_condition",
        "workload_id",
        "query_sequence",
        "baseline_repeats",
        "paging_repeats",
        "baseline_total_ms",
        "paging_total_ms",
        "total_ms_delta",
        "total_ms_speedup",
        "baseline_load_ms",
        "paging_load_ms",
        "load_ms_delta",
        "load_ms_reduction_ratio",
        "baseline_scan_work_ms",
        "paging_scan_work_ms",
        "scan_work_reduction_ratio",
        "baseline_computation_ms",
        "paging_computation_ms",
        "computation_ms_delta",
        "paging_fixed_page_cached_gb",
        "paging_fixed_page_reuse_split_count",
        "paging_fixed_page_materialize_view_count",
        "paging_fixed_page_fallback_count",
    ]
    workload_out: list[dict[str, object]] = []
    for paging in paging_conditions:
        for workload_id in workload_ids:
            b_total = avg(workload_groups, "baseline", workload_id, "total_ms")
            p_total = avg(workload_groups, paging, workload_id, "total_ms")
            b_load = avg(workload_groups, "baseline", workload_id, "load_ms")
            p_load = avg(workload_groups, paging, workload_id, "load_ms")
            b_scan = avg(workload_groups, "baseline", workload_id, "scan_materialize_work_ms")
            p_scan = avg(workload_groups, paging, workload_id, "scan_materialize_work_ms")
            b_comp = avg(workload_groups, "baseline", workload_id, "computation_ms")
            p_comp = avg(workload_groups, paging, workload_id, "computation_ms")
            group = workload_groups.get((paging, workload_id), []) or workload_groups.get(("baseline", workload_id), [])
            seq = group[0].get("query_sequence", "") if group else ""
            workload_out.append(
                {
                    "paging_condition": paging,
                    "workload_id": workload_id,
                    "query_sequence": seq,
                    "baseline_repeats": len(workload_groups.get(("baseline", workload_id), [])),
                    "paging_repeats": len(workload_groups.get((paging, workload_id), [])),
                    "baseline_total_ms": b_total,
                    "paging_total_ms": p_total,
                    "total_ms_delta": float(p_total) - float(b_total) if b_total != "" and p_total != "" else "",
                    "total_ms_speedup": float(b_total) / float(p_total) if b_total != "" and p_total != "" and float(p_total) > 0 else "",
                    "baseline_load_ms": b_load,
                    "paging_load_ms": p_load,
                    "load_ms_delta": float(p_load) - float(b_load) if b_load != "" and p_load != "" else "",
                    "load_ms_reduction_ratio": (float(b_load) - float(p_load)) / float(b_load) if b_load != "" and p_load != "" and float(b_load) > 0 else "",
                    "baseline_scan_work_ms": b_scan,
                    "paging_scan_work_ms": p_scan,
                    "scan_work_reduction_ratio": (float(b_scan) - float(p_scan)) / float(b_scan) if b_scan != "" and p_scan != "" and float(b_scan) > 0 else "",
                    "baseline_computation_ms": b_comp,
                    "paging_computation_ms": p_comp,
                    "computation_ms_delta": float(p_comp) - float(b_comp) if b_comp != "" and p_comp != "" else "",
                    "paging_fixed_page_cached_gb": avg(workload_groups, paging, workload_id, "fixed_page_cached_gb"),
                    "paging_fixed_page_reuse_split_count": avg(workload_groups, paging, workload_id, "fixed_page_reuse_split_count"),
                    "paging_fixed_page_materialize_view_count": avg(workload_groups, paging, workload_id, "fixed_page_materialize_view_count"),
                    "paging_fixed_page_fallback_count": avg(workload_groups, paging, workload_id, "fixed_page_fallback_count"),
                }
            )
    write_csv(summary_dir / "baseline_vs_paging_workload.csv", workload_out, workload_fields)

    position_fields = [
        "paging_condition",
        "workload_id",
        "position",
        "query",
        "baseline_repeats",
        "paging_repeats",
        "baseline_total_ms",
        "paging_total_ms",
        "total_ms_delta",
        "total_ms_speedup",
        "baseline_load_ms",
        "paging_load_ms",
        "load_ms_reduction_ratio",
        "paging_fixed_page_cached_gb",
    ]
    position_out: list[dict[str, object]] = []
    positions = sorted({(str(row["workload_id"]), to_int(row["position"])) for row in query_rows})
    for paging in paging_conditions:
        for workload_id, position in positions:
            b_group = query_groups.get(("baseline", workload_id, position), [])
            p_group = query_groups.get((paging, workload_id, position), [])
            b_total = mean(vals(b_group, "total_ms")) if b_group else ""
            p_total = mean(vals(p_group, "total_ms")) if p_group else ""
            b_load = mean(vals(b_group, "load_ms")) if b_group else ""
            p_load = mean(vals(p_group, "load_ms")) if p_group else ""
            sample = (p_group or b_group)
            query = sample[0].get("query", "") if sample else ""
            position_out.append(
                {
                    "paging_condition": paging,
                    "workload_id": workload_id,
                    "position": position,
                    "query": query,
                    "baseline_repeats": len(b_group),
                    "paging_repeats": len(p_group),
                    "baseline_total_ms": b_total,
                    "paging_total_ms": p_total,
                    "total_ms_delta": float(p_total) - float(b_total) if b_total != "" and p_total != "" else "",
                    "total_ms_speedup": float(b_total) / float(p_total) if b_total != "" and p_total != "" and float(p_total) > 0 else "",
                    "baseline_load_ms": b_load,
                    "paging_load_ms": p_load,
                    "load_ms_reduction_ratio": (float(b_load) - float(p_load)) / float(b_load) if b_load != "" and p_load != "" and float(b_load) > 0 else "",
                    "paging_fixed_page_cached_gb": mean(vals(p_group, "fixed_page_cached_gb")) if p_group else "",
                }
            )
    write_csv(summary_dir / "baseline_vs_paging_position.csv", position_out, position_fields)


def summarize_results(run_root: Path, conditions: list[str], workloads: list[Workload], no_plots: bool) -> None:
    all_query_rows: list[dict[str, object]] = []
    all_workload_rows: list[dict[str, object]] = []
    query_count_by_workload = {workload.workload_id: len(workload.queries) for workload in workloads}
    for condition in conditions:
        for runtime_csv in sorted((run_root / condition / "workloads").glob("*_iter*/csv/runtimes.csv")):
            bench = runtime_csv.parents[1]
            metadata = load_case_metadata(bench)
            workload_id = str(metadata.get("workload_id") or bench.name.split("_iter", 1)[0])
            expected_count = query_count_by_workload.get(workload_id, len(metadata.get("queries", [])))
            if expected_count and not runtime_csv_complete(runtime_csv, expected_count):
                continue
            query_rows, workload_row = summarize_one_case(condition, bench)
            all_query_rows.extend(query_rows)
            if workload_row:
                all_workload_rows.append(workload_row)

    summary_dir = run_root / "summary"
    write_workload_manifest(summary_dir / "workloads.csv", workloads)
    write_csv(summary_dir / "query_latency.csv", all_query_rows, QUERY_FIELDS)
    write_csv(summary_dir / "workload_latency.csv", all_workload_rows, WORKLOAD_FIELDS)
    write_workload_latency_summary(summary_dir / "workload_latency_summary.csv", all_workload_rows)
    write_baseline_vs_paging(summary_dir, all_query_rows, all_workload_rows)
    if not no_plots:
        plot_outputs(summary_dir)


def run_cmd(cmd: list[str], env: dict[str, str], timeout_s: int | None, dry_run: bool) -> int:
    print(f"==> {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    full_cmd = cmd
    if timeout_s is not None:
        full_cmd = ["timeout", "--kill-after=30s", str(timeout_s)] + cmd
    return subprocess.run(full_cmd, cwd=REPO_ROOT, env=env).returncode


def plot_outputs(summary_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] plotting skipped: {exc}", flush=True)
        return

    workload_path = summary_dir / "baseline_vs_paging_workload.csv"
    if not workload_path.exists():
        return
    df = pd.read_csv(workload_path)
    if df.empty:
        return

    for paging in sorted(df["paging_condition"].dropna().unique()):
        pdf = df[df["paging_condition"] == paging].copy()
        pdf["speedup_num"] = pd.to_numeric(pdf["total_ms_speedup"], errors="coerce")
        pdf["cached_gb_num"] = pd.to_numeric(pdf["paging_fixed_page_cached_gb"], errors="coerce")
        pdf = pdf.dropna(subset=["speedup_num"]).sort_values("speedup_num", ascending=False)
        if pdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(10, len(pdf) * 0.45), 5), constrained_layout=True)
        colors = ["#54a24b" if value >= 1.0 else "#e45756" for value in pdf["speedup_num"]]
        ax.bar(pdf["workload_id"], pdf["speedup_num"], color=colors)
        ax.axhline(1.0, color="black", linewidth=1.0)
        ax.set_ylabel("Total workload speedup (baseline / paging)")
        ax.set_xlabel("Random workload")
        ax.set_title(f"{paging}: total workload speedup")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(summary_dir / f"{paging}_workload_total_speedup.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        ax.scatter(pdf["cached_gb_num"], pdf["speedup_num"], color="#4c78a8", alpha=0.85)
        ax.axhline(1.0, color="black", linewidth=1.0)
        for _, row in pdf.iterrows():
            ax.annotate(str(row["workload_id"]), (row["cached_gb_num"], row["speedup_num"]), fontsize=7)
        ax.set_xlabel("Fixed-page cached GB used by workload")
        ax.set_ylabel("Total workload speedup")
        ax.set_title(f"{paging}: reuse volume vs speedup")
        ax.grid(True, alpha=0.25)
        fig.savefig(summary_dir / f"{paging}_reuse_vs_speedup.png", dpi=180)
        plt.close(fig)

    pos_path = summary_dir / "baseline_vs_paging_position.csv"
    if not pos_path.exists():
        return
    pos = pd.read_csv(pos_path)
    if pos.empty:
        return
    pos["speedup_num"] = pd.to_numeric(pos["total_ms_speedup"], errors="coerce")
    pos["load_reduction_num"] = pd.to_numeric(pos["load_ms_reduction_ratio"], errors="coerce")
    for paging in sorted(pos["paging_condition"].dropna().unique()):
        ppos = pos[pos["paging_condition"] == paging].copy()
        if ppos.empty:
            continue
        avg = (
            ppos.groupby("position", as_index=False)
            .agg(speedup=("speedup_num", "mean"), load_reduction=("load_reduction_num", "mean"))
            .sort_values("position")
        )
        if avg.empty:
            continue
        fig, ax1 = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax1.plot(avg["position"], avg["speedup"], marker="o", label="latency speedup", color="#4c78a8")
        ax1.axhline(1.0, color="black", linewidth=1.0)
        ax1.set_xlabel("Position in workload")
        ax1.set_ylabel("Mean latency speedup")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(avg["position"], avg["load_reduction"], marker="s", label="load reduction", color="#f58518")
        ax2.set_ylabel("Mean load reduction ratio")
        fig.suptitle(f"{paging}: effect by workload position")
        fig.savefig(summary_dir / f"{paging}_position_effect.png", dpi=180)
        plt.close(fig)


def build_workloads(args: argparse.Namespace) -> list[Workload]:
    if args.all_tpch:
        queries = list(range(1, 23))
    else:
        queries = parse_query_list(args.queries)
    if args.workloads_file:
        workloads = load_workloads(Path(args.workloads_file))
    else:
        workloads = parse_workload_specs(args.workloads)
    if not workloads:
        workloads = generate_workloads(args, queries)
    allowed = set(queries) if not args.all_tpch else set(range(1, 23))
    for workload in workloads:
        bad = [q for q in workload.queries if q not in allowed]
        if bad:
            raise SystemExit(f"{workload.workload_id} uses query outside --queries set: {bad}")
    return workloads


def run_orchestrator(args: argparse.Namespace) -> int:
    workloads = build_workloads(args)
    conditions = parse_csv_list(args.conditions)
    valid_conditions = {"baseline", "paging_key_only", "paging_budget", "paging_full_fixed"}
    bad = [condition for condition in conditions if condition not in valid_conditions]
    if bad:
        raise SystemExit(f"bad condition(s): {', '.join(bad)}")

    run_root = args.output.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    summary_dir = run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_workload_manifest(summary_dir / "workloads.csv", workloads)

    config_path = run_root / "configs" / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config_path, args)
    (run_root / "README.md").write_text(
        f"""# Fixed-page random workload experiment

Created: {datetime.now().isoformat(timespec='seconds')}
Input: `{args.input}`
Workloads: `{len(workloads)}`
Repeats: `{args.repeats}`
Conditions: `{','.join(conditions)}`
Devices: `{args.devices}`
Seed: `{args.seed}`

Each condition/workload/repeat runs in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_workload.csv`.
"""
    )

    if not args.skip_build:
        code = run_cmd(["pixi", "run", "make", "-j4"], os.environ.copy(), None, args.dry_run)
        if code != 0:
            return code

    failed_fields = ["condition", "workload_id", "repeat", "returncode"]
    failed_path = run_root / "failed_cases.csv"
    write_csv(failed_path, [], failed_fields)

    total = len(conditions) * len(workloads) * args.repeats
    current = 0
    for condition in conditions:
        for workload in workloads:
            for repeat in range(args.repeats):
                current += 1
                spec = CaseSpec(condition, workload.workload_id, workload.queries, repeat)
                bench = run_root / condition / "workloads" / spec.name
                csv_path = bench / "csv" / "runtimes.csv"
                if runtime_csv_complete(csv_path, len(workload.queries)):
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
                    "--case-workload-id",
                    workload.workload_id,
                    "--case-queries",
                    ",".join(str(q) for q in workload.queries),
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
                code = run_cmd(cmd, env, args.workload_timeout, args.dry_run)
                if not args.dry_run and (code != 0 or not runtime_csv_complete(csv_path, len(workload.queries))):
                    with failed_path.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=failed_fields)
                        writer.writerow(
                            {
                                "condition": condition,
                                "workload_id": workload.workload_id,
                                "repeat": repeat,
                                "returncode": code,
                            }
                        )
                    print(f"[FAILED] {condition} {spec.name} returncode={code}", flush=True)
                summarize_results(run_root, conditions, workloads, args.no_plots)

    summarize_results(run_root, conditions, workloads, args.no_plots)
    print(f"==> done: {run_root}", flush=True)
    print(f"==> comparison: {run_root / 'summary' / 'baseline_vs_paging_workload.csv'}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--all-tpch", action="store_true")
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workload-count", type=int, default=20)
    parser.add_argument("--workload-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--no-replacement", action="store_true")
    parser.add_argument(
        "--workloads",
        default="",
        help="Explicit workloads, e.g. 'hot:3,5,7,8;mix:10,18,21,3'.",
    )
    parser.add_argument("--workloads-file", default="")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-usage-limit", default="12GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument("--pin-rows", type=int, default=None)
    parser.add_argument("--workload-timeout", type=int, default=900)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--config", default="")
    parser.add_argument("--case-condition", default="")
    parser.add_argument("--case-workload-id", default="")
    parser.add_argument("--case-queries", default="")
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
