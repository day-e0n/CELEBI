#!/usr/bin/env python3
# wdy start
"""Run automatic motivation latency experiments for reuse opportunities.

This script can run these variants for every ordered TPC-H query pair:

1. baseline: normal Sirius execution.
2. join_reuse: experimental JOIN output retention + exact-signature reuse.
3. pinned_hot: upper-bound cached-scan experiment using existing pin_table.

Each pair repeat is a separate performance_test.py process. That keeps the
experiment simple and gives the GPU allocator/process a chance to release VRAM
before the next pair starts.

Outputs include:
- per-query latency: total/load-wall/computation latency
- per-pair latency: first-query, second-query, and pair totals
- retention/reuse logs: retained bytes, skipped bytes, hit bytes
- heatmap CSV/PNG data for second-query JOIN reuse
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUTPUT = REPO_ROOT / "experiment" / "automatic_motivation_latency"
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
QUERIES = list(range(1, 23))
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
PAIR_NAME_RE = re.compile(r"q(\d+)_then_q(\d+)_iter(\d+)")
LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")


@dataclass(frozen=True)
class RunSpec:
    variant: str
    qi: int
    qj: int
    repeat: int

    @property
    def name(self) -> str:
        return f"q{self.qi}_then_q{self.qj}_iter{self.repeat}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--gpu-usage-limit", default="4GB")
    parser.add_argument("--host-capacity", default="16GB")
    parser.add_argument("--reservation-limit-fraction", default="0.8")
    parser.add_argument("--pair-timeout", type=int, default=900)
    parser.add_argument("--join-retention-limit-bytes", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--join-retention-max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-variant", choices=("baseline", "join_reuse", "pinned_hot"), default=None)
    # wdy start
    parser.add_argument(
        "--variants",
        default="",
        help=(
            "Comma-separated variant list, e.g. baseline,pinned_hot. "
            "Default keeps the original baseline,join_reuse behavior."
        ),
    )
    # wdy end
    parser.add_argument(
        "--pairs",
        default="",
        help="Optional comma-separated subset like 2:16,19:14. Default: all ordered pairs.",
    )
    parser.add_argument(
        "--allow-diagonal",
        action="store_true",
        help="Allow pairs like 2:2 for reuse smoke tests. Default all-pairs still excludes diagonal.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_pairs(raw: str, allow_diagonal: bool = False) -> list[tuple[int, int]]:
    if not raw:
        return [(qi, qj) for qi in QUERIES for qj in QUERIES if qi != qj]
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        qi, qj = int(left), int(right)
        if qi not in QUERIES or qj not in QUERIES or (qi == qj and not allow_diagonal):
            raise SystemExit(f"bad pair: {item}")
        pairs.append((qi, qj))
    return pairs


def write_config(path: Path, args: argparse.Namespace) -> None:
    path.write_text(
        f"""sirius:
  topology:
    num_gpus: {args.num_gpus}
  memory:
    gpu:
      usage_limit_bytes: {args.gpu_usage_limit}
      reservation_limit_fraction: {args.reservation_limit_fraction}
    host:
      capacity_bytes: {args.host_capacity}
"""
    )


def run_cmd(cmd: list[str], *, env: dict[str, str], cwd: Path, timeout_s: int | None, dry_run: bool) -> int:
    printable = " ".join(cmd)
    print(f"==> {printable}", flush=True)
    if dry_run:
        return 0
    full_cmd = cmd
    if timeout_s is not None:
        full_cmd = ["timeout", "--kill-after=30s", str(timeout_s)] + cmd
    result = subprocess.run(full_cmd, cwd=cwd, env=env)
    return result.returncode


def runtime_csv_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = path.read_text(errors="replace").splitlines()
    except OSError:
        return False
    # Header + first query + second query.
    return len(rows) >= 3


def make_env(args: argparse.Namespace, config_path: Path, variant: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_LEVEL"] = "info"
    for key in (
        "SIRIUS_JOIN_OUTPUT_RETENTION",
        "SIRIUS_JOIN_OUTPUT_RETENTION_LIMIT_BYTES",
        "SIRIUS_JOIN_OUTPUT_RETENTION_MAX_BATCH_BYTES",
        "SIRIUS_JOIN_OUTPUT_REUSE",
    ):
        env.pop(key, None)
    if variant == "join_reuse":
        env["SIRIUS_JOIN_OUTPUT_RETENTION"] = "1"
        env["SIRIUS_JOIN_OUTPUT_REUSE"] = "1"
        env["SIRIUS_JOIN_OUTPUT_RETENTION_LIMIT_BYTES"] = str(args.join_retention_limit_bytes)
        env["SIRIUS_JOIN_OUTPUT_RETENTION_MAX_BATCH_BYTES"] = str(args.join_retention_max_batch_bytes)
    # wdy start
    if variant == "pinned_hot":
        env["SIRIUS_PIN_ONLY_SECOND_QUERY"] = "1"
    else:
        env.pop("SIRIUS_PIN_ONLY_SECOND_QUERY", None)
    # wdy end
    return env


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# wdy start
def parse_variants(args: argparse.Namespace) -> list[str]:
    valid = {"baseline", "join_reuse", "pinned_hot"}
    if args.only_variant:
        return [args.only_variant]
    if not args.variants:
        return ["baseline", "join_reuse"]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    bad = [item for item in variants if item not in valid]
    if bad:
        raise SystemExit(f"bad variant(s): {', '.join(bad)}; valid: {', '.join(sorted(valid))}")
    return variants
# wdy end


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def to_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def to_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def timed_query_segments(bench: Path) -> list[tuple[str, list[str]]]:
    metadata = {row.get("query", "") for row in read_csv(bench / "csv" / "runtimes.csv")}
    log_dir = bench / "log_dir"
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return []
    lines = logs[-1].read_text(errors="replace").splitlines()
    begins = []
    for idx, line in enumerate(lines):
        marker = "QueryBegin: SQL:"
        if marker not in line:
            continue
        sql = line.split(marker, 1)[1].strip().lower()
        if sql.startswith("set ") or sql.startswith("create view") or sql.startswith("call pin_table"):
            continue
        begins.append(idx)
    runtimes = [row.get("query", "") for row in read_csv(bench / "csv" / "runtimes.csv") if row.get("engine") == "sirius"]
    segments = []
    for pos, start in enumerate(begins[: len(runtimes)]):
        end = begins[pos + 1] if pos + 1 < len(begins) else len(lines)
        query = runtimes[pos] if pos < len(runtimes) else "unknown"
        if query in metadata:
            segments.append((query, lines[start:end]))
    return segments


def summarize_one_bench(variant: str, bench: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    match = PAIR_NAME_RE.match(bench.name)
    if not match:
        return [], {}
    qi, qj, repeat = map(int, match.groups())
    runtime_rows = [row for row in read_csv(bench / "csv" / "runtimes.csv") if row.get("engine") == "sirius"]
    segments = timed_query_segments(bench)
    scan_work_by_query: dict[str, float] = defaultdict(float)
    scan_intervals_by_query: dict[str, list[tuple[float, float]]] = defaultdict(list)
    retained_by_query: dict[str, int] = defaultdict(int)
    skipped_by_query: dict[str, int] = defaultdict(int)
    cached_by_query: dict[str, int] = defaultdict(int)
    hit_by_query: dict[str, int] = defaultdict(int)
    hit_count_by_query: dict[str, int] = defaultdict(int)
    miss_count_by_query: dict[str, int] = defaultdict(int)

    for query, lines in segments:
        for line in lines:
            if "[scan-audit] parquet_materialize" in line:
                duration_ms = to_int(parse_kv(line).get("duration_us")) / 1000.0
                scan_work_by_query[query] += duration_ms
                end_ms = parse_log_timestamp_ms(line)
                if end_ms is not None:
                    scan_intervals_by_query[query].append((end_ms - duration_ms, end_ms))
            elif "[join-retention] retained" in line:
                retained_by_query[query] += to_int(parse_kv(line).get("bytes"))
            elif "[join-retention] skipped" in line:
                skipped_by_query[query] += to_int(parse_kv(line).get("bytes"))
            elif "[join-reuse] cached" in line:
                cached_by_query[query] += to_int(parse_kv(line).get("bytes"))
            elif "[join-reuse] hit" in line:
                fields = parse_kv(line)
                hit_by_query[query] += to_int(fields.get("bytes"))
                hit_count_by_query[query] += 1
            elif "[join-reuse] miss" in line:
                miss_count_by_query[query] += 1

    query_rows: list[dict[str, object]] = []
    for position, row in enumerate(runtime_rows[:2], 1):
        query = row.get("query", "")
        total_ms = to_float(row.get("runtime_s")) * 1000.0
        scan_work_ms = scan_work_by_query.get(query, 0.0)
        scan_wall_ms = union_interval_ms(scan_intervals_by_query.get(query, []))
        load_ms = min(scan_wall_ms, total_ms)
        query_rows.append(
            {
                "variant": variant,
                "previous_query": f"q{qi}",
                "second_query": f"q{qj}",
                "repeat": repeat,
                "query_position": position,
                "query": query,
                "total_ms": total_ms,
                "load_ms": load_ms,
                "scan_materialize_wall_ms": scan_wall_ms,
                "scan_materialize_work_ms": scan_work_ms,
                "computation_ms": max(total_ms - load_ms, 0.0),
                "join_retained_bytes": retained_by_query.get(query, 0),
                "join_skipped_bytes": skipped_by_query.get(query, 0),
                "join_cached_bytes": cached_by_query.get(query, 0),
                "join_reuse_hit_bytes": hit_by_query.get(query, 0),
                "join_reuse_hit_count": hit_count_by_query.get(query, 0),
                "join_reuse_miss_count": miss_count_by_query.get(query, 0),
                "benchmark_dir": str(bench),
            }
        )

    first = query_rows[0] if len(query_rows) > 0 else {}
    second = query_rows[1] if len(query_rows) > 1 else {}
    pair_row = {
        "variant": variant,
        "previous_query": f"q{qi}",
        "second_query": f"q{qj}",
        "repeat": repeat,
        "first_total_ms": first.get("total_ms", ""),
        "first_load_ms": first.get("load_ms", ""),
        "first_scan_materialize_wall_ms": first.get("scan_materialize_wall_ms", ""),
        "first_scan_materialize_work_ms": first.get("scan_materialize_work_ms", ""),
        "first_computation_ms": first.get("computation_ms", ""),
        "second_total_ms": second.get("total_ms", ""),
        "second_load_ms": second.get("load_ms", ""),
        "second_scan_materialize_wall_ms": second.get("scan_materialize_wall_ms", ""),
        "second_scan_materialize_work_ms": second.get("scan_materialize_work_ms", ""),
        "second_computation_ms": second.get("computation_ms", ""),
        "pair_total_ms": to_float(str(first.get("total_ms", ""))) + to_float(str(second.get("total_ms", ""))),
        "pair_load_ms": to_float(str(first.get("load_ms", ""))) + to_float(str(second.get("load_ms", ""))),
        "pair_scan_materialize_wall_ms": to_float(str(first.get("scan_materialize_wall_ms", ""))) + to_float(str(second.get("scan_materialize_wall_ms", ""))),
        "pair_scan_materialize_work_ms": to_float(str(first.get("scan_materialize_work_ms", ""))) + to_float(str(second.get("scan_materialize_work_ms", ""))),
        "pair_computation_ms": to_float(str(first.get("computation_ms", ""))) + to_float(str(second.get("computation_ms", ""))),
        "first_join_retained_bytes": first.get("join_retained_bytes", 0),
        "second_join_reuse_hit_bytes": second.get("join_reuse_hit_bytes", 0),
        "second_join_reuse_hit_count": second.get("join_reuse_hit_count", 0),
        "second_join_reuse_miss_count": second.get("join_reuse_miss_count", 0),
        "reuse_hit_ratio_vs_first_retained": (
            to_float(str(second.get("join_reuse_hit_bytes", 0))) / to_float(str(first.get("join_retained_bytes", 0)))
            if to_float(str(first.get("join_retained_bytes", 0))) > 0
            else ""
        ),
        "benchmark_dir": str(bench),
    }
    return query_rows, pair_row


def summarize_results(run_root: Path, variants: list[str], no_plots: bool) -> None:
    all_query_rows: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    for variant in variants:
        for runtime_csv in sorted((run_root / variant / "pairs").glob("q*_then_q*_iter*/csv/runtimes.csv")):
            bench = runtime_csv.parents[1]
            if not runtime_csv_complete(runtime_csv):
                continue
            query_rows, pair_row = summarize_one_bench(variant, bench)
            all_query_rows.extend(query_rows)
            if pair_row:
                all_pair_rows.append(pair_row)

    query_fields = [
        "variant",
        "previous_query",
        "second_query",
        "repeat",
        "query_position",
        "query",
        "total_ms",
        "load_ms",
        "scan_materialize_wall_ms",
        "scan_materialize_work_ms",
        "computation_ms",
        "join_retained_bytes",
        "join_skipped_bytes",
        "join_cached_bytes",
        "join_reuse_hit_bytes",
        "join_reuse_hit_count",
        "join_reuse_miss_count",
        "benchmark_dir",
    ]
    pair_fields = [
        "variant",
        "previous_query",
        "second_query",
        "repeat",
        "first_total_ms",
        "first_load_ms",
        "first_scan_materialize_wall_ms",
        "first_scan_materialize_work_ms",
        "first_computation_ms",
        "second_total_ms",
        "second_load_ms",
        "second_scan_materialize_wall_ms",
        "second_scan_materialize_work_ms",
        "second_computation_ms",
        "pair_total_ms",
        "pair_load_ms",
        "pair_scan_materialize_wall_ms",
        "pair_scan_materialize_work_ms",
        "pair_computation_ms",
        "first_join_retained_bytes",
        "second_join_reuse_hit_bytes",
        "second_join_reuse_hit_count",
        "second_join_reuse_miss_count",
        "reuse_hit_ratio_vs_first_retained",
        "benchmark_dir",
    ]
    summary_dir = run_root / "summary"
    write_csv(summary_dir / "query_latency.csv", all_query_rows, query_fields)
    write_csv(summary_dir / "pair_latency.csv", all_pair_rows, pair_fields)
    write_latency_summary(summary_dir / "pair_latency_summary.csv", all_pair_rows)
    write_second_query_latency_summary(summary_dir / "second_query_latency_summary.csv", all_pair_rows)
    write_second_query_comparison(summary_dir / "baseline_vs_join_reuse_second_query.csv", all_pair_rows)
    write_reuse_heatmap_inputs(summary_dir, all_pair_rows)
    if not no_plots:
        plot_heatmaps(summary_dir)


def mean(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def stdev(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) >= 2 else ""


def write_latency_summary(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["previous_query"]), str(row["second_query"]))].append(row)
    fields = [
        "variant",
        "previous_query",
        "second_query",
        "repeats_completed",
        "second_total_ms_mean",
        "second_total_ms_std",
        "second_load_ms_mean",
        "second_load_ms_std",
        "second_scan_materialize_wall_ms_mean",
        "second_scan_materialize_work_ms_mean",
        "second_computation_ms_mean",
        "second_computation_ms_std",
        "pair_total_ms_mean",
        "pair_load_ms_mean",
        "pair_scan_materialize_wall_ms_mean",
        "pair_scan_materialize_work_ms_mean",
        "pair_computation_ms_mean",
        "second_join_reuse_hit_bytes_mean",
        "reuse_hit_ratio_vs_first_retained_mean",
    ]
    out = []
    for (variant, qi, qj), group in sorted(groups.items()):
        def vals(name: str) -> list[float]:
            return [to_float(str(row.get(name, ""))) for row in group if row.get(name, "") != ""]

        out.append(
            {
                "variant": variant,
                "previous_query": qi,
                "second_query": qj,
                "repeats_completed": len(group),
                "second_total_ms_mean": mean(vals("second_total_ms")),
                "second_total_ms_std": stdev(vals("second_total_ms")),
                "second_load_ms_mean": mean(vals("second_load_ms")),
                "second_load_ms_std": stdev(vals("second_load_ms")),
                "second_scan_materialize_wall_ms_mean": mean(vals("second_scan_materialize_wall_ms")),
                "second_scan_materialize_work_ms_mean": mean(vals("second_scan_materialize_work_ms")),
                "second_computation_ms_mean": mean(vals("second_computation_ms")),
                "second_computation_ms_std": stdev(vals("second_computation_ms")),
                "pair_total_ms_mean": mean(vals("pair_total_ms")),
                "pair_load_ms_mean": mean(vals("pair_load_ms")),
                "pair_scan_materialize_wall_ms_mean": mean(vals("pair_scan_materialize_wall_ms")),
                "pair_scan_materialize_work_ms_mean": mean(vals("pair_scan_materialize_work_ms")),
                "pair_computation_ms_mean": mean(vals("pair_computation_ms")),
                "second_join_reuse_hit_bytes_mean": mean(vals("second_join_reuse_hit_bytes")),
                "reuse_hit_ratio_vs_first_retained_mean": mean(vals("reuse_hit_ratio_vs_first_retained")),
            }
        )
    write_csv(path, out, fields)


def write_second_query_latency_summary(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["previous_query"]), str(row["second_query"]))].append(row)
    fields = [
        "variant",
        "previous_query",
        "second_query",
        "repeats_completed",
        "second_total_ms_mean",
        "second_total_ms_std",
        "second_load_ms_mean",
        "second_load_ms_std",
        "second_scan_materialize_wall_ms_mean",
        "second_scan_materialize_work_ms_mean",
        "second_computation_ms_mean",
        "second_computation_ms_std",
        "second_join_reuse_hit_bytes_mean",
        "reuse_hit_ratio_vs_first_retained_mean",
    ]
    out = []
    for (variant, qi, qj), group in sorted(groups.items()):
        def vals(name: str) -> list[float]:
            return [to_float(str(row.get(name, ""))) for row in group if row.get(name, "") != ""]

        out.append(
            {
                "variant": variant,
                "previous_query": qi,
                "second_query": qj,
                "repeats_completed": len(group),
                "second_total_ms_mean": mean(vals("second_total_ms")),
                "second_total_ms_std": stdev(vals("second_total_ms")),
                "second_load_ms_mean": mean(vals("second_load_ms")),
                "second_load_ms_std": stdev(vals("second_load_ms")),
                "second_scan_materialize_wall_ms_mean": mean(vals("second_scan_materialize_wall_ms")),
                "second_scan_materialize_work_ms_mean": mean(vals("second_scan_materialize_work_ms")),
                "second_computation_ms_mean": mean(vals("second_computation_ms")),
                "second_computation_ms_std": stdev(vals("second_computation_ms")),
                "second_join_reuse_hit_bytes_mean": mean(vals("second_join_reuse_hit_bytes")),
                "reuse_hit_ratio_vs_first_retained_mean": mean(vals("reuse_hit_ratio_vs_first_retained")),
            }
        )
    write_csv(path, out, fields)


def write_second_query_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), str(row["previous_query"]), str(row["second_query"]))].append(row)

    def avg(variant: str, qi: str, qj: str, field: str) -> float | str:
        vals = [
            to_float(str(row.get(field, "")))
            for row in groups.get((variant, qi, qj), [])
            if row.get(field, "") != ""
        ]
        return mean(vals)

    fields = [
        "previous_query",
        "second_query",
        "baseline_repeats",
        "join_reuse_repeats",
        "baseline_second_total_ms",
        "join_reuse_second_total_ms",
        "total_ms_delta",
        "total_ms_speedup",
        "baseline_second_load_ms",
        "join_reuse_second_load_ms",
        "load_ms_delta",
        "load_ms_reduction_ratio",
        "baseline_second_computation_ms",
        "join_reuse_second_computation_ms",
        "computation_ms_delta",
        "join_reuse_hit_gb",
        "reuse_hit_ratio_vs_first_retained",
    ]
    out = []
    labels = [f"q{i}" for i in QUERIES]
    for qi in labels:
        for qj in labels:
            if qi == qj:
                continue
            b_total = avg("baseline", qi, qj, "second_total_ms")
            r_total = avg("join_reuse", qi, qj, "second_total_ms")
            b_load = avg("baseline", qi, qj, "second_load_ms")
            r_load = avg("join_reuse", qi, qj, "second_load_ms")
            b_comp = avg("baseline", qi, qj, "second_computation_ms")
            r_comp = avg("join_reuse", qi, qj, "second_computation_ms")
            hit_bytes = avg("join_reuse", qi, qj, "second_join_reuse_hit_bytes")
            hit_ratio = avg("join_reuse", qi, qj, "reuse_hit_ratio_vs_first_retained")
            out.append(
                {
                    "previous_query": qi,
                    "second_query": qj,
                    "baseline_repeats": len(groups.get(("baseline", qi, qj), [])),
                    "join_reuse_repeats": len(groups.get(("join_reuse", qi, qj), [])),
                    "baseline_second_total_ms": b_total,
                    "join_reuse_second_total_ms": r_total,
                    "total_ms_delta": float(r_total) - float(b_total) if b_total != "" and r_total != "" else "",
                    "total_ms_speedup": float(b_total) / float(r_total) if b_total != "" and r_total != "" and float(r_total) > 0 else "",
                    "baseline_second_load_ms": b_load,
                    "join_reuse_second_load_ms": r_load,
                    "load_ms_delta": float(r_load) - float(b_load) if b_load != "" and r_load != "" else "",
                    "load_ms_reduction_ratio": (float(b_load) - float(r_load)) / float(b_load) if b_load != "" and r_load != "" and float(b_load) > 0 else "",
                    "baseline_second_computation_ms": b_comp,
                    "join_reuse_second_computation_ms": r_comp,
                    "computation_ms_delta": float(r_comp) - float(b_comp) if b_comp != "" and r_comp != "" else "",
                    "join_reuse_hit_gb": float(hit_bytes) / 1e9 if hit_bytes != "" else "",
                    "reuse_hit_ratio_vs_first_retained": hit_ratio,
                }
            )
    write_csv(path, out, fields)


def write_reuse_heatmap_inputs(summary_dir: Path, rows: list[dict[str, object]]) -> None:
    join_rows = [row for row in rows if row.get("variant") == "join_reuse"]
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in join_rows:
        groups[(str(row["previous_query"]), str(row["second_query"]))].append(row)

    long_rows = []
    for (qi, qj), group in sorted(groups.items()):
        hit = [to_float(str(row.get("second_join_reuse_hit_bytes", ""))) for row in group]
        retained = [to_float(str(row.get("first_join_retained_bytes", ""))) for row in group]
        ratio = [
            to_float(str(row.get("reuse_hit_ratio_vs_first_retained", "")))
            for row in group
            if row.get("reuse_hit_ratio_vs_first_retained", "") != ""
        ]
        long_rows.append(
            {
                "previous_query": qi,
                "second_query": qj,
                "repeats_completed": len(group),
                "reuse_hit_gb_mean": mean(hit) / 1e9 if hit else "",
                "first_retained_gb_mean": mean(retained) / 1e9 if retained else "",
                "reuse_hit_ratio_mean": mean(ratio),
            }
        )
    write_csv(
        summary_dir / "join_reuse_heatmap_long.csv",
        long_rows,
        [
            "previous_query",
            "second_query",
            "repeats_completed",
            "reuse_hit_gb_mean",
            "first_retained_gb_mean",
            "reuse_hit_ratio_mean",
        ],
    )
    for column in ("reuse_hit_gb_mean", "first_retained_gb_mean", "reuse_hit_ratio_mean"):
        write_matrix(summary_dir / f"{column}_matrix.csv", long_rows, column)


def write_matrix(path: Path, rows: list[dict[str, object]], column: str) -> None:
    lookup = {(row["previous_query"], row["second_query"]): row.get(column, "") for row in rows}
    labels = [f"q{i}" for i in QUERIES]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + labels)
        for qi in labels:
            writer.writerow([qi] + [lookup.get((qi, qj), "") for qj in labels])


def plot_heatmaps(summary_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"WARNING: could not import plotting libraries: {exc}", flush=True)
        return
    specs = [
        ("reuse_hit_gb_mean_matrix.csv", "join_reuse_hit_gb_heatmap.png", "JOIN output reused by second query", "GB", "YlOrRd"),
        ("first_retained_gb_mean_matrix.csv", "join_retained_gb_heatmap.png", "JOIN output retained after first query", "GB", "YlOrRd"),
        ("reuse_hit_ratio_mean_matrix.csv", "join_reuse_ratio_heatmap.png", "JOIN reuse hit ratio", "hit bytes / first retained bytes", "viridis"),
    ]
    for csv_name, png_name, title, cbar_label, cmap in specs:
        matrix = read_csv(summary_dir / csv_name)
        labels = [f"q{i}" for i in QUERIES]
        values = []
        for row in matrix:
            values.append([math.nan if row.get(q, "") == "" else float(row[q]) for q in labels])
        arr = np.array(values, dtype=float)
        masked = np.ma.masked_invalid(arr)
        fig, ax = plt.subplots(figsize=(13, 11))
        im = ax.imshow(masked, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Second query (Qj)")
        ax.set_ylabel("Previous query (Qi)")
        ax.set_title(title)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value = arr[i, j]
                if math.isfinite(value):
                    text = f"{value:.2f}" if "ratio" in csv_name else f"{value:.1f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")
        fig.tight_layout()
        fig.savefig(summary_dir / png_name, dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")
    pairs = parse_pairs(args.pairs, args.allow_diagonal)
    # wdy start
    variants = parse_variants(args)
    # wdy end
    run_root = args.output.resolve()
    config_dir = run_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config_path, args)

    (run_root / "README.md").write_text(
        f"""# Automatic motivation latency

Created: {datetime.now().isoformat(timespec='seconds')}
Input: `{args.input}`
Repeats: `{args.repeats}`
Pairs: `{len(pairs)} ordered pairs`
Variants: `{', '.join(variants)}`
Join retention limit bytes: `{args.join_retention_limit_bytes}`
Join retention max batch bytes: `{args.join_retention_max_batch_bytes}`

Freshness rule: every pair repeat is executed in a separate process.

Main outputs after/while the run:
- `summary/query_latency.csv`
  - `load_ms` is `min(scan_materialize_wall_ms, total_ms)` for a bounded wall-clock load estimate.
  - `scan_materialize_wall_ms` is wall-clock union of parquet materialize intervals.
  - `scan_materialize_work_ms` is summed materialize work across parallel tasks.
- `summary/pair_latency.csv`
- `summary/pair_latency_summary.csv`
- `summary/join_reuse_heatmap_long.csv`
- `summary/join_reuse_hit_gb_heatmap.png`
- `summary/join_retained_gb_heatmap.png`
- `summary/join_reuse_ratio_heatmap.png`
"""
    )

    if not args.skip_build:
        env = make_env(args, config_path, "baseline")
        code = run_cmd(["pixi", "run", "make", "-j4"], env=env, cwd=REPO_ROOT, timeout_s=None, dry_run=args.dry_run)
        if code != 0:
            raise SystemExit(code)

    for variant in variants:
        variant_root = run_root / variant
        pair_root = variant_root / "pairs"
        pair_root.mkdir(parents=True, exist_ok=True)
        failed_path = variant_root / "failed_runs.csv"
        if not failed_path.exists():
            write_csv(failed_path, [], ["variant", "previous_query", "second_query", "repeat", "status", "returncode"])
        env = make_env(args, config_path, variant)
        total = len(pairs) * args.repeats
        current = 0
        for qi, qj in pairs:
            for repeat in range(args.repeats):
                current += 1
                spec = RunSpec(variant, qi, qj, repeat)
                bench = pair_root / spec.name
                csv_path = bench / "csv" / "runtimes.csv"
                if runtime_csv_complete(csv_path):
                    print(f"[SKIP] {variant} {spec.name} ({current}/{total})", flush=True)
                    continue
                print(f"[RUN] {variant} {spec.name} ({current}/{total})", flush=True)
                # wdy start
                perf_mode = "grouped" if variant == "pinned_hot" else "sequential"
                cmd = [
                    "pixi",
                    "run",
                    "python",
                    "test/tpch_performance/performance_test.py",
                    "--input",
                    str(args.input.resolve()),
                    "--engine",
                    "gpu",
                    "--mode",
                    perf_mode,
                    "--iterations",
                    "1",
                    "--queries",
                    f"{qi},{qj}",
                    "--config",
                    str(config_path),
                    "--output",
                    str(pair_root),
                    "--name",
                    spec.name,
                ]
                if variant == "pinned_hot":
                    cmd.extend(["--pin", "gpu"])
                # wdy end
                code = run_cmd(cmd, env=env, cwd=REPO_ROOT, timeout_s=args.pair_timeout, dry_run=args.dry_run)
                if code != 0 or not runtime_csv_complete(csv_path):
                    with failed_path.open("a", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=["variant", "previous_query", "second_query", "repeat", "status", "returncode"],
                        )
                        writer.writerow(
                            {
                                "variant": variant,
                                "previous_query": f"q{qi}",
                                "second_query": f"q{qj}",
                                "repeat": repeat,
                                "status": "failed_or_partial",
                                "returncode": code,
                            }
                        )
                    print(f"[FAILED] {variant} {spec.name} returncode={code}", flush=True)
        summarize_results(run_root, [v for v in variants if (run_root / v).exists()], args.no_plots)

    summarize_results(run_root, [v for v in variants if (run_root / v).exists()], args.no_plots)
    print(f"==> Done: {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
