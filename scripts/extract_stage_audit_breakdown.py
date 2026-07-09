#!/usr/bin/env python3
"""Extract per-operator stage timing from Sirius stage-audit logs.

The regular benchmark summary stores end-to-end latency. This script replays the
logs produced by scripts/run_fixed_page_all_queries.py and aggregates
`[stage-audit]` lines per query execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_RUN_ROOT = Path("experiment/fixed_page_runs/tpch22_5x_7gb_ws3g")
DEFAULT_GRAPH_DIR = Path("experiment/graph")
DEFAULT_PREFIX = "tpch22_5x_7gb_ws3g_stage"

STAGE_ORDER = [
    "SCAN",
    "FILTER",
    "PROJECTION",
    "JOIN",
    "PARTITION",
    "CONCAT",
    "AGGREGATE",
    "SORT",
    "RESULT",
    "OTHER",
]

QUERY_RE = re.compile(r"QueryBegin: SQL:\s*(.*)$")
TS_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def parse_timestamp_ms(line: str) -> float | None:
    match = TS_RE.match(line)
    if not match:
        return None
    dt = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp() * 1000.0


def parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in KV_RE.findall(line)}


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
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return max(total, 0.0)


def query_segments(log_dir: Path, execution_count: int) -> list[list[str]]:
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return [[] for _ in range(execution_count)]
    lines = logs[-1].read_text(errors="replace").splitlines()
    begins: list[int] = []
    for idx, line in enumerate(lines):
        match = QUERY_RE.search(line)
        if not match:
            continue
        sql = match.group(1).strip().lower()
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


def summarize_stage_lines(lines: list[str]) -> dict[str, Any]:
    work_ms = defaultdict(float)
    wall_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    input_bytes = defaultdict(int)
    output_bytes = defaultdict(int)
    task_count = defaultdict(int)
    operator_work_ms = defaultdict(float)
    all_intervals: list[tuple[float, float]] = []

    for line in lines:
        if "[stage-audit]" not in line:
            continue
        fields = parse_kv(line)
        stage = fields.get("stage_kind", "OTHER")
        if stage not in STAGE_ORDER:
            stage = "OTHER"
        operator = fields.get("operator_name", "UNKNOWN")
        duration_ms = int(fields.get("duration_us", "0")) / 1000.0
        work_ms[stage] += duration_ms
        operator_work_ms[operator] += duration_ms
        input_bytes[stage] += int(fields.get("input_bytes", "0"))
        output_bytes[stage] += int(fields.get("output_bytes", "0"))
        task_count[stage] += 1

        end_ms = parse_timestamp_ms(line)
        if end_ms is not None and duration_ms >= 0:
            interval = (end_ms - duration_ms, end_ms)
            wall_intervals[stage].append(interval)
            all_intervals.append(interval)

    row: dict[str, Any] = {
        "stage_total_work_ms": sum(work_ms.values()),
        "stage_total_wall_ms": union_interval_ms(all_intervals),
        "stage_task_count": sum(task_count.values()),
    }
    for stage in STAGE_ORDER:
        lower = stage.lower()
        row[f"{lower}_work_ms"] = work_ms[stage]
        row[f"{lower}_wall_ms"] = union_interval_ms(wall_intervals[stage])
        row[f"{lower}_input_gb"] = input_bytes[stage] / 1e9
        row[f"{lower}_output_gb"] = output_bytes[stage] / 1e9
        row[f"{lower}_task_count"] = task_count[stage]
    top_ops = sorted(operator_work_ms.items(), key=lambda kv: kv[1], reverse=True)[:8]
    row["top_operator_work_ms"] = ";".join(f"{name}:{value:.3f}" for name, value in top_ops)
    return row


def series_for(condition: str, execution: int) -> str | None:
    if condition == "cold_hot":
        return "cold" if execution == 1 else "hot"
    if condition == "pin_table":
        return "pin_table" if execution > 1 else None
    if condition == "paging":
        return "paging" if execution > 1 else None
    return condition


def read_case_rows(case_result: Path) -> list[dict[str, Any]]:
    data = json.loads(case_result.read_text())
    if data.get("status") != "ok":
        return []
    return data.get("rows", [])


def collect_execution_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_result in sorted(run_root.glob("*/*/case_result.json")):
        condition = case_result.parts[-3]
        query = case_result.parts[-2]
        run_rows = read_case_rows(case_result)
        segments = query_segments(case_result.parent / "log_dir", len(run_rows))
        for base, lines in zip(run_rows, segments):
            execution = int(base["execution"])
            series = series_for(condition, execution)
            stage = summarize_stage_lines(lines)
            rows.append({
                "query": query,
                "condition": condition,
                "series": series or "",
                "execution": execution,
                "included_in_series": 1 if series else 0,
                "total_ms": to_float(base.get("total_ms")),
                **stage,
            })
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "total_ms",
        "stage_total_work_ms",
        "stage_total_wall_ms",
        "stage_task_count",
    ]
    for stage in STAGE_ORDER:
        lower = stage.lower()
        numeric_fields.extend([
            f"{lower}_work_ms",
            f"{lower}_wall_ms",
            f"{lower}_input_gb",
            f"{lower}_output_gb",
            f"{lower}_task_count",
        ])

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("included_in_series"):
            continue
        grouped[(row["query"], row["series"])].append(row)

    out: list[dict[str, Any]] = []
    for (query, series), vals in sorted(grouped.items(), key=lambda kv: (int(kv[0][0][1:]), kv[0][1])):
        item: dict[str, Any] = {
            "query": query,
            "series": series,
            "samples": len(vals),
        }
        for field in numeric_fields:
            numbers = [to_float(v.get(field)) for v in vals]
            item[field] = mean(numbers) if numbers else 0.0
            item[f"{field}_std"] = stdev(numbers) if len(numbers) > 1 else ""
        total = to_float(item["total_ms"])
        for stage in STAGE_ORDER:
            lower = stage.lower()
            item[f"{lower}_work_share_of_latency"] = (
                to_float(item[f"{lower}_work_ms"]) / total if total > 0 else ""
            )
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_speedup_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    byq: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summary:
        byq[row["query"]][row["series"]] = row
    out: list[dict[str, Any]] = []
    for query, series in sorted(byq.items(), key=lambda kv: int(kv[0][1:])):
        hot = series.get("hot")
        paging = series.get("paging")
        if not hot or not paging:
            continue
        row = {
            "query": query,
            "hot_total_ms": hot["total_ms"],
            "paging_total_ms": paging["total_ms"],
            "total_speedup": to_float(hot["total_ms"]) / to_float(paging["total_ms"])
            if to_float(paging["total_ms"]) > 0 else "",
        }
        for stage in STAGE_ORDER:
            lower = stage.lower()
            hot_work = to_float(hot.get(f"{lower}_work_ms"))
            paging_work = to_float(paging.get(f"{lower}_work_ms"))
            row[f"hot_{lower}_work_ms"] = hot_work
            row[f"paging_{lower}_work_ms"] = paging_work
            row[f"{lower}_work_ratio_hot_over_paging"] = (
                hot_work / paging_work if paging_work > 0 else ""
            )
            row[f"{lower}_work_delta_ms"] = paging_work - hot_work
        out.append(row)
    write_csv(path, out)


def write_plots(graph_dir: Path, prefix: str, summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] plot skipped: {exc}")
        return

    graph_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted({row["query"] for row in summary}, key=lambda q: int(q[1:]))
    by_key = {(row["query"], row["series"]): row for row in summary}
    stages = ["SCAN", "FILTER", "JOIN", "PARTITION", "AGGREGATE", "PROJECTION", "OTHER"]
    colors = {
        "SCAN": "#4c78a8",
        "FILTER": "#f58518",
        "JOIN": "#54a24b",
        "PARTITION": "#b279a2",
        "AGGREGATE": "#e45756",
        "PROJECTION": "#72b7b2",
        "OTHER": "#bab0ac",
    }

    for series in ["hot", "paging"]:
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.62), 5.5))
        bottoms = [0.0 for _ in labels]
        for stage in stages:
            lower = stage.lower()
            vals = [to_float(by_key.get((q, series), {}).get(f"{lower}_work_ms")) for q in labels]
            ax.bar(labels, vals, bottom=bottoms, label=stage, color=colors[stage])
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_title(f"TPC-H {series}: stage work from stage-audit")
        ax.set_ylabel("Stage work (ms, summed task duration)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(ncols=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(graph_dir / f"{prefix}_{series}_stage_work.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.62), 5.5))
    x = range(len(labels))
    width = 0.38
    for idx, series in enumerate(["hot", "paging"]):
        xs = [i + (idx - 0.5) * width for i in x]
        vals = [to_float(by_key.get((q, series), {}).get("scan_work_ms")) for q in labels]
        ax.bar(xs, vals, width, label=series)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("TPC-H scan work: hot vs paging")
    ax.set_ylabel("SCAN work (ms, summed task duration)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(graph_dir / f"{prefix}_scan_work_hot_vs_paging.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    execution_rows = collect_execution_rows(args.run_root)
    summary_rows = aggregate_rows(execution_rows)

    write_csv(args.graph_dir / f"{args.prefix}_stage_execution.csv", execution_rows)
    write_csv(args.graph_dir / f"{args.prefix}_stage_summary.csv", summary_rows)
    write_speedup_csv(args.graph_dir / f"{args.prefix}_stage_speedup.csv", summary_rows)
    write_plots(args.graph_dir, args.prefix, summary_rows)

    print(f"==> execution: {args.graph_dir / (args.prefix + '_stage_execution.csv')}")
    print(f"==> summary: {args.graph_dir / (args.prefix + '_stage_summary.csv')}")
    print(f"==> speedup: {args.graph_dir / (args.prefix + '_stage_speedup.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
