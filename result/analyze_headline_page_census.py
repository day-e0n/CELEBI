#!/usr/bin/env python3
"""Audit the EXACT three conditions in headline_cache_reorder_v2.png.

Uses existing FA_{th,cb}_{base,lru,ro_lru}_r* logs only. Execution 1 is
warmup; statistics use executions 2 and 3 (TPC-H: 3 runs, ClickBench: 2).

Page census counters are cumulative and printed repeatedly, including for
SET statements. For each column, subtract the final snapshot of execution
e-1 from the final snapshot of e. Never sum census lines directly.

Inserted/served/evicted bytes describe page ALLOCATION traffic, including
padding. Served includes full-column and partial-row-range cache serving.
It is not a request hit rate, useful payload bytes, or physical I/O avoided.
read_us + ps_read_us and other scan timers SUM overlapping scan tasks; they
must not be subtracted directly from query wall time. audit compressed_bytes
is planned demand (unchanged even on full hits), NOT actual bytes read.
Row totals sum operator inputs, so repeated table scans/self joins count again.

No page/lookup traces were enabled for these runs. Admission and eviction
summaries exist, but exact page identities and reasons for each miss do not.

Run: pixi run -e duckdb-python python result/analyze_headline_page_census.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {"th": range(1, 4), "cb": range(1, 3)}
CONDITIONS = ("base", "lru", "ro_lru")
GIB = 1024**3
CENSUS_COUNTERS = ("in_pages", "in_mib", "served_pages", "served_mib",
                   "evicted_pages", "evicted_mib")
LABEL = re.compile(r"sirius_set_query_label\('\w+_q(\d+)_exec(\d+)'\)")
NUMERIC = re.compile(r"(\w+)=(\d+(?:\.\d+)?)")
WALL = re.compile(r"exec=(\d+)/\d+ .* q(\d+) done wall=([\d.]+)s")
SCAN_TIMERS = ("read_us", "ps_read_us", "lookup_us", "materialize_us", "assemble_us",
               "ps_peek_us", "ps_assemble_us")


def values(line: str) -> dict[str, float]:
    return {k: float(v) for k, v in NUMERIC.findall(line)}


def mean_counters(samples):
    keys = set().union(*(x.keys() for x in samples))
    return {key: statistics.mean(x.get(key, 0) for x in samples) for key in sorted(keys)}


def read_run(run: str):
    folder = ROOT / "experiment" / run
    logs = list(folder.glob("*/log_dir/sirius_*.log"))
    if len(logs) != 1:
        raise ValueError(f"{run}: expected one engine log, found {len(logs)}")
    summary = {e: collections.Counter() for e in (2, 3)}
    queries = collections.defaultdict(collections.Counter)
    census = {e: {} for e in (1, 2, 3)}
    traces = collections.Counter()
    pressure_free = {e: [] for e in (2, 3)}
    pressure_target = {e: [] for e in (2, 3)}
    execution = query = None
    for line in logs[0].open(errors="replace"):
        match = LABEL.search(line)
        if match:
            query, execution = map(int, match.groups())
            continue
        for marker in ("[page-trace]", "[scan-cache]", "[scan-cache-decision]"):
            if marker in line:
                traces[marker] += 1
        if execution not in census:
            continue
        v = values(line)
        if "[page-census]   column=" in line:
            column = re.search(r"column='([^']+)'", line)[1]
            census[execution][column] = {k: v[k] for k in CENSUS_COUNTERS}
        if execution not in summary:
            continue
        dests = (summary[execution], queries[(query, execution)])
        if "[scan-audit] parquet_materialize" in line:
            for dest in dests:
                dest["scans"] += 1
                dest["full_hit_scans"] += v["full_hit"]
                dest["materialized_rows"] += v["output_rows"]
                dest["materialized_cells"] += v["output_rows"] * v["output_columns"]
                dest["planned_compressed_bytes"] += v["compressed_bytes"]
                dest["read_task_seconds"] += (v.get("read_us", 0) + v.get("ps_read_us", 0)) / 1e6
                dest["cache_serve_task_seconds"] += sum(v.get(k, 0) for k in SCAN_TIMERS[2:]) / 1e6
        elif "[scan-populate] store=pages us=" in line:
            for dest in dests:
                dest["populate_task_seconds"] += v["us"] / 1e6
        elif "[scan-stage]" in line:
            for dest in dests:
                dest["emitted_rows"] += v["rows"]
                dest["postfilter_task_seconds"] += v["postfilter_us"] / 1e6
        elif "[fixed-page-cache] partial_serve" in line:
            for dest in dests:
                dest["partial_serve_scans"] += 1
                dest["partial_serve_successes"] += v["used"]
                dest["partial_serve_read_cells"] += v["read_cells"]
                dest["partial_serve_total_cells"] += v["total_cells"]
        elif "[page-store] admission declined:" in line:
            for dest in dests:
                dest["admission_declined_batches"] += 1
                dest["admission_declined_columns"] += v["columns"]
        elif "[page-store] admission capped:" in line:
            for dest in dests:
                dest["admission_capped_batches"] += 1
        elif "[page-store] device pressure" in line:
            pressure_free[execution].append(v["free"] / GIB)
            target = re.search(r"target=(\d+)", line)
            if target:
                pressure_target[execution].append(int(target[1]) / GIB)
            for dest in dests:
                dest["pressure_events"] += 1
        elif "[page-store] pressure eviction freed nothing visible" in line:
            for dest in dests:
                dest["pressure_events_no_free_change"] += 1
        elif "[page-store] evicted pages=" in line:
            for dest in dests:
                dest["eviction_events"] += 1
                dest["evicted_pages_logged"] += v["pages"]

    for line in (folder / "runner.log").read_text().splitlines():
        match = WALL.search(line)
        if match and int(match[1]) in summary:
            e, q, wall = int(match[1]), int(match[2]), float(match[3])
            summary[e]["query_wall_seconds"] += wall
            queries[(q, e)]["query_wall_seconds"] += wall

    delta = {}
    for e in (2, 3):
        delta[e] = {}
        for col, snapshot in census[e].items():
            previous = census[e - 1].get(col, {})
            counts = {k: snapshot[k] - previous.get(k, 0) for k in CENSUS_COUNTERS}
            if any(v < 0 for v in counts.values()):
                raise ValueError(f"{run}: cumulative counter decreased: exec{e}, {col}, {counts}")
            delta[e][col] = counts
        for counter in CENSUS_COUNTERS:
            summary[e][counter] = sum(x[counter] for x in delta[e].values())
        summary[e]["pressure_free_min_gib"] = min(pressure_free[e], default=0)
        summary[e]["pressure_free_max_gib"] = max(pressure_free[e], default=0)
        summary[e]["pressure_target_min_gib"] = min(pressure_target[e], default=0)
        summary[e]["pressure_target_max_gib"] = max(pressure_target[e], default=0)
    return summary, queries, delta, traces


def write_rows(path, rows, first):
    fields = first + sorted(set().union(*(row.keys() for row in rows)) - set(first))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    column_rows, summary_rows, query_rows, execution_rows = [], [], [], []
    for bench, runs in WORKLOADS.items():
        for condition in CONDITIONS:
            sample_summaries, sample_queries, sample_columns = [], collections.defaultdict(list), []
            trace_counts = collections.Counter()
            for run_number in runs:
                run = f"FA_{bench}_{condition}_r{run_number}"
                summary, queries, columns, traces = read_run(run)
                trace_counts.update(traces)
                for e in (2, 3):
                    sample_summaries.append(summary[e])
                    sample_columns.append(columns[e])
                    execution_rows.append({"benchmark": bench, "condition": condition,
                                           "run": run_number, "execution": e, **summary[e]})
                for (q, e), row in queries.items():
                    sample_queries[q].append(row)
            averages = mean_counters(sample_summaries)
            for key in ("pressure_free_min_gib", "pressure_target_min_gib"):
                averages[key] = min(row.get(key, 0) for row in sample_summaries)
            for key in ("pressure_free_max_gib", "pressure_target_max_gib"):
                averages[key] = max(row.get(key, 0) for row in sample_summaries)
            averages["query_wall_stdev_seconds"] = statistics.stdev(
                row["query_wall_seconds"] for row in sample_summaries)
            averages["served_per_insert"] = (
                averages["served_mib"] / averages["in_mib"] if averages["in_mib"] else 0)
            summary_rows.append({"benchmark": bench, "condition": condition,
                                 "samples": len(sample_summaries), **averages})
            for col in sorted(set().union(*(sample.keys() for sample in sample_columns))):
                averages_col = mean_counters([sample.get(col, {}) for sample in sample_columns])
                averages_col["served_per_insert"] = (
                    averages_col["served_mib"] / averages_col["in_mib"]
                    if averages_col["in_mib"] else 0)
                column_rows.append({"benchmark": bench, "condition": condition,
                                    "column": col, "samples": len(sample_columns), **averages_col})
            for q, samples in sorted(sample_queries.items()):
                query_rows.append({"benchmark": bench, "condition": condition,
                                   "query": q, "samples": len(samples), **mean_counters(samples)})
            print(f"{bench}/{condition}: wall={averages['query_wall_seconds']:.3f}s, "
                  f"insert={averages['in_mib'] / 1024:.3f}GiB, "
                  f"serve={averages['served_mib'] / 1024:.3f}GiB, "
                  f"serve/insert={averages['served_per_insert']:.3f}, "
                  f"trace_lines={dict(trace_counts)}")
    write_rows(args.output_dir / "headline_page_census_columns.csv", column_rows,
               ["benchmark", "condition", "column", "samples"])
    write_rows(args.output_dir / "headline_page_census_summary.csv", summary_rows,
               ["benchmark", "condition", "samples"])
    write_rows(args.output_dir / "headline_page_census_queries.csv", query_rows,
               ["benchmark", "condition", "query", "samples"])
    write_rows(args.output_dir / "headline_page_census_executions.csv", execution_rows,
               ["benchmark", "condition", "run", "execution"])


if __name__ == "__main__":
    main()
