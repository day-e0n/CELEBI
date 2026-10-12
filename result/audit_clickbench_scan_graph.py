#!/usr/bin/env python3
"""Independently compare scan Computing intervals, scan-stage logs and query wall time."""
from __future__ import annotations

import collections
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from parse_quent_operator_breakdown import iter_ndjson, load_telemetry, resolve_query_label


def union_seconds(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / 1e9


rows = []
for condition in ("baseline", "paging_lru", "reorder_future"):
    case = ROOT / f"experiment/FULL_CACHE_PIN_20261007/runs/latency/cb_r1/{condition}"
    telemetry = case / "telemetry_data"
    query_labels, plan_query, pipeline_plan, pipeline_chain = load_telemetry(telemetry)
    events, task_pipeline = collections.defaultdict(list), {}
    for record in iter_ndjson(sorted(telemetry.rglob("task/*.ndjson"))):
        state = record.get("data", {}).get("state", {})
        events[record["id"]].append((record["timestamp"], state))
        if isinstance(state, dict) and "Created" in state:
            task_pipeline[record["id"]] = state["Created"].get("pipeline_uuid", "")
    intervals = collections.defaultdict(list)
    terminal_scan = collections.Counter()
    for task_id, task_events in events.items():
        pipeline = task_pipeline.get(task_id, "")
        label = resolve_query_label(pipeline, plan_query, pipeline_plan, query_labels)
        chain = pipeline_chain.get(pipeline, {})
        task_events.sort(key=lambda e: e[0])
        for i, (timestamp, state) in enumerate(task_events):
            if not isinstance(state, dict) or "Computing" not in state:
                continue
            if chain.get(state["Computing"].get("current_operator_id")) != "GPU_SCAN":
                continue
            assert i + 1 < len(task_events), (task_id, label)
            next_timestamp, next_state = task_events[i + 1]
            assert next_timestamp >= timestamp
            intervals[label].append((timestamp, next_timestamp))
            if not (isinstance(next_state, dict) and "Computing" in next_state):
                terminal_scan[label] += 1
    stage_us, ooms = collections.Counter(), collections.defaultdict(list)
    for log in (case / "log_dir").glob("*.log"):
        label = None
        for line_no, line in enumerate(log.read_text().splitlines(), 1):
            match = re.search(r"QueryBegin: SQL: CALL sirius_set_query_label\('([^']+)'\)", line)
            if match:
                label = match[1]
            match = re.search(r"\[scan-stage\] materialize_us=(\d+) postfilter_us=(\d+) make_batch_us=(\d+)", line)
            if match and label:
                stage_us[label] += sum(int(v) for v in match.groups())
            if "OOM at operator" in line and label:
                ooms[label].append(dict(file=str(log), line=line_no, message=line))
    with (case / "queries.csv").open() as source:
        query_records = list(csv.DictReader(source))
    plotted = list(csv.DictReader(open(ROOT / "result/cumulative_scan_work_full_samples.csv")))
    for query in query_records:
        execution = int(query["execution"])
        if execution == 1:
            continue
        q = int(query["query"])
        label = query["label"]
        segments = intervals[label]
        summed = sum(end - start for start, end in segments) / 1e9
        displayed = [float(r["scan_work_s"]) for r in plotted if r["workload"] == "ClickBench"
                     and r["condition"] == condition and int(r["execution"]) == execution
                     and r["query"] == f"q{q}"]
        assert len(displayed) == 1 and abs(summed - displayed[0]) < 1e-8, (label, summed, displayed)
        union = union_seconds(segments)
        wall = float(query["wall_s"])
        assert union <= wall + 0.02, (label, union, wall)
        rows.append(dict(condition=condition, query=f"q{q}", execution=execution,
                         scan_work_s=summed, scan_interval_union_s=union,
                         scan_stage_logged_sum_s=stage_us[label] / 1e6, query_wall_s=wall,
                         scan_intervals=len(segments), scan_intervals_ending_without_next_operator=terminal_scan[label],
                         oom_count=len(ooms[label]), oom_records=json.dumps(ooms[label])))
    for q in (28, 29):
        group = [r for r in rows if r["condition"] == condition and r["query"] == f"q{q}"]
        print(condition, q, {k: statistics.mean(r[k] for r in group)
                             for k in ("scan_work_s", "scan_interval_union_s", "scan_stage_logged_sum_s", "query_wall_s", "oom_count")})
out = ROOT / "result/clickbench_scan_graph_audit.csv"
with out.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Validated {len(rows)} steady-state query samples against independent raw interval reconstruction.")
print(out)
