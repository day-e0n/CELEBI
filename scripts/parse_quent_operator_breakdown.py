#!/usr/bin/env python3
"""Attribute Quent telemetry Task compute time to TPC-H query x operator-kind.

Chain used to attribute time (see docs/super-sirius/quent-telemetry.md for the
general telemetry model):

    Task.Computing.current_operator_id
      -> Operator.Declaration.instance_name  ("HASH_JOIN(27) -> PROJECTION(31) -> ...")
         parses into {operator_id: operator_kind}, and carries plan_id
      -> Plan.Declaration.parent.query_id
      -> Query.Init.instance_name             (our "{condition}_q{qnum}_exec{execution}" label,
                                                set via CALL sirius_set_query_label(...) in
                                                run_fixed_page_workload_sequence.py)

A Task's dwell time in each Computing entry is the delta to the *next* state
transition timestamp (state timestamps are nanoseconds since epoch), attributed
to whatever operator_id that Computing entry names. Operator kinds are bucketed
using the authoritative operator list in src/op/sirius_physical_operator_type.cpp.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

OPERATOR_RE = re.compile(r"([A-Z][A-Z0-9_]*)\((\d+)\)")

# Bucketed from the full SiriusPhysicalOperatorType list (src/op/sirius_physical_operator_type.cpp).
# Only kinds that can appear in a Super Sirius GPU pipeline chain are relevant here.
BUCKET_BY_KIND = {
    "GPU_SCAN": "scan",
    "TABLE_SCAN": "scan",
    "PARQUET_SCAN": "scan",
    "DUMMY_SCAN": "scan",
    "CHUNK_SCAN": "scan",
    "COLUMN_DATA_SCAN": "scan",
    "DELIM_SCAN": "scan",
    "DUCKDB_SCAN": "scan",
    "ICEBERG_SCAN": "scan",
    "CPU_SOURCE": "scan",
    "POSITIONAL_SCAN": "scan",
    "EXPRESSION_SCAN": "scan",
    "CTE_SCAN": "scan",
    "REC_CTE_SCAN": "scan",
    "REC_REC_CTE_SCAN": "scan",
    "HASH_JOIN": "join",
    "NESTED_LOOP_JOIN": "join",
    "BLOCKWISE_NL_JOIN": "join",
    "LEFT_DELIM_JOIN": "join",
    "RIGHT_DELIM_JOIN": "join",
    "PIECEWISE_MERGE_JOIN": "join",
    "IE_JOIN": "join",
    "ASOF_JOIN": "join",
    "CROSS_PRODUCT": "join",
    "POSITIONAL_JOIN": "join",
    "UNGROUPED_AGGREGATE": "aggregate",
    "HASH_GROUP_BY": "aggregate",
    "PERFECT_HASH_GROUP_BY": "aggregate",
    "PARTITIONED_AGGREGATE": "aggregate",
    "MERGE_GROUP_BY": "aggregate",
    "MERGE_AGGREGATE": "aggregate",
    "WINDOW": "aggregate",
    "STREAMING_WINDOW": "aggregate",
    "FILTER": "filter",
    "DYNAMIC_FILTER": "filter",
    "ORDER_BY": "sort",
    "TOP_N": "sort",
    "MERGE_TOP_N": "sort",
    "MERGE_SORT": "sort",
    "SORT_PARTITION": "sort",
    "SORT_SAMPLE": "sort",
    "LIMIT": "sort",
    "LIMIT_PERCENT": "sort",
    "STREAMING_LIMIT": "sort",
}
BUCKETS = ("scan", "join", "aggregate", "filter", "sort", "other")

LABEL_RE = re.compile(r"^(?P<condition>.+)_q(?P<qnum>\d+)(?:_s(?P<stream>\d+))?_exec(?P<execution>\d+)$")


def bucket_for(kind: str) -> str:
    return BUCKET_BY_KIND.get(kind, "other")


def iter_ndjson(paths: list[Path]):
    for path in paths:
        with path.open(errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def parse_operator_chain(instance_name: str) -> dict[int, str]:
    return {int(opid): kind for kind, opid in OPERATOR_RE.findall(instance_name)}


def load_telemetry(telemetry_dir: Path):
    """Build query_id->label, plan_id->query_id, operator_id(pipeline)->(plan_id, {opid:kind})."""

    query_label: dict[str, str] = {}
    for obj in iter_ndjson(sorted(telemetry_dir.rglob("query/*.ndjson"))):
        data = obj.get("data", {})
        state = data.get("state", {})
        if isinstance(state, dict) and "Init" in state:
            query_label[obj["id"]] = state["Init"].get("instance_name", "")

    plan_query: dict[str, str] = {}
    for obj in iter_ndjson(sorted(telemetry_dir.rglob("plan/*.ndjson"))):
        data = obj.get("data", {})
        decl = data.get("Declaration")
        if decl:
            parent = decl.get("parent") or {}
            qid = parent.get("query_id")
            if qid:
                plan_query[obj["id"]] = qid

    pipeline_plan: dict[str, str] = {}
    pipeline_chain: dict[str, dict[int, str]] = {}
    for obj in iter_ndjson(sorted(telemetry_dir.rglob("operator/*.ndjson"))):
        data = obj.get("data", {})
        decl = data.get("Declaration")
        if decl:
            pid = obj["id"]
            pipeline_plan[pid] = decl.get("plan_id", "")
            pipeline_chain[pid] = parse_operator_chain(decl.get("instance_name", ""))

    return query_label, plan_query, pipeline_plan, pipeline_chain


def resolve_query_label(pipeline_id: str, plan_query: dict[str, str], pipeline_plan: dict[str, str], query_label: dict[str, str]) -> str | None:
    plan_id = pipeline_plan.get(pipeline_id)
    if not plan_id:
        return None
    query_id = plan_query.get(plan_id)
    if not query_id:
        return None
    return query_label.get(query_id)


def accumulate_task_durations(
    telemetry_dir: Path,
    plan_query: dict[str, str],
    pipeline_plan: dict[str, str],
    pipeline_chain: dict[str, dict[int, str]],
    query_label: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Returns label -> operator_kind -> total_ms."""

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    task_pipeline: dict[str, str] = {}
    task_events: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for obj in iter_ndjson(sorted(telemetry_dir.rglob("task/*.ndjson"))):
        tid = obj["id"]
        ts = obj["timestamp"]
        data = obj.get("data", {})
        state = data.get("state", {})
        task_events[tid].append((ts, state))
        if isinstance(state, dict) and "Created" in state:
            task_pipeline[tid] = state["Created"].get("pipeline_uuid", "")

    for tid, events in task_events.items():
        pipeline_id = task_pipeline.get(tid)
        if not pipeline_id:
            continue
        label = resolve_query_label(pipeline_id, plan_query, pipeline_plan, query_label)
        if not label or label == "unnamed_query":
            continue
        chain = pipeline_chain.get(pipeline_id, {})
        events.sort(key=lambda e: e[0])
        for i, (ts, state) in enumerate(events):
            if not isinstance(state, dict) or "Computing" not in state:
                continue
            if i + 1 >= len(events):
                continue
            next_ts = events[i + 1][0]
            duration_ms = max(next_ts - ts, 0) / 1e6
            opid = state["Computing"].get("current_operator_id")
            kind = chain.get(opid, "UNKNOWN")
            totals[label][kind] += duration_ms

    return totals


def parse_label(label: str) -> tuple[str, int, int] | None:
    m = LABEL_RE.match(label)
    if not m:
        return None
    return m.group("condition"), int(m.group("qnum")), int(m.group("execution"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, required=True, help="Root telemetry_data directory for one run (may contain multiple engine sessions).")
    parser.add_argument("--out-csv", type=Path, required=True, help="Per (condition,query,execution,operator_kind) raw duration CSV.")
    parser.add_argument("--out-bucket-csv", type=Path, required=True, help="Per (condition,query,execution) pivoted bucket-ms CSV.")
    args = parser.parse_args()

    query_label, plan_query, pipeline_plan, pipeline_chain = load_telemetry(args.telemetry_dir)
    totals = accumulate_task_durations(args.telemetry_dir, plan_query, pipeline_plan, pipeline_chain, query_label)

    raw_rows: list[dict[str, object]] = []
    bucket_rows: dict[tuple[str, int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for label, kind_totals in totals.items():
        parsed = parse_label(label)
        if not parsed:
            continue
        condition, qnum, execution = parsed
        for kind, ms in kind_totals.items():
            raw_rows.append({
                "condition": condition,
                "query": f"q{qnum}",
                "execution": execution,
                "operator_kind": kind,
                "duration_ms": ms,
            })
            bucket_rows[(condition, qnum, execution)][bucket_for(kind)] += ms

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "query", "execution", "operator_kind", "duration_ms"])
        writer.writeheader()
        for row in sorted(raw_rows, key=lambda r: (r["condition"], int(str(r["query"]).lstrip("q")), r["execution"], r["operator_kind"])):
            writer.writerow(row)

    args.out_bucket_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_bucket_csv.open("w", newline="") as f:
        fields = ["condition", "query", "execution", *BUCKETS, "total_ms"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (condition, qnum, execution), kinds in sorted(bucket_rows.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            row = {"condition": condition, "query": f"q{qnum}", "execution": execution}
            for b in BUCKETS:
                row[b] = kinds.get(b, 0.0)
            row["total_ms"] = sum(kinds.values())
            writer.writerow(row)

    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_bucket_csv}")
    print(f"labeled queries found: {len(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
