#!/usr/bin/env python3
"""Every query, every task state, summed and with overlap removed.

For each query this prints how long its tasks spent in each lifecycle state --
Queued, Reserving, Computing and the rest -- two ways: added up across tasks, and
with simultaneous intervals counted once. The gap between the two is how much the
tasks overlapped, which is why summed operator time can exceed wall clock.

  pixi run -e duckdb-python python scripts/breakdown_all_queries.py <run> [exec]
"""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
STATES = ["Queued", "Routing", "Reserving", "Downgrading", "Preparing", "Computing"]


def state_name(state) -> str:
    return next(iter(state)) if isinstance(state, dict) and state else str(state)


def union_s(items) -> float:
    merged = []
    for s, e in sorted(items):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged) / 1e9


def main() -> None:
    run = sys.argv[1]
    want_exec = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("P", here / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)

    events = collections.defaultdict(list)
    pipeline_of = {}
    for path in sorted(telemetry.rglob("task/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                st = obj.get("data", {}).get("state")
                events[obj["id"]].append((obj["timestamp"], st))
                if isinstance(st, dict) and "Created" in st and isinstance(st["Created"], dict):
                    pipeline_of[obj["id"]] = st["Created"].get("pipeline_uuid", "")

    # query -> state -> list of (start, end);  query -> operator -> same
    per_q = collections.defaultdict(lambda: collections.defaultdict(list))
    per_op = collections.defaultdict(lambda: collections.defaultdict(list))
    ntask = collections.Counter()
    for tid, series in events.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed:
            continue
        _, qnum, ex = parsed
        if ex != want_exec:
            continue
        q = f"q{qnum}"
        ntask[q] += 1
        series.sort(key=lambda e: e[0])
        chain = pipeline_chain.get(pid, {})
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            span = (ts, series[i + 1][0])
            per_q[q][state_name(st)].append(span)
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                kind = chain.get(st["Computing"].get("current_operator_id"), "UNKNOWN")
                per_op[q][kind].append(span)

    order = sorted(per_q, key=lambda q: -union_s([x for v in per_q[q].values() for x in v]))
    print(f"== {run}  execution {want_exec} ==\n")
    head = f"{'쿼리':6}{'태스크':>6}{'실제시간':>10}" + "".join(f"{s[:9]:>19}" for s in STATES)
    print(head)
    print(f"{'':12}{'(겹침제거)':>10}" + "".join(f"{'합계 / 겹침제거':>19}" for _ in STATES))
    print("-" * len(head))
    tot = collections.defaultdict(lambda: [0.0, 0.0])
    for q in order:
        all_spans = [x for v in per_q[q].values() for x in v]
        row = f"{q:6}{ntask[q]:6}{union_s(all_spans):9.2f}s"
        for st in STATES:
            items = per_q[q].get(st, [])
            s = sum(e - a for a, e in items) / 1e9
            u = union_s(items)
            tot[st][0] += s
            tot[st][1] += u
            row += f"{s:10.2f}/{u:<8.2f}" if s >= 0.005 else f"{'-':>19}"
        print(row)
    print("-" * len(head))
    row = f"{'합계':6}{sum(ntask.values()):6}{'':10}"
    for st in STATES:
        row += f"{tot[st][0]:10.2f}/{tot[st][1]:<8.2f}"
    print(row)

    print(f"\n연산자별 (Computing 안에서)")
    print(f"  {'연산자':20}{'합계':>10}{'겹침제거':>11}{'겹침배수':>10}")
    agg = collections.defaultdict(list)
    for q, ops in per_op.items():
        for k, v in ops.items():
            agg[k].extend(v)
    for k, v in sorted(agg.items(), key=lambda kv: -sum(e - s for s, e in kv[1])):
        s = sum(e - a for a, e in v) / 1e9
        u = union_s(v)
        print(f"  {k:20}{s:9.2f}s{u:10.2f}s{s/u if u else 0:9.2f}x")


if __name__ == "__main__":
    main()
