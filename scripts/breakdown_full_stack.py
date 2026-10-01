#!/usr/bin/env python3
"""Everything the telemetry records, from a query's first instant to its last.

Three layers, all from the same ndjson stream:

  query  Init -> Planning -> Executing -> Exit        (per statement; Planning is
                                                       the CPU-side plan build)
  task   Created -> Queued -> Routing -> Reserving ->  (many per query, concurrent)
         [Downgrading] -> Preparing -> Computing ->
         Finalizing -> Exit
  operator  what a task was computing: GPU_SCAN, PROJECTION, HASH_GROUP_BY, ...

Each duration is given twice: summed over whatever was in that state, and with
simultaneous intervals counted once. Their ratio is the concurrency.

  pixi run -e duckdb-python python scripts/breakdown_full_stack.py <run> [exec]
"""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
TASK_STATES = ["Created", "Queued", "Routing", "Reserving", "Downgrading",
               "Preparing", "Computing", "Finalizing"]
QUERY_STATES = ["Init", "Planning", "Executing"]


def name_of(state) -> str:
    return next(iter(state)) if isinstance(state, dict) and state else str(state)


def union_s(items) -> float:
    merged = []
    for s, e in sorted(items):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(e - s for s, e in merged) / 1e9


def collect(telemetry: pathlib.Path, kind: str):
    """entity id -> sorted [(timestamp, state)]"""
    out = collections.defaultdict(list)
    for path in sorted(telemetry.rglob(f"{kind}/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                data = obj.get("data", {})
                if "state" in data:
                    out[obj["id"]].append((obj["timestamp"], data["state"]))
    for v in out.values():
        v.sort(key=lambda e: e[0])
    return out


def main() -> None:
    run = sys.argv[1]
    want_exec = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    here = pathlib.Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "P", here / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)

    # ---- query layer -------------------------------------------------
    queries = collect(telemetry, "query")
    qspans = collections.defaultdict(list)
    whole = []
    nq = 0
    for qid, series in queries.items():
        label = query_label.get(qid, "")
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != want_exec:
            continue
        nq += 1
        whole.append((series[0][0], series[-1][0]))
        for i, (ts, st) in enumerate(series):
            if i + 1 < len(series):
                qspans[name_of(st)].append((ts, series[i + 1][0]))

    # ---- task layer --------------------------------------------------
    tasks = collect(telemetry, "task")
    pipeline_of = {}
    for tid, series in tasks.items():
        for _, st in series:
            if isinstance(st, dict) and "Created" in st and isinstance(st["Created"], dict):
                pipeline_of[tid] = st["Created"].get("pipeline_uuid", "")
                break
    tspans = collections.defaultdict(list)
    ospans = collections.defaultdict(list)
    ntask = 0
    for tid, series in tasks.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        parsed = P.parse_label(label) if label else None
        if not parsed or parsed[2] != want_exec:
            continue
        ntask += 1
        chain = pipeline_chain.get(pid, {})
        for i, (ts, st) in enumerate(series):
            if i + 1 >= len(series):
                continue
            span = (ts, series[i + 1][0])
            tspans[name_of(st)].append(span)
            if isinstance(st, dict) and "Computing" in st and isinstance(st["Computing"], dict):
                ospans[chain.get(st["Computing"].get("current_operator_id"),
                                 "UNKNOWN")].append(span)

    def table(title, spans, order, count_label):
        print(f"\n{title}")
        print(f"  {'구간':24}{'횟수':>7}{'합계':>11}{'겹침제거':>11}{'동시':>8}")
        keys = [k for k in order if k in spans] + [k for k in spans if k not in order]
        for k in keys:
            items = spans[k]
            s = sum(e - a for a, e in items) / 1e9
            u = union_s(items)
            print(f"  {k:24}{len(items):7}{s:10.3f}s{u:10.3f}s{s/u if u else 0:7.2f}x")
        allit = [x for v in spans.values() for x in v]
        print(f"  {'-- 전체 --':24}{len(allit):7}{sum(e-a for a,e in allit)/1e9:10.3f}s"
              f"{union_s(allit):10.3f}s")
        print(f"  ({count_label})")

    print(f"================ {run}  execution {want_exec} ================")
    print(f"쿼리 {nq}개, 태스크 {ntask}개")
    print(f"\n쿼리 전체 구간 (Init 부터 Exit 까지) 합계 "
          f"{sum(e-s for s,e in whole)/1e9:.3f}s, 겹침제거 {union_s(whole):.3f}s")
    table("[1] 쿼리 계층", qspans, QUERY_STATES, f"쿼리 {nq}개")
    table("[2] 태스크 계층", tspans, TASK_STATES, f"태스크 {ntask}개")
    table("[3] 연산자 계층 (Computing 내부)", ospans, [], "Computing 구간을 연산자로 분해")


if __name__ == "__main__":
    main()
