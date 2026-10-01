#!/usr/bin/env python3
"""Every state every task passed through, for one query's execution.

A task moves Created -> Queued -> Routing -> Reserving -> Preparing -> Computing ->
Finalizing -> Exit, and several tasks are in flight at once. The per-operator totals
the other scripts print are sums over those tasks, so they exceed the wall clock
whenever tasks overlap. This prints the raw sequence instead: who ran, when, in which
state, and how many were in Computing at the same instant.

  pixi run -e duckdb-python python scripts/breakdown_one_query.py <run> <query> [exec]
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "experiment"
ORDER = ["Created", "Queued", "Routing", "Reserving", "Downgrading", "Preparing",
         "Computing", "Finalizing", "Exit"]


def state_name(state) -> str:
    if isinstance(state, dict) and state:
        return next(iter(state))
    return str(state)


def load(run: str):
    telemetry = next((ROOT / run).glob("*/telemetry_data"))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "P", pathlib.Path(__file__).resolve().parent / "parse_quent_operator_breakdown.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    query_label, plan_query, pipeline_plan, pipeline_chain = P.load_telemetry(telemetry)

    events = collections.defaultdict(list)
    pipeline_of = {}
    for path in sorted(telemetry.rglob("task/*.ndjson")):
        with path.open() as handle:
            for line in handle:
                obj = json.loads(line)
                state = obj.get("data", {}).get("state")
                events[obj["id"]].append((obj["timestamp"], state))
                if isinstance(state, dict) and "Created" in state:
                    created = state["Created"]
                    if isinstance(created, dict):
                        pipeline_of[obj["id"]] = created.get("pipeline_uuid", "")
    return events, pipeline_of, pipeline_plan, plan_query, query_label, pipeline_chain, P


def main() -> None:
    run, query = sys.argv[1], sys.argv[2]
    want_exec = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    events, pipeline_of, pipeline_plan, plan_query, query_label, chain, P = load(run)

    # Keep only tasks whose pipeline resolves to this query and execution.
    mine = {}
    for tid, series in events.items():
        pid = pipeline_of.get(tid)
        if not pid:
            continue
        label = P.resolve_query_label(pid, plan_query, pipeline_plan, query_label)
        if not label:
            continue
        parsed = P.parse_label(label)
        if not parsed:
            continue
        _, q, ex = parsed
        if f"q{q}" == query and ex == want_exec:
            mine[tid] = sorted(series, key=lambda e: e[0])
    if not mine:
        raise SystemExit(f"{run}: no tasks for {query} exec{want_exec}")

    t0 = min(s[0][0] for s in mine.values())
    print(f"== {run}  {query}  execution {want_exec} ==")
    print(f"   태스크 {len(mine)}개\n")

    # 1. state totals, summed and with overlap removed
    spans = collections.defaultdict(list)
    for tid, series in mine.items():
        for i, (ts, state) in enumerate(series):
            if i + 1 < len(series):
                spans[state_name(state)].append((ts, series[i + 1][0]))

    def union(items):
        merged = []
        for s, e in sorted(items):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return sum(e - s for s, e in merged) / 1e9

    print(f"   {'상태':14}{'횟수':>6}{'합계':>10}{'겹침제거':>10}{'동시':>7}")
    for name in ORDER:
        if name not in spans:
            continue
        items = spans[name]
        total = sum(e - s for s, e in items) / 1e9
        u = union(items)
        print(f"   {name:14}{len(items):6}{total:9.3f}s{u:9.3f}s{total/u if u else 0:6.2f}x")
    allspans = [x for v in spans.values() for x in v]
    print(f"   {'전체':14}{len(allspans):6}{sum(e-s for s,e in allspans)/1e9:9.3f}s"
          f"{union(allspans):9.3f}s")

    # 2. timeline per task
    print(f"\n   태스크별 타임라인 (쿼리 시작 기준 초)")
    print(f"   {'태스크':>9} {'상태':13}{'시작':>9}{'길이':>10}  연산자")
    for tid in sorted(mine, key=lambda t: mine[t][0][0])[:14]:
        series = mine[tid]
        for i, (ts, state) in enumerate(series):
            if i + 1 >= len(series):
                continue
            dur = (series[i + 1][0] - ts) / 1e6
            if dur < 1.0:
                continue
            op = ""
            if isinstance(state, dict) and "Computing" in state:
                c = state["Computing"]
                if isinstance(c, dict):
                    pid = pipeline_of.get(tid, "")
                    op = chain.get(pid, {}).get(c.get("current_operator_id"), "")
            print(f"   {tid[-8:]:>9} {state_name(state):13}{(ts-t0)/1e9:8.3f}s{dur:9.1f}ms  {op}")

    # 3. how many tasks were in Computing at once, over time
    comp = sorted(spans.get("Computing", []))
    if comp:
        points = sorted({t for span in comp for t in span})
        hist = collections.Counter()
        for a, b in zip(points, points[1:]):
            n = sum(1 for s, e in comp if s <= a < e)
            hist[n] += (b - a) / 1e9
        print(f"\n   동시에 Computing 중이던 태스크 수의 시간 분포")
        for n in sorted(hist):
            if n == 0:
                continue
            print(f"     {n}개 동시: {hist[n]:7.3f}s")
        idle = hist.get(0, 0.0)
        print(f"     아무도 계산 안 함: {idle:.3f}s")


if __name__ == "__main__":
    main()
