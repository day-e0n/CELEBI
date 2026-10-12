#!/usr/bin/env python3
"""Conditional full-cache route supply completion, independent of byte coverage."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiment/FULL_CACHE_PIN_20261007/runs/latency"
OUTPUT = ROOT / "result/conditional_cache_supply"
LABEL = re.compile(r"sirius_set_query_label\('([^']*)'\)")
QUERY = re.compile(r"(.+)_q(\d+)_exec(\d+)$")
FIELD = re.compile(r"(\w+)=('(?:[^']*)'|\S+)")
CONDITIONS = ("paging_lru", "reorder_future")


def collect_case(workload, condition):
    case = INPUT / f"{workload}_r1" / condition
    metadata = json.loads((case / "metadata.json").read_text())
    if metadata["status"] != "complete" or metadata["comparison_valid"] is not True:
        raise ValueError(f"Case incomplete/invalid: {case}")
    expected_queries = set(range(1, 23 if workload == "th" else 44))
    if set(metadata["order"]) != expected_queries or len(metadata["order"]) != len(expected_queries):
        raise ValueError(f"Not a full workload: {case}")
    expected_future = "evict" if condition == "reorder_future" else "off"
    if metadata["future_mode"] != expected_future:
        raise ValueError(f"Wrong future mode: {case}")
    with (case / "queries.csv").open(newline="") as source:
        records = list(csv.DictReader(source))
    expected = {(q, ex) for q in expected_queries for ex in (1, 2, 3)}
    if (len(records) != len(expected)
            or {(int(r["query"]), int(r["execution"])) for r in records} != expected
            or any(r["status"] != "ok" for r in records)):
        raise ValueError(f"Incomplete/failed/duplicate query records: {case}")

    selected, supplied, audited, primary = (collections.Counter() for _ in range(4))
    sources = []
    for log in sorted((case / "log_dir").glob("*.log")):
        active = None
        sources.append(dict(path=str(log.resolve()), sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        for line in log.open(errors="replace"):
            if match := LABEL.search(line):
                q = QUERY.fullmatch(match[1])
                active = (q[1], int(q[2]), int(q[3])) if q else None
                continue
            if active is None or active[2] not in (2, 3):
                continue
            if active[0] != condition or active[1] not in expected_queries:
                raise ValueError(f"Wrong label: {case}, {active}")
            fields = {k: v.strip("'") for k, v in FIELD.findall(line)}
            # The branch-selection record is emitted after page handover but before
            # construction of the scan output table. It is not a lookup-attempt record.
            if "[fixed-page-cache] column_splice " in line:
                if int(fields["cached"]) > 0 and int(fields["read"]) == 0:
                    selected[active[1], active[2], fields["file"], int(fields["row_groups"])] += 1
            elif "[fixed-page-cache] full_hit " in line:
                supplied[active[1], active[2], fields["file"], int(fields["row_groups"])] += 1
            elif "[scan-audit] parquet_materialize " in line:
                primary[active[1], active[2]] += 1
                if int(fields["full_hit"]) == 1:
                    groups = fields["row_groups"].split(",")
                    audited[active[1], active[2], fields["files"], len(groups)] += 1

    # Match within query, execution, file and row-group count, so one missing
    # completion cannot be cancelled by a completion from a different query.
    for key in set(selected) | set(supplied) | set(audited):
        if not 0 <= audited[key] <= supplied[key] <= selected[key]:
            raise ValueError(f"Unmatched/extra cache completion: {case}, {key}, "
                             f"selected={selected[key]} supplied={supplied[key]} audited={audited[key]}")
    denominator, numerator = sum(selected.values()), sum(audited.values())
    if denominator == 0:
        raise ValueError(f"No full-cache selections; conditional rate is N/A: {case}")
    row = dict(workload=workload, condition=condition, query_count=len(expected_queries),
               executions="2,3", selected_full_cache_batches=denominator,
               supplied_full_cache_batches=sum(supplied.values()),
               audited_full_cache_batches=numerator,
               conditional_supply_pct=100*numerator/denominator,
               normal_parquet_materializations=sum(primary.values()))
    query_rows = []
    for q in sorted(expected_queries):
        n = sum(v for (qq, ex, file, rg), v in audited.items() if qq == q)
        d = sum(v for (qq, ex, file, rg), v in selected.items() if qq == q)
        query_rows.append(dict(workload=workload, condition=condition, query=f"q{q}",
                               selected_full_cache_batches=d, audited_full_cache_batches=n,
                               conditional_supply_pct=100*n/d if d else None))
    audit = dict(case=str(case.resolve()), metadata=metadata, sources=sources,
                 event_counts_matched_within="query,execution,file,row_group_count")
    return row, query_rows, audit


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, queries, audit = [], [], []
    for workload in ("th", "cb"):
        for condition in CONDITIONS:
            row, qrows, source = collect_case(workload, condition)
            rows.append(row)
            queries.extend(qrows)
            audit.append(source)
    write_csv(OUTPUT.with_suffix(".csv"), rows)
    write_csv(Path(str(OUTPUT)+"_per_query.csv"), queries)
    Path(str(OUTPUT)+"_provenance.json").write_text(json.dumps(dict(cases=audit,
        definition="Completed and audited full-cache materializations / selected full-cache materializations",
        selection="column_splice cached>0 read=0, after actual page handover, before scan output table construction",
        completion="full_hit table output plus parquet_materialize full_hit=1 audit",
        scope="Normal Parquet full-cache route only; excludes misses, mixed reader/cache routes, late refetch and other providers",
        note="This measures completion after full-cache selection, not lookup effectiveness or workload byte coverage",
        warmup_excluded=1, executions=[2,3], independent_runs_per_condition=1),indent=2)+"\n")

    # Requested figure: one conditional value per workload, future eviction ON.
    plt.rcParams.update({"font.family":"DejaVu Serif", "font.size":15,
                         "svg.fonttype":"none", "pdf.fonttype":42})
    fig, ax = plt.subplots(figsize=(7.4,5.6))
    fig.subplots_adjust(left=.14,right=.97,top=.85,bottom=.24)
    for x, (w, title, color) in enumerate((("th","TPC-H SF50\n22 queries","#803e26"),
                                          ("cb","ClickBench\n43 queries","#3d6b66"))):
        r = next(r for r in rows if r["workload"]==w and r["condition"]=="reorder_future")
        y = r["conditional_supply_pct"]
        ax.bar(x,y,.5,color=color,edgecolor="#303030",linewidth=.8,zorder=3)
        ax.text(x,y+2,f"{y:.1f}%",ha="center",fontsize=18)
        ax.text(x,55,f"{r['audited_full_cache_batches']} / {r['selected_full_cache_batches']}\nbatches",
                color="white",ha="center",va="center",fontsize=16)
    ax.set_xticks((0,1),("TPC-H SF50\n22 queries","ClickBench\n43 queries"))
    ax.set_ylim(0,112)
    ax.set_yticks((0,25,50,75,100))
    ax.set_xlim(-.6,1.6)
    ax.set_ylabel("Conditional cache-path supply success (%)",fontsize=13)
    ax.grid(axis="y",color="#d8d8d8",linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Page caching + reordering + future-aware eviction",y=.95,fontsize=15)
    fig.text(.5,.065,"Denominator: selected full-cache scan batches.\n"
             "Executions 2 and 3; warmup excluded. This is not workload-wide byte coverage.",
             fontsize=9,ha="center")
    for ext in ("png","svg","pdf"):
        p=OUTPUT.with_suffix("."+ext)
        fig.savefig(p,dpi=220,facecolor="white")
        print(p)
    plt.close(fig)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
