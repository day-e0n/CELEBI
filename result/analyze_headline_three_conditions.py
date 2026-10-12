#!/usr/bin/env python3
"""Reproduce the exact baseline / paging / paging + reorder headline conditions.

Only existing FA_{cb,th}_{base,lru,ro_lru}_r* logs are read. No benchmark is run.
Execution 1 is checked for completeness and then excluded, as in the headline.
The cache + reorder condition here never includes future-aware eviction (ro_evict).

Run: pixi run python result/analyze_headline_three_conditions.py

Outputs: headline_three_conditions_queries.csv and headline_reorder_contributions.png/pdf.
Wall time is the sum of execute(...).fetchall() query timers. The scan timers below are
sums across overlapping tasks; they must not be added to wall time. read_s includes
ordinary read_us plus partial-serving ps_read_us (I/O, H2D, and decode). compressed_bytes
in scan-audit describe requested column chunks, not measured physical bytes read; they
are deliberately not presented as I/O volume here. Partial counts and row*column cells
come from explicit partial_serve used=1 records, not an inferred batch hit ratio.
"""
from __future__ import annotations

import collections
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
CONDITIONS = ("base", "lru", "ro_lru")
WORKLOADS = (("ClickBench", "cb", (1, 2), tuple(q for q in range(1, 44) if q not in (28, 29))),
             ("TPC-H SF50", "th", (1, 2, 3), tuple(range(1, 23))))
WALL_RE = re.compile(r"exec=(\d+)/(\d+) pos=(\d+)/(\d+) q(\d+) done wall=([\d.]+)s")
LABEL_RE = re.compile(r"sirius_set_query_label\('\w+_q(\d+)_exec(\d+)'\)")
FIELD_RE = re.compile(r"\b(\w+)=(\d+)\b")
METRICS = ("read_s", "cache_serve_s", "populate_s", "postfilter_s", "audit_batches",
           "full_hit_batches", "partial_used_batches", "partial_failed_batches",
           "partial_read_ranges", "partial_read_cells", "partial_total_cells")


def parse_wall(run: Path, queries: tuple[int, ...]):
    walls, positions = {}, {}
    for line in (run / "runner.log").read_text().splitlines():
        m = WALL_RE.search(line)
        if not m:
            continue
        execution, executions, position, count, query = map(int, m.groups()[:5])
        key = query, execution
        if key in walls:
            raise ValueError(f"{run.name}: duplicate timing {key}")
        if executions != 3 or count != len(queries):
            raise ValueError(f"{run.name}: unexpected execution/query count in {line}")
        walls[key] = float(m[6])
        positions[key] = position
    expected = {(q, e) for q in queries for e in (1, 2, 3)}
    if set(walls) != expected:
        raise ValueError(f"{run.name}: incomplete or extra timings; missing={expected - set(walls)}, "
                         f"extra={set(walls) - expected}")
    for e in (1, 2, 3):
        if sorted(positions[q, e] for q in queries) != list(range(1, len(queries) + 1)):
            raise ValueError(f"{run.name}: invalid position sequence, execution {e}")
    for q in queries:
        if len({positions[q, e] for e in (1, 2, 3)}) != 1:
            raise ValueError(f"{run.name}: query order changed between executions")
    return walls, {q: positions[q, 1] for q in queries}


def parse_scan(run: Path):
    logs = sorted(run.glob("*/log_dir/*.log"))
    if len(logs) != 1:
        raise ValueError(f"{run.name}: expected one engine log, found {len(logs)}")
    metrics = collections.defaultdict(collections.Counter)
    key = None
    for line in logs[0].open():
        m = LABEL_RE.search(line)
        if m:
            key = int(m[1]), int(m[2])
            continue
        if key is None:
            continue
        fields = {k: int(v) for k, v in FIELD_RE.findall(line)}
        out = metrics[key]
        if "[scan-audit] parquet_materialize" in line:
            out["read_s"] += (fields.get("read_us", 0) + fields.get("ps_read_us", 0)) / 1e6
            out["cache_serve_s"] += sum(fields.get(k, 0) for k in (
                "lookup_us", "materialize_us", "assemble_us", "ps_peek_us", "ps_assemble_us")) / 1e6
            out["audit_batches"] += 1
            out["full_hit_batches"] += fields["full_hit"]
        elif "[scan-populate] store=pages us=" in line:
            out["populate_s"] += fields["us"] / 1e6
        elif "[scan-stage]" in line:
            out["postfilter_s"] += fields["postfilter_us"] / 1e6
        elif "[fixed-page-cache] partial_serve" in line:
            if fields["used"]:
                out["partial_used_batches"] += 1
                for k in ("read_ranges", "read_cells", "total_cells"):
                    out[f"partial_{k}"] += fields[k]
            else:
                out["partial_failed_batches"] += 1
    return metrics


def main() -> None:
    rows = []
    summaries = []
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 6.8), constrained_layout=True)
    for ax, (title, bench, reps, queries) in zip(axes, WORKLOADS):
        values = {}
        runmeans = {}
        orders = {}
        for condition in CONDITIONS:
            samples = collections.defaultdict(list)
            first_pass_samples = collections.defaultdict(list)
            scan_samples = collections.defaultdict(lambda: collections.defaultdict(list))
            execution_totals = []
            first_pass_totals = []
            run_totals = []
            for r in reps:
                run = ROOT / "experiment" / f"FA_{bench}_{condition}_r{r}"
                walls, positions = parse_wall(run, queries)
                scan = parse_scan(run)
                if condition in orders and positions != orders[condition]:
                    raise ValueError(f"{run.name}: order differs across runs")
                orders[condition] = positions
                totals = [sum(walls[q, e] for q in queries) for e in (2, 3)]
                execution_totals.extend(totals)
                first_pass_totals.append(sum(walls[q, 1] for q in queries))
                run_totals.append(statistics.mean(totals))
                runmeans[condition, r] = {q: statistics.mean(walls[q, e] for e in (2, 3))
                                          for q in queries}
                for q in queries:
                    first_pass_samples[q].append(walls[q, 1])
                    for e in (2, 3):
                        samples[q].append(walls[q, e])
                        for metric in METRICS:
                            scan_samples[q][metric].append(scan[q, e][metric])
            values[condition] = {
                q: {"wall_s": statistics.mean(samples[q]),
                    "wall_sd_s": statistics.stdev(samples[q]),
                    "first_pass_wall_s": statistics.mean(first_pass_samples[q]),
                    "position": orders[condition][q],
                    **{metric: statistics.mean(scan_samples[q][metric]) for metric in METRICS}}
                for q in queries}
            summaries.append({"workload": title, "condition": condition,
                              "samples": execution_totals, "run_means": run_totals,
                              "mean_s": statistics.mean(execution_totals),
                              "sd_s": statistics.stdev(execution_totals),
                              "first_pass_mean_s": statistics.mean(first_pass_totals),
                              "scan_task_sums": {metric: sum(values[condition][q][metric]
                                                             for q in queries) for metric in METRICS}})
        for q in queries:
            row = {"workload": title, "query": f"q{q}", "samples_per_condition": 2 * len(reps)}
            for condition in CONDITIONS:
                row.update({f"{condition}_{k}": v for k, v in values[condition][q].items()})
            for before, after, label in (("base", "lru", "paging"), ("lru", "ro_lru", "reorder")):
                row[f"{label}_wall_saved_s"] = values[before][q]["wall_s"] - values[after][q]["wall_s"]
                row[f"{label}_read_task_saved_s"] = values[before][q]["read_s"] - values[after][q]["read_s"]
                for r in reps:
                    row[f"{label}_wall_saved_r{r}_s"] = runmeans[before, r][q] - runmeans[after, r][q]
            rows.append(row)
        deltas = [values["lru"][q]["wall_s"] - values["ro_lru"][q]["wall_s"] for q in queries]
        gain = sum(max(d, 0) for d in deltas)
        loss = -sum(min(d, 0) for d in deltas)
        ax.bar(range(len(queries)), deltas, color=["#3d6b66" if d >= 0 else "#b4522e" for d in deltas],
               edgecolor="#333333", linewidth=0.4)
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.set_xticks(range(len(queries)), [f"q{q}" for q in queries], fontsize=8)
        ax.set_ylabel("Query wall time saved (s)")
        ax.set_title(f"{title}: paging → paging + reorder | net {sum(deltas):.3f}s saved "
                     f"= {gain:.3f}s gains − {loss:.3f}s regressions", fontsize=11)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.015)
    fig.suptitle("Exact headline conditions: base / lru / ro_lru; future-aware eviction excluded.\n"
                 "Green: faster after reorder. Orange: slower. Execution 1 excluded; means of executions 2–3.",
                 fontsize=10)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "headline_three_conditions_queries.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"headline_reorder_contributions.{ext}", dpi=220)
    print(json.dumps(summaries, indent=2))
    print("Validated 3 complete executions per run; analyzed executions 2–3 only.")
    print(OUT / "headline_three_conditions_queries.csv")


if __name__ == "__main__":
    main()
