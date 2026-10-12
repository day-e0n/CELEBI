#!/usr/bin/env python3
"""Full-workload cumulative query wall time, CB q1-43 / TPC-H SF50 q1-22.

Use completed latency cases from FULL_CACHE_PIN_20261007. Exclude execution 1;
average executions 2 and 3, then accumulate in query-number order.
CELEBI is reorder_future with actual future-aware eviction hints, not reorder_lru.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from plot_cumulative_scan_work import CONDS, OUT, ROOT, plot, write_csv


def main():
    runs = ROOT / "experiment/FULL_CACHE_PIN_20261007/runs"
    manifest = json.loads((runs / "manifest.json").read_text())
    samples, means, cases, totals = [], [], [], {}
    for title, bench, n in (("ClickBench", "cb", 43), ("TPC-H SF50", "th", 22)):
        totals[title] = {}
        for label, condition, _, _ in CONDS:
            case = runs / f"latency/{bench}_r1/{condition}"
            metadata = json.loads((case / "metadata.json").read_text())
            assert metadata["status"] == "complete" and metadata["comparison_valid"], case
            assert metadata["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "0", case
            assert metadata["future_mode"] == ("evict" if condition == "reorder_future" else "off"), case
            assert metadata["future_horizon"] == "execution", case
            assert metadata["pin"] is None, case
            expected_cache = "0" if condition == "baseline" else "1"
            for setting in ("SIRIUS_ENABLE_FIXED_PAGE_REUSE", "SIRIUS_FIXED_PAGE_AUTO_CACHE",
                            "SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED", "SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"):
                assert metadata["environment"][setting] == expected_cache, (case, setting)
            with (case / "queries.csv").open(newline="") as source:
                records = list(csv.DictReader(source))
            expected = {(q, ex) for q in range(1, n + 1) for ex in (1, 2, 3)}
            assert len(records) == len(expected), case
            assert {(int(r["query"]), int(r["execution"])) for r in records} == expected, case
            assert all(r["status"] == "ok" and r["condition"] == condition for r in records), case
            hint_count = 0
            if condition == "reorder_future":
                for log in (case / "log_dir").glob("*.log"):
                    for line in log.read_text().splitlines():
                        if "QueryBegin: SQL: CALL sirius_set_cache_hints('" in line:
                            assert "sirius_set_cache_hints('evict;" in line, (case, line)
                            hint_count += 1
                assert hint_count == n * 3, (case, hint_count)
            for r in records:
                ex, q = int(r["execution"]), int(r["query"])
                wall = float(r["wall_s"])
                assert math.isfinite(wall) and wall > 0, (case, r)
                if ex == 1:
                    continue
                samples.append(dict(workload=title, condition=condition, query=f"q{q}", run=1,
                                    execution=ex, query_wall_s=wall))
            cumulative = 0.0
            for q in range(1, n + 1):
                group = [r for r in samples if r["workload"] == title and r["condition"] == condition
                         and r["query"] == f"q{q}"]
                assert len(group) == 2 and {r["execution"] for r in group} == {2, 3}
                values = [r["query_wall_s"] for r in group]
                wall = statistics.mean(values)
                cumulative += wall
                means.append(dict(workload=title, condition=condition, label=label, query=f"q{q}",
                                  position=q, samples=2, query_wall_s=wall,
                                  query_wall_sd_s=statistics.stdev(values), cumulative_query_wall_s=cumulative))
            assert math.isclose(cumulative, metadata["steady_state_workload_wall_s"], abs_tol=1e-8), case
            totals[title][condition] = cumulative
            cases.append(dict(workload=title, condition=condition, case=str(case), metadata=metadata,
                              actual_future_eviction_calls=hint_count,
                              queries_csv_sha256=hashlib.sha256((case / "queries.csv").read_bytes()).hexdigest()))
            print(f"{title} {condition}: {cumulative:.6f}s; {n}/{n} queries", flush=True)
    prefix = OUT / "cumulative_query_time_full"
    write_csv(prefix.with_suffix(".csv"), means)
    write_csv(Path(str(prefix) + "_samples.csv"), samples)
    Path(str(prefix) + "_provenance.json").write_text(json.dumps(dict(
        source_manifest=str(runs / "manifest.json"), extension_sha256=manifest["extension_sha256"],
        metric="Sum of per-query execute().fetchall() wall times; includes all query operators.",
        timer_scope="Query label/hint calls, SET and reorder preparation are outside the timer.",
        accumulation="Per-query means in query-number order; not actual reordered arrival order.",
        query_counts={"ClickBench": 43, "TPC-H SF50": 22}, warmup_execution=1,
        averaged_executions=[2, 3], independent_runs=1, totals=totals, cases=cases), indent=2) + "\n")
    plot(means, prefix, value_key="cumulative_query_wall_s", ylabel="Cumulative query time (s)",
         tick_step=10, label_gap_fraction=0.04, equal_data_span=True)


if __name__ == "__main__":
    main()
