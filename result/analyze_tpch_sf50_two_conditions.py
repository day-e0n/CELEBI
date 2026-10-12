#!/usr/bin/env python3
"""Extract completed SF50 baseline versus page caching/reordering/future eviction."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from parse_quent_operator_breakdown import (  # noqa: E402
    BUCKETS, accumulate_task_durations, bucket_for, load_telemetry,
)


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "experiment/FULL_CACHE_PIN_20261007/runs")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "result/tpch_sf50_two_conditions")
    parser.add_argument("--proposed-condition", choices=("reorder_future", "reorder_lru"), default="reorder_future")
    args = parser.parse_args()
    conditions = ("baseline", args.proposed_condition)
    manifest = json.loads((args.input / "manifest.json").read_text())
    samples, raw, provenance = [], [], {}
    for condition in conditions:
        case = args.input / "latency/th_r1" / condition
        metadata = json.loads((case / "metadata.json").read_text())
        if metadata["status"] != "complete" or not metadata["comparison_valid"]:
            raise ValueError(f"Incomplete experiment: {case}")
        expected_future = "evict" if condition == "reorder_future" else "off"
        if (metadata["environment"]["SIRIUS_CACHE_HIT_TRACE"] != "0"
                or metadata["future_mode"] != expected_future
                or metadata["future_horizon"] != "execution"):
            raise ValueError(f"Wrong diagnostic/future condition: {case}")
        expected_caching = "0" if condition == "baseline" else "1"
        for setting in ("SIRIUS_ENABLE_FIXED_PAGE_REUSE", "SIRIUS_FIXED_PAGE_AUTO_CACHE",
                        "SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED", "SIRIUS_VARIABLE_PAGE_CACHE_ENABLED"):
            if metadata["environment"][setting] != expected_caching:
                raise ValueError(f"Wrong cache condition: {setting}, {case}")
        with (case / "queries.csv").open(newline="") as source:
            query_rows = list(csv.DictReader(source))
        expected = {(q, ex) for ex in (1, 2, 3) for q in range(1, 23)}
        if len(query_rows) != 66 or {(int(r["query"]), int(r["execution"])) for r in query_rows} != expected:
            raise ValueError(f"Missing/duplicate queries: {case}")
        if any(row["status"] != "ok" for row in query_rows):
            raise ValueError(f"Query failure: {case}")
        hint_audit = None
        if condition == "reorder_future":
            calls = []
            for log in sorted((case / "log_dir").glob("*.log")):
                for line_number, line in enumerate(log.read_text().splitlines(), 1):
                    if "QueryBegin: SQL: CALL sirius_set_cache_hints('" in line:
                        if "sirius_set_cache_hints('evict;" not in line:
                            raise ValueError(f"Wrong future mode in actual log: {log}:{line_number}")
                        calls.append(dict(path=str(log.resolve()), line=line_number))
            if len(calls) != 66:
                raise ValueError(f"Expected 66 actual future-evict calls: {case}, got {len(calls)}")
            hint_audit = dict(mode="evict", call_count=len(calls), calls=calls)
        telemetry = case / "telemetry_data"
        mappings = load_telemetry(telemetry)
        totals = accumulate_task_durations(telemetry, mappings[1], mappings[2], mappings[3], mappings[0])
        labels = {f"{condition}_q{q}_exec{ex}" for q, ex in expected}
        if not labels.issubset(totals):
            raise ValueError(f"Missing telemetry: {case}, {sorted(labels - totals.keys())}")
        for record in query_rows:
            execution = int(record["execution"])
            if execution == 1:
                continue
            query = f'q{record["query"]}'
            kinds = totals[record["label"]]
            if kinds.get("UNKNOWN", 0) > 0:
                raise ValueError(f"Unattributed operator work: {record['label']}")
            buckets = collections.Counter()
            for kind, ms in kinds.items():
                buckets[bucket_for(kind)] += ms
                raw.append(dict(condition=condition, query=query, execution=execution,
                                operator_kind=kind, duration_ms=ms))
            sample = dict(condition=condition, query=query, run=1, execution=execution)
            sample.update({bucket: buckets[bucket] for bucket in BUCKETS})
            sample.update(total_ms=sum(buckets.values()), wall_ms=float(record["wall_s"]) * 1000)
            samples.append(sample)
        files = sorted(telemetry.rglob("*.ndjson"))
        provenance[condition] = {"case": str(case.resolve()), "metadata": metadata,
            "actual_future_hint_audit": hint_audit,
            "telemetry_files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                for path in files]}
    means = []
    for condition in conditions:
        for q in range(1, 23):
            group = [r for r in samples if r["condition"] == condition and r["query"] == f"q{q}"]
            if len(group) != 2:
                raise ValueError("Expected execution 2 and 3 for every query")
            row = dict(condition=condition, query=f"q{q}", samples=2)
            row.update({field: statistics.mean(r[field] for r in group)
                        for field in (*BUCKETS, "total_ms", "wall_ms")})
            row["total_sd_ms"] = statistics.stdev(r["total_ms"] for r in group)
            means.append(row)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(str(args.output_prefix) + "_operator_breakdown.csv"), means)
    write_csv(Path(str(args.output_prefix) + "_samples.csv"), samples)
    write_csv(Path(str(args.output_prefix) + "_operator_kinds.csv"), raw)
    result = {"source_manifest": str((args.input / "manifest.json").resolve()),
        "extension_sha256": manifest["extension_sha256"],
        "conditions": provenance, "warmup_execution": 1, "averaged_executions": [2, 3],
        "independent_runs": 1, "samples_per_query_per_condition": 2,
        "operator_work_definition": "Sum of Task Computing dwell times attributed to operators; includes host work/waits, is not GPU kernel active time or query wall time.",
        "query_wall_definition": "Sum of per-query execute().fetchall() elapsed time; query label/hint calls, SET, and reorder preparation are outside the timer.",
        "gpu_memory_note": "YAML literals GPU=20GB/HOST=32GB. Harness byte fields assume binary units; do not treat those fields as verified allocator capacities.",
        "eviction": ("Future-aware eviction ON; column next-use hints for the remainder of the current execution"
                     if args.proposed_condition == "reorder_future" else
                     "future hints OFF; variable-width-first, then fixed-width, LRU within each group"),
        "reorder": "small-first-big-last", "cache_budget_bytes": 4 * 2**30,
        "totals": {condition: {field: sum(r[field] for r in means if r["condition"] == condition)
                               for field in (*BUCKETS, "total_ms", "wall_ms")}
                   for condition in conditions}}
    Path(str(args.output_prefix) + "_provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))
    print(f"Validated: 66/66 completed queries per condition; 88 steady-state samples; 44 means")


if __name__ == "__main__":
    main()
