#!/usr/bin/env python3
"""Rank complete query wall latencies and collect four comparable byte-hit traces."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from plot_requested_cache_hits import parse_logs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result"
RUNS = ROOT / "experiment/FULL_CACHE_PIN_20261007/runs"
RETRY = ROOT / "experiment/HIT_TH_PAGING_RETRY_20261007"


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hit_case_path(bench, condition):
    if bench == "th" and condition == "paging_lru":
        return RETRY / "hit/th_r1/paging_lru"
    return RUNS / f"hit/{bench}_r1/{condition}"


def rank_queries():
    means = read_csv(OUT / "cumulative_query_time_full.csv")
    samples = read_csv(OUT / "cumulative_query_time_full_samples.csv")
    ranks = []
    for title, bench, n in (("ClickBench", "cb", 43), ("TPC-H SF50", "th", 22)):
        for condition in ("baseline", "reorder_future"):
            metadata = json.loads((RUNS / f"latency/{bench}_r1/{condition}/metadata.json").read_text())
            assert metadata["status"] == "complete" and metadata["comparison_valid"]
        for q in range(1, n + 1):
            selected = {}
            for condition in ("baseline", "reorder_future"):
                group = [r for r in means if r["workload"] == title and r["condition"] == condition and r["query"] == f"q{q}"]
                assert len(group) == 1 and int(group[0]["samples"]) == 2
                selected[condition] = float(group[0]["query_wall_s"])
            a, b = selected["baseline"], selected["reorder_future"]
            row = dict(overall_rank=0, workload_rank=0, workload=title, query=f"q{q}",
                       baseline_ms=1000*a, celebi_ms=1000*b, saved_ms=1000*(a-b),
                       latency_reduction_pct=100*(1-b/a), speedup=a/b)
            for condition, name in (("baseline", "baseline"), ("reorder_future", "celebi")):
                for ex in (2, 3):
                    values = [float(r["query_wall_s"]) for r in samples if r["workload"] == title
                              and r["condition"] == condition and r["query"] == f"q{q}" and int(r["execution"]) == ex]
                    assert len(values) == 1
                    row[f"{name}_exec{ex}_ms"] = values[0] * 1000
            ranks.append(row)
    ranks.sort(key=lambda r: (-r["latency_reduction_pct"], r["workload"], int(r["query"][1:])))
    counts = collections.Counter()
    for position, row in enumerate(ranks, 1):
        counts[row["workload"]] += 1
        row["overall_rank"] = position
        row["workload_rank"] = counts[row["workload"]]
    assert len(ranks) == 65
    write_csv(OUT / "query_latency_reduction_ranking_full.csv", ranks)
    print("Overall top two:", json.dumps(ranks[:2]), flush=True)
    for title in ("TPC-H SF50", "ClickBench"):
        print(title, "top two:", json.dumps([r for r in ranks if r["workload"] == title][:2]), flush=True)
    return ranks


def collect_hits():
    rows, cases = [], []
    old_manifest = json.loads((RUNS / "manifest.json").read_text())
    retry_manifest = json.loads((RETRY / "manifest.json").read_text())
    assert old_manifest["extension_sha256"] == retry_manifest["extension_sha256"]
    assert old_manifest["source_sha256"] == retry_manifest["source_sha256"]
    assert old_manifest["workloads"]["th"] == retry_manifest["workloads"]["th"]
    for bench, title, n in (("cb", "ClickBench", 43), ("th", "TPC-H SF50", 22)):
        for condition in ("paging_lru", "reorder_future"):
            case = hit_case_path(bench, condition)
            meta = json.loads((case / "metadata.json").read_text())
            assert meta["status"] == "complete" and meta["comparison_valid"], (case, meta["status"])
            assert meta["environment"]["SIRIUS_CACHE_HIT_TRACE"] == "1", case
            assert meta["future_mode"] == ("evict" if condition == "reorder_future" else "off"), case
            records = read_csv(case / "queries.csv")
            assert len(records) == 3*n and {(int(r["query"]), int(r["execution"])) for r in records} == {
                (q, ex) for q in range(1, n+1) for ex in (1, 2, 3)}, case
            assert all(r["status"] == "ok" for r in records), case
            latency_meta = json.loads((RUNS / f"latency/{bench}_r1/{condition}/metadata.json").read_text())
            for field in ("order", "executions", "gpu_device", "gpu_usage_limit_bytes", "host_capacity_bytes",
                          "cache_budget_bytes", "min_free_bytes_per_gpu", "future_mode", "future_horizon",
                          "late_materialization", "prefetch", "odirect"):
                assert meta[field] == latency_meta[field], (case, field)
            logs = sorted((case / "log_dir").glob("*.log"))
            data, excluded = parse_logs(logs, set(range(1, n+1)), condition, "value_bytes_v2")
            totals = collections.Counter()
            for counter in data.values():
                totals.update(counter)
            total, read = totals["total_bytes"], totals["read_bytes"]
            assert total > 0 and 0 <= read <= total
            rows.append(dict(workload=title, condition=condition, query_count=n, executions="2,3",
                             total_decoded_value_bytes=total, reader_decoded_value_bytes=read,
                             cache_served_decoded_value_bytes=total-read, cache_hit_pct=100*(total-read)/total,
                             trace_batches=totals["trace_batches"], metric="value_bytes_v2"))
            cases.append(dict(case=str(case), metadata=meta, excluded_metrics=dict(excluded),
                              logs=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in logs]))
            print(json.dumps(rows[-1]), flush=True)
    write_csv(OUT / "cache_hit_comparison_full.csv", rows)
    (OUT / "cache_hit_comparison_full_provenance.json").write_text(json.dumps(dict(
        cases=cases, extension_sha256=old_manifest["extension_sha256"],
        definition="sum(cache-served decoded value bytes) / sum(total traced decoded value bytes), access weighted",
        excluded_byte_components="STRING offsets, null masks, allocator padding/slabs, virtual columns",
        warmup_execution=1, measured_executions=[2, 3], independent_runs_per_condition=1,
        retry_note="TPC-H paging uses a separate same-configuration retry; original shutdown-crash case excluded"), indent=2)+"\n")
    return rows


def main():
    ranks = rank_queries()
    hits = collect_hits()
    text = ["# Full-query latency ranking and byte cache-hit comparison", "",
            "Baseline versus page caching + reordering + future-aware eviction (CELEBI).",
            "ClickBench all 43 queries, TPC-H SF50 all 22. Execution 1 excluded; average executions 2 and 3.",
            "Ranking = 100 * (baseline query wall time - CELEBI query wall time) / baseline query wall time.",
            "One independent run per condition; measured ranking, not a statistical significance claim.", "",
            "| Overall rank | Workload | Query | Baseline (ms) | CELEBI (ms) | Reduction | Speedup |",
            "|---|---|---|---:|---:|---:|---:|"]
    for r in ranks[:2]:
        text.append(f"| {r['overall_rank']} | {r['workload']} | {r['query']} | {r['baseline_ms']:.2f} | {r['celebi_ms']:.2f} | {r['latency_reduction_pct']:.2f}% | {r['speedup']:.2f}x |")
    text += ["", "TPC-H-only top two:"]
    for r in [r for r in ranks if r["workload"] == "TPC-H SF50"][:2]:
        text.append(f"- {r['query']}: {r['baseline_ms']:.2f} -> {r['celebi_ms']:.2f} ms ({r['latency_reduction_pct']:.2f}% reduction).")
    text += ["", "| Workload | Condition | Byte hit ratio |", "|---|---|---:|"]
    for r in hits:
        text.append(f"| {r['workload']} | {r['condition']} | {r['cache_hit_pct']:.2f}% |")
    text += ["", "Hit counts include repeated accesses across Parquet, cached-provider and late-refetch paths.",
             "They measure decoded value bytes, not distinct dataset coverage or conditional cache-path completion.",
             "Latency uses trace-off cases; hit ratios use separate trace-on cases with matching settings.",
             "Original TPC-H paging hit case crashed during shutdown and is excluded; the separately recorded retry uses identical build, inputs and configuration."]
    (OUT / "QUERY_LATENCY_RANKING_AND_HITS_2026-10-07.md").write_text("\n".join(text)+"\n")


if __name__ == "__main__":
    main()
