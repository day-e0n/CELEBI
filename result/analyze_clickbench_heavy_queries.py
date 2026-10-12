#!/usr/bin/env python3
"""Consolidate measured heavy-query wall time, scan intervals and byte hits."""
from __future__ import annotations

import collections
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "result"))
from plot_requested_cache_hits import parse_logs


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    runs = ROOT / "experiment/FULL_CACHE_PIN_20261007/runs"
    audit = read_csv(ROOT / "result/clickbench_scan_graph_audit.csv")
    columns = json.loads((ROOT / "experiment/column_bytes/clickbench.json").read_text())["hits"]
    rows = []
    for condition in ("baseline", "paging_lru", "reorder_future"):
        hits = {}
        if condition != "baseline":
            hits, _ = parse_logs(sorted((runs / f"hit/cb_r1/{condition}/log_dir").glob("*.log")),
                                 set(range(1, 44)), condition, "value_bytes_v2")
        for q in (28, 29):
            samples = [r for r in audit if r["condition"] == condition and r["query"] == f"q{q}"]
            assert len(samples) == 2
            counter = collections.Counter()
            for (qq, execution, path), c in hits.items():
                if qq == q:
                    counter.update(c)
            total, read = counter["total_bytes"], counter["read_bytes"]
            rows.append(dict(
                condition=condition, query=f"q{q}",
                query_wall_s=statistics.mean(float(r["query_wall_s"]) for r in samples),
                scan_work_s=statistics.mean(float(r["scan_work_s"]) for r in samples),
                scan_interval_union_s=statistics.mean(float(r["scan_interval_union_s"]) for r in samples),
                total_traced_value_bytes=total if hits else None,
                cache_served_value_bytes=total-read if hits else None,
                byte_hit_pct=100*(total-read)/total if total else None,
                oom_count=sum(int(r["oom_count"]) for r in samples),
                oom_records=[json.loads(r["oom_records"]) for r in samples],
            ))
    out = ROOT / "result/clickbench_heavy_query_diagnosis.json"
    out.write_text(json.dumps(dict(
        rows=rows, dataset_rows=99997497,
        decoded_column_bytes={key: columns[key] for key in ("CounterID", "URL", "Referer")},
        cache_budget_bytes=6*2**30,
        environment="Physical GPU 1, RTX A5000; 20GB GPU usage limit; 32GB host; 16MiB cache pages; prefetch off; O_DIRECT on; late materialization on.",
        warmup_excluded=1, measured_executions=[2,3], independent_runs=1,
        scope_notes=[
            "Wall/scan figures use latency cases; byte hits use separate matching trace-on cases.",
            "Scan work is summed host task Computing intervals. It includes read/decode/filter/cache/wait costs and overlapping GPU operations; it is not pure disk time or GPU kernel active time.",
            "Scan union merges overlapping scan intervals; still not the causal time saved by eliminating scan.",
            "Compressed/uncompressed scan-audit bytes are planned Parquet column-chunk volumes, not measured physical I/O traffic.",
            "The two measured CELEBI q29 executions both report projection OOM rescheduling; no attribution of its wall-time penalty is made.",
        ]
    ), indent=2) + "\n")
    print(out)
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k != "oom_records"}))


if __name__ == "__main__":
    main()
