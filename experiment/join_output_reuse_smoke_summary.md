# Join Output Retention + Reuse Smoke Summary

## Runs

| case | mode | run root | q1 runtime | q2 runtime | join reuse hits | device mismatch misses | note |
|---|---|---|---:|---:|---:|---:|---|
| q2 -> q2 | baseline | `experiment/tpch_pair_scan_audit_runs/pair_scan_20260701_044115` | 3.100650s | 1.747231s | 0 | 0 | no retention/reuse flags |
| q2 -> q2 | retention+reuse | `experiment/tpch_pair_scan_audit_runs/pair_scan_20260701_043804` | 3.221298s | 1.847831s | 2 | 10 | reuse works functionally, but slower than baseline in this single run |
| q2 -> q16 | baseline | `experiment/tpch_pair_scan_audit_runs/pair_scan_20260701_043953` | 3.169865s | 1.496681s | 0 | 0 | q16 reloads 1.550GB overlapped uncompressed scan data |
| q2 -> q16 | retention+reuse | `experiment/tpch_pair_scan_audit_runs/pair_scan_20260701_043907` | 3.260873s | 0.961467s | 0 | 0 | no join-output reuse hit; scan reload remains 1.550GB |

## Interpretation

- The experimental operator-level reuse path can cache join outputs and later hit them (`q2 -> q2` had 2 hits).
- Reuse is currently conservative: it only reuses when the cached join output is on the same GPU as the executing task; otherwise it falls back to the normal join path to avoid invalid cross-device allocator/stream usage.
- For `q2 -> q16`, table/column overlap is high, but exact join-output overlap is not observed. This means `q2 -> q16` is a better motivation case for scan/page-resident reuse than for join-output reuse.
- Current reuse copies cached cuDF tables into a fresh output batch. That makes correctness/lifetime safer, but it can erase the benefit of skipping small joins. A stronger design would need planner/source-level reuse so duplicated scans or whole subplans are skipped, not just operator execution.

## Caveat

`runtime_breakdown.csv` load/non-load numbers are not reliable for these pair runs because the log splitter saw 4 `QueryBegin` lines instead of 2 and skipped per-query log splitting. Use runtime CSV and explicit `[join-reuse]` logs for this smoke result.
