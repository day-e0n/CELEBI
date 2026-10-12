# ClickBench q28/q29: scan attribution versus whole-query latency

**Follow-up:** the deeper investigation confirmed the grouped AVG DECIMAL128
accumulation path as the main removable cost and implemented an overflow-safe
optimization. See [the measured fix and validation](CLICKBENCH_HEAVY_AVG_FIX_2026-10-07.md).
The observations and untested hypotheses below describe the earlier investigation.

The measured bottleneck is not adequately explained as disk/scan latency alone.
No engine optimization was applied in this investigation. The isolated diagnostic
uses the same extension hash and input as FULL_CACHE_PIN, but a two-query baseline
workload with cache disabled. It is not a replacement for the full-workload figures.

## Original full 43-query workload

Means of executions 2 and 3, execution 1 excluded; one independent run per condition.

| Query | Baseline wall (s) | CELEBI wall (s) | CELEBI byte hit ratio |
|---|---:|---:|---:|
| q28 | 16.7314 | 16.6773 | 0.00% |
| q29 | 40.3707 | 40.5019 | 12.5430% |

CELEBI means page caching + query reordering + future-aware eviction. Hits use
separate matching trace-on runs and the value_bytes_v2 metric; wall times use
trace-off runs. This is not the conditional cache-path success metric.

Decoded URL is 9,956,144,344 bytes and Referer 7,290,587,277 bytes. Both exceed
the 6 GiB page-store budget. Capacity alone does not prove zero reuse: the q29
trace did serve about 12.54% of its requested decoded value bytes from cache.
Both measured CELEBI q29 executions logged projection OOM and task rescheduling,
with individual failed requests of 4,043,665,824 and 3,452,802,560 bytes.
Their contribution to wall time has not been isolated.

## Isolated concurrency diagnostic

`experiment/CLICKBENCH_HEAVY_DIAG_20261007_v2`: isolated processes with 4 and 1
GPU pipeline workers, query order [28,29], three executions each. Same SQL,
input, compiled extension, physical GPU 1, 20GB GPU limit, 32GB host capacity,
O_DIRECT, no prefetch, late materialization enabled. Page cache, CPU fallback,
and the known-broken regex JIT are disabled in both cases. Only worker count
differs; output/telemetry paths are necessarily different. Twelve output result
sets agree with exact non-float comparisons and float tolerance 1e-8 relative,
1e-6 absolute. This comparison did not revalidate against a fresh CPU answer.

| Workers | Query | Whole-query wall (s) | Scan work (s) | Projection work (s) | Group-by work (s) |
|---|---|---:|---:|---:|---:|
| 4 | q28 | 16.8562 | 28.1739 | 0.0487 | 26.3730 |
| 1 | q28 | 17.8446 | 1.5167 | 0.0066 | 16.2608 |
| 4 | q29 | 39.8121 | 49.4659 | 18.1201 | 56.2966 |
| 1 | q29 | 40.1300 | 1.2564 | 9.6129 | 29.1515 |

Operator work is the sum of host task Computing intervals, not CUDA kernel
active time. It includes blocking waits and may overlap across workers. The
single-worker case reduces that overlap; it does not turn these metrics into
hardware utilization measurements. The large reduction in attributed scan
work without a query-speed improvement shows why the parallel scan bar cannot
be interpreted as removable disk I/O time. Other GPU work delaying a reader's
completion is a plausible explanation, not a measured kernel-level attribution.
Reducing workers is not an effective optimization in this diagnostic.

## Concrete optimization targets

1. **q28 local group-by.** The serial diagnostic attributes about 16.26 seconds
   to HASH_GROUP_BY. A separate CPU probe of CounterID alone found 99,997,497
   rows and 6,506 distinct keys; the largest key has 8,527,070 rows (8.5273%).
   This is the whole-table distribution, not the exact URL-nonempty subset.
   Profile the actual aggregation algorithm and test a reduction strategy
   suited to a small number of frequent keys. Atomic contention/skew is a
   candidate, not a confirmed CUDA kernel bottleneck.
2. **q29 expression and string aggregation.** Its SQL extracts a domain using
   REGEXP_REPLACE, groups by that string, and computes MIN(Referer). The serial
   diagnostic attributes 9.61 seconds to projection and 29.15 seconds to
   HASH_GROUP_BY. The existing optional hand-written regex JIT is explicitly
   documented as returning wrong results/invalid UTF-8; simply enabling it
   is not a valid fix. A replacement must preserve nulls, UTF-8, nonmatches,
   newlines and stream/resource ownership and pass output validation.
3. **Retained input and oversized batches.** q28's pre-aggregate projection
   only needs CounterID and strlen(URL), as confirmed by EXPLAIN. Mixed
   zero-copy projection holds an input read-only lock and therefore retains
   the original input batch, potentially keeping URL alive unnecessarily.
   Copying small required passthrough columns into an independent output may
   release large unused strings earlier. This candidate was not implemented
   or benchmarked. The Parquet coalescer emits whole row groups; q28 has a
   row-group contribution as large as 1,594,413,212 bytes despite a 512MiB
   target. Lowering the target alone cannot divide an individual row group.
   Smaller operation batches and accurate temporary-memory reservations are
   further candidates for memory pressure, particularly q29's OOM retries.

Key code locations:

- `src/op/aggregate/gpu_aggregate_impl.cpp`: local_grouped_aggregate / cuDF groupby.
- `src/op/sirius_physical_projection.cpp`: mixed output's projection_owner holds input_lock.
- `src/expression_evaluator/specializations/function.cpp`: strlen/count_bytes and regex dispatch.
- `src/sirius_extension.cpp`: enable_regex_jit_impl warning/default.
- `src/op/scan/parquet_gpu_ingestible.cpp`: parquet_batch_coalescer packs whole row groups.

## Artifacts and limitations

- `result/clickbench_heavy_query_diagnosis.json`: original full-workload evidence.
- `experiment/CLICKBENCH_HEAVY_DIAG_20261007_v2/analysis.json`: concurrency means and operator work.
- `experiment/CLICKBENCH_HEAVY_DIAG_20261007_v2/*/queries.csv`, `results.json`, logs and telemetry.
- `experiment/CLICKBENCH_HEAVY_DIAG_20261007_v2/counter_id_distribution.json`.
- `scripts/diagnose_clickbench_heavy_concurrency.py`: reproducible isolated runner.

The initial diagnostic directory without the v2 suffix failed at view preparation
(DuckDB does not allow a prepared parameter in CREATE VIEW) before executing
queries. It is excluded. Two additional q28 profiling attempts with installed
Nsight Systems 2023.1.2 and 2025.1.3 encountered `Unknown runtime API function
index: 509`; reports contain no CUDA kernel data. Those attempts are excluded
from performance means and no kernel-level conclusions are drawn from them.
There is only one independent process per worker setting, with two measured
executions. Configuration changes were restricted to separate diagnostic cases;
the original comparison results and running-experiment settings were preserved.
