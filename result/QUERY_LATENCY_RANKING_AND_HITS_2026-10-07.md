# Full-query latency ranking and byte cache-hit comparison

Baseline versus page caching + reordering + future-aware eviction (CELEBI).
ClickBench all 43 queries, TPC-H SF50 all 22. Execution 1 excluded; average executions 2 and 3.
Ranking = 100 * (baseline query wall time - CELEBI query wall time) / baseline query wall time.
One independent run per condition; measured ranking, not a statistical significance claim.

| Overall rank | Workload | Query | Baseline (ms) | CELEBI (ms) | Reduction | Speedup |
|---|---|---|---:|---:|---:|---:|
| 1 | ClickBench | q32 | 749.05 | 162.07 | 78.36% | 4.62x |
| 2 | ClickBench | q27 | 546.72 | 141.81 | 74.06% | 3.86x |

TPC-H-only top two:
- q2: 486.25 -> 182.37 ms (62.49% reduction).
- q17: 1886.00 -> 864.43 ms (54.17% reduction).

| Workload | Condition | Byte hit ratio |
|---|---|---:|
| ClickBench | paging_lru | 23.10% |
| ClickBench | reorder_future | 41.66% |
| TPC-H SF50 | paging_lru | 9.01% |
| TPC-H SF50 | reorder_future | 36.79% |

Hit counts include repeated accesses across Parquet, cached-provider and late-refetch paths.
They measure decoded value bytes, not distinct dataset coverage or conditional cache-path completion.
Latency uses trace-off cases; hit ratios use separate trace-on cases with matching settings.
Original TPC-H paging hit case crashed during shutdown and is excluded; the separately recorded retry uses identical build, inputs and configuration.
