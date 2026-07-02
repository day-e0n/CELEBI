# Automatic motivation latency

Created: 2026-07-01T14:08:10
Input: `/mnt/nvme/dataset`
Repeats: `1`
Pairs: `3 ordered pairs`
Variants: `baseline, join_reuse`
Join retention limit bytes: `536870912`
Join retention max batch bytes: `268435456`

Freshness rule: every pair repeat is executed in a separate process.

Main outputs after/while the run:
- `summary/query_latency.csv`
  - `load_ms` is wall-clock union of parquet materialize intervals.
  - `scan_materialize_work_ms` is summed materialize work across parallel tasks.
- `summary/pair_latency.csv`
- `summary/pair_latency_summary.csv`
- `summary/join_reuse_heatmap_long.csv`
- `summary/join_reuse_hit_gb_heatmap.png`
- `summary/join_retained_gb_heatmap.png`
- `summary/join_reuse_ratio_heatmap.png`
