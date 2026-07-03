# Automatic motivation latency

Created: 2026-07-02T02:30:23
Input: `/mnt/nvme/dataset`
Repeats: `1`
Pairs: `1 ordered pairs`
Variants: `baseline, pinned_hot`
Join retention limit bytes: `2147483648`
Join retention max batch bytes: `268435456`

Freshness rule: every pair repeat is executed in a separate process.

Main outputs after/while the run:
- `summary/query_latency.csv`
  - `load_ms` is `min(scan_materialize_wall_ms, total_ms)` for a bounded wall-clock load estimate.
  - `scan_materialize_wall_ms` is wall-clock union of parquet materialize intervals.
  - `scan_materialize_work_ms` is summed materialize work across parallel tasks.
- `summary/pair_latency.csv`
- `summary/pair_latency_summary.csv`
- `summary/join_reuse_heatmap_long.csv`
- `summary/join_reuse_hit_gb_heatmap.png`
- `summary/join_retained_gb_heatmap.png`
- `summary/join_reuse_ratio_heatmap.png`
