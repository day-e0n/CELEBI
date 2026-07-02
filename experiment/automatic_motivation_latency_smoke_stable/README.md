# Automatic motivation latency

Created: 2026-07-01T13:43:04
Input: `/mnt/nvme/dataset`
Repeats: `1`
Pairs: `10 ordered pairs`
Variants: `baseline, join_reuse`
Join retention limit bytes: `268435456`
Join retention max batch bytes: `33554432`

Freshness rule: every pair repeat is executed in a separate process.

Main outputs after/while the run:
- `summary/query_latency.csv`
- `summary/pair_latency.csv`
- `summary/pair_latency_summary.csv`
- `summary/join_reuse_heatmap_long.csv`
- `summary/join_reuse_hit_gb_heatmap.png`
- `summary/join_retained_gb_heatmap.png`
- `summary/join_reuse_ratio_heatmap.png`
