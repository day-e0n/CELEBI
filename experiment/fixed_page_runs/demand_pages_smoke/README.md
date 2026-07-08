# Fixed-page pair latency experiment

Created: 2026-07-08T05:40:34
Input: `/mnt/nvme/dataset`
Queries: `q6`
Pairs: `1`
Repeats: `1`
Conditions: `paging_filter_aware`
Devices: `2,3`

Each condition/query-pair/repeat is isolated in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_second_query.csv`.
