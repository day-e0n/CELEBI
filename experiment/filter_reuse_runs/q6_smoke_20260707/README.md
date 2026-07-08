# Fixed-page pair latency experiment

Created: 2026-07-07T08:16:21
Input: `/mnt/nvme/dataset`
Queries: `q6`
Pairs: `1`
Repeats: `1`
Conditions: `baseline,paging_key_only,paging_filter_aware`
Devices: `0,1`

Each condition/query-pair/repeat is isolated in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_second_query.csv`.
