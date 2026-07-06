# Fixed-page pair latency experiment

Created: 2026-07-03T12:28:02
Input: `/mnt/nvme/dataset`
Queries: `q5,q8`
Pairs: `2`
Repeats: `1`
Conditions: `baseline,paging_filter_aware`
Devices: `0,1`

Each condition/query-pair/repeat is isolated in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_second_query.csv`.
