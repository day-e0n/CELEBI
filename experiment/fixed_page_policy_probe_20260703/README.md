# Fixed-page pair latency experiment

Created: 2026-07-03T07:30:12
Input: `/mnt/nvme/dataset`
Queries: `q3,q5,q7,q8,q18,q21`
Pairs: `4`
Repeats: `1`
Conditions: `baseline,paging_key_only,paging_budget,paging_full_fixed`
Devices: `0,1`

Each condition/query-pair/repeat is isolated in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_second_query.csv`.
