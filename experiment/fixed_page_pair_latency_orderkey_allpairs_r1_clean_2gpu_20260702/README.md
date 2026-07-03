# Fixed-page pair latency experiment

Created: 2026-07-02T13:30:26
Input: `/mnt/nvme/dataset`
Queries: `q3,q5,q7,q8,q9,q10,q18,q21`
Pairs: `56`
Repeats: `1`
Conditions: `baseline,paging_key_only`
Devices: `0,1`

Each condition/query-pair/repeat is isolated in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_second_query.csv`.
