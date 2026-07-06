# Fixed-page paging suite

Created: 2026-07-03T12:36:23
Input: `/mnt/nvme/dataset`
Queries: `q3,q5,q7,q8,q9,q10,q18,q21`
Conditions: `baseline,paging_key_only,paging_filter_aware`
Repeats: `1`
Devices: `0,1`

Layout:
- `pairs/`: every ordered query pair, excluding same-query pairs.
- `random/`: generated random workloads from the same query pool.
- `summary/`: copied headline CSVs for quick plotting/notebook use.

Random workload design:
- query pool: `q3,q5,q7,q8,q9,q10,q18,q21`
- workload count: `8`
- workload length: `8`
- seed: `20260703`
- sampling: `with_replacement`

Primary files:
- `summary/headline.csv`
- `summary/pairs_baseline_vs_paging_second_query.csv`
- `summary/random_baseline_vs_paging_workload.csv`
- `summary/random_workloads.csv`
