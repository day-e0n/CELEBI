# Fixed-page page-size sweep

Created: `2026-07-07T01:48:09`
Input: `/mnt/nvme/dataset`
Page sizes: `1m,4m,8m,16m,64m`
Queries: `3,5,7,8,10,21`
Conditions: `baseline,paging_key_only`

Each page size is a nested `run_fixed_page_suite.py` output.
Sweep-level summary CSVs live under `summary/`.
