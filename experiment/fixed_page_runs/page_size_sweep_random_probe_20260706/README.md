# Fixed-page page-size sweep

Created: `2026-07-06T06:30:09`
Input: `/mnt/nvme/dataset`
Page sizes: `512k,1m,2m`
Queries: `3,5,7,8,9,10,18,21`
Conditions: `baseline,paging_key_only,paging_budget`

Each page size is a nested `run_fixed_page_suite.py` output.
Sweep-level summary CSVs live under `summary/`.
