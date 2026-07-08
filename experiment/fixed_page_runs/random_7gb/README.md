# Fixed-page random workload experiment

Created: 2026-07-08T03:29:08
Input: `/mnt/nvme/dataset`
Workloads: `7`
Repeats: `1`
Conditions: `baseline,pinned_hot,paging_filter_aware`
Devices: `0,1`
Seed: `20260703`

Each condition/workload/repeat runs in a separate Python process.
Primary comparison: `summary/baseline_vs_paging_workload.csv`.
