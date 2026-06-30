# TPC-H observed scan-audit query-pair run

Input: `/mnt/nvme/dataset`
Pairs: `q2->q16`
Iterations: `1`
Devices: `0,1,2,3`
Config: `configs/sirius_4gpu.yaml`

Main summary files:
- `summary/scan_audit_by_pair_second_query.csv`
- `summary/observed_overlap_reload_ratio_heatmap.png`
- `summary/observed_second_query_materialized_gb_heatmap.png`
- `summary/observed_overlap_reload_gb_heatmap.png`

Interpretation: for each Qi->Qj pair, the overlap reload bytes are the Qj
materialized table/column bytes whose table/column also appeared in Qi's TPC-H
footprint.
