# TPC-H observed scan-audit query-pair run

Input: `/mnt/nvme/dataset`
Pairs: `q2->q16`
Iterations: `1`
Devices: `0,1,2,3`
Config: `configs/sirius_4gpu.yaml`
Join output retention: `True`
Join output retention limit bytes: `8589934592`
Join output retention max batch bytes: `268435456`

Main scan summary files:
- `summary/scan_audit_by_pair_second_query.csv`
- `summary/observed_overlap_reload_ratio_heatmap.png`
- `summary/observed_second_query_materialized_gb_heatmap.png`
- `summary/observed_overlap_reload_gb_heatmap.png`

Main stage summary files:
- `stage_summary/stage_audit_by_query_stage.csv`
- `stage_summary/stage_output_gb_by_query.png`
- `stage_summary/stage_byte_ratio_heatmap.png`

Main runtime breakdown files:
- `runtime_breakdown/runtime_breakdown.csv`
- `runtime_breakdown/runtime_load_nonload_breakdown.png`
- `runtime_breakdown/runtime_load_ratio.png`

Interpretation: for each Qi->Qj pair, the overlap reload bytes are the Qj
materialized table/column bytes whose table/column also appeared in Qi's TPC-H
footprint.
