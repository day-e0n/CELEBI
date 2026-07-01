# TPC-H JOIN retention audit

Input: `/mnt/nvme/dataset`
Queries: `q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22`
Devices: `0,1,2,3`
Config: `configs/sirius_4gpu.yaml`

Outputs:
- `stage_summary/stage_audit_by_query_stage.csv`
- `join_retention_matrix/join_retention_cost_gb_heatmap.png`
- `join_retention_matrix/join_reuse_loss_gb_heatmap.png`
- `join_retention_matrix/join_retention_efficiency_heatmap.png`

Method: each query is executed once independently. All ordered query-pair JOIN
retention matrices are computed from per-query JOIN stage input/output bytes.
Diagonal cells and queries without JOIN output/input are blank.
