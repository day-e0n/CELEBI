# TPC-H observed scan-audit query-pair run

Input: `/mnt/nvme/dataset`
Pairs: `q1->q2, q1->q3, q1->q4, q1->q5, q1->q6, q1->q7, q1->q8, q1->q9, q1->q10, q1->q11, q1->q12, q1->q13, q1->q14, q1->q15, q1->q16, q1->q17, q1->q18, q1->q19, q1->q20, q1->q21, q1->q22, q2->q1, q2->q3, q2->q4, q2->q5, q2->q6, q2->q7, q2->q8, q2->q9, q2->q10, q2->q11, q2->q12, q2->q13, q2->q14, q2->q15, q2->q16, q2->q17, q2->q18, q2->q19, q2->q20, q2->q21, q2->q22, q3->q1, q3->q2, q3->q4, q3->q5, q3->q6, q3->q7, q3->q8, q3->q9, q3->q10, q3->q11, q3->q12, q3->q13, q3->q14, q3->q15, q3->q16, q3->q17, q3->q18, q3->q19, q3->q20, q3->q21, q3->q22, q4->q1, q4->q2, q4->q3, q4->q5, q4->q6, q4->q7, q4->q8, q4->q9, q4->q10, q4->q11, q4->q12, q4->q13, q4->q14, q4->q15, q4->q16, q4->q17, q4->q18, q4->q19, q4->q20, q4->q21, q4->q22, q5->q1, q5->q2, q5->q3, q5->q4, q5->q6, q5->q7, q5->q8, q5->q9, q5->q10, q5->q11, q5->q12, q5->q13, q5->q14, q5->q15, q5->q16, q5->q17, q5->q18, q5->q19, q5->q20, q5->q21, q5->q22, q6->q1, q6->q2, q6->q3, q6->q4, q6->q5, q6->q7, q6->q8, q6->q9, q6->q10, q6->q11, q6->q12, q6->q13, q6->q14, q6->q15, q6->q16, q6->q17, q6->q18, q6->q19, q6->q20, q6->q21, q6->q22, q7->q1, q7->q2, q7->q3, q7->q4, q7->q5, q7->q6, q7->q8, q7->q9, q7->q10, q7->q11, q7->q12, q7->q13, q7->q14, q7->q15, q7->q16, q7->q17, q7->q18, q7->q19, q7->q20, q7->q21, q7->q22, q8->q1, q8->q2, q8->q3, q8->q4, q8->q5, q8->q6, q8->q7, q8->q9, q8->q10, q8->q11, q8->q12, q8->q13, q8->q14, q8->q15, q8->q16, q8->q17, q8->q18, q8->q19, q8->q20, q8->q21, q8->q22, q9->q1, q9->q2, q9->q3, q9->q4, q9->q5, q9->q6, q9->q7, q9->q8, q9->q10, q9->q11, q9->q12, q9->q13, q9->q14, q9->q15, q9->q16, q9->q17, q9->q18, q9->q19, q9->q20, q9->q21, q9->q22, q10->q1, q10->q2, q10->q3, q10->q4, q10->q5, q10->q6, q10->q7, q10->q8, q10->q9, q10->q11, q10->q12, q10->q13, q10->q14, q10->q15, q10->q16, q10->q17, q10->q18, q10->q19, q10->q20, q10->q21, q10->q22, q11->q1, q11->q2, q11->q3, q11->q4, q11->q5, q11->q6, q11->q7, q11->q8, q11->q9, q11->q10, q11->q12, q11->q13, q11->q14, q11->q15, q11->q16, q11->q17, q11->q18, q11->q19, q11->q20, q11->q21, q11->q22, q12->q1, q12->q2, q12->q3, q12->q4, q12->q5, q12->q6, q12->q7, q12->q8, q12->q9, q12->q10, q12->q11, q12->q13, q12->q14, q12->q15, q12->q16, q12->q17, q12->q18, q12->q19, q12->q20, q12->q21, q12->q22, q13->q1, q13->q2, q13->q3, q13->q4, q13->q5, q13->q6, q13->q7, q13->q8, q13->q9, q13->q10, q13->q11, q13->q12, q13->q14, q13->q15, q13->q16, q13->q17, q13->q18, q13->q19, q13->q20, q13->q21, q13->q22, q14->q1, q14->q2, q14->q3, q14->q4, q14->q5, q14->q6, q14->q7, q14->q8, q14->q9, q14->q10, q14->q11, q14->q12, q14->q13, q14->q15, q14->q16, q14->q17, q14->q18, q14->q19, q14->q20, q14->q21, q14->q22, q15->q1, q15->q2, q15->q3, q15->q4, q15->q5, q15->q6, q15->q7, q15->q8, q15->q9, q15->q10, q15->q11, q15->q12, q15->q13, q15->q14, q15->q16, q15->q17, q15->q18, q15->q19, q15->q20, q15->q21, q15->q22, q16->q1, q16->q2, q16->q3, q16->q4, q16->q5, q16->q6, q16->q7, q16->q8, q16->q9, q16->q10, q16->q11, q16->q12, q16->q13, q16->q14, q16->q15, q16->q17, q16->q18, q16->q19, q16->q20, q16->q21, q16->q22, q17->q1, q17->q2, q17->q3, q17->q4, q17->q5, q17->q6, q17->q7, q17->q8, q17->q9, q17->q10, q17->q11, q17->q12, q17->q13, q17->q14, q17->q15, q17->q16, q17->q18, q17->q19, q17->q20, q17->q21, q17->q22, q18->q1, q18->q2, q18->q3, q18->q4, q18->q5, q18->q6, q18->q7, q18->q8, q18->q9, q18->q10, q18->q11, q18->q12, q18->q13, q18->q14, q18->q15, q18->q16, q18->q17, q18->q19, q18->q20, q18->q21, q18->q22, q19->q1, q19->q2, q19->q3, q19->q4, q19->q5, q19->q6, q19->q7, q19->q8, q19->q9, q19->q10, q19->q11, q19->q12, q19->q13, q19->q14, q19->q15, q19->q16, q19->q17, q19->q18, q19->q20, q19->q21, q19->q22, q20->q1, q20->q2, q20->q3, q20->q4, q20->q5, q20->q6, q20->q7, q20->q8, q20->q9, q20->q10, q20->q11, q20->q12, q20->q13, q20->q14, q20->q15, q20->q16, q20->q17, q20->q18, q20->q19, q20->q21, q20->q22, q21->q1, q21->q2, q21->q3, q21->q4, q21->q5, q21->q6, q21->q7, q21->q8, q21->q9, q21->q10, q21->q11, q21->q12, q21->q13, q21->q14, q21->q15, q21->q16, q21->q17, q21->q18, q21->q19, q21->q20, q21->q22, q22->q1, q22->q2, q22->q3, q22->q4, q22->q5, q22->q6, q22->q7, q22->q8, q22->q9, q22->q10, q22->q11, q22->q12, q22->q13, q22->q14, q22->q15, q22->q16, q22->q17, q22->q18, q22->q19, q22->q20, q22->q21`
Iterations: `1`
Devices: `0,1,2,3`
Config: `configs/sirius_4gpu.yaml`
Join output retention: `False`
Join output retention limit bytes: `8589934592`
Join output retention max batch bytes: `268435456`
Join output reuse: `False`

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
