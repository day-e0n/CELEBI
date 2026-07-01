# Sirius locality experiment outputs

This directory collects generated experiment artifacts from the Sirius multi-GPU locality work.

- `locality_baseline_runs/`: 1-GPU, multi-GPU SCHED-RR, and pinned-hot locality baseline outputs.
- `mgpu_smoke_runs/`: quick checks that multiple GPUs are actually dispatched.
- `mgpu_speedup_runs/`: Q2 1-GPU vs 4-GPU runtime, dispatch, locality, and memory graphs.
- `tpch_overlap_heatmaps/`: static TPC-H query-pair table/column overlap heatmaps.
- `tpch_pair_scan_audit_runs/`: observed query-pair scan reload and stage-audit experiments.
- `tpch_pair_scan_audit_runs/*/stage_exclusion/`: stage-output exclusion analysis such as JOIN/PARTITION/CONCAT exclusion.
- `notebooks/`: notebook-based graph inspection and plotting artifacts.
- `join_retention_runs/`: all-query JOIN output retention cost, reuse loss, and retention efficiency matrices.
- `sirius_test.yaml`: temporary Sirius experiment config.

Latest multi-pair stage exclusion run:

- `tpch_pair_scan_audit_runs/pair_scan_20260630_064409/stage_exclusion/stage_exclusion_by_pair.csv`
- `tpch_pair_scan_audit_runs/pair_scan_20260630_064409/stage_exclusion/stage_exclusion_loss_ratio.png`
- `tpch_pair_scan_audit_runs/pair_scan_20260630_064409/stage_exclusion/stage_exclusion_reuse_upper_bound_gb.png`

Latest all-query JOIN retention run:

- `join_retention_runs/join_retention_20260630_075038/join_retention_matrix/join_retention_cost_gb_heatmap.png`
- `join_retention_runs/join_retention_20260630_075038/join_retention_matrix/join_reuse_loss_gb_heatmap.png`
- `join_retention_runs/join_retention_20260630_075038/join_retention_matrix/join_retention_efficiency_heatmap.png`
