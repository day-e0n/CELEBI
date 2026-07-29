#!/usr/bin/env bash
# Resume run_expB_scaled_real_grid.sh after the N=80 paging OOM crash:
# N=20/40/60 already have comparison.csv (skip). N=80 cold_hot already
# succeeded (skip, redo paging only). N=120/140 run fresh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="experiment/expB_scaled_real"
EXECUTIONS=3
DEVICES=0

run_condition() {
  local n=$1 cond=$2 out_dir=$3
  echo "=== N=$n: $cond ==="
  pixi run -e duckdb-python python scripts/run_scaled_real_execution.py \
    --n "$n" --condition "$cond" --executions "$EXECUTIONS" --devices "$DEVICES" \
    --output "$out_dir"
}

finish_n() {
  local n=$1 out_dir="$OUT_ROOT/N$1"
  echo "=== N=$n: parse telemetry ==="
  pixi run python scripts/parse_quent_operator_breakdown.py \
    --telemetry-dir "$out_dir/cold_hot/telemetry_data" \
    --out-csv "$out_dir/cold_hot_raw.csv" \
    --out-bucket-csv "$out_dir/cold_hot_bucket.csv"
  pixi run python scripts/parse_quent_operator_breakdown.py \
    --telemetry-dir "$out_dir/paging/telemetry_data" \
    --out-csv "$out_dir/paging_raw.csv" \
    --out-bucket-csv "$out_dir/paging_bucket.csv"
  echo "=== N=$n: build comparison.csv ==="
  pixi run python scripts/build_operator_breakdown_comparison.py \
    --baseline-bucket-csv "$out_dir/cold_hot_bucket.csv" \
    --proposed-bucket-csv "$out_dir/paging_bucket.csv" \
    --out-csv "$out_dir/comparison.csv"
  echo "=== N=$n done ==="
}

# --- N=80: cold_hot already succeeded, only paging needs a fresh run ---
run_condition 80 paging "$OUT_ROOT/N80"
finish_n 80

# --- N=120, N=140: full fresh run ---
for n in 120 140; do
  run_condition "$n" cold_hot "$OUT_ROOT/N$n"
  run_condition "$n" paging "$OUT_ROOT/N$n"
  finish_n "$n"
done

echo "ALL REMAINING SIZES DONE"
