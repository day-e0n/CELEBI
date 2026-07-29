#!/usr/bin/env bash
# Fills in N=10,30,50 to go with the existing (correctly-run, single-process,
# 5-execution) N=20,40,60 data, giving a denser 6-point grid safely below the
# N=80 OOM boundary.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="experiment/expB_scaled_real"
DEVICES=0
EXECUTIONS=5
SIZES="10 30 50"

run_condition() {
  local n=$1 cond=$2 out_dir=$3
  echo "=== N=$n: $cond (single process, $EXECUTIONS executions) ==="
  pixi run -e duckdb-python python scripts/run_scaled_real_execution.py \
    --n "$n" --condition "$cond" --executions "$EXECUTIONS" --devices "$DEVICES" \
    --output "$out_dir"
  local rc=$?
  echo "=== N=$n: $cond exit=$rc ==="
  return $rc
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
  grep "^TOTAL" "$out_dir/comparison.csv"
}

for n in $SIZES; do
  rm -rf "${OUT_ROOT:?}/N$n"
  run_condition "$n" cold_hot "$OUT_ROOT/N$n" || { echo "STOPPED at N=$n cold_hot"; exit 1; }
  run_condition "$n" paging "$OUT_ROOT/N$n" || { echo "STOPPED at N=$n paging"; exit 1; }
  finish_n "$n"
done

echo "ALL FILL SIZES DONE"
