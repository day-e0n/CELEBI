#!/usr/bin/env bash
# Runs the Figure-4b grid (20,40,60,80,100,120,140) with the CORRECT cold/hot
# semantics restored: each condition runs as ONE continuous process/session
# with --executions 5 (exec1=cold, exec2-5 averaged=hot), so the fixed-page
# cache built up in exec1 stays resident and gets reused by exec2-5 -- the
# per-execution-process split tried earlier avoided the GPU OOM crash but
# silently broke this (every execution started with an empty cache).
#
# Keeps the disk-downgrade-tier + downgrade_trigger_fraction=0.8 config fixes
# (both harmless, and shown to at least delay the crash) but does NOT
# reintroduce the process-per-execution workaround. Runs sizes in ASCENDING
# order and stops at the first crash so we know exactly where the (still not
# fully understood) OOM boundary sits under this config.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="experiment/expB_scaled_real"
DEVICES=0
EXECUTIONS=5
SIZES="20 40 60 80 100 120 140"

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

echo "ALL SIZES DONE (single-process, 5 executions, correct hot semantics)"
