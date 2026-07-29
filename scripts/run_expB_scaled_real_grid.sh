#!/usr/bin/env bash
# Runs the Figure-4b real-execution scaling experiment across the SAME N grid
# as Figure 4a (20,40,60,80,100,120,140), reusing the existing N=100 run and
# only executing the new sizes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SIZES="20 40 60 80 120 140"
OUT_ROOT="experiment/expB_scaled_real"
EXECUTIONS=3
DEVICES=0

for n in $SIZES; do
  out_dir="$OUT_ROOT/N${n}"
  echo "=== N=$n: cold_hot ==="
  pixi run -e duckdb-python python scripts/run_scaled_real_execution.py \
    --n "$n" --condition cold_hot --executions "$EXECUTIONS" --devices "$DEVICES" \
    --output "$out_dir"

  echo "=== N=$n: paging ==="
  pixi run -e duckdb-python python scripts/run_scaled_real_execution.py \
    --n "$n" --condition paging --executions "$EXECUTIONS" --devices "$DEVICES" \
    --output "$out_dir"

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
done

echo "ALL SIZES DONE"
