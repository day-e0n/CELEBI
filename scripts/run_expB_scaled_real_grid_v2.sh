#!/usr/bin/env bash
# Runs the remaining Figure-4b sizes (120, 140), matching Figure 4a's grid.
# Each execution runs as its own OS process (--executions 1 --execution-start K)
# rather than 3 executions inside one long-lived connection: a sustained
# multi-hundred-query session in the paging condition was found to exhaust
# GPU memory with nothing left for the downgrade path to reclaim (root cause
# still not fully understood -- possibly related to a segfault observed in
# RMM's cuMemFreeAsync during connection teardown, which may indicate
# per-query deallocation is also silently failing during long sessions).
# A fresh process per execution sidesteps this by guaranteeing a full CUDA
# context teardown between executions, at the cost of 3x process startup
# overhead (extension load, parquet view registration) per condition.
set -uo pipefail  # no -e: a single execution's teardown crash (after its data
                   # is already written) must not abort the remaining sizes

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="experiment/expB_scaled_real"
DEVICES=0
SIZES="120 140"

run_condition_per_exec() {
  local n=$1 cond=$2 out_dir=$3
  for e in 1 2 3; do
    echo "=== N=$n: $cond exec=$e ==="
    pixi run -e duckdb-python python scripts/run_scaled_real_execution.py \
      --n "$n" --condition "$cond" --executions 1 --execution-start "$e" --devices "$DEVICES" \
      --output "$out_dir"
    echo "=== N=$n: $cond exec=$e exit=$? ==="
  done
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
  run_condition_per_exec "$n" cold_hot "$OUT_ROOT/N$n"
  run_condition_per_exec "$n" paging "$OUT_ROOT/N$n"
  finish_n "$n"
done

echo "ALL REMAINING SIZES DONE"
