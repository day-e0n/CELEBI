#!/usr/bin/env bash
# Re-runs ALL Figure-4b sizes (20,40,60,80,100,120,140) fresh, with a single
# uniform methodology (per-execution-process + disk downgrade tier +
# downgrade_trigger_fraction=0.8), so the whole grid is comparable. Earlier
# in this session, N=20/40/60 were run with the old single-process/no-disk-
# tier config, N=100 predates this session entirely, and N=80/120/140 needed
# the new config to avoid a GPU OOM crash -- mixing those methodologies
# would confound "effect of N" with "effect of which script version ran it".
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="experiment/expB_scaled_real"
DEVICES=0
SIZES="20 40 60 80 100 120 140"

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
  rm -rf "${OUT_ROOT:?}/N$n"
  run_condition_per_exec "$n" cold_hot "$OUT_ROOT/N$n"
  run_condition_per_exec "$n" paging "$OUT_ROOT/N$n"
  finish_n "$n"
done

echo "ALL SIZES DONE (uniform methodology)"
