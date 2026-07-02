#!/usr/bin/env bash
# wdy start
set -u

RUN_DIR="${1:?usage: $0 RUN_DIR [FAILED_PAIR_TO_SKIP] [START_QI]}"
FAILED_PAIR_TO_SKIP="${2:-}"
START_QI="${3:-1}"
CONFIG="$RUN_DIR/configs/sirius_4gpu.yaml"
PAIRS_DIR="$RUN_DIR/pairs"
INPUT="${INPUT:-/mnt/nvme/dataset}"
FAILED_LOG="$RUN_DIR/failed_pairs.csv"
PAIR_TIMEOUT="${PAIR_TIMEOUT:-900}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SIRIUS_CONFIG_FILE="$CONFIG"
export SIRIUS_LOG_LEVEL="${SIRIUS_LOG_LEVEL:-info}"

cd "$(dirname "$0")/.."

if [ ! -f "$FAILED_LOG" ]; then
  echo "pair,status" > "$FAILED_LOG"
fi

for qi in $(seq "$START_QI" 22); do
  for qj in $(seq 1 22); do
    [ "$qi" -eq "$qj" ] && continue
    name="q${qi}_then_q${qj}"
    csv="$PAIRS_DIR/$name/csv/runtimes.csv"

    if [ "$name" = "$FAILED_PAIR_TO_SKIP" ]; then
      if ! grep -q "^$name," "$FAILED_LOG"; then
        echo "$name,skipped_after_cuda_error" >> "$FAILED_LOG"
      fi
      echo "[FAILED-SKIP] $name"
      continue
    fi

    if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -gt 1 ]; then
      echo "[SKIP] $name"
      continue
    fi

    echo "[RUN] q${qi}->q${qj}"
    if timeout --kill-after=30s "$PAIR_TIMEOUT" pixi run python test/tpch_performance/performance_test.py \
      --input "$INPUT" \
      --engine gpu \
      --mode sequential \
      --iterations 1 \
      --queries "${qi},${qj}" \
      --config "$CONFIG" \
      --output "$PAIRS_DIR" \
      --name "$name"; then
      echo "[OK] $name"
    else
      echo "$name,failed" >> "$FAILED_LOG"
      echo "[FAILED] $name"
    fi
  done
done

echo "==> Resume done."
# wdy end
