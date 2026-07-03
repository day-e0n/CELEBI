#!/usr/bin/env bash
# wdy start
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-experiment/q5_q8_partial_pin_sweep_20260702}"
INPUT_DIR="${INPUT_DIR:-/mnt/nvme/dataset}"
CONFIG="${CONFIG:-experiment/fixed_width_page_pinned_hot_formal_20260702/configs/sirius_4gpu.yaml}"
GPUS="${GPUS:-0,1,2,3}"
REPS="${REPS:-5}"

ROWS_LIST=(500000 1000000 2000000 3000000 5000000)

mkdir -p "${OUT_ROOT}"

echo "[INFO] output root: ${OUT_ROOT}"
echo "[INFO] input dir:   ${INPUT_DIR}"
echo "[INFO] config:      ${CONFIG}"
echo "[INFO] gpus:        ${GPUS}"
echo "[INFO] reps:        ${REPS}"

run_no_pin_baseline() {
  local rep="$1"
  local name="q5_then_q8_no_pin_rep${rep}"

  echo "[RUN] no_pin baseline rep=${rep}"

  env \
    -u SIRIUS_PIN_N_ROWS \
    -u SIRIUS_PIN_ONLY_SECOND_QUERY \
    SIRIUS_ENABLE_PARTIAL_PIN_REUSE=0 \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    pixi run python test/tpch_performance/performance_test.py \
      --input "${INPUT_DIR}" \
      --engine gpu \
      --mode grouped \
      --iterations 1 \
      --queries 5,8 \
      --config "${CONFIG}" \
      --output "${OUT_ROOT}" \
      --name "${name}" \
      --pin none
}

run_partial_pin_reuse() {
  local rows="$1"
  local rep="$2"
  local name="q5_then_q8_partial_${rows}_rep${rep}"

  echo "[RUN] partial_pin_reuse rows=${rows} rep=${rep}"

  env \
    SIRIUS_ENABLE_PARTIAL_PIN_REUSE=1 \
    SIRIUS_PIN_N_ROWS="${rows}" \
    SIRIUS_PIN_ONLY_SECOND_QUERY=1 \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    pixi run python test/tpch_performance/performance_test.py \
      --input "${INPUT_DIR}" \
      --engine gpu \
      --mode grouped \
      --iterations 1 \
      --queries 5,8 \
      --config "${CONFIG}" \
      --output "${OUT_ROOT}" \
      --name "${name}" \
      --pin gpu
}

echo "[PHASE] no_pin baseline"
for rep in $(seq 1 "${REPS}"); do
  run_no_pin_baseline "${rep}"
done

echo "[PHASE] partial_pin_reuse sweep"
for rows in "${ROWS_LIST[@]}"; do
  for rep in $(seq 1 "${REPS}"); do
    run_partial_pin_reuse "${rows}" "${rep}"
  done
done

echo "[DONE] Q5 -> Q8 sweep complete"
# wdy end
