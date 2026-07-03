#!/usr/bin/env bash
# wdy start
set -uo pipefail

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-experiment/q5_q7_q8_partial_pin_sweep_2gpu_${RUN_STAMP}}"
INPUT_DIR="${INPUT_DIR:-/mnt/nvme/dataset}"
GPUS="${GPUS:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_USAGE_LIMIT="${GPU_USAGE_LIMIT:-12GB}"
HOST_CAPACITY="${HOST_CAPACITY:-32GB}"
RESERVATION_LIMIT_FRACTION="${RESERVATION_LIMIT_FRACTION:-0.85}"
REPS="${REPS:-5}"
PAIRS="${PAIRS:-5:8 7:8}"
ROWS_LIST="${ROWS_LIST:-10000 50000 100000 250000 500000 750000 1000000 1500000 2000000 3000000 4000000 6000000 8000000}"

CONFIG="${CONFIG:-${OUT_ROOT}/configs/sirius_${NUM_GPUS}gpu.yaml}"
FAILED_CSV="${OUT_ROOT}/failed_runs.csv"

mkdir -p "${OUT_ROOT}/configs"

cat > "${CONFIG}" <<YAML
sirius:
  topology:
    num_gpus: ${NUM_GPUS}
  memory:
    gpu:
      usage_limit_bytes: ${GPU_USAGE_LIMIT}
      reservation_limit_fraction: ${RESERVATION_LIMIT_FRACTION}
    host:
      capacity_bytes: ${HOST_CAPACITY}
YAML

if [ ! -f "${FAILED_CSV}" ]; then
  echo "pair,condition,n_rows,rep,returncode" > "${FAILED_CSV}"
fi

echo "[INFO] output root: ${OUT_ROOT}"
echo "[INFO] input dir:   ${INPUT_DIR}"
echo "[INFO] config:      ${CONFIG}"
echo "[INFO] gpus:        ${GPUS}"
echo "[INFO] num_gpus:    ${NUM_GPUS}"
echo "[INFO] pairs:       ${PAIRS}"
echo "[INFO] rows list:   ${ROWS_LIST}"
echo "[INFO] reps:        ${REPS}"

pair_name() {
  local pair="$1"
  local left="${pair%%:*}"
  local right="${pair##*:}"
  echo "q${left}_then_q${right}"
}

run_case() {
  local pair="$1"
  local condition="$2"
  local rows="$3"
  local rep="$4"

  local qspec="${pair/:/,}"
  local base
  base="$(pair_name "${pair}")"

  local name
  local pin_arg="none"

  if [ "${condition}" = "no_pin" ]; then
    name="${base}_no_pin_rep${rep}"
    echo "[RUN] ${name}"
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
        --queries "${qspec}" \
        --config "${CONFIG}" \
        --output "${OUT_ROOT}" \
        --name "${name}" \
        --pin "${pin_arg}"
  else
    name="${base}_partial_${rows}_rep${rep}"
    pin_arg="gpu"
    echo "[RUN] ${name}"
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
        --queries "${qspec}" \
        --config "${CONFIG}" \
        --output "${OUT_ROOT}" \
        --name "${name}" \
        --pin "${pin_arg}"
  fi
}

run_case_recording_failure() {
  local pair="$1"
  local condition="$2"
  local rows="$3"
  local rep="$4"

  run_case "${pair}" "${condition}" "${rows}" "${rep}"
  local code=$?
  if [ "${code}" -ne 0 ]; then
    echo "${pair},${condition},${rows},${rep},${code}" >> "${FAILED_CSV}"
    echo "[WARN] failed pair=${pair} condition=${condition} rows=${rows} rep=${rep} code=${code}"
  fi
}

echo "[PHASE] no_pin baseline"
for pair in ${PAIRS}; do
  for rep in $(seq 1 "${REPS}"); do
    run_case_recording_failure "${pair}" "no_pin" "" "${rep}"
  done
done

echo "[PHASE] partial_pin_reuse sweep"
for pair in ${PAIRS}; do
  for rows in ${ROWS_LIST}; do
    for rep in $(seq 1 "${REPS}"); do
      run_case_recording_failure "${pair}" "partial_pin_reuse" "${rows}" "${rep}"
    done
  done
done

echo "[PHASE] summary generation"

python3 - "${OUT_ROOT}" <<'PY'
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

out_root = Path(sys.argv[1])
all_runs_path = out_root / "summary_all_runs.csv"
summary_path = out_root / "summary_min_max_avg.csv"

run_records = []
group_values = defaultdict(list)


def parse_run_name(name: str):
    m = re.fullmatch(r"q(\d+)_then_q(\d+)_no_pin_rep(\d+)", name)
    if m:
        qi, qj, rep = m.groups()
        return {
            "pair": f"q{qi}->q{qj}",
            "previous_query": f"q{qi}",
            "second_query": f"q{qj}",
            "condition": "no_pin",
            "n_rows": "",
            "rep": int(rep),
        }

    m = re.fullmatch(r"q(\d+)_then_q(\d+)_partial_(\d+)_rep(\d+)", name)
    if m:
        qi, qj, rows, rep = m.groups()
        return {
            "pair": f"q{qi}->q{qj}",
            "previous_query": f"q{qi}",
            "second_query": f"q{qj}",
            "condition": "partial_pin_reuse",
            "n_rows": int(rows),
            "rep": int(rep),
        }

    return None


for runtime_file in sorted(out_root.glob("*/csv/runtimes.csv")):
    run_dir = runtime_file.parents[1]
    meta = parse_run_name(run_dir.name)
    if meta is None:
        print(f"[WARN] skip unknown run directory: {run_dir.name}")
        continue

    query_times = {}
    with runtime_file.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("engine") != "sirius":
                continue
            query_times[row["query"]] = float(row["runtime_s"])

    expected_queries = {meta["previous_query"], meta["second_query"]}
    if not expected_queries.issubset(query_times):
        seen = ",".join(sorted(query_times)) or "none"
        expected = ",".join(sorted(expected_queries))
        print(
            f"[WARN] skip incomplete run: {run_dir.name} "
            f"expected={expected} seen={seen}"
        )
        continue

    for query in sorted(expected_queries, key=lambda q: int(q[1:])):
        runtime_s = query_times[query]
        rec = {
            **meta,
            "run_name": run_dir.name,
            "metric": query,
            "runtime_s": runtime_s,
        }
        run_records.append(rec)
        group_values[
            (
                meta["pair"],
                meta["condition"],
                meta["n_rows"],
                query,
            )
        ].append(runtime_s)

    total_s = sum(query_times[q] for q in expected_queries)
    rec = {
        **meta,
        "run_name": run_dir.name,
        "metric": "total",
        "runtime_s": total_s,
    }
    run_records.append(rec)
    group_values[
        (
            meta["pair"],
            meta["condition"],
            meta["n_rows"],
            "total",
        )
    ].append(total_s)

if not run_records:
    raise SystemExit(f"No recognized successful runs found under {out_root}")

all_fields = [
    "run_name",
    "pair",
    "previous_query",
    "second_query",
    "condition",
    "n_rows",
    "rep",
    "metric",
    "runtime_s",
]
with all_runs_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    writer.writerows(run_records)

baseline_avg = {}
for (pair, condition, n_rows, metric), values in group_values.items():
    if condition == "no_pin":
        baseline_avg[(pair, metric)] = mean(values)


def metric_rank(metric: str) -> int:
    if metric == "total":
        return 999
    if metric.startswith("q") and metric[1:].isdigit():
        return int(metric[1:])
    return 500


def sort_key(item):
    pair, condition, n_rows, metric = item[0]
    condition_rank = 0 if condition == "no_pin" else 1
    n_rows_rank = -1 if n_rows == "" else int(n_rows)
    pair_nums = tuple(int(x[1:]) for x in pair.split("->"))
    return (*pair_nums, condition_rank, n_rows_rank, metric_rank(metric))


summary_rows = []
for (pair, condition, n_rows, metric), values in sorted(group_values.items(), key=sort_key):
    avg_s = mean(values)
    base = baseline_avg.get((pair, metric))
    if base and condition != "no_pin":
        delta = avg_s - base
        speedup = base / avg_s if avg_s > 0 else ""
    else:
        delta = ""
        speedup = ""
    qi, qj = pair.split("->")
    summary_rows.append(
        {
            "pair": pair,
            "previous_query": qi,
            "second_query": qj,
            "condition": condition,
            "n_rows": n_rows,
            "metric": metric,
            "count": len(values),
            "min_s": f"{min(values):.6f}",
            "max_s": f"{max(values):.6f}",
            "avg_s": f"{avg_s:.6f}",
            "delta_vs_no_pin_avg_s": f"{delta:.6f}" if delta != "" else "",
            "speedup_vs_no_pin_avg": f"{speedup:.4f}" if speedup != "" else "",
        }
    )

summary_fields = [
    "pair",
    "previous_query",
    "second_query",
    "condition",
    "n_rows",
    "metric",
    "count",
    "min_s",
    "max_s",
    "avg_s",
    "delta_vs_no_pin_avg_s",
    "speedup_vs_no_pin_avg",
]
with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"[OK] wrote {all_runs_path}")
print(f"[OK] wrote {summary_path}")
print()
print("=== summary_min_max_avg.csv ===")
print(summary_path.read_text())
PY

echo "[DONE] sweep + summary complete"
# wdy end
