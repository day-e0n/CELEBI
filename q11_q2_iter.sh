#!/usr/bin/env bash
# wdy start
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-experiment/fixed_page_hot_sweep_20260702}"
INPUT_DIR="${INPUT_DIR:-/mnt/nvme/dataset}"
CONFIG="${CONFIG:-experiment/fixed_width_page_pinned_hot_formal_20260702/configs/sirius_4gpu.yaml}"
GPUS="${GPUS:-0,1,2,3}"
REPS="${REPS:-5}"

ROWS_LIST=(100000 200000 500000 1000000 2000000 5000000)

mkdir -p "${OUT_ROOT}"

echo "[INFO] output root: ${OUT_ROOT}"
echo "[INFO] input dir:   ${INPUT_DIR}"
echo "[INFO] config:      ${CONFIG}"
echo "[INFO] gpus:        ${GPUS}"
echo "[INFO] reps:        ${REPS}"

run_no_pin_baseline() {
  local rep="$1"
  local name="q11_then_q2_no_pin_rep${rep}"

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
      --queries 11,2 \
      --config "${CONFIG}" \
      --output "${OUT_ROOT}" \
      --name "${name}" \
      --pin none
}

run_partial_pin_reuse() {
  local rows="$1"
  local rep="$2"
  local name="q11_then_q2_partial_${rows}_rep${rep}"

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
      --queries 11,2 \
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
    m = re.fullmatch(r"q11_then_q2_no_pin_rep(\d+)", name)
    if m:
        return {
            "condition": "no_pin",
            "n_rows": "",
            "rep": int(m.group(1)),
        }

    m = re.fullmatch(r"q11_then_q2_partial_(\d+)_rep(\d+)", name)
    if m:
        return {
            "condition": "partial_pin_reuse",
            "n_rows": int(m.group(1)),
            "rep": int(m.group(2)),
        }

    return None

runtime_files = sorted(out_root.glob("*/csv/runtimes.csv"))

if not runtime_files:
    raise SystemExit(f"No runtimes.csv files found under {out_root}")

for runtime_file in runtime_files:
    run_dir = runtime_file.parents[1]
    run_name = run_dir.name
    meta = parse_run_name(run_name)

    if meta is None:
        print(f"[WARN] skip unknown run directory: {run_name}")
        continue

    query_times = {}

    with runtime_file.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            runtime_s = float(row["runtime_s"])
            query_times[query] = runtime_s

            rec = {
                "run_name": run_name,
                "condition": meta["condition"],
                "n_rows": meta["n_rows"],
                "rep": meta["rep"],
                "metric": query,
                "runtime_s": runtime_s,
            }
            run_records.append(rec)
            group_values[(meta["condition"], meta["n_rows"], query)].append(runtime_s)

    if query_times:
        total_s = sum(query_times.values())
        rec = {
            "run_name": run_name,
            "condition": meta["condition"],
            "n_rows": meta["n_rows"],
            "rep": meta["rep"],
            "metric": "total",
            "runtime_s": total_s,
        }
        run_records.append(rec)
        group_values[(meta["condition"], meta["n_rows"], "total")].append(total_s)

if not run_records:
    raise SystemExit("No recognized runs found. Check --name patterns.")

with all_runs_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_name",
            "condition",
            "n_rows",
            "rep",
            "metric",
            "runtime_s",
        ],
    )
    writer.writeheader()
    writer.writerows(run_records)

# Baseline averages for optional relative comparison.
baseline_avg = {}
for (condition, n_rows, metric), values in group_values.items():
    if condition == "no_pin":
        baseline_avg[metric] = mean(values)

def sort_key(item):
    condition, n_rows, metric = item[0]
    condition_rank = 0 if condition == "no_pin" else 1
    n_rows_rank = -1 if n_rows == "" else int(n_rows)
    metric_rank = {"q11": 0, "q2": 1, "total": 2}.get(metric, 99)
    return (condition_rank, n_rows_rank, metric_rank)

summary_rows = []

for (condition, n_rows, metric), values in sorted(group_values.items(), key=sort_key):
    avg_s = mean(values)
    base = baseline_avg.get(metric)

    if base and condition != "no_pin":
        delta_vs_no_pin_avg_s = avg_s - base
        speedup_vs_no_pin_avg = base / avg_s if avg_s > 0 else ""
    else:
        delta_vs_no_pin_avg_s = ""
        speedup_vs_no_pin_avg = ""

    summary_rows.append({
        "condition": condition,
        "n_rows": n_rows,
        "metric": metric,
        "count": len(values),
        "min_s": f"{min(values):.6f}",
        "max_s": f"{max(values):.6f}",
        "avg_s": f"{avg_s:.6f}",
        "delta_vs_no_pin_avg_s": (
            f"{delta_vs_no_pin_avg_s:.6f}"
            if delta_vs_no_pin_avg_s != ""
            else ""
        ),
        "speedup_vs_no_pin_avg": (
            f"{speedup_vs_no_pin_avg:.4f}"
            if speedup_vs_no_pin_avg != ""
            else ""
        ),
    })

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "condition",
            "n_rows",
            "metric",
            "count",
            "min_s",
            "max_s",
            "avg_s",
            "delta_vs_no_pin_avg_s",
            "speedup_vs_no_pin_avg",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"[OK] wrote {all_runs_path}")
print(f"[OK] wrote {summary_path}")

print()
print("=== summary_min_max_avg.csv ===")
with summary_path.open() as f:
    print(f.read())
PY

echo "[DONE] sweep + summary complete"
# wdy end
