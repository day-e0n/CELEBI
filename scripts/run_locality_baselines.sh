#!/usr/bin/env bash
# wdy start
# Run Sirius locality baseline experiments with comparable settings.
# Produces runtime CSVs, Sirius logs, generated configs, and locality summaries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT=""
OUTPUT_ROOT="$PROJECT_DIR/locality_baseline_runs/$(date +%Y%m%d_%H%M%S)"
QUERIES="1,3,6,9,10,12"
ITERATIONS=4
MODE="sequential"
MGPU_DEVICES="0,1"
ONEGPU_DEVICE="0"
GPU_USAGE_LIMIT="4GB"
HOST_CAPACITY="16GB"
RESERVATION_LIMIT_FRACTION="0.8"
RUNS="1gpu,mgpu,pinned-gpu"
SKIP_BUILD=0

usage() {
  cat <<USAGE
Usage: $0 --input <tpch_parquet_dir> [options]

Options:
  --output <dir>              Output root (default: locality_baseline_runs/<timestamp>)
  --queries <spec>            Query spec for performance_test.py (default: $QUERIES)
  --iterations <N>            Iterations per query (default: $ITERATIONS)
  --mode <mode>               grouped|sequential|isolated (default: $MODE)
  --mgpu-devices <ids>        CUDA_VISIBLE_DEVICES for multi-GPU runs (default: $MGPU_DEVICES)
  --onegpu-device <id>        CUDA_VISIBLE_DEVICES for 1-GPU run (default: $ONEGPU_DEVICE)
  --gpu-usage-limit <bytes>   Config memory.gpu.usage_limit_bytes (default: $GPU_USAGE_LIMIT)
  --host-capacity <bytes>     Config memory.host.capacity_bytes (default: $HOST_CAPACITY)
  --runs <list>               Comma list: 1gpu,mgpu,pinned-gpu,pinned-host (default: $RUNS)
  --skip-build                Do not run pixi run make -j4 first
  -h, --help                  Show this help

Examples:
  $0 --input /data/tpch/sf100 --mgpu-devices 0,1 --iterations 4
  $0 --input test_datasets/tpch_parquet_sf10 --runs mgpu,pinned-gpu --queries 1,6
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --mgpu-devices) MGPU_DEVICES="$2"; shift 2 ;;
    --onegpu-device) ONEGPU_DEVICE="$2"; shift 2 ;;
    --gpu-usage-limit) GPU_USAGE_LIMIT="$2"; shift 2 ;;
    --host-capacity) HOST_CAPACITY="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$INPUT" ]]; then
  echo "ERROR: --input is required" >&2
  usage
  exit 1
fi
if [[ ! -d "$INPUT" ]]; then
  echo "ERROR: input directory does not exist: $INPUT" >&2
  exit 1
fi
INPUT="$(cd "$INPUT" && pwd)"
OUTPUT_ROOT="$(mkdir -p "$OUTPUT_ROOT" && cd "$OUTPUT_ROOT" && pwd)"
case "$MODE" in
  grouped|sequential|isolated) ;;
  *) echo "ERROR: --mode must be grouped, sequential, or isolated" >&2; exit 1 ;;
esac

mkdir -p "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/summaries"

count_devices() {
  local csv="$1"
  awk -F',' '{print NF}' <<< "$csv"
}

write_config() {
  local path="$1"
  local num_gpus="$2"
  cat > "$path" <<YAML
sirius:
  topology:
    num_gpus: $num_gpus
  memory:
    gpu:
      usage_limit_bytes: $GPU_USAGE_LIMIT
      reservation_limit_fraction: $RESERVATION_LIMIT_FRACTION
    host:
      capacity_bytes: $HOST_CAPACITY
YAML
}

ONEGPU_CONFIG="$OUTPUT_ROOT/configs/sirius_1gpu.yaml"
MGPU_CONFIG="$OUTPUT_ROOT/configs/sirius_mgpu$(count_devices "$MGPU_DEVICES").yaml"
write_config "$ONEGPU_CONFIG" 1
write_config "$MGPU_CONFIG" "$(count_devices "$MGPU_DEVICES")"

cat > "$OUTPUT_ROOT/README.md" <<EOF
# Sirius locality baseline run

Input: \\`$INPUT\\`
Queries: \\`$QUERIES\\`
Iterations: \\`$ITERATIONS\\`
Mode: \\`$MODE\\`
Runs: \\`$RUNS\\`
1GPU devices: \\`$ONEGPU_DEVICE\\`
MGPU devices: \\`$MGPU_DEVICES\\`

Runtime CSVs live under each experiment's \\`csv/runtimes.csv\\`.
Locality summaries live under each experiment's \\`locality_summary/\\` and are copied into \\`summaries/\\`.
EOF

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "==> Building Sirius"
  (cd "$PROJECT_DIR" && pixi run make -j4)
fi

run_one() {
  local name="$1"
  local devices="$2"
  local config="$3"
  local pin="$4"

  echo "==> Running $name (CUDA_VISIBLE_DEVICES=$devices, pin=$pin)"
  (
    cd "$PROJECT_DIR"
    export CUDA_VISIBLE_DEVICES="$devices"
    export SIRIUS_CONFIG_FILE="$config"
    export SIRIUS_LOG_LEVEL=info
    pixi run python test/tpch_performance/performance_test.py \
      --input "$INPUT" \
      --engine gpu \
      --mode "$MODE" \
      --iterations "$ITERATIONS" \
      --queries "$QUERIES" \
      --config "$config" \
      --pin "$pin" \
      --output "$OUTPUT_ROOT" \
      --name "$name"
  )

  "$PROJECT_DIR/scripts/summarize_locality_audit.py" \
    --benchmark-dir "$OUTPUT_ROOT/$name" \
    --experiment "$name"

  cp "$OUTPUT_ROOT/$name/locality_summary/prepare_summary.csv" \
    "$OUTPUT_ROOT/summaries/${name}_prepare_summary.csv"
  cp "$OUTPUT_ROOT/$name/locality_summary/dispatch_summary.csv" \
    "$OUTPUT_ROOT/summaries/${name}_dispatch_summary.csv"
  cp "$OUTPUT_ROOT/$name/locality_summary/task_create_summary.csv" \
    "$OUTPUT_ROOT/summaries/${name}_task_create_summary.csv"
}

IFS=',' read -r -a RUN_ARRAY <<< "$RUNS"
for run in "${RUN_ARRAY[@]}"; do
  case "$run" in
    1gpu) run_one baseline_1gpu "$ONEGPU_DEVICE" "$ONEGPU_CONFIG" none ;;
    mgpu) run_one baseline_mgpu_schedrr "$MGPU_DEVICES" "$MGPU_CONFIG" none ;;
    pinned-gpu) run_one baseline_mgpu_pinned_gpu "$MGPU_DEVICES" "$MGPU_CONFIG" gpu ;;
    pinned-host) run_one baseline_mgpu_pinned_host "$MGPU_DEVICES" "$MGPU_CONFIG" host ;;
    *) echo "ERROR: unknown run '$run' in --runs" >&2; exit 1 ;;
  esac
done

echo "==> Done"
echo "Output root: $OUTPUT_ROOT"
echo "Runtime CSV examples:"
find "$OUTPUT_ROOT" -path '*/csv/runtimes.csv' -print | sort
echo "Locality summaries:"
find "$OUTPUT_ROOT/summaries" -type f -name '*.csv' -print | sort
# wdy end
