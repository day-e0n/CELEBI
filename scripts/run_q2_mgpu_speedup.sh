#!/usr/bin/env bash
# wdy start
# Run TPC-H Q2 on 1 GPU and 4 GPUs for repeated measurements, then draw a speedup graph.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT=""
OUTPUT_ROOT="$PROJECT_DIR/mgpu_speedup_runs/q2_$(date +%Y%m%d_%H%M%S)"
ITERATIONS=10
ONEGPU_DEVICE="0"
FOURGPU_DEVICES="0,1,2,3"
GPU_USAGE_LIMIT="4GB"
HOST_CAPACITY="16GB"
RESERVATION_LIMIT_FRACTION="0.8"
MODE="grouped"
SKIP_BUILD=0

usage() {
  cat <<USAGE
Usage: $0 --input <tpch_parquet_dir> [options]

Options:
  --output <dir>              Output root (default: mgpu_speedup_runs/q2_<timestamp>)
  --iterations <N>            Repetitions per experiment (default: $ITERATIONS)
  --onegpu-device <id>        CUDA_VISIBLE_DEVICES for 1-GPU run (default: $ONEGPU_DEVICE)
  --fourgpu-devices <ids>     CUDA_VISIBLE_DEVICES for 4-GPU run (default: $FOURGPU_DEVICES)
  --gpu-usage-limit <bytes>   Config memory.gpu.usage_limit_bytes (default: $GPU_USAGE_LIMIT)
  --host-capacity <bytes>     Config memory.host.capacity_bytes (default: $HOST_CAPACITY)
  --mode <mode>               grouped|sequential|isolated (default: $MODE)
  --skip-build                Do not run pixi run make -j4 first
  -h, --help                  Show this help

Example:
  $0 --input /data/tpch/sf100 --fourgpu-devices 0,1,2,3 --iterations 10
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --onegpu-device) ONEGPU_DEVICE="$2"; shift 2 ;;
    --fourgpu-devices) FOURGPU_DEVICES="$2"; shift 2 ;;
    --gpu-usage-limit) GPU_USAGE_LIMIT="$2"; shift 2 ;;
    --host-capacity) HOST_CAPACITY="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
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
case "$MODE" in
  grouped|sequential|isolated) ;;
  *) echo "ERROR: --mode must be grouped, sequential, or isolated" >&2; exit 1 ;;
esac

INPUT="$(cd "$INPUT" && pwd)"
OUTPUT_ROOT="$(mkdir -p "$OUTPUT_ROOT" && cd "$OUTPUT_ROOT" && pwd)"
mkdir -p "$OUTPUT_ROOT/configs" "$OUTPUT_ROOT/graphs"

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
FOURGPU_CONFIG="$OUTPUT_ROOT/configs/sirius_4gpu.yaml"
write_config "$ONEGPU_CONFIG" 1
write_config "$FOURGPU_CONFIG" 4

cat > "$OUTPUT_ROOT/README.md" <<EOF
# TPC-H Q2 multi-GPU speedup run

Input: \`$INPUT\`
Iterations: \`$ITERATIONS\`
Mode: \`$MODE\`
1GPU devices: \`$ONEGPU_DEVICE\`
4GPU devices: \`$FOURGPU_DEVICES\`

Runtime CSVs:
- \`q2_1gpu/csv/runtimes.csv\`
- \`q2_4gpu/csv/runtimes.csv\`

Graph:
- \`graphs/q2_speedup.svg\`
- \`graphs/q2_speedup_summary.csv\`
EOF

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "==> Building Sirius"
  (cd "$PROJECT_DIR" && pixi run make -j4)
fi

run_experiment() {
  local name="$1"
  local devices="$2"
  local config="$3"

  echo "==> Running $name with CUDA_VISIBLE_DEVICES=$devices"
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
      --queries 2 \
      --config "$config" \
      --output "$OUTPUT_ROOT" \
      --name "$name"
  )

  "$PROJECT_DIR/scripts/summarize_locality_audit.py" \
    --benchmark-dir "$OUTPUT_ROOT/$name" \
    --experiment "$name"
}

run_experiment q2_1gpu "$ONEGPU_DEVICE" "$ONEGPU_CONFIG"
run_experiment q2_4gpu "$FOURGPU_DEVICES" "$FOURGPU_CONFIG"

"$PROJECT_DIR/scripts/plot_mgpu_speedup.py" \
  --one-gpu-runtime "$OUTPUT_ROOT/q2_1gpu/csv/runtimes.csv" \
  --four-gpu-runtime "$OUTPUT_ROOT/q2_4gpu/csv/runtimes.csv" \
  --query 2 \
  --out-dir "$OUTPUT_ROOT/graphs" \
  --title "TPC-H Q2 1GPU vs 4GPU Runtime"

echo "==> Done"
echo "Output root: $OUTPUT_ROOT"
echo "Graph: $OUTPUT_ROOT/graphs/q2_speedup.svg"
echo "Summary: $OUTPUT_ROOT/graphs/q2_speedup_summary.csv"
# wdy end
