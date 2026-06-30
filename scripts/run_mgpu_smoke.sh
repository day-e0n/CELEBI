#!/usr/bin/env bash
# wdy start
# Minimal multi-GPU smoke test for Sirius parquet execution.
# Verifies that tasks are dispatched to more than one GPU by scanning locality-audit logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT=""
OUTPUT_ROOT="$PROJECT_DIR/mgpu_smoke_runs/$(date +%Y%m%d_%H%M%S)"
DEVICES="0,1"
QUERIES="6"
ITERATIONS=2
GPU_USAGE_LIMIT="4GB"
HOST_CAPACITY="16GB"
RESERVATION_LIMIT_FRACTION="0.8"
PIN="none"
SKIP_BUILD=0

usage() {
  cat <<USAGE
Usage: $0 --input <tpch_parquet_dir> [options]

Options:
  --output <dir>              Output root (default: mgpu_smoke_runs/<timestamp>)
  --devices <ids>             CUDA_VISIBLE_DEVICES list (default: $DEVICES)
  --queries <spec>            Query spec for performance_test.py (default: $QUERIES)
  --iterations <N>            Iterations per query (default: $ITERATIONS)
  --pin <none|gpu|host>       Pin mode forwarded to performance_test.py (default: $PIN)
  --gpu-usage-limit <bytes>   Config memory.gpu.usage_limit_bytes (default: $GPU_USAGE_LIMIT)
  --host-capacity <bytes>     Config memory.host.capacity_bytes (default: $HOST_CAPACITY)
  --skip-build                Do not run pixi run make -j4 first
  -h, --help                  Show this help

Example:
  $0 --input /path/to/tpch_parquet --devices 0,1 --queries 6 --iterations 2 --skip-build
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --pin) PIN="$2"; shift 2 ;;
    --gpu-usage-limit) GPU_USAGE_LIMIT="$2"; shift 2 ;;
    --host-capacity) HOST_CAPACITY="$2"; shift 2 ;;
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
case "$PIN" in
  none|gpu|host) ;;
  *) echo "ERROR: --pin must be one of: none, gpu, host" >&2; exit 1 ;;
esac

count_devices() {
  local csv="$1"
  awk -F',' '{print NF}' <<< "$csv"
}

INPUT="$(cd "$INPUT" && pwd)"
OUTPUT_ROOT="$(mkdir -p "$OUTPUT_ROOT" && cd "$OUTPUT_ROOT" && pwd)"
CONFIG="$OUTPUT_ROOT/sirius_mgpu.yaml"
NUM_GPUS="$(count_devices "$DEVICES")"

cat > "$CONFIG" <<YAML
sirius:
  topology:
    num_gpus: $NUM_GPUS
  memory:
    gpu:
      usage_limit_bytes: $GPU_USAGE_LIMIT
      reservation_limit_fraction: $RESERVATION_LIMIT_FRACTION
    host:
      capacity_bytes: $HOST_CAPACITY
YAML

cat > "$OUTPUT_ROOT/README.md" <<EOF
# Sirius multi-GPU smoke run

Input: \`$INPUT\`
Devices: \`$DEVICES\`
num_gpus: \`$NUM_GPUS\`
Queries: \`$QUERIES\`
Iterations: \`$ITERATIONS\`
Pin: \`$PIN\`
Config: \`$CONFIG\`
EOF

echo "==> Visible GPUs requested: $DEVICES"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "==> Building Sirius"
  (cd "$PROJECT_DIR" && pixi run make -j4)
fi

echo "==> Running multi-GPU smoke workload"
(
  cd "$PROJECT_DIR"
  export CUDA_VISIBLE_DEVICES="$DEVICES"
  export SIRIUS_CONFIG_FILE="$CONFIG"
  export SIRIUS_LOG_LEVEL=info
  pixi run python test/tpch_performance/performance_test.py \
    --input "$INPUT" \
    --engine gpu \
    --mode sequential \
    --iterations "$ITERATIONS" \
    --queries "$QUERIES" \
    --config "$CONFIG" \
    --pin "$PIN" \
    --output "$OUTPUT_ROOT" \
    --name mgpu_smoke
)

echo "==> Summarizing locality logs"
"$PROJECT_DIR/scripts/summarize_locality_audit.py" \
  --benchmark-dir "$OUTPUT_ROOT/mgpu_smoke" \
  --experiment mgpu_smoke || true

LOG_MATCHES="$OUTPUT_ROOT/locality_actual_gpus.txt"
grep -Rho "actual_gpu=[0-9-]*" "$OUTPUT_ROOT/mgpu_smoke" 2>/dev/null \
  | sort -u > "$LOG_MATCHES" || true

echo "==> actual_gpu values observed"
if [[ -s "$LOG_MATCHES" ]]; then
  cat "$LOG_MATCHES"
else
  echo "No actual_gpu entries found. Check Sirius logs under: $OUTPUT_ROOT/mgpu_smoke"
fi

OBSERVED_COUNT="$(wc -l < "$LOG_MATCHES" | tr -d ' ')"
if [[ "$OBSERVED_COUNT" -ge 2 ]]; then
  echo "PASS: locality logs show work on at least two GPUs."
else
  echo "WARN: locality logs did not show at least two actual GPUs."
  echo "      This can mean the query was too small, logs were not emitted, or multi-GPU was not active."
fi

echo "==> Output root: $OUTPUT_ROOT"
find "$OUTPUT_ROOT" -maxdepth 4 -type f \( -name 'runtimes.csv' -o -name '*summary.csv' -o -name '*.log' \) | sort
# wdy end
