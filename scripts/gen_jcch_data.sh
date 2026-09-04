#!/usr/bin/env bash
# Generate a JCC-H (skewed TPC-H) parquet dataset for Sirius benchmarking.
#
#   scripts/gen_jcch_data.sh <SF> [OUT_DIR] [JCCH_DIR]
#
# JCC-H is schema-identical to TPC-H, so OUT_DIR drops straight into
# performance_test.py --input. Data skew comes from dbgen's -k flag; the
# matching skewed query constants come from scripts/gen_jcch_queries.py.
set -euo pipefail

SF="${1:?usage: gen_jcch_data.sh <SF> [OUT_DIR] [JCCH_DIR]}"
OUT="${2:-/mnt/nvme/sirius_tpch/jcch_parquet_sf${SF}}"
JCCH="${3:-$HOME/dbgen.JCC-H}"
TBL="${JCCH_TBL_DIR:-/mnt/nvme/sirius_tpch/jcch_tbl_sf${SF}}"
JOBS="${JCCH_JOBS:-$(nproc)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -x "$JCCH/dbgen" ]] || { echo "building dbgen in $JCCH"; make -C "$JCCH" >/dev/null; }

echo "=== 1/3  dbgen -k  (SF=$SF, ${JOBS} chunks) -> $TBL"
mkdir -p "$TBL"
if [[ "$JOBS" -gt 1 ]]; then
  # -C/-S shard the generation; each child writes <table>.tbl.<S>.
  for s in $(seq 1 "$JOBS"); do
    ( cd "$TBL" && DSS_PATH="$TBL" DSS_CONFIG="$JCCH" "$JCCH/dbgen" -k -f -s "$SF" -C "$JOBS" -S "$s" >/dev/null 2>&1 ) &
  done
  wait
else
  ( cd "$TBL" && DSS_PATH="$TBL" DSS_CONFIG="$JCCH" "$JCCH/dbgen" -k -f -s "$SF" >/dev/null 2>&1 )
fi
du -sh "$TBL"

echo "=== 2/3  tbl -> parquet -> $OUT.raw"
pixi run -e duckdb-python python "$ROOT/scripts/gen_jcch_parquet.py" "$TBL" "$OUT.raw"

# DuckDB's parquet writer emits V1 PLAIN_DICTIONARY pages. Sirius's GPU decoder is
# built for the V2 RLE_DICTIONARY layout the rest of the benchmark datasets use, and
# feeding it V1 measured 6.9x slower end to end on JCC-H SF50 (GPU 145s vs 21s on the
# byte-identical V2 data). rewrite_parquet.py is the documented step that produces the
# V2 layout -- it is not optional for anything that will be benchmarked.
echo "=== 3/3  GPU-optimized V2 rewrite -> $OUT"
( cd "$ROOT/test/tpch_performance" && pixi run python rewrite_parquet.py "$OUT.raw" "$OUT" 10000000 )
echo "raw V1 staging left at $OUT.raw (safe to delete)"

echo
echo "done. benchmark with:"
echo "  pixi run -e duckdb-python python test/tpch_performance/performance_test.py \\"
echo "      --input $OUT --data-source parquet"
echo "keep or remove the raw .tbl staging at $TBL"
