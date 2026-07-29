#!/usr/bin/env bash
# Regenerates experiment/expB_qgen_streams/stream_{1..10}.sql via the official
# TPC-H qgen binary (test_datasets/tpch-dbgen/qgen), one file per stream.
#
# qgen picks a fresh RANDOM seed every run unless -r <seed> is given explicitly
# (confirmed empirically: no -r => different substitution literals every
# invocation) -- so a fixed, recorded seed per stream is required for the
# output to be reproducible later. Seed for stream N is 1000+N.
#
# Query numbers are passed explicitly in ascending order (1 2 3 ... 22) rather
# than via qgen's -p <stream> permutation flag, because scripts/parse_qgen_streams.py
# assigns qnum by POSITION in the file (Q1..Q22 order), not by parsing a query
# number out of the SQL text.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DBGEN_DIR="$REPO_ROOT/test_datasets/tpch-dbgen"
OUT_DIR="$REPO_ROOT/experiment/expB_qgen_streams"
NUM_STREAMS=10
SEED_BASE=1000

mkdir -p "$OUT_DIR"
cd "$DBGEN_DIR"
export DSS_QUERY=./queries

for stream in $(seq 1 "$NUM_STREAMS"); do
  seed=$((SEED_BASE + stream))
  out="$OUT_DIR/stream_${stream}.sql"
  ./qgen -r "$seed" $(seq 1 22) > "$out"
  echo "wrote $out (seed=$seed)"
done
