#!/usr/bin/env python3
"""Does the page cache ever serve a query fewer rows than the table has?

`pinned_entry::num_rows` is a running total of the chunks that happen to have been
inserted, and `fixed_page_databatch_provider::usable()` only asks whether the
provider covers that same running total -- so an entry caught mid-population
satisfies it at every intermediate state. A reader that arrives then is served the
chunks so far, silently, with no error and no log line.

This runs each query on the GPU with auto-caching on, repeatedly (so later
executions hit a cache that earlier ones populated), and compares every result
against DuckDB's CPU answer for the same query. A mismatch that appears only on
execution 2+ is the cache serving partial data.

Usage:
  SIRIUS_FIXED_PAGE_REQUIRE_COMPLETE_ENTRY=0 \\
    pixi run -e duckdb-python python scripts/check_clickbench_correctness.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "test" / "tpch_performance"))
sys.path.insert(0, str(REPO / "scripts"))

DATA = os.environ.get("CLICKBENCH_PARQUET", "/mnt/nvme/clickbench/hits_v2.parquet")
EXECUTIONS = int(os.environ.get("CHECK_EXECUTIONS", "3"))
# Same exclusions the benchmark uses: unsupported on the GPU path, or fatal.
SKIP = {1, 5, 6, 24, 29, 30}


def results_match(cpu, gpu) -> tuple[bool, str]:
    """Compare two result sets as multisets, with a tolerance on floats.

    Row ORDER is not comparable: `ORDER BY count(*) DESC LIMIT 10` leaves ties
    unordered, and the GPU breaks them differently from DuckDB. Float VALUES are
    not bit-comparable either -- a GPU aggregate sums in a different order, so
    AVG() lands a few ULPs away. Neither is a defect; both would drown out the
    thing this script exists to catch, which is the cache serving FEWER ROWS than
    the table has.
    """
    if len(cpu) != len(gpu):
        return False, f"row count: cpu {len(cpu)}, gpu {len(gpu)}"

    def norm(rows):
        out = []
        for row in rows:
            out.append(tuple(
                round(v, 6) if isinstance(v, float) else v for v in row))
        return sorted(out, key=lambda r: tuple((v is None, str(v)) for v in r))

    a, b = norm(cpu), norm(gpu)
    if a == b:
        return True, ""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"row {i}: cpu {x!r} vs gpu {y!r}"
    return False, "differs"


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import duckdb
    from clickbench_queries import QUERIES

    # CPU reference first, in its own connection with the extension never loaded.
    ref = duckdb.connect(":memory:")
    ref.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{DATA}')")
    expected = {}
    for q in range(1, 44):
        if q in SKIP:
            continue
        expected[q] = ref.execute(QUERIES[f"q{q}"]).fetchall()
    ref.close()
    print(f"CPU reference: {len(expected)} queries", flush=True)

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{DATA}')")
    con.execute(f"LOAD '{REPO}/build/release/extension/sirius/sirius.duckdb_extension'")
    con.execute("SET gpu_execution = true;")

    mismatches = []
    for execution in range(1, EXECUTIONS + 1):
        for q in sorted(expected):
            got = con.execute(QUERIES[f"q{q}"]).fetchall()
            same, why = results_match(expected[q], got)
            if not same:
                mismatches.append((execution, q, why))
                print(f"  MISMATCH exec={execution} q{q}: {why}", flush=True)
        print(f"execution {execution}/{EXECUTIONS} done "
              f"({len(mismatches)} mismatches so far)", flush=True)
    con.close()

    if mismatches:
        print(f"\nFAIL: {len(mismatches)} mismatching (execution, query) pairs")
        return 1
    print(f"\nOK: {len(expected)} queries x {EXECUTIONS} executions all match CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
