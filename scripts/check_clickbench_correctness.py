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
# Exclusions default to the benchmark's, but the two lists must not be welded
# together: q29 was skipped by the benchmark, so the checker skipped it too, and
# the string corruption it hits (Invalid unicode on a cached Referer) was never
# looked for. CHECK_SKIP overrides -- "5,6,30" leaves in the three that only fail
# with the cache ON, which is exactly where a cache bug would hide.
# Nothing is skipped by default any more. q1, q5, q6, q24, q29 and q30 were all here and all
# six now match DuckDB on the full table: q30 once the CAST to HUGEINT stopped falling off the
# AST path, q29 once the hand-written regex kernel stopped being the default, and q5/q6 once
# the ungrouped aggregate learned COUNT(DISTINCT).
SKIP = {int(x) for x in os.environ.get("CHECK_SKIP", "").split(",") if x.strip()}


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
    # Build the engine environment with the SAME function the benchmark runner
    # uses. Setting a couple of variables by hand is not enough and fails silently:
    # without SIRIUS_FIXED_PAGE_BACKED_PROVIDER the page-cache provider never runs
    # at all, so the run reports "all match" having exercised none of the code it
    # was meant to check. Flags this script does not own (pre/post-filter caching,
    # partial residency, entry cap) are read from the caller's environment, which
    # make_env copies rather than overwrites.
    from run_random22_breakdown import write_config
    from run_random22_breakdown_var import make_env

    out = Path(os.environ.get("CHECK_OUT_DIR", ".tmp/check_correctness")).resolve()
    log_dir, config_path = out / "log_dir", out / "sirius.yaml"
    out.mkdir(parents=True, exist_ok=True)
    write_config(config_path, out / "telemetry_data", os.environ.get("CHECK_GPU_LIMIT", "20GB"))
    os.environ.update(make_env(
        os.environ.get("CHECK_CONDITION", "celebi_fixed_variable"),
        os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        log_dir, config_path,
        os.environ.get("CHECK_FIXED_BUDGET", "6GB"),
        os.environ.get("CHECK_VARIABLE_BUDGET", "4GB"),
        os.environ.get("CHECK_VARIABLE_PAGE_BYTES", "16777216"),
        os.environ.get("CHECK_MIN_FREE_BYTES", "0"),
    ))
    print(f"engine log dir: {log_dir}", flush=True)
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

    # Arrival order matters to what ends up resident: the benchmark's adversarial
    # order is what produces partially-covered entries, and query-number order can
    # miss that path entirely -- a correctness run that never reaches the code it
    # is checking reports "all match" for the wrong reason.
    order_env = os.environ.get("CHECK_ORDER", "")
    if order_env:
        order = [int(x) for x in order_env.split(",") if x.strip()]
        order = [q for q in order if q in expected]
        missing = [q for q in sorted(expected) if q not in order]
        order += missing
        print(f"arrival order: {','.join(f'q{q}' for q in order)}", flush=True)
    else:
        order = sorted(expected)

    mismatches = []
    for execution in range(1, EXECUTIONS + 1):
        for q in order:
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
