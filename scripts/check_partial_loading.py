#!/usr/bin/env python3
"""Is the page cache's cached half plus its residual half exactly the table?

The provider serves the row groups it holds in full and tells the reader to skip
exactly those, so the two halves must partition the scan. Overlap shows up as
counts ABOVE the truth, a gap as counts below it. Neither is visible in a query
whose answer depends on ordering or float association, which is why this uses
only exact integer aggregates -- count(*) and integer sums --
over a whole table scan.

Run it with a cache budget small enough that entries are evicted mid-workload;
that is what makes coverage partial, and partial coverage is the path under test.

Usage:
  SIRIUS_ENABLE_FIXED_PAGE_REUSE=1 SIRIUS_FIXED_PAGE_AUTO_CACHE=1 \\
  SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU=2GB \\
    pixi run -e duckdb-python python scripts/check_partial_loading.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "test" / "tpch_performance"))
sys.path.insert(0, str(REPO / "scripts"))
DATA = os.environ.get("CLICKBENCH_PARQUET", "/mnt/nvme/clickbench/hits_v2.parquet")
EXECUTIONS = int(os.environ.get("CHECK_EXECUTIONS", "4"))

# Exact-integer probes only. Each one reads the whole table, so a row group served
# twice or not at all changes the answer; none of them depends on row order.
# No count(DISTINCT): the GPU path rejects it outright ("Distinct aggregates not
# supported in GPU path yet"), which aborts the run instead of testing anything.
PROBES = {
    "count_all":        "SELECT count(*) FROM hits",
    "count_filtered":   "SELECT count(*) FROM hits WHERE SearchEngineID = 2",
    "sum_resolution":   "SELECT sum(ResolutionWidth::BIGINT) FROM hits",
    "sum_refresh":      "SELECT sum(IsRefresh::BIGINT) FROM hits",
    "count_nonempty":   "SELECT count(*) FROM hits WHERE SearchPhrase <> ''",
    "sum_userid_hi":    "SELECT count(*) FROM hits WHERE UserID > 0",
}


def main() -> int:
    # Same engine environment the benchmark runner builds -- see the note in
    # check_clickbench_correctness.py. Hand-setting a couple of variables leaves
    # the page-cache provider switched off and the run proves nothing.
    from run_random22_breakdown import write_config
    from run_random22_breakdown_var import make_env

    out = Path(os.environ.get("CHECK_OUT_DIR", ".tmp/check_partial")).resolve()
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

    ref = duckdb.connect(":memory:")
    ref.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{DATA}')")
    truth = {name: ref.execute(sql).fetchone()[0] for name, sql in PROBES.items()}
    ref.close()
    for name, value in truth.items():
        print(f"  CPU {name:18} {value}", flush=True)

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{DATA}')")
    con.execute(f"LOAD '{REPO}/build/release/extension/sirius/sirius.duckdb_extension'")
    con.execute("SET gpu_execution = true;")

    bad = 0
    for execution in range(1, EXECUTIONS + 1):
        for name, sql in PROBES.items():
            got = con.execute(sql).fetchone()[0]
            if got == truth[name]:
                continue
            bad += 1
            delta = got - truth[name]
            side = "OVERLAP (중복)" if delta > 0 else "GAP (누락)"
            print(f"  MISMATCH exec={execution} {name}: {got} vs {truth[name]} "
                  f"({delta:+}) -> {side}", flush=True)
        print(f"execution {execution}/{EXECUTIONS} done ({bad} mismatches so far)", flush=True)
    con.close()

    if bad:
        print(f"\nFAIL: {bad} mismatching probes")
        return 1
    print(f"\nOK: {len(PROBES)} probes x {EXECUTIONS} executions all exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
