#!/usr/bin/env python3
"""Does the page cache ever serve a query fewer rows than the table has?

Comparing ClickBench query RESULTS cannot answer this: most of them are
`ORDER BY count(*) DESC LIMIT N` over thousands of tied rows, so which rows come
back is unspecified -- DuckDB itself returns different ties run to run. So this
checks the one thing that is exactly defined: how many rows the scan produced.

The probes are order-independent aggregates with a known exact answer over the
whole table. They are interleaved with the real ClickBench queries so the cache is
populated and exercised between probes; if an entry is ever served while only
partly populated, the probe that reads its columns comes back short.

Usage:
  pixi run -e duckdb-python python scripts/check_clickbench_row_conservation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "test" / "tpch_performance"))

DATA = os.environ.get("CLICKBENCH_PARQUET", "/mnt/nvme/clickbench/hits_v2.parquet")
ROUNDS = int(os.environ.get("CHECK_ROUNDS", "3"))
SKIP = {1, 5, 6, 24, 29, 30}

# Exact, tie-free, order-independent. Each names columns that the workload also
# caches, so a short entry shows up here as a short count.
PROBES = {
    "rows_regionid": "SELECT COUNT(*), SUM(RegionID::HUGEINT) FROM hits",
    "rows_userid": "SELECT COUNT(*), SUM(UserID::HUGEINT) FROM hits",
    "rows_searchphrase": "SELECT COUNT(*), SUM(LENGTH(SearchPhrase)::HUGEINT) FROM hits",
    "rows_adveng": "SELECT COUNT(*), SUM(AdvEngineID::HUGEINT) FROM hits",
    "rows_counterid": "SELECT COUNT(*), SUM(CounterID::HUGEINT) FROM hits",
}


def main() -> int:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import duckdb
    from clickbench_queries import QUERIES

    ref = duckdb.connect(":memory:")
    ref.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{DATA}')")
    truth = {name: ref.execute(sql).fetchone() for name, sql in PROBES.items()}
    ref.close()
    for name, (n, s) in truth.items():
        print(f"  reference {name}: {n:,} rows, sum={s}", flush=True)

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    con.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{DATA}')")
    con.execute(f"LOAD '{REPO}/build/release/extension/sirius/sirius.duckdb_extension'")
    con.execute("SET gpu_execution = true;")

    bad = 0
    for rnd in range(1, ROUNDS + 1):
        for name, sql in PROBES.items():
            got = con.execute(sql).fetchone()
            if got != truth[name]:
                bad += 1
                print(f"  SHORT round={rnd} {name}: expected {truth[name]}, got {got}",
                      flush=True)
        for q in range(1, 44):
            if q in SKIP:
                continue
            con.execute(QUERIES[f"q{q}"]).fetchall()
        print(f"round {rnd}/{ROUNDS} done ({bad} short reads so far)", flush=True)

    # One last pass after the workload has fully warmed the cache.
    for name, sql in PROBES.items():
        got = con.execute(sql).fetchone()
        if got != truth[name]:
            bad += 1
            print(f"  SHORT final {name}: expected {truth[name]}, got {got}", flush=True)
    con.close()

    print(f"\n{'FAIL' if bad else 'OK'}: {bad} short reads")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
