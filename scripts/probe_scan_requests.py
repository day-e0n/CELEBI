#!/usr/bin/env python3
"""Extract each query's parquet scans as (table, pushed-down filter, columns).

The page cache keys an entry by (file, filter signature) and an entry serves a
scan only if it covers ALL of that scan's columns. A reorder that scores
"columns shared between two queries" therefore scores something the cache does
not implement: two queries reading identical columns of the same table share
nothing when their pushed-down filters differ, and a query sharing 3 of its 4
columns with the resident entry gets zero, not 75%.

Reading the plan is the only way to know which predicates actually reach the
scan -- a predicate in the SQL text may be pushed down, turned into a join, or
left above the scan, and only the pushed-down ones become part of the key.

  python scripts/probe_scan_requests.py --parquet-dir <dir> --output <json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "test" / "tpch_performance"))

import duckdb  # noqa: E402

TABLES = ("lineitem", "orders", "customer", "part", "supplier", "partsupp",
          "nation", "region")

# TPC-H prefixes every column with its table's initials; the plan box can wrap or
# truncate a file path, so the columns are the reliable way back to the table.
TABLE_BY_PREFIX = {
    "l": "lineitem", "o": "orders", "c": "customer", "p": "part",
    "s": "supplier", "ps": "partsupp", "n": "nation", "r": "region",
}


def parse_plan(plan: str) -> list[dict]:
    """Pull one record per READ_PARQUET node out of EXPLAIN's box drawing."""

    # Strip the box borders; each node then survives as a run of text lines.
    lines = [re.sub(r"^[\s│┌└├┬┴┼─┐┘]+|[\s│┌└├┬┴┼─┐┘]+$", "", ln) for ln in plan.splitlines()]
    scans: list[dict] = []
    current: dict | None = None
    section: str | None = None
    for line in lines:
        if line.startswith("READ_PARQUET"):
            current = {"file": None, "columns": [], "filters": []}
            scans.append(current)
            section = None
            continue
        if current is None or not line:
            continue
        if line.startswith("Projections"):
            section = "columns"
            continue
        if line.startswith("Filters"):
            section = "filters"
            continue
        if line.startswith(("Function", "~", "Total Files", "File Filters", "EC:")):
            section = None
            continue
        if section:
            current[section].append(line)
    return scans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-set", default="queries")
    args = parser.parse_args()

    queries = __import__(args.query_set).QUERIES
    con = duckdb.connect(":memory:")
    for table in TABLES:
        con.execute(f"CREATE OR REPLACE VIEW {table} AS "
                    f"SELECT * FROM read_parquet('{args.parquet_dir}/{table}.parquet')")

    out: dict[str, list[dict]] = {}
    for qnum in range(1, 23):
        key = f"q{qnum}"
        if key not in queries:
            continue
        plan = con.execute("EXPLAIN " + queries[key]).fetchall()[0][1]
        requests = []
        for scan in parse_plan(plan):
            columns = [c for c in scan["columns"] if c]
            if not columns:
                continue
            # EXPLAIN wraps long identifiers, so the table name is recovered from
            # the column prefix rather than from a file path the box may have cut.
            prefixes = {c.split("_", 1)[0] for c in columns}
            table = next((TABLE_BY_PREFIX[p] for p in sorted(prefixes, key=len, reverse=True)
                          if p in TABLE_BY_PREFIX), None)
            requests.append({
                "table": table,
                "columns": sorted(columns),
                # Only the presence and identity of the predicate matters: two
                # scans share an entry iff these strings match.
                "filter": " ".join(scan["filters"]),
            })
        out[key] = requests
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True))
    n_scans = sum(len(v) for v in out.values())
    n_filtered = sum(1 for v in out.values() for s in v if s["filter"])
    print(f"  {len(out)} queries, {n_scans} parquet scans, {n_filtered} with a pushed-down filter")
    print(f"  -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
