#!/usr/bin/env python3
"""Which table scans in each query carry a predicate, and which do not.

The cache-aware reorder scores a pair of queries by the columns they share, and a
column名 is all it looks at: `WHERE l_shipdate < '1995-01-01'` and
`WHERE l_shipdate >= '1998-01-01'` count as a full overlap even though neither
query can use a single row the other cached. That is fine for an unfiltered scan,
whose pages serve any later query over the same columns, and wrong for a filtered
one, whose pages serve only a narrower predicate.

This script asks DuckDB's own optimizer which scans end up with pushed-down
filters -- so the answer matches what the engine will actually do, including
predicates the planner moves or drops -- and writes

    {"q6": {"lineitem": ["l_quantity<24.00", ...]}, "q1": {"lineitem": []}, ...}

An empty list means that table was scanned unfiltered in that query.

  pixi run -e duckdb-python python scripts/extract_query_filters.py \
      --workload tpch --input /mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized \
      --out experiment/query_table_filters_sf50.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "test" / "tpch_performance"))

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def plan_nodes(node):
    """Every node of an EXPLAIN (FORMAT JSON) tree, children first."""
    if isinstance(node, list):
        for item in node:
            yield from plan_nodes(item)
        return
    if not isinstance(node, dict):
        return
    for key in ("children", "Children"):
        for child in node.get(key) or []:
            yield from plan_nodes(child)
    yield node


def column_to_table(con, tables: list[str]) -> dict[str, str]:
    """Map each column name to its table, so a scan node can be attributed."""
    owner: dict[str, str] = {}
    for table in tables:
        for row in con.execute(f"DESCRIBE {table}").fetchall():
            owner.setdefault(row[0], table)
    return owner


def info_list(info: dict, key: str) -> list[str]:
    """`extra_info` entries are a list for some operators and a bare string for
    others (a one-filter READ_PARQUET gives `"Filters": "AdvEngineID!=0"`), and
    iterating the string yields characters. Normalise to a list of expressions."""
    value = info.get(key)
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part for part in value.split("\n") if part.strip()]
    return [str(part) for part in value]


def scan_table(node: dict, owner: dict[str, str]) -> str | None:
    """The table a READ_PARQUET node reads, named by the columns it touches."""
    info = node.get("extra_info") or {}
    names: list[str] = []
    for value in info_list(info, "Projections"):
        names.append(value)
    for value in info_list(info, "Filters"):
        names.extend(IDENT.findall(value))
    for name in names:
        if name in owner:
            return owner[name]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", choices=("tpch", "clickbench", "ssb"), required=True)
    parser.add_argument("--input", required=True, help="parquet directory (tpch) or file (clickbench)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.workload == "tpch":
        from run_random22_breakdown_var import QUERIES, open_connection
        con = open_connection(args.input, gpu_execution=False)
        tables = ["customer", "lineitem", "nation", "orders",
                  "part", "partsupp", "region", "supplier"]
    elif args.workload == "ssb":
        sys.path.insert(0, str(REPO_ROOT / "test" / "ssb_performance"))
        from run_ssb_breakdown import open_ssb_connection  # type: ignore
        # test/ssb_performance/queries.py IS the SSB query set -- it shadows the
        # TPC-H module of the same name via the sys.path entry above.
        import importlib
        QUERIES = importlib.import_module("queries").QUERIES  # type: ignore
        con = open_ssb_connection(args.input, gpu_execution=False)
        tables = ["customer", "date", "lineorder", "part", "supplier"]
    else:
        import duckdb
        from performance_test import load_query_set  # type: ignore
        from queries import QUERIES  # type: ignore

        # load_query_set swaps QUERIES' contents in place, the same call the
        # ClickBench runner makes -- without it QUERIES still holds TPC-H.
        load_query_set("clickbench_queries")
        # Plain DuckDB, no Sirius extension: the CPU planner is what decides
        # pushdown, and loading the GPU extension would take a device this script
        # has no use for.
        con = duckdb.connect(":memory:")
        con.execute("CREATE OR REPLACE VIEW hits AS "
                    f"SELECT * FROM read_parquet('{args.input}')")
        tables = ["hits"]

    owner = column_to_table(con, tables)
    out: dict[str, dict[str, list[str]]] = {}
    for label, sql in QUERIES.items():
        try:
            raw = con.execute("EXPLAIN (FORMAT JSON) " + sql).fetchall()[0][1]
        except Exception as exc:  # a query the planner rejects has no scans to record
            print(f"{label}: EXPLAIN failed ({exc})", file=sys.stderr)
            continue
        per_table: dict[str, list[str]] = {}
        for node in plan_nodes(json.loads(raw)):
            if "PARQUET" not in str(node.get("name", "")).upper():
                continue
            table = scan_table(node, owner)
            if table is None:
                continue
            filters = info_list(node.get("extra_info") or {}, "Filters")
            # A table read twice in one query (self-join) is unfiltered only if
            # every one of its scans is: the union is what the cache can hold.
            per_table.setdefault(table, [])
            per_table[table].extend(filters)
        # SSB labels are "q1.1"; the reorder addresses queries by the integer id
        # "11" that run_ssb_breakdown.py derives, so key the file the same way.
        out[label.replace(".", "")] = per_table

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    unfiltered = sum(1 for q in out.values() for f in q.values() if not f)
    total = sum(len(q) for q in out.values())
    print(f"wrote {args.out}: {len(out)} queries, {unfiltered}/{total} table scans unfiltered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
