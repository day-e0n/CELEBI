#!/usr/bin/env python3
"""Convert TPC-H dbgen .tbl files to one Parquet file per table."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


SCHEMAS = {
    "nation": """
        n_nationkey INTEGER,
        n_name CHAR(25),
        n_regionkey INTEGER,
        n_comment VARCHAR
    """,
    "region": """
        r_regionkey INTEGER,
        r_name CHAR(25),
        r_comment VARCHAR
    """,
    "part": """
        p_partkey BIGINT,
        p_name VARCHAR,
        p_mfgr CHAR(25),
        p_brand CHAR(10),
        p_type VARCHAR,
        p_size INTEGER,
        p_container CHAR(10),
        p_retailprice DECIMAL(15,2),
        p_comment VARCHAR
    """,
    "supplier": """
        s_suppkey BIGINT,
        s_name CHAR(25),
        s_address VARCHAR,
        s_nationkey INTEGER,
        s_phone CHAR(15),
        s_acctbal DECIMAL(15,2),
        s_comment VARCHAR
    """,
    "partsupp": """
        ps_partkey BIGINT,
        ps_suppkey BIGINT,
        ps_availqty INTEGER,
        ps_supplycost DECIMAL(15,2),
        ps_comment VARCHAR
    """,
    "customer": """
        c_custkey INTEGER,
        c_name VARCHAR,
        c_address VARCHAR,
        c_nationkey INTEGER,
        c_phone CHAR(15),
        c_acctbal DECIMAL(15,2),
        c_mktsegment CHAR(10),
        c_comment VARCHAR
    """,
    "orders": """
        o_orderkey BIGINT,
        o_custkey INTEGER,
        o_orderstatus CHAR(1),
        o_totalprice DECIMAL(15,2),
        o_orderdate DATE,
        o_orderpriority CHAR(15),
        o_clerk CHAR(15),
        o_shippriority INTEGER,
        o_comment VARCHAR
    """,
    "lineitem": """
        l_orderkey BIGINT,
        l_partkey BIGINT,
        l_suppkey BIGINT,
        l_linenumber INTEGER,
        l_quantity DECIMAL(15,2),
        l_extendedprice DECIMAL(15,2),
        l_discount DECIMAL(15,2),
        l_tax DECIMAL(15,2),
        l_returnflag CHAR(1),
        l_linestatus CHAR(1),
        l_shipdate DATE,
        l_commitdate DATE,
        l_receiptdate DATE,
        l_shipinstruct CHAR(25),
        l_shipmode CHAR(10),
        l_comment VARCHAR
    """,
}

TABLE_ORDER = [
    "nation",
    "region",
    "part",
    "supplier",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
]


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tbl-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        if args.database.exists():
            args.database.unlink()
        for table in TABLE_ORDER:
            parquet = args.out_dir / f"{table}.parquet"
            if parquet.exists():
                parquet.unlink()

    missing = [table for table in TABLE_ORDER if not (args.tbl_dir / f"{table}.tbl").exists()]
    if missing:
        raise SystemExit(f"missing .tbl files: {', '.join(missing)}")

    con = duckdb.connect(str(args.database))
    try:
        con.execute("SET memory_limit='32GB'")
        con.execute("SET preserve_insertion_order=false")
        for table in TABLE_ORDER:
            print(f"[LOAD] {table}", flush=True)
            con.execute(f"DROP TABLE IF EXISTS {table}")
            con.execute(f"CREATE TABLE {table} ({SCHEMAS[table]})")
            tbl_path = sql_path(args.tbl_dir / f"{table}.tbl")
            con.execute(
                f"COPY {table} FROM '{tbl_path}' "
                "(HEADER false, DELIMITER '|')"
            )
            parquet_path = sql_path(args.out_dir / f"{table}.parquet")
            print(f"[PARQUET] {table}", flush=True)
            con.execute(f"COPY {table} TO '{parquet_path}' (FORMAT PARQUET)")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
