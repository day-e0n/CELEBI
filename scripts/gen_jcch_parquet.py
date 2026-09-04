#!/usr/bin/env python3
"""JCC-H (skewed TPC-H) .tbl -> parquet, laid out like the existing TPC-H sets.

JCC-H shares TPC-H's schema exactly, so the resulting directory is a drop-in
--input for test/tpch_performance/performance_test.py. Run dbgen with -k first
(gen_jcch_data.sh does both steps).
"""
import argparse, os, sys, glob, duckdb

# dss.ddl, in dbgen's column order. CHAR(n) is loaded as VARCHAR: dbgen emits
# unpadded values and TPC-H comparisons are on the trimmed text.
SCHEMA = {
 "nation":   [("n_nationkey","INTEGER"),("n_name","VARCHAR"),("n_regionkey","INTEGER"),("n_comment","VARCHAR")],
 "region":   [("r_regionkey","INTEGER"),("r_name","VARCHAR"),("r_comment","VARCHAR")],
 "part":     [("p_partkey","INTEGER"),("p_name","VARCHAR"),("p_mfgr","VARCHAR"),("p_brand","VARCHAR"),
              ("p_type","VARCHAR"),("p_size","INTEGER"),("p_container","VARCHAR"),
              ("p_retailprice","DECIMAL(15,2)"),("p_comment","VARCHAR")],
 "supplier": [("s_suppkey","INTEGER"),("s_name","VARCHAR"),("s_address","VARCHAR"),("s_nationkey","INTEGER"),
              ("s_phone","VARCHAR"),("s_acctbal","DECIMAL(15,2)"),("s_comment","VARCHAR")],
 "partsupp": [("ps_partkey","INTEGER"),("ps_suppkey","INTEGER"),("ps_availqty","INTEGER"),
              ("ps_supplycost","DECIMAL(15,2)"),("ps_comment","VARCHAR")],
 "customer": [("c_custkey","INTEGER"),("c_name","VARCHAR"),("c_address","VARCHAR"),("c_nationkey","INTEGER"),
              ("c_phone","VARCHAR"),("c_acctbal","DECIMAL(15,2)"),("c_mktsegment","VARCHAR"),("c_comment","VARCHAR")],
 "orders":   [("o_orderkey","BIGINT"),("o_custkey","INTEGER"),("o_orderstatus","VARCHAR"),
              ("o_totalprice","DECIMAL(15,2)"),("o_orderdate","DATE"),("o_orderpriority","VARCHAR"),
              ("o_clerk","VARCHAR"),("o_shippriority","INTEGER"),("o_comment","VARCHAR")],
 "lineitem": [("l_orderkey","BIGINT"),("l_partkey","INTEGER"),("l_suppkey","INTEGER"),("l_linenumber","INTEGER"),
              ("l_quantity","DECIMAL(15,2)"),("l_extendedprice","DECIMAL(15,2)"),("l_discount","DECIMAL(15,2)"),
              ("l_tax","DECIMAL(15,2)"),("l_returnflag","VARCHAR"),("l_linestatus","VARCHAR"),
              ("l_shipdate","DATE"),("l_commitdate","DATE"),("l_receiptdate","DATE"),
              ("l_shipinstruct","VARCHAR"),("l_shipmode","VARCHAR"),("l_comment","VARCHAR")],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tbl_dir", help="directory holding dbgen's *.tbl output")
    ap.add_argument("out_dir", help="destination parquet directory")
    ap.add_argument("--row-group-rows", type=int, default=10_000_000,
                    help="rows per row group (default 10M, matching tpch_parquet_sf*_optimized)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--memory-limit", default="64GB")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads={args.threads}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    con.execute(f"SET preserve_insertion_order=false")

    for table, cols in SCHEMA.items():
        # dbgen shards large tables as lineitem.tbl.1, .2, ... under -C/-S.
        parts = sorted(glob.glob(os.path.join(args.tbl_dir, f"{table}.tbl*")))
        if not parts:
            print(f"  {table:10} SKIP (no .tbl)", flush=True)
            continue
        # Every dbgen line ends with the delimiter, so the reader sees a trailing
        # empty field; name it and drop it in the projection.
        names = [c for c, _ in cols] + ["_trailing"]
        types = {c: t for c, t in cols} | {"_trailing": "VARCHAR"}
        src = ", ".join(f"'{p}'" for p in parts)
        proj = ", ".join(f'"{c}"' for c, _ in cols)
        out = os.path.join(args.out_dir, f"{table}.parquet")
        con.execute(f"""
            COPY (SELECT {proj} FROM read_csv([{src}], delim='|', header=false,
                                              columns={ {n: types[n] for n in names} },
                                              parallel=true))
            TO '{out}' (FORMAT parquet, COMPRESSION snappy,
                        ROW_GROUP_SIZE {args.row_group_rows})
        """)
        n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
        mb = os.path.getsize(out) / 2**20
        print(f"  {table:10} {n:>13,} rows  {mb:9.1f} MiB  <- {len(parts)} tbl", flush=True)

    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
