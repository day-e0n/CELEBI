# wdy start
import os
import glob
import duckdb
import traceback

REPO_ROOT = "/home/dy1013/sirius_multi-gpu"
EXTENSION_PATH = f"{REPO_ROOT}/build/release/extension/sirius/sirius.duckdb_extension"
PARQUET_DIR = "/mnt/nvme/dataset"

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]


def resolve_parquet_files(parquet_dir, table):
    candidates = []
    patterns = [
        os.path.join(parquet_dir, f"{table}.parquet"),
        os.path.join(parquet_dir, f"{table}_*.parquet"),
        os.path.join(parquet_dir, table, "*.parquet"),
    ]
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern)))
    return candidates


def open_con():
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})

    for table in TPCH_TABLES:
        files = resolve_parquet_files(PARQUET_DIR, table)
        if not files:
            raise RuntimeError(f"No parquet files for table={table}")

        file_list = ",".join(f"'{f}'" for f in files)
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_parquet([{file_list}])
            """
        )

    con.execute(f"LOAD '{EXTENSION_PATH}'")
    con.execute("SET gpu_execution = true;")
    return con


TESTS = [
    (
        "T01_region_filter",
        """
        select r_regionkey
        from region
        where r_name = 'EUROPE'
        """
    ),

    (
        "T02_nation_region_join",
        """
        select n.n_nationkey, n.n_name
        from nation n, region r
        where n.n_regionkey = r.r_regionkey
          and r.r_name = 'EUROPE'
        """
    ),

    (
        "T03_supplier_nation_region_join",
        """
        select s.s_suppkey, n.n_name
        from supplier s, nation n, region r
        where s.s_nationkey = n.n_nationkey
          and n.n_regionkey = r.r_regionkey
          and r.r_name = 'EUROPE'
        limit 10
        """
    ),

    (
        "T04_customer_orders_date_join",
        """
        select c.c_custkey, o.o_orderkey
        from customer c, orders o
        where c.c_custkey = o.o_custkey
          and o.o_orderdate >= date '1997-01-01'
          and o.o_orderdate < date '1998-01-01'
        limit 10
        """
    ),

    (
        "T05_lineitem_orders_date_join",
        """
        select l.l_orderkey, l.l_suppkey, l.l_extendedprice, l.l_discount
        from lineitem l, orders o
        where l.l_orderkey = o.o_orderkey
          and o.o_orderdate >= date '1997-01-01'
          and o.o_orderdate < date '1998-01-01'
        limit 10
        """
    ),

    (
        "T06_lineitem_supplier_join",
        """
        select l.l_orderkey, l.l_suppkey, s.s_nationkey
        from lineitem l, supplier s
        where l.l_suppkey = s.s_suppkey
        limit 10
        """
    ),

    (
        "T07_customer_supplier_nationkey_join",
        """
        select c.c_custkey, s.s_suppkey
        from customer c, supplier s
        where c.c_nationkey = s.s_nationkey
        limit 10
        """
    ),

    (
        "T08_q5_core_joins_no_agg",
        """
        select
          n.n_name,
          l.l_extendedprice,
          l.l_discount
        from
          customer c,
          orders o,
          lineitem l,
          supplier s,
          nation n,
          region r
        where
          c.c_custkey = o.o_custkey
          and l.l_orderkey = o.o_orderkey
          and l.l_suppkey = s.s_suppkey
          and c.c_nationkey = s.s_nationkey
          and s.s_nationkey = n.n_nationkey
          and n.n_regionkey = r.r_regionkey
          and r.r_name = 'EUROPE'
          and o.o_orderdate >= date '1997-01-01'
          and o.o_orderdate < date '1998-01-01'
        limit 10
        """
    ),

    (
        "T09_q5_expression_no_agg",
        """
        select
          n.n_name,
          l.l_extendedprice * (1 - l.l_discount) as revenue_part
        from
          customer c,
          orders o,
          lineitem l,
          supplier s,
          nation n,
          region r
        where
          c.c_custkey = o.o_custkey
          and l.l_orderkey = o.o_orderkey
          and l.l_suppkey = s.s_suppkey
          and c.c_nationkey = s.s_nationkey
          and s.s_nationkey = n.n_nationkey
          and n.n_regionkey = r.r_regionkey
          and r.r_name = 'EUROPE'
          and o.o_orderdate >= date '1997-01-01'
          and o.o_orderdate < date '1998-01-01'
        limit 10
        """
    ),

    (
        "T10_q5_group_by_no_order",
        """
        select
          n.n_name,
          sum(l.l_extendedprice * (1 - l.l_discount)) as revenue
        from
          customer c,
          orders o,
          lineitem l,
          supplier s,
          nation n,
          region r
        where
          c.c_custkey = o.o_custkey
          and l.l_orderkey = o.o_orderkey
          and l.l_suppkey = s.s_suppkey
          and c.c_nationkey = s.s_nationkey
          and s.s_nationkey = n.n_nationkey
          and n.n_regionkey = r.r_regionkey
          and r.r_name = 'EUROPE'
          and o.o_orderdate >= date '1997-01-01'
          and o.o_orderdate < date '1998-01-01'
        group by
          n.n_name
        """
    ),

    (
        "T11_q5_full",
        """
        select
          n.n_name,
          sum(l.l_extendedprice * (1 - l.l_discount)) as revenue
        from
          customer c,
          orders o,
          lineitem l,
          supplier s,
          nation n,
          region r
        where
          c.c_custkey = o.o_custkey
          and l.l_orderkey = o.o_orderkey
          and l.l_suppkey = s.s_suppkey
          and c.c_nationkey = s.s_nationkey
          and s.s_nationkey = n.n_nationkey
          and n.n_regionkey = r.r_regionkey
          and r.r_name = 'EUROPE'
          and o.o_orderdate >= date '1997-01-01'
          and o.o_orderdate < date '1998-01-01'
        group by
          n.n_name
        order by
          revenue desc
        """
    ),
]


def main():
    con = open_con()

    for name, sql in TESTS:
        print(f"\n===== {name} =====", flush=True)
        try:
            rows = con.execute(sql).fetchall()
            print(f"OK rows={len(rows)}", flush=True)
            if rows:
                print(rows[:3], flush=True)
        except Exception:
            print(f"FAIL at {name}", flush=True)
            traceback.print_exc()
            break

    con.close()


if __name__ == "__main__":
    main()
# wdy end
