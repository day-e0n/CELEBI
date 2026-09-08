#!/usr/bin/env python3
"""Measure per-column DECODED bytes, i.e. what a column costs once it is in VRAM.

The reorder's overlap metric counts columns, which makes a 25-row
`nation.n_nationkey` worth as much as a 600M-row `lineitem.l_orderkey`. A
byte-weighted reorder needs real sizes, and hardcoding them would be wrong for
every scale factor and useless for ClickBench -- so read them off the files.

Deliberately NOT parquet's `total_uncompressed_size`: that is the *encoded* page
size, and the page cache holds *decoded* columns. On TPC-H SF100 the two differ
by up to 16x -- `l_discount` has 11 distinct values so it is 0.28 GB of
dictionary-encoded pages but 4.47 GB of int64 in VRAM. Ranking a cache by
encoded size would systematically prefer exactly the low-cardinality columns
that are cheapest to re-decode and most expensive to keep.

So: fixed-width columns cost `rows * sizeof(type)`. STRING costs
`rows * 4` (the int32 offsets child) plus the measured character bytes, which
is what cudf materialises.

  python scripts/probe_column_bytes.py --parquet-dir <dir> --output <json>
  python scripts/probe_column_bytes.py --parquet-file <file> --table hits --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


# Decoded width in VRAM. cudf materialises DATE32 as int32 and DECIMAL(p<=18) as
# int64, so these are widths of the physical GPU column, not of the SQL type.
TYPE_WIDTHS: dict[str, int] = {
    "BOOLEAN": 1, "TINYINT": 1, "UTINYINT": 1,
    "SMALLINT": 2, "USMALLINT": 2,
    "INTEGER": 4, "UINTEGER": 4, "DATE": 4, "FLOAT": 4,
    "BIGINT": 8, "UBIGINT": 8, "DOUBLE": 8,
    "TIMESTAMP": 8, "TIMESTAMP_S": 8, "TIMESTAMP_MS": 8, "TIMESTAMP_NS": 8,
    "HUGEINT": 16, "UUID": 16, "INTERVAL": 16,
}


def decoded_width(sql_type: str) -> int | None:
    """None means "variable width" (STRING/BLOB/nested), measured separately."""

    upper = sql_type.upper()
    if upper.startswith("DECIMAL"):
        try:
            precision = int(upper.split("(")[1].split(",")[0])
        except (IndexError, ValueError):
            precision = 18
        return 4 if precision <= 9 else (8 if precision <= 18 else 16)
    return TYPE_WIDTHS.get(upper)


def probe(con, table: str, source: str, string_sample: float) -> dict[str, int]:
    rows = con.execute(
        f"SELECT count(*)::BIGINT FROM read_parquet('{source}')"
    ).fetchone()[0]
    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{source}')"
    ).fetchall()

    out: dict[str, int] = {}
    variable: list[str] = []
    for name, sql_type, *_ in schema:
        width = decoded_width(sql_type)
        if width is None:
            variable.append(name)
        else:
            out[name] = rows * width

    if variable:
        # One pass for every STRING column; sampled because SF100 lineitem is
        # 600M rows and the average length is stable well before that.
        using = f" USING SAMPLE {string_sample}%" if 0 < string_sample < 100 else ""
        projection = ", ".join(
            f'avg(octet_length(CAST("{name}" AS BLOB)))' for name in variable
        )
        averages = con.execute(
            f"SELECT {projection} FROM read_parquet('{source}'){using}"
        ).fetchone()
        for name, average in zip(variable, averages):
            # chars + the int32 offsets child cudf allocates per row
            out[name] = int(rows * ((average or 0.0) + 4))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", help="TPC-H style directory of <table>/*.parquet or <table>.parquet")
    parser.add_argument("--parquet-file", help="single parquet file")
    parser.add_argument("--table", default="hits", help="table name for --parquet-file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--string-sample", type=float, default=1.0,
                        help="percent of rows sampled to average STRING length")
    args = parser.parse_args()

    con = duckdb.connect(":memory:")
    sizes: dict[str, dict[str, int]] = {}
    if args.parquet_file:
        sizes[args.table] = probe(con, args.table, args.parquet_file, args.string_sample)
    else:
        root = Path(args.parquet_dir)
        for table in ("lineitem", "orders", "partsupp", "part", "customer",
                      "supplier", "nation", "region"):
            for pattern in (f"{table}/*.parquet", f"{table}.parquet", f"{table}/**/*.parquet"):
                if list(root.glob(pattern)):
                    sizes[table] = probe(con, table, str(root / pattern), args.string_sample)
                    break
    args.output.write_text(json.dumps(sizes, indent=2, sort_keys=True))
    total = sum(sum(c.values()) for c in sizes.values())
    for table, cols in sorted(sizes.items()):
        print(f"  {table:<10} {len(cols):3d} cols  {sum(cols.values())/2**30:8.2f} GB")
    print(f"  {'TOTAL':<10} {'':3}       {total/2**30:8.2f} GB -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
