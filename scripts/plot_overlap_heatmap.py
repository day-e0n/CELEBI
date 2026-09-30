#!/usr/bin/env python3
"""How many bytes each pair of queries could share, as a matrix, per workload.

Cell (i, j) is the decoded size of the columns query i and query j both read --
what the cache could hand the second one if the first ran just before it. Bytes,
not column counts: ClickBench's URL is 8.8GB and its EventDate 0.4GB, and a
count treats them alike, which is why a name-counting overlap metric mispredicts
what the cache does.

Queries are laid out in query-number order so a pair can be found by name; the
arrival order the benchmark runs is a permutation of this and does not change any
cell, only which cells happen to be adjacent.

  pixi run -e duckdb-python python scripts/plot_overlap_heatmap.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "test" / "tpch_performance"))

import paper_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import duckdb  # noqa: E402

TPCH_DIR = "/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized"
TPCH_ROWS = {"customer": 7500000, "lineitem": 300005811, "nation": 25,
             "orders": 75000000, "part": 10000000, "partsupp": 40000000,
             "region": 5, "supplier": 500000}
TPCH_ORDER = list(range(1, 23))

CB_PARQUET = "/mnt/nvme/clickbench/hits_v2.parquet"
CB_SKIP = {1, 5, 6, 24, 29, 30}   # GPU 경로가 못 도는 쿼리
CB_ORDER = [q for q in range(1, 44) if q not in CB_SKIP]
FIXED = {"INTEGER": 4, "BIGINT": 8, "DOUBLE": 8, "DATE": 4, "SMALLINT": 2,
         "TINYINT": 1, "TIMESTAMP": 8, "BOOLEAN": 1}


def column_bytes(con, tables: dict[str, str], rows: dict[str, int]) -> dict[str, float]:
    """Decoded size per column, in GiB. STRING adds its 4-byte-per-row offsets."""
    out: dict[str, float] = {}
    for table, parquet in tables.items():
        types = {r[0]: r[1] for r in con.execute(f"DESCRIBE {table}").fetchall()}
        encoded = dict(con.execute(
            f"select path_in_schema, sum(total_uncompressed_size) "
            f"from parquet_metadata(\'{parquet}\') group by 1").fetchall())
        n_rows = rows[table]
        for name, t in types.items():
            if t.startswith("VARCHAR"):
                out[name] = (encoded.get(name, 0) + 4 * n_rows) / 2**30
            else:
                out[name] = FIXED.get(t, 16 if t.startswith("DECIMAL") else 0) * n_rows / 2**30
    return out


def draw(title: str, stem: str, order, queries, size) -> None:
    cols = {q: {c for c in size if re.search(rf"\b{re.escape(c)}\b", queries[f"q{q}"])}
            for q in order}
    n = len(order)
    m = np.zeros((n, n))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            m[i, j] = sum(size[c] for c in cols[a] & cols[b])
    # The diagonal is a query against itself -- its own footprint, not an overlap,
    # and the largest number in the matrix. Leaving it in sets the colour scale and
    # flattens everything that actually matters, so blank it out.
    shown = m.copy()
    np.fill_diagonal(shown, np.nan)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(shown, cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [f"q{q}" for q in order]
    ax.set_xticklabels(labels, fontsize=5 if n > 25 else 7, rotation=90)
    ax.set_yticklabels(labels, fontsize=5 if n > 25 else 7)
    ax.set_xlabel("query", fontsize=8)
    ax.set_ylabel("query", fontsize=8)
    ax.set_title(title, fontsize=9, pad=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("shared column bytes (GiB)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    fig.tight_layout()
    ps.save(fig, stem)
    off = m[~np.eye(n, dtype=bool)]
    print(f"{title}: 쌍 겹침 평균 {off.mean():.2f} GiB, 중앙값 {np.median(off):.2f} GiB, "
          f"최대 {off.max():.2f} GiB  |  (참고) 각 쿼리 자기 크기 중앙값 {np.median(np.diag(m)):.2f} GiB")


def main() -> int:
    from performance_test import load_query_set
    from queries import QUERIES

    load_query_set("clickbench_queries")
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE VIEW hits AS SELECT * FROM read_parquet('{CB_PARQUET}')")
    n_rows = con.execute("select count(*) from hits").fetchone()[0]
    draw("ClickBench: bytes two queries both read", "fig_clickbench_overlap_heatmap",
         CB_ORDER, QUERIES, column_bytes(con, {"hits": CB_PARQUET}, {"hits": n_rows}))

    import importlib
    import queries as tpch_queries
    importlib.reload(tpch_queries)
    con2 = duckdb.connect(":memory:")
    tables = {t: f"{TPCH_DIR}/{t}.parquet" for t in TPCH_ROWS}
    for t, p in tables.items():
        con2.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{p}')")
    draw("TPC-H SF50: bytes two queries both read", "fig_tpch_overlap_heatmap",
         TPCH_ORDER, tpch_queries.QUERIES, column_bytes(con2, tables, TPCH_ROWS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
