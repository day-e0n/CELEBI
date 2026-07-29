#!/usr/bin/env python3
"""Plot observed query-column overlap from Sirius scan-audit materialization logs.

Input is scan_audit_materializations.csv from scripts/summarize_scan_audit.py.
This is stricter than table overlap: each loaded item is table.column from the
actual [scan-audit] parquet_materialize events.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

QUERY_ORDER = [f"q{i}" for i in range(1, 23)]
TPCH_TABLE_ORDER = [
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
]
FIXED_WIDTH_COLUMNS: dict[str, set[str]] = {
    "customer": {"c_custkey", "c_nationkey", "c_acctbal"},
    "lineitem": {
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
    },
    "nation": {"n_nationkey", "n_regionkey"},
    "orders": {"o_orderkey", "o_custkey", "o_totalprice", "o_orderdate"},
    "part": {"p_partkey", "p_size", "p_retailprice"},
    "partsupp": {"ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"},
    "region": {"r_regionkey"},
    "supplier": {"s_suppkey", "s_nationkey", "s_acctbal"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materializations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/graph"))
    parser.add_argument("--prefix", default="observed_gpu_loaded_columns")
    parser.add_argument(
        "--fixed-width-only",
        action="store_true",
        help="Only count fixed-width TPC-H columns, excluding strings/variable-width columns.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def query_sort_key(query: str) -> tuple[int, str]:
    if query.startswith("q") and query[1:].isdigit():
        return int(query[1:]), query
    return 999, query


def column_sort_key(column: str) -> tuple[int, str, str]:
    table, _, col = column.partition(".")
    try:
        table_idx = TPCH_TABLE_ORDER.index(table)
    except ValueError:
        table_idx = len(TPCH_TABLE_ORDER)
    return table_idx, table, col


def to_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def is_fixed_width_column(table: str, column: str) -> bool:
    return column in FIXED_WIDTH_COLUMNS.get(table, set())


def read_query_columns(path: Path, *, fixed_width_only: bool = False):
    by_query: dict[str, set[str]] = defaultdict(set)
    by_query_table: dict[str, set[str]] = defaultdict(set)
    events_by_query: dict[str, int] = defaultdict(int)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            query = row.get("query", "")
            table = row.get("table", "") or "unknown"
            columns = [c for c in (row.get("columns", "") or "").split("|") if c]
            compressed = to_int(row.get("compressed_bytes"))
            uncompressed = to_int(row.get("uncompressed_bytes"))
            output_rows = to_int(row.get("output_rows"))
            if not query or query == "unknown" or not columns:
                continue
            if compressed == 0 and uncompressed == 0 and output_rows == 0:
                continue
            events_by_query[query] += 1
            by_query_table[query].add(table)
            for column in columns:
                if fixed_width_only and not is_fixed_width_column(table, column):
                    continue
                by_query[query].add(f"{table}.{column}")
    queries = sorted(by_query, key=query_sort_key)
    columns = sorted({c for qcols in by_query.values() for c in qcols}, key=column_sort_key)
    return by_query, by_query_table, events_by_query, queries, columns


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_matrix(path: Path, labels: list[str], matrix: dict[tuple[str, str], float | str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + labels)
        for qi in labels:
            writer.writerow([qi] + [matrix.get((qi, qj), "") for qj in labels])


def plot_heatmap(
    path: Path,
    labels: list[str],
    matrix: dict[tuple[str, str], float | str],
    cbar_label: str,
    fmt: str,
    cmap_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import LinearSegmentedColormap
    except Exception as exc:
        print(f"WARNING: plot skipped for {path.name}: {exc}")
        return

    arr = np.array(
        [
            [float("nan") if matrix.get((qi, qj), "") == "" else float(matrix[(qi, qj)]) for qj in labels]
            for qi in labels
        ],
        dtype=float,
    )
    finite = [v for v in arr.flatten() if math.isfinite(v)]
    if not finite:
        return

    palettes = {
        "soft_blue": ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c"],
        "soft_teal": ["#f7fcfd", "#e0f3f8", "#99d8c9", "#41ae76", "#005824"],
    }
    cmap = LinearSegmentedColormap.from_list(cmap_name, palettes[cmap_name])
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=19)
    cbar.ax.tick_params(labelsize=17)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=17)
    ax.set_yticklabels(labels, fontsize=17)
    ax.set_xlabel("Second query (Qj)", fontsize=19)
    ax.set_ylabel("Previous query (Qi)", fontsize=19)

    threshold = max(finite) * 0.55 if finite else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if not math.isfinite(value):
                continue
            color = "white" if value > threshold else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_query, by_query_table, events_by_query, observed_queries, columns = read_query_columns(
        args.materializations, fixed_width_only=args.fixed_width_only
    )
    labels = [q for q in QUERY_ORDER if q in by_query] + [q for q in observed_queries if q not in QUERY_ORDER]

    per_query_rows = []
    for query in labels:
        qcols = sorted(by_query.get(query, set()), key=column_sort_key)
        per_query_rows.append(
            {
                "query": query,
                "loaded_column_count": len(qcols),
                "loaded_table_count": len(by_query_table.get(query, set())),
                "materialization_events": events_by_query.get(query, 0),
                "loaded_columns": " ".join(qcols),
            }
        )
    write_csv(
        args.output_dir / f"{args.prefix}_per_query_columns.csv",
        per_query_rows,
        ["query", "loaded_column_count", "loaded_table_count", "materialization_events", "loaded_columns"],
    )

    presence_rows = []
    for query in labels:
        row = {"query": query}
        qcols = by_query.get(query, set())
        for column in columns:
            row[column] = 1 if column in qcols else 0
        presence_rows.append(row)
    write_csv(args.output_dir / f"{args.prefix}_presence_matrix.csv", presence_rows, ["query"] + columns)

    count_matrix: dict[tuple[str, str], float | str] = {}
    ratio_matrix: dict[tuple[str, str], float | str] = {}
    pair_rows = []
    for previous_query in labels:
        prev_cols = by_query.get(previous_query, set())
        for second_query in labels:
            if previous_query == second_query:
                continue
            second_cols = by_query.get(second_query, set())
            shared_cols = prev_cols & second_cols
            count = len(shared_cols)
            ratio = count / len(second_cols) if second_cols else ""
            count_matrix[(previous_query, second_query)] = count if second_cols else ""
            ratio_matrix[(previous_query, second_query)] = ratio
            pair_rows.append(
                {
                    "previous_query": previous_query,
                    "second_query": second_query,
                    "previous_loaded_column_count": len(prev_cols),
                    "second_loaded_column_count": len(second_cols),
                    "shared_column_count": count,
                    "overlap_column_ratio_of_second": ratio,
                    "shared_loaded_columns": " ".join(sorted(shared_cols, key=column_sort_key)),
                    "second_loaded_columns": " ".join(sorted(second_cols, key=column_sort_key)),
                }
            )

    write_csv(
        args.output_dir / f"{args.prefix}_pair_long.csv",
        pair_rows,
        [
            "previous_query",
            "second_query",
            "previous_loaded_column_count",
            "second_loaded_column_count",
            "shared_column_count",
            "overlap_column_ratio_of_second",
            "shared_loaded_columns",
            "second_loaded_columns",
        ],
    )
    write_matrix(args.output_dir / f"{args.prefix}_overlap_column_count_matrix.csv", labels, count_matrix)
    write_matrix(args.output_dir / f"{args.prefix}_overlap_column_ratio_matrix.csv", labels, ratio_matrix)

    if not args.no_plots:
        plot_heatmap(
            args.output_dir / f"{args.prefix}_overlap_column_count_heatmap.png",
            labels,
            count_matrix,
            "Shared loaded columns",
            ".0f",
            "soft_blue",
        )
        plot_heatmap(
            args.output_dir / f"{args.prefix}_overlap_column_ratio_heatmap.png",
            labels,
            ratio_matrix,
            "Shared Qj loaded columns / Qj loaded columns",
            ".2f",
            "soft_teal",
        )

    print(f"queries: {len(labels)}")
    print(f"columns: {len(columns)}")
    print(f"wrote:   {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
