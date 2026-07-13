#!/usr/bin/env python3
"""Plot observed query-table overlap from Sirius scan-audit materialization logs.

Input is scan_audit_materializations.csv from scripts/summarize_scan_audit.py.
Unlike SQL-footprint heatmaps, this uses the tables Sirius actually materialized
through GPU Parquet scan. Zero-byte/zero-row fallback scans are excluded because
they do not load table data into GPU memory.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


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
QUERY_ORDER = [f"q{i}" for i in range(1, 23)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materializations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment/graph"))
    parser.add_argument("--prefix", default="observed_gpu_loaded_tables")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def to_int(value: str | None) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def query_sort_key(query: str) -> tuple[int, str]:
    if query.startswith("q") and query[1:].isdigit():
        return int(query[1:]), query
    return 999, query


def table_sort_key(table: str) -> tuple[int, str]:
    try:
        return TPCH_TABLE_ORDER.index(table), table
    except ValueError:
        return len(TPCH_TABLE_ORDER), table


def read_table_bytes(path: Path):
    by_query = defaultdict(
        lambda: defaultdict(
            lambda: {"compressed_bytes": 0, "uncompressed_bytes": 0, "events": 0}
        )
    )
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            query = row.get("query", "")
            table = row.get("table", "") or "unknown"
            compressed = to_int(row.get("compressed_bytes"))
            uncompressed = to_int(row.get("uncompressed_bytes"))
            output_rows = to_int(row.get("output_rows"))
            if not query or query == "unknown":
                continue
            if compressed == 0 and uncompressed == 0 and output_rows == 0:
                continue
            item = by_query[query][table]
            item["compressed_bytes"] += compressed
            item["uncompressed_bytes"] += uncompressed
            item["events"] += 1
    queries = sorted(by_query, key=query_sort_key)
    tables = sorted({t for tq in by_query.values() for t in tq}, key=table_sort_key)
    return by_query, queries, tables


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_matrix(
    path: Path, labels: list[str], matrix: dict[tuple[str, str], float | str]
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["previous_query"] + labels)
        for qi in labels:
            writer.writerow([qi] + [matrix.get((qi, qj), "") for qj in labels])


def plot_heatmap(
    path: Path,
    labels: list[str],
    matrix: dict[tuple[str, str], float | str],
    title: str,
    cbar_label: str,
    fmt: str,
    cmap: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"WARNING: plot skipped for {path.name}: {exc}")
        return

    values = []
    for qi in labels:
        values.append(
            [
                float("nan") if matrix.get((qi, qj), "") == "" else float(matrix[(qi, qj)])
                for qj in labels
            ]
        )
    arr = np.array(values, dtype=float)
    finite = [v for v in arr.flatten() if math.isfinite(v)]
    if not finite:
        return

    palettes = {
        "soft_blue": ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c"],
        "soft_teal": ["#f7fcfd", "#e0f3f8", "#99d8c9", "#41ae76", "#005824"],
        "soft_purple": ["#fcfbfd", "#efedf5", "#bcbddc", "#807dba", "#4a1486"],
    }

    fig, ax = plt.subplots(figsize=(13, 11))
    if cmap in palettes:
        from matplotlib.colors import LinearSegmentedColormap

        cmap_obj = LinearSegmentedColormap.from_list(cmap, palettes[cmap])
    else:
        cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    im = ax.imshow(arr, cmap=cmap_obj, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=13)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Second query (Qj)", fontsize=13)
    ax.set_ylabel("Previous query (Qi)", fontsize=13)
    ax.set_title(title, fontsize=16)

    threshold = max(finite) * 0.55 if finite else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if not math.isfinite(value):
                continue
            color = "white" if value > threshold else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_query, observed_queries, tables = read_table_bytes(args.materializations)
    labels = [q for q in QUERY_ORDER if q in by_query] + [
        q for q in observed_queries if q not in QUERY_ORDER
    ]

    per_query_rows = []
    for query in labels:
        table_bytes = by_query.get(query, {})
        loaded = sorted(table_bytes, key=table_sort_key)
        total_uncompressed = sum(v["uncompressed_bytes"] for v in table_bytes.values())
        total_compressed = sum(v["compressed_bytes"] for v in table_bytes.values())
        per_query_rows.append(
            {
                "query": query,
                "loaded_table_count": len(loaded),
                "loaded_tables": " ".join(loaded),
                "materialized_uncompressed_gb": total_uncompressed / 1e9,
                "materialized_compressed_gb": total_compressed / 1e9,
            }
        )
    write_csv(
        args.output_dir / f"{args.prefix}_per_query_tables.csv",
        per_query_rows,
        [
            "query",
            "loaded_table_count",
            "loaded_tables",
            "materialized_uncompressed_gb",
            "materialized_compressed_gb",
        ],
    )

    presence_rows = []
    bytes_rows = []
    for query in labels:
        presence_row = {"query": query}
        bytes_row = {"query": query}
        for table in tables:
            presence_row[table] = 1 if table in by_query.get(query, {}) else 0
            bytes_row[table] = (
                by_query.get(query, {}).get(table, {}).get("uncompressed_bytes", 0) / 1e9
            )
        presence_rows.append(presence_row)
        bytes_rows.append(bytes_row)
    write_csv(
        args.output_dir / f"{args.prefix}_presence_matrix.csv",
        presence_rows,
        ["query"] + tables,
    )
    write_csv(
        args.output_dir / f"{args.prefix}_uncompressed_gb_by_query_table.csv",
        bytes_rows,
        ["query"] + tables,
    )

    count_matrix: dict[tuple[str, str], float | str] = {}
    ratio_matrix: dict[tuple[str, str], float | str] = {}
    gb_matrix: dict[tuple[str, str], float | str] = {}
    pair_rows = []
    for previous_query in labels:
        previous_tables = set(by_query.get(previous_query, {}))
        for second_query in labels:
            if previous_query == second_query:
                continue
            second_tables = set(by_query.get(second_query, {}))
            shared_tables = previous_tables & second_tables
            second_total = sum(
                v["uncompressed_bytes"] for v in by_query.get(second_query, {}).values()
            )
            overlap_uncompressed = sum(
                by_query[second_query][table]["uncompressed_bytes"]
                for table in shared_tables
                if table in by_query[second_query]
            )
            shared_count = len(shared_tables)
            table_ratio = shared_count / len(second_tables) if second_tables else ""
            byte_ratio = overlap_uncompressed / second_total if second_total else ""
            count_matrix[(previous_query, second_query)] = (
                shared_count if second_tables else ""
            )
            ratio_matrix[(previous_query, second_query)] = byte_ratio
            gb_matrix[(previous_query, second_query)] = (
                overlap_uncompressed / 1e9 if second_total else ""
            )
            pair_rows.append(
                {
                    "previous_query": previous_query,
                    "second_query": second_query,
                    "previous_loaded_tables": " ".join(
                        sorted(previous_tables, key=table_sort_key)
                    ),
                    "second_loaded_tables": " ".join(
                        sorted(second_tables, key=table_sort_key)
                    ),
                    "shared_loaded_tables": " ".join(
                        sorted(shared_tables, key=table_sort_key)
                    ),
                    "shared_table_count": shared_count,
                    "second_table_count": len(second_tables),
                    "overlap_table_ratio_of_second": table_ratio,
                    "second_materialized_uncompressed_gb": (
                        second_total / 1e9 if second_total else ""
                    ),
                    "overlap_second_uncompressed_gb": (
                        overlap_uncompressed / 1e9 if second_total else ""
                    ),
                    "overlap_byte_ratio_of_second": byte_ratio,
                }
            )

    pair_fields = [
        "previous_query",
        "second_query",
        "previous_loaded_tables",
        "second_loaded_tables",
        "shared_loaded_tables",
        "shared_table_count",
        "second_table_count",
        "overlap_table_ratio_of_second",
        "second_materialized_uncompressed_gb",
        "overlap_second_uncompressed_gb",
        "overlap_byte_ratio_of_second",
    ]
    write_csv(args.output_dir / f"{args.prefix}_pair_long.csv", pair_rows, pair_fields)
    write_matrix(
        args.output_dir / f"{args.prefix}_overlap_table_count_matrix.csv",
        labels,
        count_matrix,
    )
    write_matrix(
        args.output_dir / f"{args.prefix}_overlap_byte_ratio_matrix.csv",
        labels,
        ratio_matrix,
    )
    write_matrix(
        args.output_dir / f"{args.prefix}_overlap_uncompressed_gb_matrix.csv",
        labels,
        gb_matrix,
    )

    if not args.no_plots:
        plot_heatmap(
            args.output_dir / f"{args.prefix}_overlap_table_count_heatmap.png",
            labels,
            count_matrix,
            "Observed GPU-loaded table overlap count",
            "Shared loaded tables",
            ".0f",
            "soft_blue",
        )
        plot_heatmap(
            args.output_dir / f"{args.prefix}_overlap_byte_ratio_heatmap.png",
            labels,
            ratio_matrix,
            "Observed GPU-loaded table overlap ratio of second query bytes",
            "Overlapped Qj materialized bytes / Qj bytes",
            ".2f",
            "soft_teal",
        )
        plot_heatmap(
            args.output_dir / f"{args.prefix}_overlap_uncompressed_gb_heatmap.png",
            labels,
            gb_matrix,
            "Observed overlapped GPU materialization by second query",
            "Overlapped Qj materialized GB",
            ".1f",
            "soft_purple",
        )

    print(f"queries: {len(labels)}")
    print(f"tables:  {len(tables)}")
    print(f"wrote:   {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
