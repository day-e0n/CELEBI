#!/usr/bin/env python3
# wdy start
"""Generate TPC-H query-pair overlap heatmaps for paged GPU memory motivation.

The script uses the per-query table/column footprint from
test/tpch_performance/tpch_pin_columns.py and Parquet metadata from a TPC-H
dataset. It estimates how much data two consecutive queries could share if GPU
VRAM were managed at a finer, page-like granularity.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_PERF_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(TPCH_PERF_DIR))

from tpch_pin_columns import QUERY_COLUMNS  # noqa: E402


@dataclass(frozen=True)
class ColumnStats:
    table: str
    column: str
    compressed_bytes: int
    uncompressed_bytes: int
    rows: int
    is_fixed_width: bool
    type_name: str


# wdy start
@dataclass(frozen=True)
class FixedWidthPage:
    table: str
    column: str
    page_id: int
    bytes: int


# wdy end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="TPC-H parquet directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tpch_overlap_heatmaps",
        help="Output directory",
    )
    parser.add_argument(
        "--metric",
        choices=("compressed", "uncompressed"),
        default="uncompressed",
        help="Byte metric: compressed approximates disk load; uncompressed approximates GPU footprint",
    )
    parser.add_argument(
        "--fixed-width-only",
        action="store_true",
        help="Only count fixed-width columns, matching the proposed first target",
    )
    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Mask Qi->Qi entries in heatmaps",
    )
    # wdy start
    parser.add_argument(
        "--page-size-mib",
        type=int,
        default=64,
        help="Fixed-width GPU page size used for page-level reuse opportunity estimates.",
    )
    # wdy end
    return parser.parse_args()


def resolve_parquet_files(parquet_dir: Path, table: str) -> list[Path]:
    patterns = [
        parquet_dir / f"{table}.parquet",
        parquet_dir / f"{table}_*.parquet",
        parquet_dir / table / "*.parquet",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in sorted(glob.glob(str(pattern))))
    if not files:
        raise FileNotFoundError(f"No parquet files found for table '{table}' in {parquet_dir}")
    return files


def is_fixed_width_type(dtype: pa.DataType) -> bool:
    if (
        pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_binary(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_struct(dtype)
        or pa.types.is_map(dtype)
    ):
        return False
    return True


def parquet_column_stats(parquet_dir: Path) -> dict[tuple[str, str], ColumnStats]:
    stats: dict[tuple[str, str], ColumnStats] = {}
    tables = sorted({table for cols_by_table in QUERY_COLUMNS.values() for table in cols_by_table})
    for table in tables:
        table_totals: dict[str, dict[str, int]] = {}
        schema_types: dict[str, pa.DataType] = {}
        for path in resolve_parquet_files(parquet_dir, table):
            pf = pq.ParquetFile(path)
            for field in pf.schema_arrow:
                schema_types[field.name] = field.type
                table_totals.setdefault(
                    field.name, {"compressed": 0, "uncompressed": 0, "rows": 0}
                )
            md = pf.metadata
            for rg_idx in range(md.num_row_groups):
                rg = md.row_group(rg_idx)
                for col_idx in range(rg.num_columns):
                    col = rg.column(col_idx)
                    name = col.path_in_schema
                    if name not in table_totals:
                        table_totals[name] = {"compressed": 0, "uncompressed": 0, "rows": 0}
                    table_totals[name]["compressed"] += int(col.total_compressed_size)
                    table_totals[name]["uncompressed"] += int(col.total_uncompressed_size)
                for name in table_totals:
                    table_totals[name]["rows"] += int(rg.num_rows)
        for column, totals in table_totals.items():
            dtype = schema_types.get(column, pa.string())
            stats[(table, column)] = ColumnStats(
                table=table,
                column=column,
                compressed_bytes=totals["compressed"],
                uncompressed_bytes=totals["uncompressed"],
                rows=totals["rows"],
                is_fixed_width=is_fixed_width_type(dtype),
                type_name=str(dtype),
            )
    return stats


def query_columns(fixed_width_only: bool, stats: dict[tuple[str, str], ColumnStats]) -> dict[int, set[tuple[str, str]]]:
    result: dict[int, set[tuple[str, str]]] = {}
    missing: list[tuple[int, str, str]] = []
    for q, cols_by_table in QUERY_COLUMNS.items():
        cols: set[tuple[str, str]] = set()
        for table, columns in cols_by_table.items():
            for column in columns:
                key = (table, column)
                if key not in stats:
                    missing.append((q, table, column))
                    continue
                if fixed_width_only and not stats[key].is_fixed_width:
                    continue
                cols.add(key)
        result[q] = cols
    if missing:
        msg = "\n".join(f"q{q}: {table}.{column}" for q, table, column in missing[:20])
        raise RuntimeError(f"Missing parquet metadata for referenced columns:\n{msg}")
    return result


def bytes_for(
    cols: set[tuple[str, str]], stats: dict[tuple[str, str], ColumnStats], metric: str
) -> int:
    attr = "compressed_bytes" if metric == "compressed" else "uncompressed_bytes"
    return sum(getattr(stats[key], attr) for key in cols)


# wdy start
def fixed_width_pages_for_column(
    stat: ColumnStats, metric: str, page_size_bytes: int
) -> list[FixedWidthPage]:
    total_bytes = stat.compressed_bytes if metric == "compressed" else stat.uncompressed_bytes
    if total_bytes <= 0 or page_size_bytes <= 0:
        return []
    pages: list[FixedWidthPage] = []
    page_count = math.ceil(total_bytes / page_size_bytes)
    for page_id in range(page_count):
        start = page_id * page_size_bytes
        page_bytes = min(page_size_bytes, total_bytes - start)
        pages.append(
            FixedWidthPage(
                table=stat.table,
                column=stat.column,
                page_id=page_id,
                bytes=page_bytes,
            )
        )
    return pages


def build_fixed_width_page_inventory(
    stats: dict[tuple[str, str], ColumnStats], metric: str, page_size_bytes: int
) -> dict[tuple[str, str], list[FixedWidthPage]]:
    inventory: dict[tuple[str, str], list[FixedWidthPage]] = {}
    for key, stat in stats.items():
        if not stat.is_fixed_width:
            continue
        inventory[key] = fixed_width_pages_for_column(stat, metric, page_size_bytes)
    return inventory


def query_pages(
    cols: set[tuple[str, str]], page_inventory: dict[tuple[str, str], list[FixedWidthPage]]
) -> set[tuple[str, str, int]]:
    pages: set[tuple[str, str, int]] = set()
    for table, column in cols:
        for page in page_inventory.get((table, column), []):
            pages.add((table, column, page.page_id))
    return pages


def bytes_for_pages(
    pages: set[tuple[str, str, int]],
    page_inventory: dict[tuple[str, str], list[FixedWidthPage]],
) -> int:
    lookup = {
        (page.table, page.column, page.page_id): page.bytes
        for column_pages in page_inventory.values()
        for page in column_pages
    }
    return sum(lookup[page] for page in pages)


# wdy end


def matrix_to_csv(path: Path, matrix: pd.DataFrame) -> None:
    matrix.to_csv(path, index=True)


def write_long_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_heatmap(path: Path, matrix: pd.DataFrame, title: str, cbar_label: str, fmt: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    values = matrix.to_numpy(dtype=float)
    masked = values.copy()
    im = ax.imshow(masked, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    labels = list(matrix.index)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Second query (Qj)")
    ax.set_ylabel("Previous query (Qi)")
    ax.set_title(title)

    finite_values = [v for row in values for v in row if math.isfinite(v)]
    threshold = (max(finite_values) * 0.55) if finite_values else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not math.isfinite(value):
                continue
            color = "white" if value > threshold else "black"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=6, color=color)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    stats = parquet_column_stats(args.input)
    qcols = query_columns(args.fixed_width_only, stats)
    # wdy start
    all_qcols = query_columns(False, stats)
    # wdy end
    queries = sorted(qcols)
    labels = [f"q{q}" for q in queries]

    footprints = {
        q: bytes_for(qcols[q], stats, args.metric)
        for q in queries
    }
    # wdy start
    page_size_bytes = args.page_size_mib * 1024 * 1024
    page_inventory = build_fixed_width_page_inventory(stats, args.metric, page_size_bytes)
    fixed_qcols = query_columns(True, stats)
    fixed_query_pages = {q: query_pages(fixed_qcols[q], page_inventory) for q in queries}
    fixed_page_footprints = {
        q: bytes_for_pages(fixed_query_pages[q], page_inventory) for q in queries
    }
    variable_footprints = {
        q: bytes_for({col for col in all_qcols[q] if not stats[col].is_fixed_width}, stats, args.metric)
        for q in queries
    }
    # wdy end
    records: list[dict[str, object]] = []
    # wdy start
    page_records: list[dict[str, object]] = []
    # wdy end
    overlap_bytes = pd.DataFrame(0.0, index=labels, columns=labels)
    overlap_ratio_next = pd.DataFrame(0.0, index=labels, columns=labels)
    reuse_per_retained = pd.DataFrame(0.0, index=labels, columns=labels)
    # wdy start
    fixed_page_overlap_gb = pd.DataFrame(0.0, index=labels, columns=labels)
    fixed_page_overlap_ratio_next = pd.DataFrame(0.0, index=labels, columns=labels)
    fixed_page_reuse_per_retained = pd.DataFrame(0.0, index=labels, columns=labels)
    fixed_page_overlap_count = pd.DataFrame(0.0, index=labels, columns=labels)
    # wdy end

    for qi in queries:
        for qj in queries:
            if args.no_diagonal and qi == qj:
                overlap_bytes.loc[f"q{qi}", f"q{qj}"] = float("nan")
                overlap_ratio_next.loc[f"q{qi}", f"q{qj}"] = float("nan")
                reuse_per_retained.loc[f"q{qi}", f"q{qj}"] = float("nan")
                # wdy start
                fixed_page_overlap_gb.loc[f"q{qi}", f"q{qj}"] = float("nan")
                fixed_page_overlap_ratio_next.loc[f"q{qi}", f"q{qj}"] = float("nan")
                fixed_page_reuse_per_retained.loc[f"q{qi}", f"q{qj}"] = float("nan")
                fixed_page_overlap_count.loc[f"q{qi}", f"q{qj}"] = float("nan")
                # wdy end
                continue
            overlap_cols = qcols[qi] & qcols[qj]
            overlap = bytes_for(overlap_cols, stats, args.metric)
            fp_i = footprints[qi]
            fp_j = footprints[qj]
            overlap_bytes.loc[f"q{qi}", f"q{qj}"] = overlap / 1e9
            overlap_ratio_next.loc[f"q{qi}", f"q{qj}"] = overlap / fp_j if fp_j else 0.0
            reuse_per_retained.loc[f"q{qi}", f"q{qj}"] = overlap / fp_i if fp_i else 0.0
            records.append(
                {
                    "previous_query": f"q{qi}",
                    "second_query": f"q{qj}",
                    "overlap_bytes": overlap,
                    "overlap_gb": overlap / 1e9,
                    "second_query_footprint_bytes": fp_j,
                    "previous_query_footprint_bytes": fp_i,
                    "overlap_ratio_of_second_query": overlap / fp_j if fp_j else 0.0,
                    "reuse_benefit_per_retained_footprint": overlap / fp_i if fp_i else 0.0,
                    "overlap_columns": " ".join(f"{t}.{c}" for t, c in sorted(overlap_cols)),
                }
            )
            # wdy start
            overlap_pages = fixed_query_pages[qi] & fixed_query_pages[qj]
            page_overlap = bytes_for_pages(overlap_pages, page_inventory)
            page_fp_i = fixed_page_footprints[qi]
            page_fp_j = fixed_page_footprints[qj]
            fixed_page_overlap_gb.loc[f"q{qi}", f"q{qj}"] = page_overlap / 1e9
            fixed_page_overlap_ratio_next.loc[f"q{qi}", f"q{qj}"] = (
                page_overlap / page_fp_j if page_fp_j else 0.0
            )
            fixed_page_reuse_per_retained.loc[f"q{qi}", f"q{qj}"] = (
                page_overlap / page_fp_i if page_fp_i else 0.0
            )
            fixed_page_overlap_count.loc[f"q{qi}", f"q{qj}"] = len(overlap_pages)
            page_records.append(
                {
                    "previous_query": f"q{qi}",
                    "second_query": f"q{qj}",
                    "page_size_mib": args.page_size_mib,
                    "fixed_width_overlap_pages": len(overlap_pages),
                    "fixed_width_overlap_bytes": page_overlap,
                    "fixed_width_overlap_gb": page_overlap / 1e9,
                    "second_query_fixed_width_page_footprint_bytes": page_fp_j,
                    "previous_query_fixed_width_page_footprint_bytes": page_fp_i,
                    "variable_width_second_query_footprint_bytes": variable_footprints[qj],
                    "fixed_width_page_overlap_ratio_of_second_query": (
                        page_overlap / page_fp_j if page_fp_j else 0.0
                    ),
                    "fixed_width_page_reuse_benefit_per_retained_footprint": (
                        page_overlap / page_fp_i if page_fp_i else 0.0
                    ),
                    "overlap_pages": " ".join(
                        f"{t}.{c}:p{pid}" for t, c, pid in sorted(overlap_pages)
                    ),
                }
            )
            # wdy end

    prefix = "fixed_width_" if args.fixed_width_only else "all_columns_"
    metric_prefix = f"{prefix}{args.metric}_"

    column_rows = [
        {
            "table": s.table,
            "column": s.column,
            "type": s.type_name,
            "is_fixed_width": s.is_fixed_width,
            "compressed_bytes": s.compressed_bytes,
            "uncompressed_bytes": s.uncompressed_bytes,
            "rows": s.rows,
        }
        for s in sorted(stats.values(), key=lambda x: (x.table, x.column))
    ]
    write_long_csv(args.output / "parquet_column_stats.csv", column_rows)
    write_long_csv(args.output / f"{metric_prefix}query_pair_overlap_long.csv", records)
    # wdy start
    page_prefix = f"fixed_width_page_{args.page_size_mib}mib_{args.metric}_"
    page_inventory_rows = [
        {
            "table": page.table,
            "column": page.column,
            "page_id": page.page_id,
            "page_size_mib": args.page_size_mib,
            "bytes": page.bytes,
            "gb": page.bytes / 1e9,
        }
        for pages in page_inventory.values()
        for page in pages
    ]
    write_long_csv(args.output / f"{page_prefix}page_inventory.csv", page_inventory_rows)
    write_long_csv(args.output / f"{page_prefix}query_pair_page_overlap_long.csv", page_records)
    # wdy end
    pd.DataFrame(
        [
            {
                "query": f"q{q}",
                "footprint_bytes": footprints[q],
                "footprint_gb": footprints[q] / 1e9,
                # wdy start
                "fixed_width_page_footprint_bytes": fixed_page_footprints[q],
                "fixed_width_page_footprint_gb": fixed_page_footprints[q] / 1e9,
                "fixed_width_page_count": len(fixed_query_pages[q]),
                "variable_width_footprint_bytes": variable_footprints[q],
                "variable_width_footprint_gb": variable_footprints[q] / 1e9,
                # wdy end
                "columns": " ".join(f"{t}.{c}" for t, c in sorted(qcols[q])),
            }
            for q in queries
        ]
    ).to_csv(args.output / f"{metric_prefix}query_footprints.csv", index=False)

    matrix_to_csv(args.output / f"{metric_prefix}overlap_gb_matrix.csv", overlap_bytes)
    matrix_to_csv(
        args.output / f"{metric_prefix}overlap_ratio_of_second_query_matrix.csv",
        overlap_ratio_next,
    )
    matrix_to_csv(
        args.output / f"{metric_prefix}reuse_benefit_per_retained_footprint_matrix.csv",
        reuse_per_retained,
    )
    # wdy start
    matrix_to_csv(args.output / f"{page_prefix}overlap_gb_matrix.csv", fixed_page_overlap_gb)
    matrix_to_csv(
        args.output / f"{page_prefix}overlap_ratio_of_second_query_matrix.csv",
        fixed_page_overlap_ratio_next,
    )
    matrix_to_csv(
        args.output / f"{page_prefix}reuse_benefit_per_retained_footprint_matrix.csv",
        fixed_page_reuse_per_retained,
    )
    matrix_to_csv(args.output / f"{page_prefix}overlap_page_count_matrix.csv", fixed_page_overlap_count)
    # wdy end

    plot_heatmap(
        args.output / f"{metric_prefix}overlap_gb_heatmap.png",
        overlap_bytes,
        f"TPC-H query-pair overlap ({args.metric}, GB)",
        "Overlapped GB",
        ".1f",
    )
    plot_heatmap(
        args.output / f"{metric_prefix}overlap_ratio_of_second_query_heatmap.png",
        overlap_ratio_next,
        f"Overlap ratio of second query footprint ({args.metric})",
        "Overlap / Qj footprint",
        ".2f",
    )
    plot_heatmap(
        args.output / f"{metric_prefix}reuse_benefit_per_retained_footprint_heatmap.png",
        reuse_per_retained,
        f"Reuse benefit per retained previous-query footprint ({args.metric})",
        "Overlap / Qi footprint",
        ".2f",
    )
    # wdy start
    plot_heatmap(
        args.output / f"{page_prefix}overlap_gb_heatmap.png",
        fixed_page_overlap_gb,
        f"Fixed-width page overlap ({args.metric}, {args.page_size_mib} MiB pages)",
        "Overlapped fixed-width page GB",
        ".1f",
    )
    plot_heatmap(
        args.output / f"{page_prefix}overlap_ratio_of_second_query_heatmap.png",
        fixed_page_overlap_ratio_next,
        f"Fixed-width page overlap ratio of second query ({args.metric})",
        "Page overlap / Qj fixed-width footprint",
        ".2f",
    )
    plot_heatmap(
        args.output / f"{page_prefix}reuse_benefit_per_retained_footprint_heatmap.png",
        fixed_page_reuse_per_retained,
        f"Fixed-width page reuse benefit per retained footprint ({args.metric})",
        "Page overlap / Qi fixed-width footprint",
        ".2f",
    )
    plot_heatmap(
        args.output / f"{page_prefix}overlap_page_count_heatmap.png",
        fixed_page_overlap_count,
        f"Fixed-width overlapped page count ({args.page_size_mib} MiB pages)",
        "Pages",
        ".0f",
    )
    # wdy end

    print(f"wrote heatmaps to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
