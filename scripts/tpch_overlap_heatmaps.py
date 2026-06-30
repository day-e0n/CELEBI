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
    queries = sorted(qcols)
    labels = [f"q{q}" for q in queries]

    footprints = {
        q: bytes_for(qcols[q], stats, args.metric)
        for q in queries
    }
    records: list[dict[str, object]] = []
    overlap_bytes = pd.DataFrame(0.0, index=labels, columns=labels)
    overlap_ratio_next = pd.DataFrame(0.0, index=labels, columns=labels)
    reuse_per_retained = pd.DataFrame(0.0, index=labels, columns=labels)

    for qi in queries:
        for qj in queries:
            if args.no_diagonal and qi == qj:
                overlap_bytes.loc[f"q{qi}", f"q{qj}"] = float("nan")
                overlap_ratio_next.loc[f"q{qi}", f"q{qj}"] = float("nan")
                reuse_per_retained.loc[f"q{qi}", f"q{qj}"] = float("nan")
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
    pd.DataFrame(
        [
            {
                "query": f"q{q}",
                "footprint_bytes": footprints[q],
                "footprint_gb": footprints[q] / 1e9,
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

    print(f"wrote heatmaps to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
