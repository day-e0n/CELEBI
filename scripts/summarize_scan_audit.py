#!/usr/bin/env python3
# wdy start
"""Summarize Sirius [scan-audit] parquet materialization logs.

This is the observed counterpart to scripts/tpch_overlap_heatmaps.py. The heatmap
script estimates reuse opportunity statically from TPC-H query footprints; this
script parses actual Sirius logs and reports what parquet splits/columns were
materialized during a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_PERF_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(TPCH_PERF_DIR))

from tpch_pin_columns import QUERY_COLUMNS  # noqa: E402

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
QUERY_RE = re.compile(r"/q(\d+)(?:/|$)")
PAIR_RE = re.compile(r"q(\d+)_then_q(\d+)")
TPCH_TABLES = tuple(sorted({table for cols in QUERY_COLUMNS.values() for table in cols}, key=len, reverse=True))


@dataclass(frozen=True)
class Materialization:
    experiment: str
    benchmark_dir: str
    query: str
    previous_query: str
    second_query: str
    target_gpu: int
    table: str
    files: str
    columns: str
    row_groups: str
    compressed_bytes: int
    uncompressed_bytes: int
    column_compressed_bytes: int
    column_uncompressed_bytes: int
    output_rows: int
    output_columns: int
    split_count: int


@dataclass(frozen=True)
class ColumnMaterialization:
    experiment: str
    benchmark_dir: str
    query: str
    previous_query: str
    second_query: str
    target_gpu: int
    table: str
    column: str
    compressed_bytes: int
    uncompressed_bytes: int
    files: str
    row_groups: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir",
        action="append",
        type=Path,
        default=[],
        help="Benchmark directory from performance_test.py. Can be repeated.",
    )
    parser.add_argument(
        "--log-file",
        action="append",
        type=Path,
        default=[],
        help="Raw Sirius log file to parse. Can be repeated.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment label. Defaults to benchmark directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <benchmark>/csv for one benchmark, otherwise ./scan_audit_summary.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate observed heatmap PNGs.",
    )
    return parser.parse_args()


def parse_kv(line: str) -> dict[str, str]:
    return {k: v for k, v in KV_RE.findall(line)}


def to_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def query_from_path(path: Path) -> str:
    match = QUERY_RE.search(str(path))
    return f"q{match.group(1)}" if match else "unknown"


def pair_from_experiment(experiment: str) -> tuple[str, str]:
    match = PAIR_RE.search(experiment)
    if not match:
        return "", ""
    return f"q{match.group(1)}", f"q{match.group(2)}"


def benchmark_metadata(benchmark_dir: Path) -> dict[str, object]:
    path = benchmark_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def query_order_from_metadata(metadata: dict[str, object]) -> list[str]:
    raw = metadata.get("queries", [])
    result = []
    for item in raw if isinstance(raw, list) else []:
        text = str(item)
        if text.startswith("q"):
            result.append(text)
        elif text.isdigit():
            result.append(f"q{text}")
    return result


def raw_query_sequence(metadata: dict[str, object], span_count: int) -> list[str]:
    queries = query_order_from_metadata(metadata)
    if not queries:
        return ["unknown"] * span_count
    mode = str(metadata.get("mode", "sequential"))
    iterations = int(metadata.get("iterations", 1) or 1)
    expected = len(queries) * iterations
    if span_count != expected:
        return ["unknown"] * span_count
    if mode in ("grouped", "isolated"):
        return [q for q in queries for _ in range(iterations)]
    return [q for _ in range(iterations) for q in queries]


def is_timed_query_begin(line: str) -> bool:
    marker = "QueryBegin: SQL:"
    if marker not in line:
        return False
    sql = line.split(marker, 1)[1].strip().lower()
    return not (
        sql.startswith("set ")
        or sql.startswith("call pin_table")
        or sql.startswith("call unpin_table")
        or sql.startswith("create view")
    )


def iter_log_segments(benchmark_dir: Path):
    split_logs = sorted((benchmark_dir / "sirius").glob("q*/sirius.log"))
    if split_logs:
        for log_file in split_logs:
            try:
                yield query_from_path(log_file), log_file, log_file.read_text(errors="replace").splitlines()
            except OSError:
                continue
        return

    metadata = benchmark_metadata(benchmark_dir)
    log_dir = benchmark_dir / "log_dir"
    if not log_dir.exists():
        return
    for log_file in sorted(log_dir.rglob("*.log")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        begin_indices = [i for i, line in enumerate(lines) if is_timed_query_begin(line)]
        query_sequence = raw_query_sequence(metadata, len(begin_indices))
        for span_idx, start in enumerate(begin_indices):
            end = begin_indices[span_idx + 1] if span_idx + 1 < len(begin_indices) else len(lines)
            yield query_sequence[span_idx], log_file, lines[start:end]


def infer_table(files_field: str) -> str:
    tokens = [token for token in files_field.split("|") if token]
    for token in tokens:
        path = Path(token)
        stem = path.name
        if stem.endswith(".parquet"):
            stem = stem[: -len(".parquet")]
        for table in TPCH_TABLES:
            if stem == table or stem.startswith(f"{table}_") or path.parent.name == table:
                return table
    return Path(tokens[0]).stem if tokens else "unknown"


def parse_column_bytes(field: str) -> list[tuple[str, int, int]]:
    if not field:
        return []
    result = []
    for item in field.split(","):
        if not item:
            continue
        parts = item.rsplit(":", 2)
        if len(parts) != 3:
            continue
        column, compressed, uncompressed = parts
        result.append((column, to_int(compressed), to_int(uncompressed)))
    return result


def collect(args: argparse.Namespace) -> tuple[list[Materialization], list[ColumnMaterialization]]:
    materializations: list[Materialization] = []
    column_rows: list[ColumnMaterialization] = []

    sources: list[tuple[str, Path | None, str, Path, list[str]]] = []
    for benchmark_dir in args.benchmark_dir:
        experiment = args.experiment or benchmark_dir.name
        for query, log_file, lines in iter_log_segments(benchmark_dir):
            sources.append((experiment, benchmark_dir, query, log_file, lines))
    for log_file in args.log_file:
        experiment = args.experiment or log_file.stem
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        sources.append((experiment, None, query_from_path(log_file), log_file, lines))

    for experiment, benchmark_dir, query, log_file, lines in sources:
        previous_query, second_query = pair_from_experiment(experiment)
        for line in lines:
            if "[scan-audit] parquet_materialize" not in line:
                continue
            fields = parse_kv(line)
            table = infer_table(fields.get("files", ""))
            col_bytes = parse_column_bytes(fields.get("column_bytes", ""))
            column_compressed = sum(v for _, v, _ in col_bytes)
            column_uncompressed = sum(v for _, _, v in col_bytes)
            materializations.append(
                Materialization(
                    experiment=experiment,
                    benchmark_dir=str(benchmark_dir or log_file.parent),
                    query=query,
                    previous_query=previous_query,
                    second_query=second_query,
                    target_gpu=to_int(fields.get("target_gpu"), -1),
                    table=table,
                    files=fields.get("files", ""),
                    columns=fields.get("columns", ""),
                    row_groups=fields.get("row_groups", ""),
                    compressed_bytes=to_int(fields.get("compressed_bytes")),
                    uncompressed_bytes=to_int(fields.get("uncompressed_bytes")),
                    column_compressed_bytes=column_compressed,
                    column_uncompressed_bytes=column_uncompressed,
                    output_rows=to_int(fields.get("output_rows")),
                    output_columns=to_int(fields.get("output_columns")),
                    split_count=to_int(fields.get("split_count")),
                )
            )
            for column, compressed, uncompressed in col_bytes:
                column_rows.append(
                    ColumnMaterialization(
                        experiment=experiment,
                        benchmark_dir=str(benchmark_dir or log_file.parent),
                        query=query,
                        previous_query=previous_query,
                        second_query=second_query,
                        target_gpu=to_int(fields.get("target_gpu"), -1),
                        table=table,
                        column=column,
                        compressed_bytes=compressed,
                        uncompressed_bytes=uncompressed,
                        files=fields.get("files", ""),
                        row_groups=fields.get("row_groups", ""),
                    )
                )
    return materializations, column_rows


def write_csv(path: Path, rows, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                writer.writerow({name: getattr(row, name) for name in fieldnames})
            else:
                writer.writerow(row)


def aggregate(rows, key_fn, value_fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = key_fn(row)
        if key not in grouped:
            grouped[key] = {"count": 0}
            for i, value in enumerate(key):
                grouped[key][f"key_{i}"] = value
            for field in value_fields:
                grouped[key][field] = 0
        grouped[key]["count"] += 1
        for field in value_fields:
            grouped[key][field] += getattr(row, field)
    return list(grouped.values())


def pair_summary(column_rows: list[ColumnMaterialization]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    query_columns = {
        f"q{q}": {(table, column) for table, columns in cols.items() for column in columns}
        for q, cols in QUERY_COLUMNS.items()
    }

    for row in column_rows:
        if not row.previous_query or not row.second_query or row.query != row.second_query:
            continue
        key = (row.experiment, row.previous_query, row.second_query)
        if key not in grouped:
            grouped[key] = {
                "experiment": row.experiment,
                "previous_query": row.previous_query,
                "second_query": row.second_query,
                "second_query_materialized_compressed_bytes": 0,
                "second_query_materialized_uncompressed_bytes": 0,
                "overlap_reload_compressed_bytes": 0,
                "overlap_reload_uncompressed_bytes": 0,
                "second_query_materialized_columns": set(),
                "overlap_reloaded_columns": set(),
            }
        out = grouped[key]
        col_key = (row.table, row.column)
        out["second_query_materialized_compressed_bytes"] += row.compressed_bytes
        out["second_query_materialized_uncompressed_bytes"] += row.uncompressed_bytes
        out["second_query_materialized_columns"].add(f"{row.table}.{row.column}")
        if col_key in query_columns.get(row.previous_query, set()):
            out["overlap_reload_compressed_bytes"] += row.compressed_bytes
            out["overlap_reload_uncompressed_bytes"] += row.uncompressed_bytes
            out["overlap_reloaded_columns"].add(f"{row.table}.{row.column}")

    records = []
    for out in grouped.values():
        total_comp = out["second_query_materialized_compressed_bytes"]
        total_unc = out["second_query_materialized_uncompressed_bytes"]
        overlap_comp = out["overlap_reload_compressed_bytes"]
        overlap_unc = out["overlap_reload_uncompressed_bytes"]
        records.append(
            {
                "experiment": out["experiment"],
                "previous_query": out["previous_query"],
                "second_query": out["second_query"],
                "second_query_materialized_compressed_bytes": total_comp,
                "second_query_materialized_uncompressed_bytes": total_unc,
                "second_query_materialized_compressed_gb": total_comp / 1e9,
                "second_query_materialized_uncompressed_gb": total_unc / 1e9,
                "overlap_reload_compressed_bytes": overlap_comp,
                "overlap_reload_uncompressed_bytes": overlap_unc,
                "overlap_reload_compressed_gb": overlap_comp / 1e9,
                "overlap_reload_uncompressed_gb": overlap_unc / 1e9,
                "overlap_reload_ratio_of_second_load_compressed": overlap_comp / total_comp if total_comp else "",
                "overlap_reload_ratio_of_second_load_uncompressed": overlap_unc / total_unc if total_unc else "",
                "second_query_materialized_columns": " ".join(sorted(out["second_query_materialized_columns"])),
                "overlap_reloaded_columns": " ".join(sorted(out["overlap_reloaded_columns"])),
            }
        )
    return sorted(records, key=lambda r: (r["previous_query"], r["second_query"]))


def rename_keyed(rows: list[dict[str, object]], names: list[str]) -> list[dict[str, object]]:
    renamed = []
    for row in rows:
        out = {name: row.pop(f"key_{i}") for i, name in enumerate(names)}
        out.update(row)
        renamed.append(out)
    return sorted(renamed, key=lambda r: tuple(str(r[name]) for name in names))


def maybe_plot_pair_heatmaps(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"WARNING: could not import plotting libraries: {exc}", file=sys.stderr)
        return

    labels = [f"q{i}" for i in range(1, 23)]
    metrics = [
        (
            "observed_overlap_reload_ratio_heatmap.png",
            "overlap_reload_ratio_of_second_load_uncompressed",
            "Observed overlap reload ratio of second query load",
            "Overlap reload / Qj materialized bytes",
            ".2f",
        ),
        (
            "observed_second_query_materialized_gb_heatmap.png",
            "second_query_materialized_uncompressed_gb",
            "Observed second-query materialized data",
            "Qj materialized GB",
            ".1f",
        ),
        (
            "observed_overlap_reload_gb_heatmap.png",
            "overlap_reload_uncompressed_gb",
            "Observed overlapped data reloaded by second query",
            "Overlapped reload GB",
            ".1f",
        ),
    ]

    for filename, field, title, cbar_label, fmt in metrics:
        matrix = pd.DataFrame(float("nan"), index=labels, columns=labels)
        for row in rows:
            value = row[field]
            if value == "":
                continue
            matrix.loc[row["previous_query"], row["second_query"]] = float(value)
        values = matrix.to_numpy(dtype=float)
        if all(not math.isfinite(v) for row_values in values for v in row_values):
            continue
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Second query (Qj)")
        ax.set_ylabel("Previous query (Qi)")
        ax.set_title(title)
        finite = [v for row_values in values for v in row_values if math.isfinite(v)]
        threshold = max(finite) * 0.55 if finite else 0.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not math.isfinite(value):
                    continue
                color = "white" if value > threshold else "black"
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=6, color=color)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.benchmark_dir and not args.log_file:
        raise SystemExit("provide --benchmark-dir or --log-file")

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif len(args.benchmark_dir) == 1 and not args.log_file:
        output_dir = args.benchmark_dir[0] / "csv"
    else:
        output_dir = Path("scan_audit_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    materializations, column_rows = collect(args)

    materialization_fields = list(Materialization.__dataclass_fields__)
    column_fields = list(ColumnMaterialization.__dataclass_fields__)
    write_csv(output_dir / "scan_audit_materializations.csv", materializations, materialization_fields)
    write_csv(output_dir / "scan_audit_columns.csv", column_rows, column_fields)

    table_rows = rename_keyed(
        aggregate(
            materializations,
            lambda r: (r.experiment, r.query, r.table, r.target_gpu),
            ("compressed_bytes", "uncompressed_bytes", "column_compressed_bytes", "column_uncompressed_bytes", "output_rows"),
        ),
        ["experiment", "query", "table", "target_gpu"],
    )
    column_summary_rows = rename_keyed(
        aggregate(
            column_rows,
            lambda r: (r.experiment, r.query, r.table, r.column, r.target_gpu),
            ("compressed_bytes", "uncompressed_bytes"),
        ),
        ["experiment", "query", "table", "column", "target_gpu"],
    )
    pair_rows = pair_summary(column_rows)

    write_csv(
        output_dir / "scan_audit_by_query_table.csv",
        table_rows,
        ["experiment", "query", "table", "target_gpu", "count", "compressed_bytes", "uncompressed_bytes", "column_compressed_bytes", "column_uncompressed_bytes", "output_rows"],
    )
    write_csv(
        output_dir / "scan_audit_by_query_column.csv",
        column_summary_rows,
        ["experiment", "query", "table", "column", "target_gpu", "count", "compressed_bytes", "uncompressed_bytes"],
    )
    write_csv(
        output_dir / "scan_audit_by_pair_second_query.csv",
        pair_rows,
        [
            "experiment",
            "previous_query",
            "second_query",
            "second_query_materialized_compressed_bytes",
            "second_query_materialized_uncompressed_bytes",
            "second_query_materialized_compressed_gb",
            "second_query_materialized_uncompressed_gb",
            "overlap_reload_compressed_bytes",
            "overlap_reload_uncompressed_bytes",
            "overlap_reload_compressed_gb",
            "overlap_reload_uncompressed_gb",
            "overlap_reload_ratio_of_second_load_compressed",
            "overlap_reload_ratio_of_second_load_uncompressed",
            "second_query_materialized_columns",
            "overlap_reloaded_columns",
        ],
    )

    if not args.no_plots:
        maybe_plot_pair_heatmaps(output_dir, pair_rows)

    print(f"materializations: {len(materializations)}")
    print(f"column rows:      {len(column_rows)}")
    print(f"wrote:            {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
