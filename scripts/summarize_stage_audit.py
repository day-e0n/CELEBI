#!/usr/bin/env python3
# wdy start
"""Summarize Sirius [stage-audit] operator input/output logs.

The scan-audit experiment shows parquet materialization. This script summarizes
operator-stage outputs after scan/filter/join/aggregate/etc. so we can reason
about which intermediate data scope is worth retaining.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
QUERY_RE = re.compile(r"/q(\d+)(?:/|$)")


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
    parser.add_argument("--experiment", default=None, help="Experiment label")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <benchmark>/csv for one benchmark, otherwise ./stage_audit_summary.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots")
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


def to_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def query_from_path(path: Path) -> str:
    match = QUERY_RE.search(str(path))
    return f"q{match.group(1)}" if match else "unknown"


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
    result: list[str] = []
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


def collect(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sources: list[tuple[str, str, Path, list[str]]] = []
    for benchmark_dir in args.benchmark_dir:
        experiment = args.experiment or benchmark_dir.name
        for query, log_file, lines in iter_log_segments(benchmark_dir):
            sources.append((experiment, query, log_file, lines))
    for log_file in args.log_file:
        experiment = args.experiment or log_file.stem
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        sources.append((experiment, query_from_path(log_file), log_file, lines))

    for experiment, query, log_file, lines in sources:
        for line in lines:
            if "[stage-audit]" not in line:
                continue
            fields = parse_kv(line)
            input_bytes = to_int(fields.get("input_bytes"))
            output_bytes = to_int(fields.get("output_bytes"))
            input_rows = to_int(fields.get("input_rows"))
            output_rows = to_int(fields.get("output_rows"))
            rows.append(
                {
                    "experiment": experiment,
                    "query": query,
                    "log_file": str(log_file),
                    "pipeline_id": to_int(fields.get("pipeline_id")),
                    "task_id": to_int(fields.get("task_id")),
                    "operator_id": to_int(fields.get("operator_id")),
                    "operator_name": fields.get("operator_name", ""),
                    "stage_kind": fields.get("stage_kind", "OTHER"),
                    "operator_type": to_int(fields.get("operator_type"), -1),
                    "input_type": fields.get("input_type", ""),
                    "output_type": fields.get("output_type", ""),
                    "actual_gpu": to_int(fields.get("actual_gpu"), -1),
                    "pipeline_operator_count": to_int(fields.get("pipeline_operator_count")),
                    "input_batches": to_int(fields.get("input_batches")),
                    "input_rows": input_rows,
                    "input_columns": to_int(fields.get("input_columns")),
                    "input_bytes": input_bytes,
                    "input_gb": input_bytes / 1e9,
                    "output_batches": to_int(fields.get("output_batches")),
                    "output_rows": output_rows,
                    "output_columns": to_int(fields.get("output_columns")),
                    "output_bytes": output_bytes,
                    "output_gb": output_bytes / 1e9,
                    "byte_ratio": to_float(fields.get("byte_ratio")),
                    "row_ratio": to_float(fields.get("row_ratio")),
                    "duration_us": to_int(fields.get("duration_us")),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key not in grouped:
            out = {k: row[k] for k in keys}
            out.update(
                {
                    "events": 0,
                    "input_bytes": 0,
                    "output_bytes": 0,
                    "input_rows": 0,
                    "output_rows": 0,
                    "input_batches": 0,
                    "output_batches": 0,
                    "duration_us": 0,
                }
            )
            grouped[key] = out
        out = grouped[key]
        out["events"] += 1
        for field in (
            "input_bytes",
            "output_bytes",
            "input_rows",
            "output_rows",
            "input_batches",
            "output_batches",
            "duration_us",
        ):
            out[field] += int(row[field])
    result = []
    for out in grouped.values():
        input_bytes = int(out["input_bytes"])
        input_rows = int(out["input_rows"])
        output_bytes = int(out["output_bytes"])
        out["input_gb"] = input_bytes / 1e9
        out["output_gb"] = output_bytes / 1e9
        out["byte_ratio"] = output_bytes / input_bytes if input_bytes else ""
        out["row_ratio"] = int(out["output_rows"]) / input_rows if input_rows else ""
        out["duration_ms"] = int(out["duration_us"]) / 1000.0
        result.append(out)
    return sorted(result, key=lambda r: tuple(str(r[k]) for k in keys))


def maybe_plot(output_dir: Path, by_stage: list[dict[str, object]]) -> None:
    if not by_stage:
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not import plotting libraries: {exc}")
        return

    df = pd.DataFrame(by_stage)
    stage_order = [
        "SCAN",
        "FILTER",
        "PROJECTION",
        "JOIN",
        "AGGREGATE",
        "PARTITION",
        "CONCAT",
        "SORT",
        "LIMIT",
        "CTE",
        "RESULT",
        "OTHER",
    ]
    queries = sorted(df["query"].unique(), key=lambda q: int(q[1:]) if str(q).startswith("q") and str(q)[1:].isdigit() else 999)
    pivot = df.pivot_table(index="query", columns="stage_kind", values="output_gb", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(index=queries, columns=[s for s in stage_order if s in pivot.columns], fill_value=0.0)
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6), width=0.8)
    ax.set_xlabel("Query")
    ax.set_ylabel("Output GB")
    ax.set_title("Operator-stage output volume by query")
    ax.legend(title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "stage_output_gb_by_query.png", dpi=180)
    plt.close(ax.figure)

    ratio_df = df[df["input_bytes"] > 0].copy()
    if not ratio_df.empty:
        ratio_pivot = ratio_df.pivot_table(index="query", columns="stage_kind", values="byte_ratio", aggfunc="mean")
        ratio_pivot = ratio_pivot.reindex(index=queries, columns=[s for s in stage_order if s in ratio_pivot.columns])
        fig, ax = plt.subplots(figsize=(12, 6))
        values = ratio_pivot.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="viridis", aspect="auto")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Output / input bytes")
        ax.set_xticks(range(len(ratio_pivot.columns)))
        ax.set_yticks(range(len(ratio_pivot.index)))
        ax.set_xticklabels(ratio_pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(ratio_pivot.index)
        ax.set_title("Average byte ratio by query and stage")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if math.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if value > 0.6 else "black")
        fig.tight_layout()
        fig.savefig(output_dir / "stage_byte_ratio_heatmap.png", dpi=180)
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
        output_dir = Path("stage_audit_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(args)
    event_fields = [
        "experiment",
        "query",
        "log_file",
        "pipeline_id",
        "task_id",
        "operator_id",
        "operator_name",
        "stage_kind",
        "operator_type",
        "input_type",
        "output_type",
        "actual_gpu",
        "pipeline_operator_count",
        "input_batches",
        "input_rows",
        "input_columns",
        "input_bytes",
        "input_gb",
        "output_batches",
        "output_rows",
        "output_columns",
        "output_bytes",
        "output_gb",
        "byte_ratio",
        "row_ratio",
        "duration_us",
    ]
    write_csv(output_dir / "stage_audit_events.csv", rows, event_fields)

    by_stage = aggregate(rows, ["experiment", "query", "stage_kind"])
    by_operator = aggregate(rows, ["experiment", "query", "stage_kind", "operator_name", "operator_id"])
    summary_fields = [
        "experiment",
        "query",
        "stage_kind",
        "events",
        "input_bytes",
        "output_bytes",
        "input_gb",
        "output_gb",
        "input_rows",
        "output_rows",
        "input_batches",
        "output_batches",
        "byte_ratio",
        "row_ratio",
        "duration_us",
        "duration_ms",
    ]
    write_csv(output_dir / "stage_audit_by_query_stage.csv", by_stage, summary_fields)
    write_csv(
        output_dir / "stage_audit_by_query_operator.csv",
        by_operator,
        ["experiment", "query", "stage_kind", "operator_name", "operator_id"] + summary_fields[3:],
    )

    if not args.no_plots:
        maybe_plot(output_dir, by_stage)

    print(f"stage events: {len(rows)}")
    print(f"wrote:        {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
