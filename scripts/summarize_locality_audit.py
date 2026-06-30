#!/usr/bin/env python3
# wdy start
"""Summarize Sirius [locality-audit] logs emitted by locality instrumentation."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
QUERY_RE = re.compile(r"/q(\d+)(?:/|$)")

BYTE_FIELDS = (
    "input_bytes",
    "local_bytes",
    "remote_gpu_bytes",
    "host_bytes",
    "disk_bytes",
    "gpu_source_bytes",
    "host_source_bytes",
    "disk_source_bytes",
)


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


def iter_log_files(benchmark_dir: Path):
    # Prefer split per-query Sirius logs; fall back to raw log_dir files.
    split_logs = sorted((benchmark_dir / "sirius").glob("q*/sirius.log"))
    if split_logs:
        yield from split_logs
        return
    log_dir = benchmark_dir / "log_dir"
    if log_dir.exists():
        for path in sorted(log_dir.rglob("*.log")):
            yield path


def summarize(benchmark_dir: Path, experiment: str):
    phase_rows = defaultdict(lambda: defaultdict(int))
    dispatch_rows = defaultdict(lambda: defaultdict(int))
    task_create_rows = defaultdict(lambda: defaultdict(int))

    for log_file in iter_log_files(benchmark_dir):
        query = query_from_path(log_file)
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if "[locality-audit]" not in line:
                continue
            fields = parse_kv(line)
            if " dispatch " in line:
                key = (experiment, query)
                row = dispatch_rows[key]
                row["dispatches"] += 1
                preferred = to_int(fields.get("preferred_device"), -1)
                actual = to_int(fields.get("actual_gpu"), -1)
                if preferred >= 0:
                    row["preferred_dispatches"] += 1
                if preferred >= 0 and actual >= 0 and preferred != actual:
                    row["preferred_actual_mismatches"] += 1
                continue

            if " task_create " in line:
                key = (experiment, query)
                row = task_create_rows[key]
                row["task_creates"] += 1
                for field in BYTE_FIELDS:
                    row[field] += to_int(fields.get(field))
                if to_int(fields.get("preferred_device"), -1) >= 0:
                    row["preferred_tasks"] += 1
                continue

            phase = None
            if " prepare_before " in line:
                phase = "prepare_before"
            elif " prepare_after " in line:
                phase = "prepare_after"
            if phase is not None:
                key = (experiment, query, phase)
                row = phase_rows[key]
                row["tasks"] += 1
                for field in BYTE_FIELDS:
                    row[field] += to_int(fields.get(field))
                row["batch_count"] += to_int(fields.get("batch_count"))

    return phase_rows, dispatch_rows, task_create_rows


def write_phase_csv(path: Path, phase_rows):
    cols = [
        "experiment",
        "query",
        "phase",
        "tasks",
        "input_bytes",
        "local_bytes",
        "remote_gpu_bytes",
        "host_bytes",
        "disk_bytes",
        "batch_count",
        "locality_ratio",
        "movement_candidate_bytes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for (experiment, query, phase), row in sorted(phase_rows.items()):
            input_bytes = row["input_bytes"]
            movement = row["remote_gpu_bytes"] + row["host_bytes"] + row["disk_bytes"]
            out = {c: row.get(c, 0) for c in cols}
            out.update(
                {
                    "experiment": experiment,
                    "query": query,
                    "phase": phase,
                    "locality_ratio": (row["local_bytes"] / input_bytes) if input_bytes else "",
                    "movement_candidate_bytes": movement,
                }
            )
            writer.writerow(out)


def write_dispatch_csv(path: Path, dispatch_rows):
    cols = [
        "experiment",
        "query",
        "dispatches",
        "preferred_dispatches",
        "preferred_actual_mismatches",
        "mismatch_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for (experiment, query), row in sorted(dispatch_rows.items()):
            preferred = row["preferred_dispatches"]
            out = {c: row.get(c, 0) for c in cols}
            out.update(
                {
                    "experiment": experiment,
                    "query": query,
                    "mismatch_ratio": (row["preferred_actual_mismatches"] / preferred)
                    if preferred
                    else "",
                }
            )
            writer.writerow(out)


def write_task_create_csv(path: Path, task_create_rows):
    cols = [
        "experiment",
        "query",
        "task_creates",
        "preferred_tasks",
        "input_bytes",
        "local_bytes",
        "gpu_source_bytes",
        "host_source_bytes",
        "disk_source_bytes",
        "locality_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for (experiment, query), row in sorted(task_create_rows.items()):
            input_bytes = row["input_bytes"]
            out = {c: row.get(c, 0) for c in cols}
            out.update(
                {
                    "experiment": experiment,
                    "query": query,
                    "locality_ratio": (row["local_bytes"] / input_bytes) if input_bytes else "",
                }
            )
            writer.writerow(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    experiment = args.experiment or benchmark_dir.name
    out_dir = (args.out_dir or benchmark_dir / "locality_summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_rows, dispatch_rows, task_create_rows = summarize(benchmark_dir, experiment)
    write_phase_csv(out_dir / "prepare_summary.csv", phase_rows)
    write_dispatch_csv(out_dir / "dispatch_summary.csv", dispatch_rows)
    write_task_create_csv(out_dir / "task_create_summary.csv", task_create_rows)

    print(f"wrote {out_dir / 'prepare_summary.csv'}")
    print(f"wrote {out_dir / 'dispatch_summary.csv'}")
    print(f"wrote {out_dir / 'task_create_summary.csv'}")


if __name__ == "__main__":
    main()
# wdy end
