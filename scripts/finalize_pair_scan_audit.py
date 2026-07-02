#!/usr/bin/env python3
# wdy start
"""Finalize a resumed TPC-H pair scan-audit run.

The normal runner summarizes results only after all pairs finish in one process.
Our long all-pair baseline can be resumed and can skip unstable pairs, so this
helper rebuilds the summaries from whatever completed benchmark directories
exist under a run root.

It also builds a representative per-query stage summary. This matters because an
all-pair run executes each query many times; summing JOIN bytes across all pairs
would inflate the retained-output cost. The representative summary uses the
median per query/stage by default and is the right input for JOIN-retention
opportunity heatmaps.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_RE_PREFIX = "q"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="pair_scan_* run root")
    parser.add_argument("--query-dir", type=Path, default=Path("/home/dy1013/queries"))
    parser.add_argument(
        "--representative",
        choices=("median", "mean"),
        default="median",
        help="How to collapse repeated query/stage rows from all-pair runs.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("==>", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, check=True)


def completed_benchmark_dirs(run_root: Path) -> list[Path]:
    pair_root = run_root / "pairs"
    dirs: list[Path] = []
    for runtime in sorted(pair_root.glob("q*_then_q*/csv/runtimes.csv")):
        bench = runtime.parents[1]
        try:
            rows = runtime.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if len(rows) > 1 and (bench / "log_dir").exists():
            dirs.append(bench)
    return dirs


def pair_from_name(name: str) -> tuple[str, str] | None:
    if "_then_" not in name:
        return None
    left, right = name.split("_then_", 1)
    if not left.startswith(PAIR_RE_PREFIX) or not right.startswith(PAIR_RE_PREFIX):
        return None
    return left, right


def write_manifest(run_root: Path, dirs: list[Path]) -> None:
    with (run_root / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["previous_query", "second_query", "benchmark_dir"])
        writer.writeheader()
        for bench in dirs:
            pair = pair_from_name(bench.name)
            if pair is None:
                continue
            writer.writerow(
                {
                    "previous_query": pair[0],
                    "second_query": pair[1],
                    "benchmark_dir": str(bench),
                }
            )


def to_float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def summarize_values(values: list[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "mean":
        return sum(values) / len(values)
    return statistics.median(values)


def write_representative_stage_summary(stage_dir: Path, mode: str) -> Path:
    source = stage_dir / "stage_audit_by_query_stage.csv"
    output = stage_dir / "stage_audit_by_query_stage_representative.csv"
    rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            query = row.get("query", "")
            stage = row.get("stage_kind", "")
            if not query or not stage:
                continue
            rows[(query, stage)].append(row)

    numeric_fields = [
        "events",
        "input_bytes",
        "output_bytes",
        "input_gb",
        "output_gb",
        "input_rows",
        "output_rows",
        "input_batches",
        "output_batches",
        "duration_us",
        "duration_ms",
    ]
    passthrough = [field for field in fieldnames if field not in numeric_fields]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows, key=lambda item: (int(item[0][1:]) if item[0].startswith("q") else 999, item[1])):
            group = rows[key]
            out = {field: group[0].get(field, "") for field in passthrough}
            out["experiment"] = f"representative_{mode}"
            for field in numeric_fields:
                value = summarize_values([to_float(row.get(field)) for row in group], mode)
                if field in ("events", "input_bytes", "output_bytes", "input_rows", "output_rows", "input_batches", "output_batches", "duration_us"):
                    out[field] = str(int(round(value)))
                else:
                    out[field] = f"{value:.9f}"
            input_bytes = to_float(out.get("input_bytes"))
            input_rows = to_float(out.get("input_rows"))
            out["input_gb"] = f"{input_bytes / 1e9:.9f}"
            out["output_gb"] = f"{to_float(out.get('output_bytes')) / 1e9:.9f}"
            out["byte_ratio"] = f"{to_float(out.get('output_bytes')) / input_bytes:.9f}" if input_bytes else ""
            out["row_ratio"] = f"{to_float(out.get('output_rows')) / input_rows:.9f}" if input_rows else ""
            out["duration_ms"] = f"{to_float(out.get('duration_us')) / 1000.0:.9f}"
            writer.writerow(out)
    return output


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.exists():
        raise SystemExit(f"run root does not exist: {run_root}")
    benchmark_dirs = completed_benchmark_dirs(run_root)
    if not benchmark_dirs:
        raise SystemExit(f"no completed benchmark dirs under {run_root / 'pairs'}")

    summary_dir = run_root / "summary"
    stage_dir = run_root / "stage_summary"
    runtime_dir = run_root / "runtime_breakdown"
    semantic_dir = run_root / "join_semantic_retention_matrix"
    byte_dir = run_root / "join_retention_matrix_observed_upper_bound"
    for path in (summary_dir, stage_dir, runtime_dir, semantic_dir, byte_dir):
        path.mkdir(parents=True, exist_ok=True)

    write_manifest(run_root, benchmark_dirs)

    scan_cmd = [sys.executable, REPO_ROOT / "scripts" / "summarize_scan_audit.py"]
    for bench in benchmark_dirs:
        scan_cmd.extend(["--benchmark-dir", bench])
    scan_cmd.extend(["--output-dir", summary_dir])
    if args.no_plots:
        scan_cmd.append("--no-plots")
    run(scan_cmd)

    stage_cmd = [sys.executable, REPO_ROOT / "scripts" / "summarize_stage_audit.py"]
    for bench in benchmark_dirs:
        stage_cmd.extend(["--benchmark-dir", bench])
    stage_cmd.extend(["--output-dir", stage_dir])
    if args.no_plots:
        stage_cmd.append("--no-plots")
    run(stage_cmd)

    representative_stage = write_representative_stage_summary(stage_dir, args.representative)

    run(
        [
            sys.executable,
            REPO_ROOT / "scripts" / "summarize_runtime_breakdown.py",
            "--run-root",
            run_root,
            "--output-dir",
            runtime_dir,
        ]
        + (["--no-plots"] if args.no_plots else [])
    )
    run(
        [
            sys.executable,
            REPO_ROOT / "scripts" / "analyze_semantic_join_retention_matrix.py",
            "--stage-summary",
            representative_stage,
            "--query-dir",
            args.query_dir,
            "--output-dir",
            semantic_dir,
        ]
        + (["--no-plots"] if args.no_plots else [])
    )
    run(
        [
            sys.executable,
            REPO_ROOT / "scripts" / "analyze_join_retention_matrix.py",
            "--stage-summary",
            representative_stage,
            "--output-dir",
            byte_dir,
        ]
        + (["--no-plots"] if args.no_plots else [])
    )

    print(f"completed pairs:     {len(benchmark_dirs)}")
    print(f"scan summary:        {summary_dir}")
    print(f"stage summary:       {stage_dir}")
    print(f"representative CSV:  {representative_stage}")
    print(f"runtime breakdown:   {runtime_dir}")
    print(f"semantic JOIN maps:  {semantic_dir}")
    print(f"byte upper-bound:    {byte_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
