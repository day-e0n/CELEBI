#!/usr/bin/env python3
# wdy start
"""Build a coarse Sirius runtime breakdown from benchmark runtime + audit logs.

The intent is a practical baseline for retention experiments:
- load_ms: time spent inside parquet read/materialization, from [scan-audit]
- non_load_ms: total query runtime minus load_ms
- non_scan_operator_ms: summed [stage-audit] duration for non-SCAN operators

`non_load_ms` is the clean two-way split for slides. `non_scan_operator_ms`
is diagnostic: it excludes scheduler/sink/result collection overhead and can be
smaller than non_load_ms.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run root containing manifest.csv, summary/, stage_summary/, and pair/query benchmark dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-root>/runtime_breakdown.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def find_benchmark_dirs(run_root: Path) -> dict[str, Path]:
    manifest = read_csv(run_root / "manifest.csv")
    result: dict[str, Path] = {}
    for row in manifest:
        path = Path(row.get("benchmark_dir", ""))
        if not path.is_absolute():
            path = run_root / path
        if path.exists():
            result[path.name] = path
    if result:
        return result

    for runtime in run_root.rglob("csv/runtimes.csv"):
        bench = runtime.parents[1]
        result[bench.name] = bench
    return result


def load_runtime_rows(benchmark_dirs: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for experiment, bench in sorted(benchmark_dirs.items()):
        runtime_path = bench / "csv" / "runtimes.csv"
        for row in read_csv(runtime_path):
            if row.get("engine") != "sirius":
                continue
            query = row.get("query", "unknown")
            iteration = int(to_float(row.get("iteration"), 0))
            runtime_ms = to_float(row.get("runtime_s")) * 1000.0
            rows.append(
                {
                    "experiment": experiment,
                    "query": query,
                    "iteration": iteration,
                    "runtime_ms": runtime_ms,
                    "benchmark_dir": str(bench),
                }
            )
    return rows


def load_scan_ms(run_root: Path) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], float] = defaultdict(float)
    for row in read_csv(run_root / "summary" / "scan_audit_materializations.csv"):
        key = (row.get("experiment", ""), row.get("query", ""))
        grouped[key] += to_float(row.get("duration_us")) / 1000.0
    return grouped


def load_stage_ms(run_root: Path) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    scan_stage: dict[tuple[str, str], float] = defaultdict(float)
    non_scan_stage: dict[tuple[str, str], float] = defaultdict(float)
    for row in read_csv(run_root / "stage_summary" / "stage_audit_by_query_stage.csv"):
        key = (row.get("experiment", ""), row.get("query", ""))
        duration_ms = to_float(row.get("duration_ms"))
        if row.get("stage_kind") == "SCAN":
            scan_stage[key] += duration_ms
        else:
            non_scan_stage[key] += duration_ms
    return scan_stage, non_scan_stage


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_rows(run_root: Path) -> list[dict[str, object]]:
    benchmark_dirs = find_benchmark_dirs(run_root)
    runtime_rows = load_runtime_rows(benchmark_dirs)
    scan_load = load_scan_ms(run_root)
    scan_stage, non_scan_stage = load_stage_ms(run_root)

    rows: list[dict[str, object]] = []
    for row in runtime_rows:
        key = (str(row["experiment"]), str(row["query"]))
        runtime_ms = float(row["runtime_ms"])
        load_ms = scan_load.get(key, 0.0)
        non_load_ms = max(runtime_ms - load_ms, 0.0)
        operator_ms = non_scan_stage.get(key, 0.0)
        scan_operator_ms = scan_stage.get(key, 0.0)
        unaccounted_ms = runtime_ms - load_ms - operator_ms
        rows.append(
            {
                "experiment": row["experiment"],
                "query": row["query"],
                "iteration": row["iteration"],
                "runtime_ms": runtime_ms,
                "load_ms": load_ms,
                "non_load_ms": non_load_ms,
                "load_ratio": load_ms / runtime_ms if runtime_ms else "",
                "non_load_ratio": non_load_ms / runtime_ms if runtime_ms else "",
                "scan_stage_ms": scan_operator_ms,
                "non_scan_operator_ms": operator_ms,
                "unaccounted_ms": unaccounted_ms,
                "benchmark_dir": row["benchmark_dir"],
            }
        )
    return rows


def maybe_plot(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not import plotting libraries: {exc}")
        return

    df = pd.DataFrame(rows)
    df["label"] = df["experiment"].astype(str) + " / " + df["query"].astype(str)
    df = df.sort_values(["experiment", "query", "iteration"])

    fig_h = max(4.0, min(14.0, 0.35 * len(df) + 2.0))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = range(len(df))
    ax.barh(y, df["load_ms"], label="load/materialize", color="#e76f51")
    ax.barh(y, df["non_load_ms"], left=df["load_ms"], label="non-load", color="#457b9d")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("Runtime (ms)")
    ax.set_title("Sirius baseline runtime breakdown")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_load_nonload_breakdown.png", dpi=180)
    plt.close(fig)

    ratio = df.copy()
    ratio["load_ratio_pct"] = ratio["load_ratio"].apply(lambda v: float(v) * 100.0 if v != "" and math.isfinite(float(v)) else 0.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(range(len(ratio)), ratio["load_ratio_pct"], color="#e76f51")
    ax.set_yticks(list(range(len(ratio))))
    ax.set_yticklabels(ratio["label"], fontsize=8)
    ax.set_xlabel("Load/materialize share of runtime (%)")
    ax.set_title("Sirius baseline load share")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "runtime_load_ratio.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir or run_root / "runtime_breakdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = make_rows(run_root)
    fields = [
        "experiment",
        "query",
        "iteration",
        "runtime_ms",
        "load_ms",
        "non_load_ms",
        "load_ratio",
        "non_load_ratio",
        "scan_stage_ms",
        "non_scan_operator_ms",
        "unaccounted_ms",
        "benchmark_dir",
    ]
    write_csv(output_dir / "runtime_breakdown.csv", rows, fields)
    if not args.no_plots:
        maybe_plot(output_dir, rows)
    print(f"rows:  {len(rows)}")
    print(f"wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
