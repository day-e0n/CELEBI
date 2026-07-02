#!/usr/bin/env python3
# wdy start
"""Run per-query TPC-H stage audit and generate all-pair JOIN retention matrices."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "experiment" / "join_retention_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--queries", default="1-22", help="Query spec for performance_test.py")
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--gpu-usage-limit", default="18GB")
    parser.add_argument("--host-capacity", default="64GB")
    parser.add_argument("--reservation-limit-fraction", default="0.8")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--query-dir", type=Path, default=Path("/home/dy1013/queries"), help="Directory containing q1.sql ... q22.sql for semantic JOIN graph analysis")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_query_spec(spec: str) -> list[int]:
    out: list[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(item))
    return [q for q in out if 1 <= q <= 22]


def write_config(path: Path, args: argparse.Namespace) -> None:
    path.write_text(
        f"""sirius:
  topology:
    num_gpus: {args.num_gpus}
  memory:
    gpu:
      usage_limit_bytes: {args.gpu_usage_limit}
      reservation_limit_fraction: {args.reservation_limit_fraction}
    host:
      capacity_bytes: {args.host_capacity}
"""
    )


def run(cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("==> " + " ".join(cmd), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, cwd=cwd, env=env, check=check, text=True)


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")
    queries = parse_query_spec(args.queries)
    if not queries:
        raise SystemExit("no queries selected")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output / f"join_retention_{ts}"
    config_dir = run_root / "configs"
    query_root = run_root / "queries"
    stage_summary_dir = run_root / "stage_summary"
    matrix_dir = run_root / "join_retention_matrix"
    semantic_matrix_dir = run_root / "join_semantic_retention_matrix"
    for path in (config_dir, query_root, stage_summary_dir, matrix_dir, semantic_matrix_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config_path, args)

    (run_root / "README.md").write_text(
        f"""# TPC-H JOIN retention audit

Input: `{args.input.resolve()}`
Queries: `{','.join(f'q{q}' for q in queries)}`
Devices: `{args.devices}`
Config: `{config_path.relative_to(run_root)}`

Outputs:
- `stage_summary/stage_audit_by_query_stage.csv`
- `join_retention_matrix/join_retention_cost_gb_heatmap.png`
- `join_retention_matrix/join_reuse_loss_gb_heatmap.png`
- `join_retention_matrix/join_retention_efficiency_heatmap.png`
- `join_semantic_retention_matrix/join_semantic_similarity_heatmap.png`
- `join_semantic_retention_matrix/join_semantic_reuse_gb_heatmap.png`
- `join_semantic_retention_matrix/join_semantic_useful_retention_cost_gb_heatmap.png`

Method: each query is executed once independently. The `join_retention_matrix`
outputs are byte-capacity upper bounds from per-query JOIN input/output bytes.
The `join_semantic_retention_matrix` outputs additionally require SQL JOIN graph
overlap from the query files, so unrelated joins do not receive reuse credit.
Diagonal cells and queries without JOIN output/input are blank.
"""
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_LEVEL"] = "info"

    if not args.skip_build:
        run(["pixi", "run", "make", "-j4"], env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    manifest_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    benchmark_dirs: list[Path] = []
    for q in queries:
        name = f"q{q}"
        benchmark_dir = query_root / name
        cmd = [
            "pixi",
            "run",
            "python",
            "test/tpch_performance/performance_test.py",
            "--input",
            str(args.input.resolve()),
            "--engine",
            "gpu",
            "--mode",
            "sequential",
            "--iterations",
            "1",
            "--queries",
            str(q),
            "--config",
            str(config_path),
            "--output",
            str(query_root),
            "--name",
            name,
        ]
        proc = run(cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run, check=False)
        if proc.returncode == 0:
            benchmark_dirs.append(benchmark_dir)
            manifest_rows.append({"query": f"q{q}", "benchmark_dir": str(benchmark_dir), "status": "ok"})
        else:
            failed_rows.append({"query": f"q{q}", "benchmark_dir": str(benchmark_dir), "status": f"failed:{proc.returncode}"})
            manifest_rows.append({"query": f"q{q}", "benchmark_dir": str(benchmark_dir), "status": f"failed:{proc.returncode}"})

    with (run_root / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "benchmark_dir", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    if not benchmark_dirs:
        raise SystemExit("all query runs failed; no stage summary generated")

    stage_cmd = ["python3", str(REPO_ROOT / "scripts" / "summarize_stage_audit.py")]
    for benchmark_dir in benchmark_dirs:
        stage_cmd.extend(["--benchmark-dir", str(benchmark_dir)])
    stage_cmd.extend(["--output-dir", str(stage_summary_dir)])
    run(stage_cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    matrix_cmd = [
        "python3",
        str(REPO_ROOT / "scripts" / "analyze_join_retention_matrix.py"),
        "--stage-summary",
        str(stage_summary_dir / "stage_audit_by_query_stage.csv"),
        "--output-dir",
        str(matrix_dir),
    ]
    run(matrix_cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    if args.query_dir.is_dir():
        semantic_matrix_cmd = [
            "python3",
            str(REPO_ROOT / "scripts" / "analyze_semantic_join_retention_matrix.py"),
            "--stage-summary",
            str(stage_summary_dir / "stage_audit_by_query_stage.csv"),
            "--query-dir",
            str(args.query_dir),
            "--output-dir",
            str(semantic_matrix_dir),
        ]
        run(semantic_matrix_cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)
    else:
        print(f"==> WARNING: query dir not found; skipped semantic JOIN matrix: {args.query_dir}")

    print(f"==> Done. Output root: {run_root}")
    if failed_rows:
        print("==> Failed queries: " + ", ".join(row["query"] for row in failed_rows))
    print(f"==> JOIN byte upper-bound matrix: {matrix_dir}")
    if args.query_dir.is_dir():
        print(f"==> JOIN semantic matrix: {semantic_matrix_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
