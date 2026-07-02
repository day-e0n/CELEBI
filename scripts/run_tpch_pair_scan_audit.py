#!/usr/bin/env python3
# wdy start
"""Run observed TPC-H query-pair scan-audit experiments.

Each pair runs as Qi -> Qj in one Sirius connection. The scan audit logs then
answer: when Qj runs after Qi, how many parquet bytes did Sirius actually
materialize, and how much of that materialized data overlaps Qi's table/column
footprint?
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT = REPO_ROOT / "tpch_pair_scan_audit_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="TPC-H parquet directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output root directory")
    parser.add_argument(
        "--pairs",
        default="2:16",
        help="Comma-separated ordered pairs, e.g. '2:16,1:6'. Ignored with --all-ordered.",
    )
    parser.add_argument("--all-ordered", action="store_true", help="Run all ordered Qi->Qj pairs")
    parser.add_argument("--include-diagonal", action="store_true", help="Include Qi->Qi with --all-ordered")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per pair")
    parser.add_argument("--devices", default="0,1,2,3", help="CUDA_VISIBLE_DEVICES for the run")
    parser.add_argument("--num-gpus", type=int, default=4, help="Sirius topology.num_gpus")
    parser.add_argument("--gpu-usage-limit", default="4GB", help="Sirius memory.gpu.usage_limit_bytes")
    parser.add_argument("--host-capacity", default="16GB", help="Sirius memory.host.capacity_bytes")
    parser.add_argument(
        "--reservation-limit-fraction",
        default="0.8",
        help="Sirius memory.gpu.reservation_limit_fraction",
    )
    parser.add_argument("--skip-build", action="store_true", help="Do not run pixi run make -j4 first")
    # wdy start
    parser.add_argument(
        "--join-output-retention",
        action="store_true",
        help="Enable experimental retain-only HASH_JOIN output retention.",
    )
    parser.add_argument(
        "--join-output-retention-limit-bytes",
        default="8589934592",
        help="Maximum retained HASH_JOIN output bytes when --join-output-retention is set.",
    )
    parser.add_argument(
        "--join-output-retention-max-batch-bytes",
        default="268435456",
        help="Maximum single HASH_JOIN output batch retained when --join-output-retention is set.",
    )
    parser.add_argument(
        "--join-output-reuse",
        action="store_true",
        help="Enable experimental HASH_JOIN output reuse on exact signature hit.",
    )
    # wdy end
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def parse_pairs(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.all_ordered:
        return [
            (qi, qj)
            for qi in range(1, 23)
            for qj in range(1, 23)
            if args.include_diagonal or qi != qj
        ]

    pairs: list[tuple[int, int]] = []
    for item in args.pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"bad pair {item!r}; expected Qi:Qj")
        left, right = item.split(":", 1)
        qi, qj = int(left), int(right)
        if not 1 <= qi <= 22 or not 1 <= qj <= 22:
            raise SystemExit(f"bad pair {item!r}; query numbers must be 1..22")
        pairs.append((qi, qj))
    if not pairs:
        raise SystemExit("no pairs selected")
    return pairs


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


def run(cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"==> {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_pair(
    cmd: list[str], *, env: dict[str, str], cwd: Path, dry_run: bool
) -> bool:
    """Like run() but returns False on failure instead of raising."""
    printable = " ".join(cmd)
    print(f"==> {printable}", flush=True)
    if dry_run:
        return True
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        print(
            f"[WARN] command failed with exit code {result.returncode}, skipping pair.",
            flush=True,
        )
        return False
    return True


def main() -> int:
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")

    pairs = parse_pairs(args)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output / f"pair_scan_{ts}"
    config_dir = run_root / "configs"
    pair_root = run_root / "pairs"
    summary_dir = run_root / "summary"
    stage_summary_dir = run_root / "stage_summary"
    runtime_breakdown_dir = run_root / "runtime_breakdown"
    config_dir.mkdir(parents=True, exist_ok=True)
    pair_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    stage_summary_dir.mkdir(parents=True, exist_ok=True)
    runtime_breakdown_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / f"sirius_{args.num_gpus}gpu.yaml"
    write_config(config_path, args)

    readme_pairs = ", ".join(f"q{qi}->q{qj}" for qi, qj in pairs)
    (run_root / "README.md").write_text(
        f"""# TPC-H observed scan-audit query-pair run

Input: `{args.input.resolve()}`
Pairs: `{readme_pairs}`
Iterations: `{args.iterations}`
Devices: `{args.devices}`
Config: `{config_path.relative_to(run_root)}`
Join output retention: `{args.join_output_retention}`
Join output retention limit bytes: `{args.join_output_retention_limit_bytes}`
Join output retention max batch bytes: `{args.join_output_retention_max_batch_bytes}`
Join output reuse: `{args.join_output_reuse}`

Main scan summary files:
- `summary/scan_audit_by_pair_second_query.csv`
- `summary/observed_overlap_reload_ratio_heatmap.png`
- `summary/observed_second_query_materialized_gb_heatmap.png`
- `summary/observed_overlap_reload_gb_heatmap.png`

Main stage summary files:
- `stage_summary/stage_audit_by_query_stage.csv`
- `stage_summary/stage_output_gb_by_query.png`
- `stage_summary/stage_byte_ratio_heatmap.png`

Main runtime breakdown files:
- `runtime_breakdown/runtime_breakdown.csv`
- `runtime_breakdown/runtime_load_nonload_breakdown.png`
- `runtime_breakdown/runtime_load_ratio.png`

Interpretation: for each Qi->Qj pair, the overlap reload bytes are the Qj
materialized table/column bytes whose table/column also appeared in Qi's TPC-H
footprint.
"""
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices
    env["SIRIUS_CONFIG_FILE"] = str(config_path)
    env["SIRIUS_LOG_LEVEL"] = "info"
    # wdy start
    if args.join_output_retention:
        env["SIRIUS_JOIN_OUTPUT_RETENTION"] = "1"
        env["SIRIUS_JOIN_OUTPUT_RETENTION_LIMIT_BYTES"] = str(args.join_output_retention_limit_bytes)
        env["SIRIUS_JOIN_OUTPUT_RETENTION_MAX_BATCH_BYTES"] = str(
            args.join_output_retention_max_batch_bytes
        )
        if args.join_output_reuse:
            env["SIRIUS_JOIN_OUTPUT_REUSE"] = "1"
        else:
            env.pop("SIRIUS_JOIN_OUTPUT_REUSE", None)
    else:
        env.pop("SIRIUS_JOIN_OUTPUT_RETENTION", None)
        env.pop("SIRIUS_JOIN_OUTPUT_RETENTION_LIMIT_BYTES", None)
        env.pop("SIRIUS_JOIN_OUTPUT_RETENTION_MAX_BATCH_BYTES", None)
        env.pop("SIRIUS_JOIN_OUTPUT_REUSE", None)
    # wdy end

    if not args.skip_build:
        run(["pixi", "run", "make", "-j4"], env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    benchmark_dirs: list[Path] = []
    manifest_rows = []
    failed_rows = []
    for idx, (qi, qj) in enumerate(pairs, 1):
        name = f"q{qi}_then_q{qj}"
        benchmark_dir = pair_root / name
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
            str(args.iterations),
            "--queries",
            f"{qi},{qj}",
            "--config",
            str(config_path),
            "--output",
            str(pair_root),
            "--name",
            name,
        ]
        print(f"[{idx}/{len(pairs)}] q{qi}->q{qj}", flush=True)
        ok = run_pair(cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)
        if ok:
            benchmark_dirs.append(benchmark_dir)
            manifest_rows.append({"previous_query": f"q{qi}", "second_query": f"q{qj}", "benchmark_dir": str(benchmark_dir)})
        else:
            failed_rows.append({"previous_query": f"q{qi}", "second_query": f"q{qj}"})

    if failed_rows:
        with (run_root / "failed_pairs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["previous_query", "second_query"])
            writer.writeheader()
            writer.writerows(failed_rows)
        print(f"[WARN] {len(failed_rows)} pair(s) failed. See {run_root / 'failed_pairs.csv'}", flush=True)

    with (run_root / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["previous_query", "second_query", "benchmark_dir"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    summarize_cmd = ["python3", str(REPO_ROOT / "scripts" / "summarize_scan_audit.py")]
    for benchmark_dir in benchmark_dirs:
        summarize_cmd.extend(["--benchmark-dir", str(benchmark_dir)])
    summarize_cmd.extend(["--output-dir", str(summary_dir)])
    run(summarize_cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    stage_cmd = ["python3", str(REPO_ROOT / "scripts" / "summarize_stage_audit.py")]
    for benchmark_dir in benchmark_dirs:
        stage_cmd.extend(["--benchmark-dir", str(benchmark_dir)])
    stage_cmd.extend(["--output-dir", str(stage_summary_dir)])
    run(stage_cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    breakdown_cmd = [
        "python3",
        str(REPO_ROOT / "scripts" / "summarize_runtime_breakdown.py"),
        "--run-root",
        str(run_root),
        "--output-dir",
        str(runtime_breakdown_dir),
    ]
    run(breakdown_cmd, env=env, cwd=REPO_ROOT, dry_run=args.dry_run)

    print(f"==> Done. Output root: {run_root}")
    print(f"==> Pair summary: {summary_dir / 'scan_audit_by_pair_second_query.csv'}")
    print(f"==> Stage summary: {stage_summary_dir / 'stage_audit_by_query_stage.csv'}")
    print(f"==> Runtime breakdown: {runtime_breakdown_dir / 'runtime_breakdown.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
