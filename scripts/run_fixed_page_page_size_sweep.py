#!/usr/bin/env python3
# wdy start
"""Run fixed-page experiments across several logical page sizes.

This is a thin orchestration layer over run_fixed_page_suite.py.  Each page
size gets its own compact suite directory, then the key summary CSVs are
combined into sweep-level CSVs for notebook plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("/mnt/nvme/dataset")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiment" / "fixed_page_runs"
DEFAULT_PAGE_SIZES = "262144,524288,1048576,2097152,4194304"
DEFAULT_QUERIES = "3,5,7,8,9,10,18,21"
DEFAULT_CONDITIONS = "baseline,paging_key_only,paging_budget"


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_page_size(value: str) -> int:
    raw = value.strip().lower()
    multiplier = 1
    for suffix, factor in (("kib", 1024), ("kb", 1024), ("mib", 1024 * 1024), ("mb", 1024 * 1024)):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
            multiplier = factor
            break
    return int(float(raw) * multiplier)


def page_size_label(page_bytes: int) -> str:
    if page_bytes % (1024 * 1024) == 0:
        return f"{page_bytes // (1024 * 1024)}m"
    if page_bytes % 1024 == 0:
        return f"{page_bytes // 1024}k"
    return str(page_bytes)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prefixed_row(page_bytes: int, suite_name: str, row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {
        "page_bytes": page_bytes,
        "page_label": page_size_label(page_bytes),
        "suite_name": suite_name,
    }
    out.update(row)
    return out


def collect_summary(sweep_root: Path, suite_name: str, page_bytes: int) -> None:
    suite_root = sweep_root / suite_name
    summary_dir = suite_root / "summary"
    sweep_summary = sweep_root / "summary"
    sweep_summary.mkdir(parents=True, exist_ok=True)

    targets = [
        ("headline.csv", "sweep_headline.csv"),
        ("pairs_baseline_vs_paging_second_query.csv", "sweep_pairs_baseline_vs_paging_second_query.csv"),
        ("random_baseline_vs_paging_workload.csv", "sweep_random_baseline_vs_paging_workload.csv"),
        ("random_workloads.csv", "sweep_random_workloads.csv"),
    ]
    for src_name, dst_name in targets:
        src = summary_dir / src_name
        if not src.exists():
            continue
        rows = [prefixed_row(page_bytes, suite_name, row) for row in read_csv_rows(src)]
        dst = sweep_summary / dst_name
        if not rows:
            continue
        if dst.exists():
            existing = read_csv_rows(dst)
            fieldnames = list(existing[0].keys()) if existing else list(rows[0].keys())
            with dst.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(rows)
        else:
            write_csv(dst, rows, list(rows[0].keys()))


def run_cmd(cmd: list[str], env: dict[str, str], dry_run: bool) -> int:
    print(f"==> {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sweep-name", default="")
    parser.add_argument("--page-sizes", default=DEFAULT_PAGE_SIZES)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-usage-limit", default="12GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument("--pin-rows", type=int, default=None)
    parser.add_argument("--pair-timeout", type=int, default=300)
    parser.add_argument("--workload-timeout", type=int, default=900)
    parser.add_argument("--random-count", type=int, default=6)
    parser.add_argument("--random-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--no-replacement", action="store_true")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--workloads-file", default="")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-pairs", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"input directory does not exist: {args.input}")

    page_sizes = [parse_page_size(item) for item in parse_csv_list(args.page_sizes)]
    sweep_name = args.sweep_name or f"page_size_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_root = (args.output_root / sweep_name).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(args.input.resolve()),
        "sweep_root": str(sweep_root),
        "page_sizes": page_sizes,
        "queries": parse_csv_list(args.queries),
        "conditions": parse_csv_list(args.conditions),
        "repeats": args.repeats,
        "devices": args.devices,
        "num_gpus": args.num_gpus,
        "random_count": args.random_count,
        "random_length": args.random_length,
        "seed": args.seed,
    }
    (sweep_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (sweep_root / "README.md").write_text(
        f"""# Fixed-page page-size sweep

Created: `{manifest['created_at']}`
Input: `{manifest['input']}`
Page sizes: `{','.join(page_size_label(size) for size in page_sizes)}`
Queries: `{args.queries}`
Conditions: `{args.conditions}`

Each page size is a nested `run_fixed_page_suite.py` output.
Sweep-level summary CSVs live under `summary/`.
"""
    )

    if not args.skip_build:
        code = run_cmd(["pixi", "run", "make", "-j4"], os.environ.copy(), args.dry_run)
        if code != 0:
            return code

    for page_bytes in page_sizes:
        label = page_size_label(page_bytes)
        suite_name = f"page_{label}"
        env = os.environ.copy()
        env["SIRIUS_FIXED_WIDTH_PAGE_BYTES"] = str(page_bytes)
        cmd = [
            "pixi",
            "run",
            "python",
            str(REPO_ROOT / "scripts" / "run_fixed_page_suite.py"),
            "--input",
            str(args.input.resolve()),
            "--output-root",
            str(sweep_root),
            "--suite-name",
            suite_name,
            "--queries",
            args.queries,
            "--conditions",
            args.conditions,
            "--repeats",
            str(args.repeats),
            "--devices",
            args.devices,
            "--num-gpus",
            str(args.num_gpus),
            "--gpu-usage-limit",
            args.gpu_usage_limit,
            "--host-capacity",
            args.host_capacity,
            "--reservation-limit-fraction",
            args.reservation_limit_fraction,
            "--pair-timeout",
            str(args.pair_timeout),
            "--workload-timeout",
            str(args.workload_timeout),
            "--random-count",
            str(args.random_count),
            "--random-length",
            str(args.random_length),
            "--seed",
            str(args.seed),
            "--skip-build",
            "--log-level",
            args.log_level,
        ]
        if args.pin_rows is not None:
            cmd.extend(["--pin-rows", str(args.pin_rows)])
        if args.no_replacement:
            cmd.append("--no-replacement")
        if args.workloads:
            cmd.extend(["--workloads", args.workloads])
        if args.workloads_file:
            cmd.extend(["--workloads-file", args.workloads_file])
        if args.no_pairs:
            cmd.append("--no-pairs")
        if args.no_random:
            cmd.append("--no-random")
        if args.no_plots:
            cmd.append("--no-plots")
        if args.dry_run:
            cmd.append("--dry-run")

        code = run_cmd(cmd, env, args.dry_run)
        collect_summary(sweep_root, suite_name, page_bytes)
        if code != 0 and not args.continue_on_error:
            return code

    print(f"==> sweep done: {sweep_root}", flush=True)
    print(f"==> summary: {sweep_root / 'summary'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
