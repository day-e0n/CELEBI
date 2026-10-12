#!/usr/bin/env python3
"""Frozen, isolated q28/q29 investigation; never edits original experiment cases."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_full_cache_pin_comparison import case_environment, config_text

CASES = {
    "phase_auto": (
        1,
        1,
        {"SIRIUS_GROUPBY_DIAGNOSTIC": "1", "SIRIUS_GROUPBY_SAFE_AVG_INT64": "0"},
    ),
    "phase_native": (
        1,
        1,
        {"SIRIUS_GROUPBY_DIAGNOSTIC": "1", "SIRIUS_GROUPBY_SAFE_AVG_INT64": "1"},
    ),
    "latency_auto": (4, 3, {"SIRIUS_GROUPBY_SAFE_AVG_INT64": "0"}),
    "latency_native": (4, 3, {"SIRIUS_GROUPBY_SAFE_AVG_INT64": "1"}),
    "latency_default": (4, 3, {}),
    "latency_future": (4, 3, {}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", default="phase_auto,phase_native,latency_auto,latency_native"
    )
    parser.add_argument("--wait-pid", type=int)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        while Path(f"/proc/{args.wait_pid}").exists():
            print(f"Waiting for existing experiment PID {args.wait_pid}", flush=True)
            time.sleep(20)
    # No GPU workload may overlap this comparison; also protect CPU/NVMe work.
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert not apps, f"GPU processes still active: {apps}"
    ext = args.extension.resolve()
    sha = hashlib.sha256(ext.read_bytes()).hexdigest()
    for name in args.cases.split(","):
        threads, executions, overrides = CASES[name]
        condition = "reorder_future" if name == "latency_future" else "baseline"
        case = out / name
        case.mkdir(exist_ok=False)
        (case / "sirius.yaml").write_text(
            config_text(case).replace(
                "  executor:\n",
                f"  executor:\n    pipeline:\n      num_threads: {threads}\n",
            )
        )
        env = case_environment(case, condition, 6 * 2**30, "latency")
        env.update(overrides)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/diagnose_clickbench_heavy_concurrency.py"),
            "--output",
            str(case),
            "--worker-threads",
            str(threads),
            "--executions",
            str(executions),
            "--extension",
            str(ext),
            "--condition",
            condition,
        ]
        meta = dict(
            diagnostic=True,
            name=name,
            condition=condition,
            query_numbers=[28, 29],
            query_sql_unchanged=True,
            extension=str(ext),
            extension_sha256=sha,
            threads=threads,
            executions=executions,
            warmup_excluded=1 if executions > 1 else None,
            environment={
                k: v
                for k, v in env.items()
                if k.startswith(("SIRIUS_", "BENCH_", "CUDA_VISIBLE"))
            },
            command=cmd,
            status="running",
            started=time.time(),
        )
        (case / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"START {name}", flush=True)
        with (case / "runner.log").open("w") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        meta.update(
            status="complete" if proc.returncode == 0 else "failed",
            returncode=proc.returncode,
            finished=time.time(),
        )
        (case / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"DONE {name} exit={proc.returncode}", flush=True)
        assert proc.returncode == 0, case
        assert (
            hashlib.sha256(ext.read_bytes()).hexdigest() == sha
        ), "Build changed during comparison"


if __name__ == "__main__":
    main()
