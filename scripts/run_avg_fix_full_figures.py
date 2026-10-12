#!/usr/bin/env python3
"""Wait for existing measurements, run frozen plans sequentially, then render."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def running(pid):
    path = Path(f"/proc/{pid}/stat")
    try:
        return path.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def gpu_pids():
    data = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    return {int(line.strip()) for line in data.splitlines() if line.strip().isdigit()}


def descendant(pid, ancestor):
    while pid > 1:
        if pid == ancestor:
            return True
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except FileNotFoundError:
            return True  # Process ended during the snapshot.
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--plans", default="latency_runs,hit_runs")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--runner-script", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    state = dict(
        status="waiting",
        started_utc=now(),
        wait_pid=args.wait_pid,
        plans=[],
        overlaps=[],
    )

    def save():
        temp = root / "job_status.json.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(root / "job_status.json")

    save()
    while args.wait_pid and running(args.wait_pid):
        print(f"Waiting for existing measurement group PID {args.wait_pid}", flush=True)
        time.sleep(15)
    for folder in args.plans.split(","):
        while gpu_pids():
            print(
                "Waiting for GPU workloads to finish before shared-NVMe measurement",
                flush=True,
            )
            time.sleep(15)
        state["status"] = "running"
        record = dict(plan=folder, status="running", started_utc=now())
        state["plans"].append(record)
        save()
        print(f"START {folder}", flush=True)
        cmd = [
            sys.executable,
            str(
                args.runner_script.resolve()
                if args.runner_script
                else repo / "scripts/run_full_cache_pin_comparison.py"
            ),
            "--execute-plan",
            str(root / folder / "manifest.json"),
        ]
        with (root / f"{folder}_runner.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                external = sorted(
                    pid for pid in gpu_pids() if not descendant(pid, proc.pid)
                )
                if external:
                    state["overlaps"].append(
                        dict(plan=folder, time_utc=now(), external_pids=external)
                    )
                    save()
                time.sleep(3)
        record.update(
            returncode=proc.returncode,
            ended_utc=now(),
            status="complete" if proc.returncode == 0 else "failed",
        )
        save()
        if proc.returncode or state["overlaps"]:
            state["status"] = "failed_or_contaminated"
            save()
            raise RuntimeError(
                f"Invalid measurements; inspect {root / 'job_status.json'}"
            )
        print(f"DONE {folder}", flush=True)
    state["status"] = "rendering"
    save()
    if not args.no_render:
        subprocess.run(
            [
                sys.executable,
                str(repo / "result/plot_avg_fix_query_latency_and_hits.py"),
                "--input-root",
                str(root),
                "--output",
                str(root / "figures"),
            ],
            cwd=repo,
            check=True,
        )
    state.update(status="complete", ended_utc=now())
    save()


if __name__ == "__main__":
    main()
