#!/usr/bin/env python3
"""Repeated batch-count sweep for ClickBench or TPC-H, batch-confined reorder and hints.

Generalises scripts/run_clickbench43_batch_comparison.py to either workload and any
batch counts, and runs the cases interleaved (run, then batch count, then condition) so
that drift over the hours a sweep takes does not land on one batch count.

The queries are split by Random(seed=run).shuffle into strided groups, each keeping the
arrival order. Baseline and Page run that order; CELEBI reorders each batch on its own
(small-first-big-last) and its future hints see only the rest of the current batch.
Batches run back to back on one connection, cache contents persist across batches and
executions. Everything else (cache budget, engine, memory limits, O_DIRECT, executions)
is the canonical scripts/run_full_cache_pin_comparison.py setup.

  # prepare (CPU only), one root per workload:
  pixi run -e duckdb-python python scripts/run_batch_sweep.py --prepare ROOT/cb \
      --workload cb --batches 1,2,4 --runs 10 --extension EXT
  # execute, roots in order:
  pixi run -e duckdb-python python scripts/run_batch_sweep.py --execute-sweep ROOT/cb ROOT/th
"""

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import cache_aware_query_reorder as reorder
import future_cache_hints
import run_full_cache_pin_comparison as runner

CONDITIONS = "baseline,paging_lru,reorder_future"
GPU = "1"


def partition(queries, count, seed):
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    position = {q: i for i, q in enumerate(queries)}
    return [sorted(shuffled[i::count], key=position.__getitem__) for i in range(count)]


def local_position(batches, position):
    for batch_number, batch in enumerate(batches, 1):
        if 1 <= position <= len(batch):
            return batch_number, batch, position
        position -= len(batch)
    raise ValueError("Query position is outside the batch sequence")


def prepare_sweep(root, workload, counts, extension, runs):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    wrapper = Path(__file__).resolve()
    for count in counts:
        args = argparse.Namespace(
            output=root / f"b{count}",
            extension=extension,
            workloads=workload,
            conditions=CONDITIONS,
            runs=runs,
            trace_mode="latency",
            th_input=Path("/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized"),
            cb_input=Path("/mnt/nvme/clickbench/hits_v2.parquet"),
            condition_timeout=1800,
        )
        path = runner.prepare(args)
        manifest = json.loads(path.read_text())
        plan = manifest["workloads"][workload]
        manifest.update(
            batch_count=count,
            batch_split="Random(seed=run).shuffle, strided groups, arrival order within each group",
            cache_reset_between_batches=False,
            future_horizon="remaining queries of the current batch only",
            post_workload_exit_policy="synchronize_flush_then_process_exit_after_all_queries",
        )
        manifest["source_sha256"][str(wrapper)] = runner.digest(wrapper)
        if workload == "cb":
            reorder.load_column_set("clickbench_pin_columns")
        reorder.load_column_bytes(plan["reorder_column_bytes_file"])
        config = reorder.ReorderConfig(
            policy=plan["reorder_policy"],
            scope="fixed_width",
            cache_budget_bytes=plan["cache_budget_bytes"],
        )
        for name in manifest["cases"]:
            metadata = Path(name)
            meta = json.loads(metadata.read_text())
            arrivals = partition(plan["arrival_order"], count, meta["run"])
            orders = (
                [list(reorder.reorder_query_sequence(b, config).reordered_queries) for b in arrivals]
                if meta["condition"] == "reorder_future"
                else arrivals
            )
            flat = [q for b in orders for q in b]
            runner.validate_order(flat, plan["query_count"])
            assert all(sorted(a) == sorted(b) for a, b in zip(arrivals, orders))
            meta.update(
                batch_count=count,
                batch_seed=meta["run"],
                arrival_batches=arrivals,
                batch_orders=orders,
                order=flat,
                future_horizon="batch",
            )
            runner.save_json(metadata, meta)
        runner.save_json(path, manifest)
        sizes = sorted({tuple(len(b) for b in json.loads(Path(c).read_text())["batch_orders"])
                        for c in manifest["cases"]})
        print(f"b{count}: batch sizes {sizes}", flush=True)


def check_manifest(path):
    manifest = json.loads(path.read_text())
    if runner.digest(Path(manifest["extension"])) != manifest["extension_sha256"]:
        raise ValueError("Extension changed after preparation")
    for source, expected in manifest["source_sha256"].items():
        if runner.digest(Path(source)) != expected:
            raise ValueError(f"Source changed after preparation: {source}")
    for plan in manifest["workloads"].values():
        for source, expected in ((plan["column_bytes_file"], plan["column_bytes_sha256"]),
                                 (plan["reorder_column_bytes_file"], plan["reorder_column_bytes_sha256"])):
            if runner.digest(Path(source)) != expected:
                raise ValueError(f"Column-size manifest changed after preparation: {source}")
        for record in plan["input_file_metadata"]:
            stat = Path(record["path"]).stat()
            if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise ValueError(f"Input file changed after preparation: {record['path']}")
    return manifest


def gpu_apps():
    """(physical gpu index, pid) of every compute process."""
    uuids = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    index = {u.strip(): i.strip() for i, u in (l.split(",") for l in uuids.splitlines() if l)}
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    return {(index.get(u.strip(), "?"), int(p)) for u, p in
            (l.split(",") for l in apps.splitlines() if l.strip())}


def descendant(pid, ancestor):
    while pid > 1:
        if pid == ancestor:
            return True
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except FileNotFoundError:
            return True  # ended during the snapshot
    return False


def foreign(apps):
    return sorted((g, p) for g, p in apps if not descendant(p, os.getpid()))


def run_case(manifest_path, manifest, metadata_path, log):
    meta = json.loads(metadata_path.read_text())
    if meta["status"] != "prepared":
        return None  # already ran (resumed sweep)
    if runner.digest(Path(manifest["extension"])) != manifest["extension_sha256"]:
        raise ValueError("Extension changed during the sweep")
    waited = 0
    while any(g == GPU for g, _ in foreign(gpu_apps())):
        if waited % 300 == 0:
            log(f"WAIT gpu {GPU} busy: {foreign(gpu_apps())}")
        time.sleep(30)
        waited += 30
    meta.update(status="running", started_utc=runner.utc_now(), waited_for_gpu_s=waited)
    runner.save_json(metadata_path, meta)
    case = Path(meta["case_directory"])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SIRIUS_", "BENCH_"))}
    env.update(meta["environment"])
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--execute-plan", str(manifest_path), "--metadata", str(metadata_path)]
    seen, stop = set(), threading.Event()

    def sample():
        while not stop.wait(5):
            try:
                seen.update(foreign(gpu_apps()))
            except subprocess.CalledProcessError:
                pass

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    timeout = False
    with (case / "runner.log").open("w") as out:
        try:
            proc = subprocess.run(command, env=env, cwd=runner.ROOT, stdout=out,
                                  stderr=subprocess.STDOUT,
                                  timeout=manifest["condition_timeout_s"] or None, check=False)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timeout, returncode = True, -1
    stop.set()
    sampler.join()
    record = runner.finalize_case(metadata_path, returncode, timeout)
    meta = json.loads(metadata_path.read_text())
    meta["foreign_gpu_processes_during_case"] = [list(x) for x in sorted(seen)]
    runner.save_json(metadata_path, meta)
    record["foreign_gpu_processes"] = meta["foreign_gpu_processes_during_case"]
    return record


def execute_sweep(roots):
    status_path = roots[0].parent / "sweep_status.jsonl"

    def log(message):
        line = f"{runner.utc_now()} {message}"
        print(line, flush=True)
        with status_path.open("a") as f:
            f.write(line + "\n")

    for root in roots:
        manifests = sorted(root.glob("b*/manifest.json"),
                           key=lambda p: int(p.parent.name[1:]))
        loaded = {p: check_manifest(p) for p in manifests}
        cases = []
        for p, m in loaded.items():
            for c in m["cases"]:
                meta = json.loads(Path(c).read_text())
                cases.append((meta["run"], m["batch_count"],
                              m["conditions"].index(meta["condition"]), p, Path(c)))
        cases.sort(key=lambda x: x[:3])
        log(f"START {root} ({len(cases)} cases)")
        for run, count, _, path, metadata in cases:
            meta = json.loads(metadata.read_text())
            tag = f"{meta['workload']} b{count} r{run} {meta['condition']}"
            record = run_case(path.resolve(), loaded[path], metadata.resolve(), log)
            if record is None:
                continue
            summary_path = path.parent / "summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
            summary.append(record)
            runner.save_json(summary_path, summary)
            log(f"DONE {tag} {record['status']} wall={record['steady_state_workload_wall_s']} "
                f"foreign={record['foreign_gpu_processes']}")
        log(f"END {root}")
    return 0


def worker():
    def argument(name):
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve()

    manifest = json.loads(argument("--execute-plan").read_text())
    wrapper = Path(__file__).resolve()
    assert runner.digest(wrapper) == manifest["source_sha256"][str(wrapper)]
    import duckdb

    meta = json.loads(argument("--metadata").read_text())
    count = manifest["workloads"][meta["workload"]]["query_count"]
    assert meta["order"] == [q for b in meta["batch_orders"] for q in b]
    runner.validate_order(meta["order"], count)
    original_hint_spec = future_cache_hints.hint_spec

    def batch_hint_spec(mode, order, executions, execution, position, horizon):
        assert order == meta["order"]
        number, batch, local = local_position(meta["batch_orders"], position)
        spec = original_hint_spec(mode, batch, executions, execution, local, "execution")
        runner.event(Path(meta["journal"]), stage="batch_hint", execution=execution,
                     query=order[position - 1], batch=number, local_position=local, hint=spec)
        return spec

    future_cache_hints.hint_spec = batch_hint_spec
    original_connect = duckdb.connect

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def close(self):
            rows = list(csv.DictReader(Path(meta["query_csv"]).open()))
            expected = {(ex, q) for ex in (1, 2, 3) for q in meta["order"]}
            events = [json.loads(line) for line in Path(meta["journal"]).read_text().splitlines()]
            complete = (len(rows) == len(expected)
                        and {(int(r["execution"]), int(r["query"])) for r in rows} == expected
                        and all(r["status"] == "ok" for r in rows)
                        and events[-1]["stage"] == "worker_finished")
            if not complete:
                self.connection.close()
                return
            # The frozen engine faults in its final scan-manager destructor; exit only
            # after every query and journal record has succeeded.
            assert ctypes.CDLL("libcudart.so.13").cudaDeviceSynchronize() == 0
            runner.event(Path(meta["journal"]), stage="post_workload_process_exit",
                         all_queries_completed=True, cuda_synchronized=True,
                         reason="known frozen-engine destructor fault, after all measurements")
            sys.stdout.flush()
            sys.stderr.flush()
            libc = ctypes.CDLL(None)
            libc.fflush.argtypes = [ctypes.c_void_p]
            assert libc.fflush(None) == 0
            os._exit(0)

    duckdb.connect = lambda *a, **kw: Connection(original_connect(*a, **kw))
    runner.__file__ = __file__
    return runner.main()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(worker())
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--workload", choices=("cb", "th"))
    parser.add_argument("--batches", default="1")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--execute-sweep", type=Path, nargs="+")
    args = parser.parse_args()
    if args.prepare:
        prepare_sweep(args.prepare.resolve(), args.workload,
                      [int(b) for b in args.batches.split(",")], args.extension.resolve(), args.runs)
    elif args.execute_sweep:
        raise SystemExit(execute_sweep([r.resolve() for r in args.execute_sweep]))
    else:
        parser.error("--prepare or --execute-sweep")
