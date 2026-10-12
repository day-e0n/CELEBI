#!/usr/bin/env python3
"""Frozen 43-query ClickBench comparison with future hints confined to each batch.

Uses the original whole-query worker and original batch partition/reorder rules.
Only partition/order and the horizon passed to future_cache_hints differ. Cache
contents persist between batches and executions. A validated, synchronized exit
after all measurements avoids the known frozen engine's final destructor fault.
"""

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import random
import sys

import cache_aware_query_reorder as reorder
import future_cache_hints
import run_full_cache_pin_comparison as runner


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


def prepare_sweep(root, extension, runs):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for count in (2, 4):
        for trace in ("latency", "hit"):
            args = argparse.Namespace(
                output=root / f"b{count}_{trace}",
                extension=extension,
                workloads="cb",
                conditions=(
                    "baseline,paging_lru,reorder_future"
                    if trace == "latency"
                    else "paging_lru,reorder_future"
                ),
                runs=runs,
                trace_mode=trace,
                th_input=Path("/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized"),
                cb_input=Path("/mnt/nvme/clickbench/hits_v2.parquet"),
                condition_timeout=0,
            )
            path = runner.prepare(args)
            manifest = json.loads(path.read_text())
            plan = manifest["workloads"]["cb"]
            manifest.update(
                batch_count=count,
                batch_sizes=[
                    len(b) for b in partition(plan["arrival_order"], count, 1)
                ],
                batch_split="Random(seed=run).shuffle, strided groups, arrival order within each group",
                cache_reset_between_batches=False,
                future_horizon="remaining queries of the current batch only",
                post_workload_exit_policy="synchronize_flush_then_process_exit_after_all_queries",
            )
            wrapper = Path(__file__).resolve()
            manifest["source_sha256"][str(wrapper)] = runner.digest(wrapper)
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
                    [
                        list(
                            reorder.reorder_query_sequence(b, config).reordered_queries
                        )
                        for b in arrivals
                    ]
                    if meta["condition"] == "reorder_future"
                    else arrivals
                )
                flat = [q for b in orders for q in b]
                runner.validate_order(flat, 43)
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


def execute():
    def argument(name):
        return Path(sys.argv[sys.argv.index(name) + 1]).resolve()

    manifest = json.loads(argument("--execute-plan").read_text())
    wrapper = Path(__file__).resolve()
    assert runner.digest(wrapper) == manifest["source_sha256"][str(wrapper)]
    assert (
        manifest["post_workload_exit_policy"]
        == "synchronize_flush_then_process_exit_after_all_queries"
    )
    if "--worker" in sys.argv:
        import duckdb

        meta = json.loads(argument("--metadata").read_text())
        assert meta["order"] == [q for b in meta["batch_orders"] for q in b]
        runner.validate_order(meta["order"], 43)
        original_hint_spec = future_cache_hints.hint_spec

        def batch_hint_spec(mode, order, executions, execution, position, horizon):
            assert order == meta["order"]
            number, batch, local = local_position(meta["batch_orders"], position)
            spec = original_hint_spec(
                mode, batch, executions, execution, local, "execution"
            )
            runner.event(
                Path(meta["journal"]),
                stage="batch_hint",
                execution=execution,
                query=order[position - 1],
                batch=number,
                local_position=local,
                hint=spec,
            )
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
                events = [
                    json.loads(line)
                    for line in Path(meta["journal"]).read_text().splitlines()
                ]
                complete = (
                    len(rows) == len(expected)
                    and {(int(r["execution"]), int(r["query"])) for r in rows}
                    == expected
                    and all(r["status"] == "ok" for r in rows)
                    and events[-1]["stage"] == "worker_finished"
                )
                if not complete:
                    self.connection.close()
                    return
                runtime = ctypes.CDLL("libcudart.so.13")
                assert runtime.cudaDeviceSynchronize() == 0
                runner.event(
                    Path(meta["journal"]),
                    stage="post_workload_process_exit",
                    all_queries_completed=True,
                    cuda_synchronized=True,
                    reason="known frozen-engine destructor fault, after all measurements",
                )
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
    if "--prepare-sweep" in sys.argv:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--prepare-sweep", type=Path, required=True)
        parser.add_argument("--extension", type=Path, required=True)
        parser.add_argument("--runs", type=int, default=1)
        args = parser.parse_args()
        prepare_sweep(args.prepare_sweep.resolve(), args.extension.resolve(), args.runs)
    else:
        raise SystemExit(execute())
