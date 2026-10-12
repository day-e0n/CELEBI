#!/usr/bin/env python3
"""Isolate q28/q29 with unchanged query SQL, comparing GPU pipeline worker counts.

Diagnostic only: the [28,29] workload differs from the full 43-query cache history.
Reuses the full comparison's environment/config generators and existing extension.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "test/tpch_performance"))
from run_full_cache_pin_comparison import case_environment, config_text


def worker(
    case: Path,
    threads: int,
    executions: int = 3,
    queries=(28, 29),
    extension_override=None,
    condition="baseline",
) -> None:
    import duckdb
    from clickbench_queries import QUERIES

    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    manifest = json.loads(
        (ROOT / "experiment/FULL_CACHE_PIN_20261007/runs/manifest.json").read_text()
    )
    source = manifest["workloads"]["cb"]["input"]
    con.execute(
        "CREATE VIEW hits AS SELECT * FROM read_parquet('"
        + source.replace("'", "''")
        + "')"
    )
    extension = str(extension_override) if extension_override else manifest["extension"]
    if not extension_override:
        assert (
            hashlib.sha256(Path(extension).read_bytes()).hexdigest()
            == manifest["extension_sha256"]
        )
    con.execute("LOAD '" + extension.replace("'", "''") + "'")
    con.execute("SET gpu_execution=true")
    con.execute("SET enable_duckdb_fallback=false")
    con.execute("SET enable_regex_jit_impl=false")
    if condition == "reorder_future":
        import cache_aware_query_reorder as reorder
        from future_cache_hints import hint_spec

        reorder.load_column_set("clickbench_pin_columns")
    for q in queries:
        (case / f"q{q}.sql").write_text(QUERIES[f"q{q}"] + "\n")
    all_results = []
    with (case / "queries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["threads", "query", "execution", "wall_s", "rows"]
        )
        writer.writeheader()
        for execution in range(1, executions + 1):
            for position, q in enumerate(queries, start=1):
                if condition == "reorder_future":
                    spec = hint_spec(
                        "evict",
                        list(queries),
                        executions,
                        execution,
                        position,
                        "execution",
                    )
                    con.execute(f"CALL sirius_set_cache_hints('{spec}')")
                con.execute(
                    f"CALL sirius_set_query_label('{condition}_t{threads}_q{q}_exec{execution}')"
                )
                start = time.perf_counter()
                sql = QUERIES[f"q{q}"]
                rows = con.execute(sql).fetchall()
                wall = time.perf_counter() - start
                writer.writerow(
                    dict(
                        threads=threads,
                        query=q,
                        execution=execution,
                        wall_s=wall,
                        rows=len(rows),
                    )
                )
                handle.flush()
                all_results.append(dict(query=q, execution=execution, results=rows))
                print(
                    f"threads={threads} q{q} execution={execution} wall={wall:.6f}s rows={len(rows)}",
                    flush=True,
                )
    (case / "results.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2) + "\n"
    )
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-threads", type=int)
    parser.add_argument(
        "--executions", type=int, default=3, help="Worker-only override for profiling"
    )
    parser.add_argument(
        "--queries", default="28,29", help="Worker-only override for profiling"
    )
    parser.add_argument(
        "--extension", type=Path, help="Worker-only explicit diagnostic build"
    )
    parser.add_argument(
        "--condition", choices=("baseline", "reorder_future"), default="baseline"
    )
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker_threads:
        worker(
            out,
            args.worker_threads,
            args.executions,
            tuple(map(int, args.queries.split(","))),
            args.extension.resolve() if args.extension else None,
            args.condition,
        )
        return
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    for threads in (4, 1):
        case = out / f"baseline_threads_{threads}"
        case.mkdir()
        config = config_text(case).replace(
            "  executor:\n",
            f"  executor:\n    pipeline:\n      num_threads: {threads}\n",
        )
        (case / "sirius.yaml").write_text(config)
        env = case_environment(case, "baseline", 6 * 2**30, "latency")
        (case / "metadata.json").write_text(
            json.dumps(
                dict(
                    diagnostic=True,
                    workload=[28, 29],
                    executions=3,
                    warmup=1,
                    condition="baseline",
                    gpu_pipeline_threads=threads,
                    environment={
                        k: v
                        for k, v in env.items()
                        if k.startswith(("SIRIUS_", "BENCH_", "CUDA_VISIBLE"))
                    },
                    note="Two-query diagnostic, not the full-workload comparison. Cache disabled; fallback disabled; regex JIT disabled.",
                ),
                indent=2,
            )
            + "\n"
        )
        with (case / "runner.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(case),
                    "--worker-threads",
                    str(threads),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        assert result.returncode == 0, (case, result.returncode)
        with (case / "queries.csv").open() as handle:
            samples = list(csv.DictReader(handle))
        assert len(samples) == 6
        for q in (28, 29):
            times = [
                float(r["wall_s"])
                for r in samples
                if int(r["query"]) == q and int(r["execution"]) > 1
            ]
            row = dict(
                threads=threads,
                query=q,
                mean_wall_s=statistics.mean(times),
                exec2_s=times[0],
                exec3_s=times[1],
            )
            summary.append(row)
            print(json.dumps(row), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
