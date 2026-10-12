#!/usr/bin/env python3
"""Prepare or run a whole-query cache/pin comparison in isolated subprocesses.

Preparation is CPU-only and is the default. Add --run to launch Sirius. Every
condition runs CB q1..q43 or all 22 TPC-H queries, in one connection for three
executions. Recoverable query exceptions are recorded and the next query is
attempted. A process crash/timeout leaves explicit fatal/unattempted rows;
such a condition is never reported as a complete workload measurement.

Examples (from the repository root, through pixi):
  pixi run -e duckdb-python python scripts/run_full_cache_pin_comparison.py \
      --output experiment/full_cache_pin_20261007 --trace-mode both
  # Review manifest.json and each metadata.json, then launch the frozen plan:
  pixi run -e duckdb-python python scripts/run_full_cache_pin_comparison.py \
      --execute-plan experiment/full_cache_pin_20261007/manifest.json

Static GPU pinning uses an order-independent column subset selected before any
query runs, within 90% of the corresponding 4/6 GiB cache budget. Candidates are
ranked by number of queries referencing a column / measured decoded bytes,
then query count, bytes and name. This is a reproducible heuristic, not an
optimal or full-dataset pin. The byte manifest excludes allocation overhead;
the 10% reserve does not guarantee admission. A pin setup failure is recorded
and invalidates the comparison, without changing the selected columns/budget.
Pin-preparation time is recorded separately from query latency. Provider logs,
rather than page-cache [cache-hit] lines alone, establish actual pin reuse.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(TPCH_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

CONDITIONS = ("baseline", "paging_lru", "reorder_lru", "reorder_future", "pin_gpu")
TH_ARRIVAL = [8, 1, 4, 7, 17, 5, 6, 2, 10, 16, 12, 19, 21, 14, 18, 15, 22,
              9, 13, 20, 3, 11]
FIELDS = ("workload", "condition", "trace_mode", "run", "execution", "position",
          "query", "label", "warmup", "status", "wall_s", "rows", "error",
          "started_utc", "ended_utc")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sql_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate_order(order: list[int], count: int) -> None:
    if len(order) != count or sorted(order) != list(range(1, count + 1)):
        raise ValueError(f"Order must contain each of q1..q{count} exactly once: {order}")


def select_static_pin(columns, sizes, budget: int) -> dict:
    uses = Counter((table, col) for tables in columns.values()
                   for table, cols in tables.items() for col in set(cols))
    ranked = sorted(uses, key=lambda k: (-uses[k] / max(sizes[k[0]][k[1]], 1),
                                         -uses[k], sizes[k[0]][k[1]], k))
    limit = budget * 9 // 10
    selected, used, candidates = {}, 0, []
    for table, col in ranked:
        size = int(sizes[table][col])
        chosen = used + size <= limit
        if chosen:
            selected.setdefault(table, []).append(col)
            used += size
        candidates.append({"table": table, "column": col, "estimated_decoded_bytes": size,
                           "query_use_count": uses[(table, col)], "selected": chosen})
    for cols in selected.values():
        cols.sort()
    coverage = {}
    for q, tables in columns.items():
        coverage[str(q)] = {
            table: {"required_columns": sorted(set(cols)),
                    "selected_columns": sorted(set(cols) & set(selected.get(table, []))),
                    "static_all_columns_selected": bool(cols) and
                       set(cols).issubset(selected.get(table, []))}
            for table, cols in tables.items()
        }
    return {"strategy": "query_use_count_per_estimated_decoded_byte_greedy",
            "budget_bytes": budget, "selection_limit_bytes": limit,
            "estimated_decoded_bytes": used, "columns": selected,
            "candidates": candidates, "static_scan_coverage": coverage,
            "coverage_is_static_prediction_not_measured_reuse": True}


def workload_plan(name: str, source: Path) -> dict:
    import cache_aware_query_reorder as reorder

    # Snapshot before load_column_set mutates the imported TPC-H mapping in place.
    module_name = "clickbench_pin_columns" if name == "cb" else "tpch_pin_columns"
    columns = copy.deepcopy(importlib.import_module(module_name).QUERY_COLUMNS)
    query_module = "clickbench_queries" if name == "cb" else "queries"
    queries = dict(importlib.import_module(query_module).QUERIES)
    size_path = ROOT / "experiment" / "column_bytes" / (
        "clickbench.json" if name == "cb" else "tpch_sf50.json")
    reorder_size_path = ROOT / "experiment" / "frag_replay" / (
        "clickbench_column_bytes.json" if name == "cb" else "tpch_sf50_column_bytes.json")
    sizes = json.loads(size_path.read_text())
    count = 43 if name == "cb" else 22
    if sorted(queries) != sorted(f"q{q}" for q in range(1, count + 1)):
        raise ValueError(f"Unexpected query set in {query_module}")
    # q24 SELECT * fetches the whole schema. q1 COUNT(*) has no explicit AST
    # column and is left unknown, so it is never predicted as a full pin hit.
    if name == "cb":
        columns[24] = {"hits": list(sizes["hits"])}
        reorder.load_column_set("clickbench_pin_columns")
    reorder.load_column_bytes(reorder_size_path)
    arrival = list(range(1, 44)) if name == "cb" else list(TH_ARRIVAL)
    budget = (6 if name == "cb" else 4) * 2 ** 30
    cfg = reorder.ReorderConfig(policy="small-first-big-last", scope="fixed_width",
                                cache_budget_bytes=budget)
    # The existing reorder uses its existing map, including q24's scan-stage
    # columns. Only static pin coverage uses the SELECT * expansion above.
    reordered = list(reorder.reorder_query_sequence(arrival, cfg).reordered_queries)
    validate_order(arrival, count)
    validate_order(reordered, count)
    pin = select_static_pin(columns, sizes, budget)
    files, pin_paths = {}, {}
    if name == "cb":
        if not source.is_file():
            raise FileNotFoundError(source)
        files = {"hits": [str(source)]}
        pin_paths = {"hits": str(source)}
    else:
        from tpch_pin_columns import detect_pin_glob
        import glob
        for table in ("customer", "lineitem", "nation", "orders", "part", "partsupp",
                      "region", "supplier"):
            paths = sorted(str(p.resolve()) for pattern in
                           (f"{table}.parquet", f"{table}_*.parquet", f"{table}/*.parquet")
                           for p in source.glob(pattern))
            if not paths:
                raise FileNotFoundError(f"No parquet files for {table} in {source}")
            files[table] = paths
            if table in pin["columns"]:
                pattern = detect_pin_glob(str(source), table)
                if sorted(str(Path(p).resolve()) for p in glob.glob(pattern)) != paths:
                    raise ValueError(f"Pin and view file sets differ for {table}")
                pin_paths[table] = pattern
    return {"workload": name, "input": str(source), "query_count": count,
            "query_module": query_module, "queries": queries, "arrival_order": arrival,
            "reordered_order": reordered, "reorder_policy": "small-first-big-last",
            "cache_budget_bytes": budget, "column_bytes_file": str(size_path),
            "column_bytes_sha256": digest(size_path),
            "reorder_column_bytes_file": str(reorder_size_path),
            "reorder_column_bytes_sha256": digest(reorder_size_path),
            "pin": pin, "pin_paths": pin_paths,
            "view_files": files,
            "input_file_metadata": [{"path": f, "bytes": Path(f).stat().st_size,
                                      "mtime_ns": Path(f).stat().st_mtime_ns}
                                    for fs in files.values() for f in fs],
            "notes": ["All queries run, including CB q24, q28 and q29.",
                      "CB q24 pin coverage includes all SELECT * columns; COUNT(*) coverage unknown.",
                      "Future hints follow the existing scan-stage column map."]}


def case_environment(case: Path, condition: str, budget: int, trace: str) -> dict[str, str]:
    # Freeze relevant experiment variables instead of leaking previous exported
    # cache settings into a new condition. Preserve normal pixi/CUDA paths.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SIRIUS_", "BENCH_"))}
    env.update({"CUDA_VISIBLE_DEVICES": "1", "SIRIUS_CONFIG_FILE": str(case / "sirius.yaml"),
                "SIRIUS_LOG_DIR": str(case / "log_dir"), "SIRIUS_LOG_LEVEL": "info",
                "SIRIUS_ENABLE_LATE_MATERIALIZATION": "1",
                "SIRIUS_FIXED_PAGE_COLUMN_KEYED_CACHE": "1",
                "SIRIUS_FIXED_WIDTH_PAGE_BYTES": str(16 * 2 ** 20),
                "SIRIUS_VARIABLE_WIDTH_PAGE_BYTES": str(16 * 2 ** 20),
                "SIRIUS_FIXED_PAGE_CACHE_BYTES_PER_GPU": str(budget),
                "SIRIUS_VARIABLE_PAGE_CACHE_BYTES_PER_GPU": str(budget),
                "SIRIUS_FIXED_PAGE_CACHE_MIN_FREE_BYTES_PER_GPU": str(6 * 2 ** 30),
                "SIRIUS_PAGE_TRACE": "0",
                "SIRIUS_CACHE_HIT_TRACE": "1" if trace == "hit" else "0",
                "BENCH_USE_ODIRECT": "1", "BENCH_PREFETCH_CACHE": "0"})
    caching = condition in ("paging_lru", "reorder_lru", "reorder_future")
    env.update({"SIRIUS_ENABLE_FIXED_PAGE_REUSE": "1" if caching or condition == "pin_gpu" else "0",
                "SIRIUS_FIXED_PAGE_AUTO_CACHE": "1" if caching else "0",
                "SIRIUS_FIXED_PAGE_OWNED_PAGES": "1" if caching else "0",
                "SIRIUS_FIXED_PAGE_BACKED_PROVIDER": "1" if caching else "0",
                "SIRIUS_FIXED_PAGE_HYBRID_PROVIDER": "1" if caching else "0",
                "SIRIUS_FIXED_WIDTH_PAGE_CACHE_ENABLED": "1" if caching else "0",
                "SIRIUS_VARIABLE_PAGE_CACHE_ENABLED": "1" if caching else "0"})
    # A shared page-store budget constrains both widths in fixed+variable mode,
    # as in the existing latest-A runner; the variable legacy budget is frozen too.
    return env


def config_text(case: Path) -> str:
    return ("sirius:\n  topology:\n    num_gpus: 1\n  memory:\n    gpu:\n"
            "      usage_limit_bytes: 20GB\n      reservation_limit_fraction: 0.85\n"
            "      downgrade_trigger_fraction: 0.8\n      downgrade_stop_fraction: 0.7\n"
            "    host:\n      capacity_bytes: 32GB\n    disk:\n      disk_id: 0\n"
            "      capacity_bytes: 100GB\n"
            f"      downgrade_root_dirs: {json.dumps(str(case / 'disk_downgrade'))}\n"
            "  executor:\n    scan_manager:\n      enable_prefetch_cache: false\n"
            "      local:\n        use_odirect: true\n"
            "  telemetry:\n    enable_quent: true\n"
            f"    output_directory: {json.dumps(str(case / 'telemetry_data'))}\n"
            "    engine_name: siriusDB\n")


def prepare(args) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    extension = args.extension.resolve()
    if not extension.is_file():
        raise FileNotFoundError(extension)
    # TPC-H first: load_column_set(cb) mutates the helper's shared TPC-H map.
    names = [n for n in ("th", "cb") if n in args.workloads.split(",")]
    if len(names) != len(args.workloads.split(",")):
        raise ValueError("--workloads must contain th and/or cb once each")
    conditions = args.conditions.split(",")
    if len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError(f"--conditions must be unique members of {CONDITIONS}")
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    plans = {n: workload_plan(n, (args.th_input if n == "th" else args.cb_input).resolve())
             for n in names}
    traces = ("latency", "hit") if args.trace_mode == "both" else (args.trace_mode,)
    output.mkdir(parents=True)
    source_files = [Path(__file__), ROOT / "scripts/cache_aware_query_reorder.py",
                    ROOT / "scripts/future_cache_hints.py", TPCH_DIR / "queries.py",
                    TPCH_DIR / "clickbench_queries.py", TPCH_DIR / "tpch_pin_columns.py",
                    TPCH_DIR / "clickbench_pin_columns.py",
                    ROOT / "src/include/op/scan/cache_hit_metrics.hpp",
                    ROOT / "src/op/scan/gpu_ingestible.cpp",
                    ROOT / "src/op/scan/parquet_gpu_ingestible.cpp"]
    manifest = {"schema_version": 1, "prepared_utc": utc_now(), "root": str(ROOT),
                "output": str(output), "python_executable": sys.executable,
                "extension": str(extension), "extension_sha256": digest(extension),
                "source_sha256": {str(p): digest(p) for p in source_files},
                "executions": 3, "warmup_execution": 1, "report_executions": [2, 3],
                "runs": args.runs, "conditions": conditions, "trace_modes": list(traces),
                "hit_metric": "value_bytes_v2",
                "hit_metric_definition": "fixed-width value data + STRING character payload; "
                    "excludes offsets, null masks, allocation padding, slabs and virtual columns",
                "workloads": plans, "condition_timeout_s": args.condition_timeout,
                "prepared_only": True, "cases": []}
    for trace in traces:
        for run in range(1, args.runs + 1):
            for name in names:
                plan = plans[name]
                for condition in conditions:
                    case = output / trace / f"{name}_r{run}" / condition
                    for directory in ("log_dir", "telemetry_data", "disk_downgrade"):
                        (case / directory).mkdir(parents=True)
                    (case / "sirius.yaml").write_text(config_text(case))
                    order = plan["reordered_order"] if condition.startswith("reorder_") else plan["arrival_order"]
                    meta = {"workload": name, "condition": condition, "trace_mode": trace,
                            "run": run, "case_directory": str(case), "order": order,
                            "executions": 3, "gpu_device": "1", "gpu_usage_limit_bytes": 20 * 2 ** 30,
                            "host_capacity_bytes": 32 * 2 ** 30,
                            "cache_budget_bytes": plan["cache_budget_bytes"],
                            "min_free_bytes_per_gpu": 6 * 2 ** 30,
                            "future_mode": "evict" if condition == "reorder_future" else "off",
                            "future_horizon": "execution", "late_materialization": True,
                            "prefetch": False, "odirect": True, "environment":
                            case_environment(case, condition, plan["cache_budget_bytes"], trace),
                            "query_csv": str(case / "queries.csv"), "journal": str(case / "events.jsonl"),
                            "status": "prepared", "pin": plan["pin"] if condition == "pin_gpu" else None}
                    # Export only experiment-relevant env in readable metadata;
                    # the full inherited process env is never serialized.
                    meta["environment"] = {k: v for k, v in meta["environment"].items()
                                           if k.startswith(("SIRIUS_", "BENCH_")) or k == "CUDA_VISIBLE_DEVICES"}
                    save_json(case / "metadata.json", meta)
                    manifest["cases"].append(str(case / "metadata.json"))
    path = output / "manifest.json"
    save_json(path, manifest)
    print(f"Prepared {len(manifest['cases'])} isolated cases: {path}", flush=True)
    print("No engine loaded or GPU queries launched during preparation.", flush=True)
    return path


def event(path: Path, **record) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"utc": utc_now(), **record}) + "\n")
        f.flush()


def append_row(path: Path, row: dict) -> None:
    first = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if first:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def query_row(meta: dict, execution: int, position: int, q: int) -> dict:
    return {"workload": meta["workload"], "condition": meta["condition"],
            "trace_mode": meta["trace_mode"], "run": meta["run"], "execution": execution,
            "position": position, "query": q,
            "label": f"{meta['condition']}_q{q}_exec{execution}",
            "warmup": int(execution == 1), "status": "", "wall_s": "", "rows": "",
            "error": "", "started_utc": "", "ended_utc": ""}


def worker(manifest_path: Path, metadata_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    meta = json.loads(metadata_path.read_text())
    plan = manifest["workloads"][meta["workload"]]
    case = Path(meta["case_directory"])
    os.environ.update(meta["environment"])
    # Imports which load DuckDB happen only inside the launched worker.
    import duckdb
    from future_cache_hints import hint_spec
    if meta["workload"] == "cb":
        import cache_aware_query_reorder as reorder
        reorder.load_column_set("clickbench_pin_columns")
    con = None
    pin_complete = True
    pinned = []
    try:
        con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
        for table, files in plan["view_files"].items():
            file_list = ",".join(sql_string(f) for f in files)
            con.execute(f'CREATE VIEW "{table}" AS SELECT * FROM read_parquet([{file_list}])')
        con.execute(f"LOAD {sql_string(manifest['extension'])}")
        con.execute("SET gpu_execution = true")
        con.execute("CALL sirius_set_query_label('preparation')")
        event(case / "events.jsonl", stage="connection_ready")
        pin_start = time.perf_counter()
        if meta["condition"] == "pin_gpu":
            for table, cols in plan["pin"]["columns"].items():
                statement = (f"CALL pin_table({sql_string(plan['pin_paths'][table])}, "
                             f"tier='gpu', name={sql_string(table)}, "
                             f"cols=[{','.join(sql_string(c) for c in cols)}])")
                started = time.perf_counter()
                event(case / "events.jsonl", stage="pin_start", table=table, sql=statement)
                try:
                    con.execute(statement).fetchall()
                    pinned.append(table)
                    event(case / "events.jsonl", stage="pin_end", table=table, status="ok",
                          wall_s=time.perf_counter() - started)
                except Exception as exc:
                    pin_complete = False
                    event(case / "events.jsonl", stage="pin_end", table=table, status="error",
                          error=f"{type(exc).__name__}: {exc}", wall_s=time.perf_counter() - started)
            meta["pin_preparation_wall_s"] = time.perf_counter() - pin_start
            meta["pin_setup_complete"] = pin_complete
            meta["successfully_pinned_tables"] = pinned
            save_json(metadata_path, meta)
        for execution in range(1, 4):
            for position, q in enumerate(meta["order"], 1):
                row = query_row(meta, execution, position, q)
                row["started_utc"] = utc_now()
                event(case / "events.jsonl", stage="query_start", execution=execution,
                      position=position, query=q, label=row["label"])
                started = None
                try:
                    con.execute(f"CALL sirius_set_query_label({sql_string(row['label'])})")
                    if meta["future_mode"] != "off":
                        spec = hint_spec(meta["future_mode"], meta["order"], 3, execution,
                                         position, "execution")
                        con.execute(f"CALL sirius_set_cache_hints({sql_string(spec)})")
                    con.execute("SET gpu_execution = true")
                    started = time.perf_counter()
                    result = con.execute(plan["queries"][f"q{q}"]).fetchall()
                    row.update(status="ok", wall_s=time.perf_counter() - started, rows=len(result))
                except Exception as exc:
                    row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    if started is not None:
                        row["wall_s"] = time.perf_counter() - started
                row["ended_utc"] = utc_now()
                append_row(case / "queries.csv", row)
                event(case / "events.jsonl", stage="query_end", execution=execution,
                      position=position, query=q, status=row["status"])
                print(f"[{meta['condition']}] exec={execution}/3 pos={position}/{len(meta['order'])} "
                      f"q{q} {row['status']} wall={row['wall_s']} error={row['error']}", flush=True)
        con.execute("CALL sirius_set_query_label('cleanup')")
        for table in pinned:
            try:
                con.execute(f"CALL unpin_table({sql_string(table)})").fetchall()
            except Exception as exc:
                event(case / "events.jsonl", stage="unpin_error", table=table, error=str(exc))
        event(case / "events.jsonl", stage="worker_finished", pin_setup_complete=pin_complete)
        return 0
    except Exception as exc:
        event(case / "events.jsonl", stage="worker_error", error=f"{type(exc).__name__}: {exc}")
        print(f"Worker aborted: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if con is not None:
            con.close()


def finalize_case(path: Path, returncode: int, timeout: bool) -> dict:
    meta = json.loads(path.read_text())
    case = Path(meta["case_directory"])
    csv_path = case / "queries.csv"
    rows = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    seen = {(int(r["execution"]), int(r["query"])) for r in rows}
    if len(seen) != len(rows):
        raise ValueError(f"Duplicate query rows in {csv_path}")
    current = None
    if (case / "events.jsonl").exists():
        for line in (case / "events.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item.get("stage") == "query_start":
                current = (item["execution"], item["query"])
            elif item.get("stage") == "query_end":
                current = None
    for execution in range(1, 4):
        for position, q in enumerate(meta["order"], 1):
            if (execution, q) not in seen:
                row = query_row(meta, execution, position, q)
                row["status"] = "fatal" if current == (execution, q) else "unattempted"
                row["error"] = f"condition timeout" if timeout else f"worker exit {returncode}"
                append_row(csv_path, row)
                rows.append({k: str(v) for k, v in row.items()})
    counts = Counter(r["status"] for r in rows)
    complete = counts["ok"] == len(meta["order"]) * 3 and returncode == 0
    if meta["condition"] == "pin_gpu" and not meta.get("pin_setup_complete", False):
        complete = False
    totals = {}
    for execution in range(1, 4):
        subset = [r for r in rows if int(r["execution"]) == execution]
        totals[str(execution)] = sum(float(r["wall_s"]) for r in subset) if all(
            r["status"] == "ok" for r in subset) else None
    meta.update(status="complete" if complete else "incomplete", returncode=returncode,
                timed_out=timeout, status_counts=dict(counts), workload_wall_s_by_execution=totals,
                ended_utc=utc_now(), comparison_valid=complete,
                steady_state_workload_wall_s=(totals["2"] + totals["3"]) / 2
                if complete else None)
    save_json(path, meta)
    return {k: meta[k] for k in ("workload", "condition", "trace_mode", "run", "status",
                                "status_counts", "comparison_valid", "steady_state_workload_wall_s")}


def execute_plan(path: Path) -> int:
    manifest = json.loads(path.read_text())
    if digest(Path(manifest["extension"])) != manifest["extension_sha256"]:
        raise ValueError("Extension changed after preparation; prepare a new output")
    for source, expected in manifest["source_sha256"].items():
        if digest(Path(source)) != expected:
            raise ValueError(f"Source changed after preparation: {source}")
    for plan in manifest["workloads"].values():
        for source, expected in ((plan["column_bytes_file"], plan["column_bytes_sha256"]),
                                 (plan["reorder_column_bytes_file"], plan["reorder_column_bytes_sha256"])):
            if digest(Path(source)) != expected:
                raise ValueError(f"Column-size manifest changed after preparation: {source}")
        for record in plan["input_file_metadata"]:
            stat = Path(record["path"]).stat()
            if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
                raise ValueError(f"Input file changed after preparation: {record['path']}")
    # No automatic retries or concurrent cases: both alter the workload/cache state.
    summary = []
    for metadata in manifest["cases"]:
        metadata_path = Path(metadata)
        meta = json.loads(metadata_path.read_text())
        if meta["status"] != "prepared":
            raise ValueError(f"Refusing to rerun existing case {metadata_path}")
        meta.update(status="running", started_utc=utc_now())
        save_json(metadata_path, meta)
        case = Path(meta["case_directory"])
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("SIRIUS_", "BENCH_"))}
        env.update(meta["environment"])
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--execute-plan", str(path.resolve()), "--metadata", str(metadata_path)]
        timeout = False
        print(f"Running {meta['trace_mode']} {meta['workload']} r{meta['run']} {meta['condition']}", flush=True)
        with (case / "runner.log").open("w") as log:
            try:
                proc = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=manifest["condition_timeout_s"] or None, check=False)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timeout, returncode = True, -1
        record = finalize_case(metadata_path, returncode, timeout)
        summary.append(record)
        save_json(Path(manifest["output"]) / "summary.json", summary)
        print(json.dumps(record), flush=True)
    manifest["prepared_only"] = False
    manifest["executed_utc"] = utc_now()
    save_json(path, manifest)
    return 0 if all(r["comparison_valid"] for r in summary) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workloads", default="th,cb")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--trace-mode", choices=("latency", "hit", "both"), default="both")
    parser.add_argument("--th-input", type=Path, default=Path("/mnt/nvme/sirius_tpch/tpch_parquet_sf50_optimized"))
    parser.add_argument("--cb-input", type=Path, default=Path("/mnt/nvme/clickbench/hits_v2.parquet"))
    parser.add_argument("--extension", type=Path, default=ROOT / "build/release/extension/sirius/sirius.duckdb_extension")
    parser.add_argument("--condition-timeout", type=int, default=0, help="Whole-condition timeout seconds; 0 is unlimited.")
    parser.add_argument("--run", action="store_true", help="Launch the newly prepared plan (default is preparation only).")
    parser.add_argument("--execute-plan", type=Path, help="Launch an existing immutable plan once.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--metadata", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.execute_plan.resolve(), args.metadata.resolve())
    if args.execute_plan:
        return execute_plan(args.execute_plan.resolve())
    if args.output is None:
        parser.error("--output is required for preparation")
    if args.condition_timeout < 0:
        parser.error("--condition-timeout must be nonnegative")
    path = prepare(args)
    return execute_plan(path) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
