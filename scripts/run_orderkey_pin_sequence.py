#!/usr/bin/env python3
# wdy start
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
TPCH_DIR = REPO_ROOT / "test" / "tpch_performance"
sys.path.insert(0, str(TPCH_DIR))

from performance_test import _execute_multi, open_connection, time_query  # noqa: E402
from tpch_pin_columns import QUERY_COLUMNS, detect_pin_glob  # noqa: E402


DEFAULT_QUERIES = "3,5,7,8,9,10,18,21"
DEFAULT_CONDITIONS = "no_pin,key_only_pin"

# TPC-H logical fixed-width columns. CHAR/VARCHAR columns are intentionally
# excluded even when their declared width is bounded because the GPU layout is
# still offset/value-buffer based.
FIXED_WIDTH_COLUMNS: dict[str, set[str]] = {
    "customer": {
        "c_custkey",
        "c_nationkey",
        "c_acctbal",
    },
    "lineitem": {
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
    },
    "nation": {
        "n_nationkey",
        "n_regionkey",
    },
    "orders": {
        "o_orderkey",
        "o_custkey",
        "o_totalprice",
        "o_orderdate",
    },
    "part": {
        "p_partkey",
        "p_size",
        "p_retailprice",
    },
    "partsupp": {
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
        "ps_supplycost",
    },
    "region": {
        "r_regionkey",
    },
    "supplier": {
        "s_suppkey",
        "s_nationkey",
        "s_acctbal",
    },
}

# A budgeted prototype policy: cache reusable fact-table keys, not every
# fixed-width measure/date column. This avoids polluting/exhausting VRAM while
# still testing page reuse for normalized join-key style columns.
FACT_TABLE_FIXED_KEY_COLUMNS: dict[str, set[str]] = {
    "lineitem": {"l_orderkey", "l_partkey", "l_suppkey"},
}


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_query_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def write_config(
    out_root: Path,
    num_gpus: int,
    gpu_usage_limit: str,
    host_capacity: str,
    reservation_limit_fraction: str,
) -> Path:
    config = out_root / "configs" / f"sirius_{num_gpus}gpu.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "\n".join(
            [
                "sirius:",
                "  topology:",
                f"    num_gpus: {num_gpus}",
                "  memory:",
                "    gpu:",
                f"      usage_limit_bytes: {gpu_usage_limit}",
                f"      reservation_limit_fraction: {reservation_limit_fraction}",
                "    host:",
                f"      capacity_bytes: {host_capacity}",
                "",
            ]
        )
    )
    return config


def union_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    by_table: dict[str, set[str]] = {}
    for qnum in queries:
        for table, cols in QUERY_COLUMNS[qnum].items():
            by_table.setdefault(table, set()).update(cols)
    return {table: sorted(cols) for table, cols in sorted(by_table.items())}


def fixed_width_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    cols_by_table = union_columns_for_queries(queries)
    fixed_by_table: dict[str, list[str]] = {}
    for table, cols in cols_by_table.items():
        fixed = FIXED_WIDTH_COLUMNS.get(table, set())
        selected = sorted(col for col in cols if col in fixed)
        if selected:
            fixed_by_table[table] = selected
    return fixed_by_table


def budgeted_fixed_width_columns_for_queries(queries: list[int]) -> dict[str, list[str]]:
    cols_by_table = union_columns_for_queries(queries)
    selected_by_table: dict[str, list[str]] = {}
    for table, cols in cols_by_table.items():
        fixed = FIXED_WIDTH_COLUMNS.get(table, set())
        allowed = FACT_TABLE_FIXED_KEY_COLUMNS.get(table, fixed)
        selected = sorted(col for col in cols if col in fixed and col in allowed)
        if selected:
            selected_by_table[table] = selected
    return selected_by_table


def pin_sql_for_condition(
    condition: str,
    parquet_dir: str,
    queries: list[int],
    pin_rows: int | None,
) -> tuple[str, list[str]]:
    if condition == "no_pin":
        return "", []

    if condition == "key_only_pin":
        cols_by_table = {
            "lineitem": ["l_orderkey"],
            "orders": ["o_orderkey"],
        }
    elif condition == "fixed_width_pin":
        cols_by_table = fixed_width_columns_for_queries(queries)
    elif condition == "fixed_width_budget_pin":
        cols_by_table = budgeted_fixed_width_columns_for_queries(queries)
    elif condition == "query_union_pin":
        cols_by_table = union_columns_for_queries(queries)
    else:
        raise ValueError(f"unknown condition: {condition}")

    n_rows_clause = f", n_rows={pin_rows}" if pin_rows is not None else ""
    lines = []
    tables = []
    for table, cols in cols_by_table.items():
        path = detect_pin_glob(parquet_dir, table)
        col_literals = ",".join(f"'{col}'" for col in cols)
        lines.append(
            f"CALL pin_table('{path}', tier='gpu', name='{table}', "
            f"cols=[{col_literals}]{n_rows_clause});"
        )
        tables.append(table)
    return "\n".join(lines) + "\n", tables


def unpin_sql(tables: list[str]) -> str:
    return "\n".join(f"CALL unpin_table('{table}');" for table in tables) + "\n"


def run_case(args: argparse.Namespace) -> int:
    out_root = Path(args.output_root)
    case_dir = out_root / f"{args.case_condition}_rep{args.case_rep}"
    csv_dir = case_dir / "csv"
    log_dir = case_dir / "log_dir"
    csv_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    queries = parse_query_list(args.queries)
    runtime_csv = csv_dir / "runtimes.csv"

    os.environ["SIRIUS_CONFIG_FILE"] = str(Path(args.config).resolve())
    os.environ["SIRIUS_LOG_DIR"] = str(log_dir.resolve())
    os.environ["SIRIUS_PIN_TIER"] = "gpu"
    os.environ["SIRIUS_ENABLE_PARTIAL_PIN_REUSE"] = (
        "1" if args.pin_rows is not None and args.case_condition != "no_pin" else "0"
    )

    metadata = {
        "condition": args.case_condition,
        "rep": args.case_rep,
        "queries": queries,
        "pin_rows": args.pin_rows,
        "input": args.input,
        "config": args.config,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "partial_pin_reuse": os.environ["SIRIUS_ENABLE_PARTIAL_PIN_REUSE"],
        "date": datetime.now().isoformat(timespec="seconds"),
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[CASE] {args.case_condition} rep={args.case_rep}", flush=True)
    print(f"[CASE] output={case_dir}", flush=True)

    con = open_connection(args.input, gpu_execution=True)
    pinned_tables: list[str] = []
    try:
        pin_sql, pinned_tables = pin_sql_for_condition(
            args.case_condition,
            args.input,
            queries,
            args.pin_rows,
        )
        if pin_sql:
            print("[CASE] pinning before timed query sequence", flush=True)
            (case_dir / "pin.sql").write_text(pin_sql)
            _execute_multi(con, pin_sql)

        with runtime_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["engine", "query", "iteration", "runtime_s"])
            for qnum in queries:
                print(f"[CASE] running q{qnum}", flush=True)
                elapsed, _rows = time_query(con, qnum, use_gpu=True)
                writer.writerow(["sirius", f"q{qnum}", 0, f"{elapsed:.6f}"])
                f.flush()
                print(f"[CASE] q{qnum} runtime={elapsed:.4f}s", flush=True)

    finally:
        if pinned_tables:
            try:
                print("[CASE] unpinning", flush=True)
                _execute_multi(con, unpin_sql(pinned_tables))
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] unpin failed: {exc}", flush=True)
        con.close()

    return 0


def summarize(out_root: Path, queries: list[int]) -> tuple[Path, Path, Path]:
    all_rows = []
    incomplete = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    case_re = re.compile(r"(.+)_rep(\d+)$")

    expected = {f"q{q}" for q in queries}

    for runtime_csv in sorted(out_root.glob("*/csv/runtimes.csv")):
        case_dir = runtime_csv.parents[1]
        match = case_re.fullmatch(case_dir.name)
        if not match:
            continue
        condition, rep_s = match.groups()
        rep = int(rep_s)

        query_times = {}
        with runtime_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("engine") == "sirius":
                    query_times[row["query"]] = float(row["runtime_s"])

        if not expected.issubset(query_times):
            incomplete.append(
                {
                    "run_name": case_dir.name,
                    "condition": condition,
                    "rep": rep,
                    "expected": ",".join(sorted(expected, key=lambda q: int(q[1:]))),
                    "seen": ",".join(sorted(query_times, key=lambda q: int(q[1:])))
                    or "none",
                }
            )
            continue

        cumulative = 0.0
        for idx, qnum in enumerate(queries, start=1):
            metric = f"q{qnum}"
            runtime = query_times[metric]
            cumulative += runtime
            all_rows.append(
                {
                    "run_name": case_dir.name,
                    "condition": condition,
                    "rep": rep,
                    "query_index": idx,
                    "query": metric,
                    "runtime_s": runtime,
                    "cumulative_runtime_s": cumulative,
                }
            )
            grouped[(condition, metric)].append(runtime)
        grouped[(condition, "total")].append(cumulative)

    all_runs_path = out_root / "orderkey_sequence_all_runs.csv"
    summary_path = out_root / "orderkey_sequence_summary.csv"
    incomplete_path = out_root / "orderkey_sequence_incomplete_runs.csv"

    if all_rows:
        with all_runs_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        all_runs_path.write_text(
            "run_name,condition,rep,query_index,query,runtime_s,cumulative_runtime_s\n"
        )

    summary_rows = []
    baseline = {}
    for (condition, metric), values in grouped.items():
        if condition == "no_pin":
            baseline[metric] = mean(values)
    for (condition, metric), values in sorted(grouped.items()):
        avg_s = mean(values)
        base = baseline.get(metric)
        summary_rows.append(
            {
                "condition": condition,
                "metric": metric,
                "count": len(values),
                "min_s": f"{min(values):.6f}",
                "max_s": f"{max(values):.6f}",
                "avg_s": f"{avg_s:.6f}",
                "delta_vs_no_pin_avg_s": f"{avg_s - base:.6f}"
                if base is not None and condition != "no_pin"
                else "",
                "speedup_vs_no_pin_avg": f"{base / avg_s:.4f}"
                if base is not None and condition != "no_pin" and avg_s > 0
                else "",
            }
        )
    with summary_path.open("w", newline="") as f:
        fields = [
            "condition",
            "metric",
            "count",
            "min_s",
            "max_s",
            "avg_s",
            "delta_vs_no_pin_avg_s",
            "speedup_vs_no_pin_avg",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    if incomplete:
        with incomplete_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(incomplete[0].keys()))
            writer.writeheader()
            writer.writerows(incomplete)
    else:
        incomplete_path.write_text("run_name,condition,rep,expected,seen\n")

    return all_runs_path, summary_path, incomplete_path


def plot(out_root: Path, all_runs_path: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(all_runs_path)
    if df.empty:
        return None

    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    avg = (
        df.groupby(["condition", "query_index", "query"], as_index=False)
        .agg(runtime_s=("runtime_s", "mean"), cumulative_runtime_s=("cumulative_runtime_s", "mean"))
        .sort_values(["query_index", "condition"])
    )
    queries = avg[["query_index", "query"]].drop_duplicates().sort_values("query_index")
    conditions = list(avg["condition"].drop_duplicates())
    colors = {
        "no_pin": "#4c78a8",
        "key_only_pin": "#f58518",
        "fixed_width_pin": "#72b7b2",
        "fixed_width_budget_pin": "#b279a2",
        "query_union_pin": "#54a24b",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    x = list(range(len(queries)))
    width = 0.8 / max(len(conditions), 1)
    for idx, condition in enumerate(conditions):
        cdf = avg[avg["condition"] == condition].set_index("query")
        offsets = [pos - 0.4 + width / 2 + idx * width for pos in x]
        values = [cdf.loc[q, "runtime_s"] if q in cdf.index else 0.0 for q in queries["query"]]
        axes[0].bar(
            offsets,
            values,
            width=width,
            label=condition,
            color=colors.get(condition),
        )

    axes[0].set_title("Runtime per query")
    axes[0].set_xlabel("TPC-H query sequence")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([q.upper() for q in queries["query"]])
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)

    for condition in conditions:
        cdf = avg[avg["condition"] == condition].sort_values("query_index")
        axes[1].plot(
            cdf["query_index"],
            cdf["cumulative_runtime_s"],
            marker="o",
            linewidth=2.2,
            label=condition,
            color=colors.get(condition),
        )

    axes[1].set_title("Cumulative runtime")
    axes[1].set_xlabel("Query sequence position")
    axes[1].set_ylabel("Cumulative runtime (s)")
    axes[1].set_xticks(list(queries["query_index"]))
    axes[1].set_xticklabels([q.upper() for q in queries["query"]])
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.suptitle("Shared orderkey join sequence: Q3,Q5,Q7,Q8,Q9,Q10,Q18,Q21")
    plot_path = plot_dir / "orderkey_sequence_runtime.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def run_orchestrator(args: argparse.Namespace) -> int:
    stamp = args.run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root or f"experiment/orderkey_pin_sequence_2gpu_{stamp}")
    out_root.mkdir(parents=True, exist_ok=True)

    queries = parse_query_list(args.queries)
    conditions = parse_csv_list(args.conditions)
    config = write_config(
        out_root,
        args.num_gpus,
        args.gpu_usage_limit,
        args.host_capacity,
        args.reservation_limit_fraction,
    )

    print(f"[INFO] output root: {out_root}")
    print(f"[INFO] input:       {args.input}")
    print(f"[INFO] queries:     {queries}")
    print(f"[INFO] conditions:  {conditions}")
    print(f"[INFO] gpus:        {args.gpus}")
    print(f"[INFO] config:      {config}")
    print(f"[INFO] pin rows:    {args.pin_rows if args.pin_rows is not None else 'full'}")

    failed_path = out_root / "failed_cases.csv"
    with failed_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "rep", "returncode"])

    for condition in conditions:
        for rep in range(1, args.reps + 1):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case-condition",
                condition,
                "--case-rep",
                str(rep),
                "--input",
                args.input,
                "--output-root",
                str(out_root),
                "--queries",
                args.queries,
                "--config",
                str(config),
                "--gpus",
                args.gpus,
                "--num-gpus",
                str(args.num_gpus),
            ]
            if args.pin_rows is not None:
                cmd.extend(["--pin-rows", str(args.pin_rows)])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpus
            env["SIRIUS_ENABLE_PARTIAL_PIN_REUSE"] = (
                "1" if args.pin_rows is not None and condition != "no_pin" else "0"
            )

            print(f"[RUN] condition={condition} rep={rep}", flush=True)
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
            if proc.returncode != 0:
                print(
                    f"[WARN] failed condition={condition} rep={rep} code={proc.returncode}",
                    flush=True,
                )
                with failed_path.open("a", newline="") as f:
                    csv.writer(f).writerow([condition, rep, proc.returncode])

    all_runs_path, summary_path, incomplete_path = summarize(out_root, queries)
    plot_path = plot(out_root, all_runs_path)

    print(f"[OK] wrote {all_runs_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {incomplete_path}")
    print(f"[OK] wrote {failed_path}")
    if plot_path:
        print(f"[OK] wrote {plot_path}")
    else:
        print("[WARN] no complete successful runs; plot skipped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a TPC-H sequence sharing l_orderkey=o_orderkey and compare "
            "no pin, key-only pin, fixed-width pin, budgeted fixed-width pin, and query-union pin."
        )
    )
    parser.add_argument("--input", default="/mnt/nvme/dataset")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--run-stamp", default="")
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--gpu-usage-limit", default="12GB")
    parser.add_argument("--host-capacity", default="32GB")
    parser.add_argument("--reservation-limit-fraction", default="0.85")
    parser.add_argument(
        "--pin-rows",
        type=int,
        default=None,
        help="Optional n_rows cap for pin_table. Default pins full key columns.",
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--case-condition", default="")
    parser.add_argument("--case-rep", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.case_condition:
        return run_case(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
