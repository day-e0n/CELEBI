#!/usr/bin/env python3
"""Plot Figure 1: cold/warm/cache cost breakdown for Sirius.

The preferred input is a run where each TPC-H query is executed five times in a
fresh process, e.g. workloads q1x5, q2x5, ... q22x5. Baseline execution 1 is
treated as cold, baseline executions 2+ are averaged as warm, pinned_hot
executions 2+ are averaged as the existing Sirius pin_table cached-scan path,
and paging_filter_aware executions 2+ are averaged as the proposed no-pin
fixed-page reuse path.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "experiment" / "motivation_runs" / "figure1_sirius_baseline_cold_hot_5x_20260707"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiment" / "graph"
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


COMPONENTS = [
    ("load_materialize_ms", "Load / materialize", "#4C78A8"),
    ("gpu_compute_ms", "GPU compute + orchestration", "#72B7B2"),
    ("result_collection_proxy_ms", "Result collection proxy", "#F2CF5B"),
    ("cache_management_ms", "Cache management", "#E45756"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="Queries in one repeated sequence. Defaults to the shortest repeated query pattern.",
    )
    parser.add_argument(
        "--hot-start-execution",
        type=int,
        default=2,
        help="1-based repeated-sequence index where hot averaging begins. Default: 2.",
    )
    parser.add_argument(
        "--figure",
        choices=("all", "query", "workload"),
        default="all",
        help="Which figure to render. Default: all.",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="Comma-separated output formats. Default: png,pdf,svg.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write derived CSV files.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_failed_conditions(run_root: Path) -> set[str]:
    failed_path = run_root / "failed_cases.csv"
    if not failed_path.exists():
        return set()
    return {
        row.get("condition", "")
        for row in read_csv(failed_path)
        if row.get("condition") and row.get("returncode", "0") not in {"", "0", "0.0"}
    }


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_kv(line: str) -> dict[str, str]:
    return {k: v for k, v in KV_RE.findall(line)}


def is_timed_query_begin(line: str) -> bool:
    marker = "QueryBegin: SQL:"
    if marker not in line:
        return False
    sql = line.split(marker, 1)[1].strip().lower()
    return not (
        sql.startswith("set ")
        or sql.startswith("create view")
        or sql.startswith("call pin_table")
        or sql.startswith("call unpin_table")
    )


def stage_work_by_position(bench: Path) -> dict[int, dict[str, float]]:
    runtime_rows = [row for row in read_csv(bench / "csv" / "runtimes.csv") if row.get("engine") == "sirius"]
    logs = sorted((bench / "log_dir").glob("*.log"))
    if not logs:
        return {}
    lines = logs[-1].read_text(errors="replace").splitlines()
    begins = [idx for idx, line in enumerate(lines) if is_timed_query_begin(line)]

    by_position: dict[int, dict[str, float]] = {}
    for pos, start in enumerate(begins[: len(runtime_rows)], 1):
        end = begins[pos] if pos < len(begins) else len(lines)
        stage_ms: dict[str, float] = defaultdict(float)
        for line in lines[start:end]:
            if "[stage-audit]" not in line:
                continue
            fields = parse_kv(line)
            stage = fields.get("stage_kind", "OTHER")
            stage_ms[stage] += to_int(fields.get("duration_us")) / 1000.0
        by_position[pos] = {
            "stage_scan_work_ms": stage_ms.get("SCAN", 0.0),
            "stage_result_work_ms": stage_ms.get("RESULT", 0.0),
            "stage_non_scan_operator_work_ms": sum(
                value for stage, value in stage_ms.items() if stage not in {"SCAN", "RESULT"}
            ),
        }
    return by_position


def infer_sequence_length(query_rows: list[dict[str, str]], explicit: int) -> int:
    queries = [row.get("query", "") for row in query_rows]
    total = len(queries)
    if explicit > 0:
        if total % explicit != 0:
            raise SystemExit("query_latency.csv length must be a multiple of --sequence-length.")
        pattern = queries[:explicit]
        if queries != pattern * (total // explicit):
            raise SystemExit("query sequence does not repeat according to --sequence-length.")
        return explicit

    for candidate in range(1, total // 2 + 1):
        if total % candidate != 0:
            continue
        pattern = queries[:candidate]
        if queries == pattern * (total // candidate):
            return candidate
    raise SystemExit("Could not infer repeated cold/hot query sequence. Pass --sequence-length.")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parse_query_numbers(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value).split(","):
        token = token.strip().lower().removeprefix("q")
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            return []
    return out


def single_query_workload_positions(run_root: Path) -> dict[str, int]:
    workloads_path = run_root / "summary" / "workloads.csv"
    if not workloads_path.exists():
        return {}
    rows = read_csv(workloads_path)
    entries: list[tuple[int, str]] = []
    for idx, row in enumerate(rows, 1):
        queries = parse_query_numbers(row.get("queries", ""))
        if not queries or any(query != queries[0] for query in queries):
            return {}
        entries.append((queries[0], row.get("workload_id", f"w{idx:03d}")))
    return {workload_id: pos for pos, (_query, workload_id) in enumerate(sorted(entries), 1)}


def aggregate_breakdown_rows(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["execution"]), int(row["position"]))].append(row)

    numeric_fields = [
        "total_ms",
        "load_materialize_ms",
        "gpu_compute_ms",
        "result_collection_proxy_ms",
        "cache_management_ms",
        "scan_materialize_work_ms",
        "scan_uncompressed_gb",
        "post_filter_select_ms",
        "stage_scan_work_ms",
        "stage_non_scan_operator_work_ms",
        "stage_result_work_ms",
    ]
    out: list[dict[str, object]] = []
    for (execution, position), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        first = group[0]
        row: dict[str, object] = {
            "execution": execution,
            "condition": first.get("condition", "baseline"),
            "workload_id": first.get("workload_id", ""),
            "position": position,
            "raw_position": ";".join(str(item.get("raw_position", "")) for item in group),
            "query": first.get("query", ""),
            "sample_count": len(group),
            "execution_indices": ";".join(str(item.get("execution_index", "")) for item in group),
            "benchmark_dir": first.get("benchmark_dir", ""),
        }
        for field in numeric_fields:
            row[field] = mean([to_float(item.get(field)) for item in group])
        out.append(row)
    return rows_with_fraction(out)


def rows_with_fraction(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        total = to_float(row["total_ms"])
        for field in (
            "load_materialize_ms",
            "gpu_compute_ms",
            "result_collection_proxy_ms",
            "cache_management_ms",
        ):
            row[f"{field.removesuffix('_ms')}_pct"] = (
                to_float(row[field]) / total * 100.0 if total > 0 else ""
            )
    return rows


def make_breakdown_rows(
    run_root: Path,
    sequence_length: int = 0,
    hot_start_execution: int = 2,
) -> list[dict[str, object]]:
    all_query_rows = read_csv(run_root / "summary" / "query_latency.csv")
    rows_by_case: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_query_rows:
        rows_by_case[(row.get("condition", "baseline"), row.get("workload_id", ""))].append(row)

    stage_cache: dict[Path, dict[int, dict[str, float]]] = {}
    raw_out: list[dict[str, object]] = []
    single_query_positions = single_query_workload_positions(run_root)

    for (condition, workload_id), query_rows in sorted(rows_by_case.items()):
        query_rows = sorted(query_rows, key=lambda row: to_int(row.get("position")))
        if not query_rows:
            continue
        condition_sequence_length = infer_sequence_length(query_rows, sequence_length)
        execution_count = len(query_rows) // condition_sequence_length
        if condition == "baseline" and execution_count < hot_start_execution:
            raise SystemExit("baseline does not have enough repeated executions to build warm average")
        if condition == "pinned_hot" and execution_count < hot_start_execution:
            raise SystemExit("pinned_hot does not have enough repeated executions to build pin_table average")

        for row in query_rows:
            bench = Path(row["benchmark_dir"])
            raw_position = to_int(row.get("position"))
            if raw_position <= 0:
                continue

            execution_index = ((raw_position - 1) // condition_sequence_length) + 1
            sequence_position = ((raw_position - 1) % condition_sequence_length) + 1
            plot_position = single_query_positions.get(workload_id, sequence_position)
            if condition == "baseline":
                if execution_index == 1:
                    execution = "cold_baseline"
                elif execution_index >= hot_start_execution:
                    execution = "warm_baseline"
                else:
                    continue
            elif condition == "pinned_hot":
                if execution_index < hot_start_execution:
                    continue
                execution = "pinned_hot"
            elif condition == "paging_filter_aware":
                if execution_index < hot_start_execution:
                    continue
                execution = "paging_filter_aware"
            else:
                if execution_index < hot_start_execution:
                    continue
                execution = condition

            if bench not in stage_cache:
                stage_cache[bench] = stage_work_by_position(bench)
            stage = stage_cache[bench].get(raw_position, {})

            total_ms = to_float(row["total_ms"])
            result_proxy_ms = min(to_float(stage.get("stage_result_work_ms")), total_ms)
            cache_management_ms = (
                to_float(row.get("cache_column_view_ms"))
                + to_float(row.get("cache_column_materialize_ms"))
                + to_float(row.get("fixed_page_extra_stage_ms"))
            )
            cache_management_ms = min(cache_management_ms, max(total_ms - result_proxy_ms, 0.0))

            scan_work_ms = to_float(stage.get("stage_scan_work_ms")) or to_float(row.get("scan_materialize_work_ms"))
            non_scan_work_ms = max(to_float(stage.get("stage_non_scan_operator_work_ms")), 0.0)
            attributed_ms = max(total_ms - result_proxy_ms - cache_management_ms, 0.0)
            if scan_work_ms > 0.0 and non_scan_work_ms > 0.0:
                load_ms = attributed_ms * scan_work_ms / (scan_work_ms + non_scan_work_ms)
                gpu_compute_ms = max(attributed_ms - load_ms, 0.0)
            else:
                load_ms = min(to_float(row.get("load_ms")), attributed_ms)
                gpu_compute_ms = max(attributed_ms - load_ms, 0.0)

            raw_out.append(
                {
                    "execution": execution,
                    "execution_index": execution_index,
                    "condition": condition,
                    "workload_id": row.get("workload_id", ""),
                    "position": plot_position,
                    "raw_position": raw_position,
                    "query": row.get("query", ""),
                    "total_ms": total_ms,
                    "load_materialize_ms": load_ms,
                    "gpu_compute_ms": gpu_compute_ms,
                    "result_collection_proxy_ms": result_proxy_ms,
                    "cache_management_ms": cache_management_ms,
                    "scan_materialize_work_ms": to_float(row.get("scan_materialize_work_ms")),
                    "scan_uncompressed_gb": to_float(row.get("scan_uncompressed_gb")),
                    "post_filter_select_ms": to_float(row.get("post_filter_select_ms")),
                    "stage_scan_work_ms": to_float(stage.get("stage_scan_work_ms")),
                    "stage_non_scan_operator_work_ms": to_float(stage.get("stage_non_scan_operator_work_ms")),
                    "stage_result_work_ms": to_float(stage.get("stage_result_work_ms")),
                    "benchmark_dir": str(bench),
                }
            )
    return sorted(aggregate_breakdown_rows(raw_out), key=lambda r: (int(r["position"]), str(r["execution"])))

def workload_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["execution"])].append(row)
    out: list[dict[str, object]] = []
    for execution, group in sorted(grouped.items()):
        totals = {
            "total_ms": sum(to_float(row["total_ms"]) for row in group),
            "load_materialize_ms": sum(to_float(row["load_materialize_ms"]) for row in group),
            "gpu_compute_ms": sum(to_float(row["gpu_compute_ms"]) for row in group),
            "result_collection_proxy_ms": sum(to_float(row["result_collection_proxy_ms"]) for row in group),
            "cache_management_ms": sum(to_float(row["cache_management_ms"]) for row in group),
            "scan_materialize_work_ms": sum(to_float(row["scan_materialize_work_ms"]) for row in group),
            "scan_uncompressed_gb": sum(to_float(row["scan_uncompressed_gb"]) for row in group),
            "post_filter_select_ms": sum(to_float(row["post_filter_select_ms"]) for row in group),
        }
        total_ms = totals["total_ms"]
        out.append(
            {
                "execution": execution,
                "queries": ",".join(str(row["query"]) for row in sorted(group, key=lambda r: int(r["position"]))),
                **totals,
                "load_materialize_pct": totals["load_materialize_ms"] / total_ms * 100.0 if total_ms else "",
                "gpu_compute_pct": totals["gpu_compute_ms"] / total_ms * 100.0 if total_ms else "",
                "result_collection_proxy_pct": totals["result_collection_proxy_ms"] / total_ms * 100.0 if total_ms else "",
                "cache_management_pct": totals["cache_management_ms"] / total_ms * 100.0 if total_ms else "",
            }
        )
    return out


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def seconds_formatter(value: float, _pos: int) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def save_figure(fig: plt.Figure, output_path: Path, formats: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_query_breakdown(
    rows: list[dict[str, object]],
    output_path: Path,
    formats: list[str],
    failed_conditions: set[str] | None = None,
) -> None:
    apply_paper_style()
    failed_conditions = failed_conditions or set()
    positions = sorted({int(row["position"]) for row in rows})
    queries = {int(row["position"]): str(row["query"]) for row in rows if row["execution"] == "cold_baseline"}
    if not queries:
        queries = {int(row["position"]): str(row["query"]) for row in rows}
    rows_by_execution = {
        execution: {int(row["position"]): row for row in rows if row["execution"] == execution}
        for execution in sorted({str(row["execution"]) for row in rows})
    }
    display_order = [
        ("cold_baseline", "Cold"),
        ("warm_baseline", "Warm avg"),
        ("pinned_hot", "pin_table"),
        ("paging_filter_aware", "baseline+paging"),
    ]
    display_order = [
        (key, label)
        for key, label in display_order
        if key in rows_by_execution or (key == "pinned_hot" and key in failed_conditions)
    ]

    max_total = max((to_float(row["total_ms"]) / 1000.0 for row in rows), default=1.0)
    fig_width = max(8.4, 0.78 * len(positions) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    width = 0.19 if len(display_order) >= 4 else (0.24 if len(display_order) >= 3 else 0.34)
    center = (len(display_order) - 1) / 2.0
    offsets = {key: (idx - center) * (width + 0.035) for idx, (key, _label) in enumerate(display_order)}
    legend_seen: set[str] = set()

    for idx, pos in enumerate(positions):
        base_x = float(idx)
        for execution_key, execution_label in display_order:
            row = rows_by_execution.get(execution_key, {}).get(pos)
            x = base_x + offsets[execution_key]
            if row is None:
                if execution_key in failed_conditions:
                    oom_label = "pin_table preload OOM"
                    if execution_key != "pinned_hot":
                        oom_label = f"{execution_label} failed"
                    ax.bar(
                        x,
                        max_total * 0.045,
                        width,
                        color="#F7D7D4",
                        edgecolor="#E45756",
                        linewidth=0.9,
                        hatch="///",
                        label=oom_label if oom_label not in legend_seen else None,
                        zorder=3,
                    )
                    legend_seen.add(oom_label)
                    ax.text(
                        x,
                        max_total * 0.075,
                        "OOM",
                        ha="center",
                        va="bottom",
                        fontsize=7.0,
                        color="#B33A3A",
                        rotation=90,
                    )
                continue
            bottom = 0.0
            for field, component_label, color in COMPONENTS:
                value = to_float(row[field]) / 1000.0
                if value <= 0:
                    continue
                ax.bar(
                    x,
                    value,
                    width,
                    bottom=bottom,
                    color=color,
                    edgecolor="white",
                    linewidth=0.65,
                    label=component_label if component_label not in legend_seen else None,
                    zorder=3,
                )
                legend_seen.add(component_label)
                bottom += value
            ax.text(
                x,
                -0.075,
                execution_label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.2,
                color="#444444",
                rotation=32 if len(display_order) >= 3 else 0,
            )
        ax.text(base_x, -0.20, f"{queries.get(pos, '')}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8.3, rotation=45)

    ax.set_xticks([])
    ax.set_xlim(-0.7, len(positions) - 0.3)
    ax.set_ylim(0, max_total * 1.08)
    ax.set_ylabel("Latency (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    ax.grid(axis="y")
    title = "Cold, Warm, pin_table, and baseline+paging Cost Breakdown in Sirius"
    ax.set_title(title, pad=13, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.34), frameon=False, columnspacing=1.4, handlelength=1.2)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.40)
    save_figure(fig, output_path, formats)

def plot_workload_breakdown(summary_rows: list[dict[str, object]], output_path: Path, formats: list[str]) -> None:
    apply_paper_style()
    order = [
        ("cold_baseline", "Cold"),
        ("warm_baseline", "Warm avg"),
        ("pinned_hot", "pin_table"),
        ("paging_filter_aware", "baseline+paging"),
    ]
    lookup = {str(row["execution"]): row for row in summary_rows}
    max_total = max((to_float(row["total_ms"]) / 1000.0 for row in summary_rows), default=1.0)
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    legend_seen: set[str] = set()
    totals: dict[str, float] = {}

    for x, (execution, display_label) in enumerate(order):
        row = lookup.get(execution)
        if row is None:
            continue
        bottom = 0.0
        for field, component_label, color in COMPONENTS:
            value = to_float(row[field]) / 1000.0
            if value <= 0:
                continue
            ax.bar(x, value, width=0.58, bottom=bottom, color=color, edgecolor="white", linewidth=0.75, label=component_label if component_label not in legend_seen else None, zorder=3)
            legend_seen.add(component_label)
            bottom += value
        totals[execution] = bottom

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label for _, label in order])
    ax.set_xlim(-0.55, max(len(order) - 0.45, 1.55))
    ax.set_ylim(0, max_total * 1.08)
    ax.set_ylabel("Total workload latency (s)")
    ax.yaxis.set_major_formatter(FuncFormatter(seconds_formatter))
    ax.grid(axis="y")
    ax.set_title("Workload-Level Cost Breakdown", pad=13, fontweight="bold")
    ax.legend(ncols=1, loc="center left", bbox_to_anchor=(1.02, 0.55), frameon=False, handlelength=1.2)
    fig.subplots_adjust(left=0.12, right=0.74, top=0.86, bottom=0.13)
    save_figure(fig, output_path, formats)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.run_root.resolve()
    rows = make_breakdown_rows(run_root, args.sequence_length, args.hot_start_execution)
    summary = workload_summary(rows)
    failed_conditions = read_failed_conditions(run_root)
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    if not formats:
        raise SystemExit("--formats must contain at least one format")

    breakdown_fields = [
        "execution",
        "condition",
        "workload_id",
        "position",
        "raw_position",
        "query",
        "sample_count",
        "execution_indices",
        "total_ms",
        "load_materialize_ms",
        "gpu_compute_ms",
        "result_collection_proxy_ms",
        "cache_management_ms",
        "load_materialize_pct",
        "gpu_compute_pct",
        "result_collection_proxy_pct",
        "cache_management_pct",
        "scan_materialize_work_ms",
        "scan_uncompressed_gb",
        "post_filter_select_ms",
        "stage_scan_work_ms",
        "stage_non_scan_operator_work_ms",
        "stage_result_work_ms",
        "benchmark_dir",
    ]
    summary_fields = [
        "execution",
        "queries",
        "total_ms",
        "load_materialize_ms",
        "gpu_compute_ms",
        "result_collection_proxy_ms",
        "cache_management_ms",
        "scan_materialize_work_ms",
        "scan_uncompressed_gb",
        "post_filter_select_ms",
        "load_materialize_pct",
        "gpu_compute_pct",
        "result_collection_proxy_pct",
        "cache_management_pct",
    ]
    if not args.no_csv:
        write_csv(args.output_dir / "figure1_cold_hot_breakdown.csv", rows, breakdown_fields)
        write_csv(args.output_dir / "figure1_cold_hot_workload_summary.csv", summary, summary_fields)
    if args.figure in {"all", "query"}:
        plot_query_breakdown(rows, args.output_dir / "figure1_cold_hot_breakdown.png", formats, failed_conditions)
    if args.figure in {"all", "workload"}:
        plot_workload_breakdown(summary, args.output_dir / "figure1_cold_hot_workload_breakdown.png", formats)

    print(f"wrote: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
