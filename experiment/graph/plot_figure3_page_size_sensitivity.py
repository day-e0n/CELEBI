#!/usr/bin/env python3
"""Plot Figure 3: fixed-page size sensitivity."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_ROOT = (
    REPO_ROOT / "experiment" / "fixed_page_runs" / "figure3_page_size_sensitivity_20260707"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiment" / "graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(sweep_root: Path) -> list[dict[str, object]]:
    source = sweep_root / "summary" / "sweep_random_baseline_vs_paging_workload.csv"
    rows: list[dict[str, object]] = []
    for row in read_csv(source):
        page_bytes = int(float(row["page_bytes"]))
        page_mib = page_bytes / (1024 * 1024)
        baseline_total = to_float(row["baseline_total_ms"])
        paging_total = to_float(row["paging_total_ms"])
        rows.append(
            {
                "page_bytes": page_bytes,
                "page_mib": page_mib,
                "page_label": row["page_label"],
                "workload_id": row["workload_id"],
                "query_sequence": row["query_sequence"],
                "baseline_total_ms": baseline_total,
                "paging_total_ms": paging_total,
                "total_ms_speedup": to_float(row["total_ms_speedup"]),
                "load_reduction_ratio": to_float(row["load_ms_reduction_ratio"]),
                "scan_work_reduction_ratio": to_float(row["scan_work_reduction_ratio"]),
                "paging_fixed_page_cached_gb": to_float(row["paging_fixed_page_cached_gb"]),
                "paging_fixed_page_extra_stage_ms": to_float(
                    row["paging_fixed_page_extra_stage_ms"]
                ),
                "paging_post_filter_select_ms": to_float(row["paging_post_filter_select_ms"]),
                "paging_assembly_ms": to_float(row["paging_assembly_ms"]),
                "paging_fixed_page_reuse_split_count": to_float(
                    row["paging_fixed_page_reuse_split_count"]
                ),
            }
        )
    rows.sort(key=lambda r: float(r["page_mib"]))
    best_latency = min(to_float(row["paging_total_ms"]) for row in rows)
    for row in rows:
        row["latency_over_best_pct"] = (
            (to_float(row["paging_total_ms"]) / best_latency - 1.0) * 100.0
            if best_latency > 0
            else 0.0
        )
    return rows


def plot(rows: list[dict[str, object]], output_path: Path) -> None:
    x = [to_float(row["page_mib"]) for row in rows]
    labels = [str(row["page_label"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Figure 3: Fixed-page size sensitivity", fontsize=15)

    ax = axes[0, 0]
    ax.plot(x, [to_float(r["baseline_total_ms"]) / 1000.0 for r in rows], marker="o", label="baseline")
    ax.plot(x, [to_float(r["paging_total_ms"]) / 1000.0 for r in rows], marker="s", label="paging")
    ax.set_ylabel("Workload latency (s)")
    ax.set_title("End-to-end latency")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(x, [to_float(r["total_ms_speedup"]) for r in rows], marker="o", color="#1b9e77")
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_ylabel("Speedup (baseline / paging)")
    ax.set_title("Latency speedup")

    ax = axes[1, 0]
    ax.plot(
        x,
        [to_float(r["load_reduction_ratio"]) * 100.0 for r in rows],
        marker="o",
        label="load wall reduction",
        color="#d95f02",
    )
    ax.plot(
        x,
        [to_float(r["scan_work_reduction_ratio"]) * 100.0 for r in rows],
        marker="s",
        label="scan work reduction",
        color="#7570b3",
    )
    ax.set_ylabel("Reduction (%)")
    ax.set_title("Data movement reduction")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.bar(
        [v * 0.92 for v in x],
        [to_float(r["paging_fixed_page_extra_stage_ms"]) for r in rows],
        width=[max(v * 0.18, 0.5) for v in x],
        label="fixed-page extra stage",
        color="#e7298a",
        alpha=0.75,
    )
    ax.bar(
        [v * 1.08 for v in x],
        [to_float(r["paging_post_filter_select_ms"]) for r in rows],
        width=[max(v * 0.18, 0.5) for v in x],
        label="post-filter select",
        color="#66a61e",
        alpha=0.75,
    )
    ax.set_ylabel("Overhead (ms)")
    ax.set_title("Paging overheads")
    ax.legend(frameon=False)

    for ax in axes.ravel():
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Page size")
        ax.grid(True, alpha=0.25)

    best = min(rows, key=lambda r: to_float(r["paging_total_ms"]))
    fig.text(
        0.5,
        0.01,
        (
            f"Best observed page size: {best['page_label']} "
            f"({to_float(best['paging_total_ms']) / 1000.0:.2f}s paging latency). "
            "Single-repeat run; use repeated runs for publication-quality error bars."
        ),
        ha="center",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.sweep_root.resolve())
    fields = [
        "page_bytes",
        "page_mib",
        "page_label",
        "workload_id",
        "query_sequence",
        "baseline_total_ms",
        "paging_total_ms",
        "total_ms_speedup",
        "load_reduction_ratio",
        "scan_work_reduction_ratio",
        "paging_fixed_page_cached_gb",
        "paging_fixed_page_extra_stage_ms",
        "paging_post_filter_select_ms",
        "paging_assembly_ms",
        "paging_fixed_page_reuse_split_count",
        "latency_over_best_pct",
    ]
    csv_path = args.output_dir / "figure3_page_size_sensitivity.csv"
    png_path = args.output_dir / "figure3_page_size_sensitivity.png"
    write_csv(csv_path, rows, fields)
    plot(rows, png_path)
    best = min(rows, key=lambda r: to_float(r["paging_total_ms"]))
    print(f"best_page_size={best['page_label']} paging_total_ms={to_float(best['paging_total_ms']):.3f}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
