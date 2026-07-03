#!/usr/bin/env python3
# wdy start
"""Summarize baseline vs pinned-hot cached-scan upper-bound experiments.

The pinned-hot variant uses Sirius' existing full-column ``pin_table`` path as
an upper bound for a future fixed-width page cache. This script compares only
the second query in each ordered pair, because that is where prior residency or
preloaded cached data should reduce scan materialization latency.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "experiment" / "fixed_width_page_pinned_hot_formal_20260702"
DEFAULT_OVERLAP = (
    REPO_ROOT
    / "experiment"
    / "fixed_width_page_overlap_20260702"
    / "fixed_width_page_64mib_uncompressed_query_pair_page_overlap_long.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--overlap", type=Path, default=DEFAULT_OVERLAP)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def load_overlap(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.exists():
        return {}
    overlap: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_csv(path):
        key = (row["previous_query"], row["second_query"])
        overlap[key] = {
            "fixed_width_overlap_gb": f(row, "fixed_width_overlap_gb"),
            "fixed_width_overlap_ratio": f(row, "fixed_width_page_overlap_ratio_of_second_query"),
            "second_query_fixed_width_page_footprint_gb": f(
                row, "second_query_fixed_width_page_footprint_bytes"
            )
            / 1_000_000_000.0,
            "previous_query_fixed_width_page_footprint_gb": f(
                row, "previous_query_fixed_width_page_footprint_bytes"
            )
            / 1_000_000_000.0,
        }
    return overlap


def build_comparison(run_root: Path, overlap_path: Path) -> list[dict[str, object]]:
    summary = run_root / "summary" / "second_query_latency_summary.csv"
    rows = read_csv(summary)
    by_pair: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        by_pair.setdefault((row["previous_query"], row["second_query"]), {})[row["variant"]] = row

    overlap = load_overlap(overlap_path)
    out: list[dict[str, object]] = []
    for key, variants in sorted(by_pair.items(), key=lambda item: (int(item[0][0][1:]), int(item[0][1][1:]))):
        if "baseline" not in variants or "pinned_hot" not in variants:
            continue
        base = variants["baseline"]
        hot = variants["pinned_hot"]
        base_total = f(base, "second_total_ms_mean")
        hot_total = f(hot, "second_total_ms_mean")
        base_load = f(base, "second_load_ms_mean")
        hot_load = f(hot, "second_load_ms_mean")
        base_compute = f(base, "second_computation_ms_mean")
        hot_compute = f(hot, "second_computation_ms_mean")
        base_repeats = int(float(base["repeats_completed"]))
        hot_repeats = int(float(hot["repeats_completed"]))
        overlap_info = overlap.get(key, {})
        out.append(
            {
                "previous_query": key[0],
                "second_query": key[1],
                "baseline_repeats": base_repeats,
                "pinned_hot_repeats": hot_repeats,
                "baseline_second_total_ms": base_total,
                "pinned_hot_second_total_ms": hot_total,
                "total_speedup": base_total / hot_total if hot_total > 0 else math.nan,
                "baseline_second_load_ms": base_load,
                "pinned_hot_second_load_ms": hot_load,
                "load_saved_ms": base_load - hot_load,
                "load_reduction_ratio": (base_load - hot_load) / base_load if base_load > 0 else math.nan,
                "baseline_second_computation_ms": base_compute,
                "pinned_hot_second_computation_ms": hot_compute,
                "fixed_width_overlap_gb": overlap_info.get("fixed_width_overlap_gb", math.nan),
                "fixed_width_overlap_ratio": overlap_info.get("fixed_width_overlap_ratio", math.nan),
                "second_query_fixed_width_page_footprint_gb": overlap_info.get(
                    "second_query_fixed_width_page_footprint_gb", math.nan
                ),
                "previous_query_fixed_width_page_footprint_gb": overlap_info.get(
                    "previous_query_fixed_width_page_footprint_gb", math.nan
                ),
            }
        )
    return out


def try_plot(rows: list[dict[str, object]], summary_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[WARN] matplotlib unavailable; wrote CSV only: {exc}")
        return

    labels = [f"{row['previous_query']}->{row['second_query']}" for row in rows]
    x = list(range(len(rows)))
    base_load = [float(row["baseline_second_load_ms"]) for row in rows]
    base_compute = [float(row["baseline_second_computation_ms"]) for row in rows]
    hot_load = [float(row["pinned_hot_second_load_ms"]) for row in rows]
    hot_compute = [float(row["pinned_hot_second_computation_ms"]) for row in rows]

    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar([i - width / 2 for i in x], base_load, width, label="baseline load", color="#e76f51")
    ax.bar(
        [i - width / 2 for i in x],
        base_compute,
        width,
        bottom=base_load,
        label="baseline computation",
        color="#f4a261",
    )
    ax.bar([i + width / 2 for i in x], hot_load, width, label="pinned-hot load", color="#2a9d8f")
    ax.bar(
        [i + width / 2 for i in x],
        hot_compute,
        width,
        bottom=hot_load,
        label="pinned-hot computation",
        color="#8ecae6",
    )
    ax.set_title("Second-query latency: baseline vs pinned-hot cached scan")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(summary_dir / "pinned_hot_second_query_latency_breakdown.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    overlap = [float(row["fixed_width_overlap_gb"]) for row in rows]
    speedup = [float(row["total_speedup"]) for row in rows]
    ax.scatter(overlap, speedup, s=90, color="#457b9d")
    for row, ox, sy in zip(rows, overlap, speedup):
        ax.annotate(f"{row['previous_query']}->{row['second_query']}", (ox, sy), xytext=(5, 4), textcoords="offset points")
    ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    ax.set_title("Pinned-hot upper-bound speedup vs fixed-width page overlap")
    ax.set_xlabel("Fixed-width page overlap (GB)")
    ax.set_ylabel("Second-query speedup")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(summary_dir / "pinned_hot_speedup_vs_fixed_width_overlap.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = build_comparison(args.run, args.overlap)
    fields = [
        "previous_query",
        "second_query",
        "baseline_repeats",
        "pinned_hot_repeats",
        "baseline_second_total_ms",
        "pinned_hot_second_total_ms",
        "total_speedup",
        "baseline_second_load_ms",
        "pinned_hot_second_load_ms",
        "load_saved_ms",
        "load_reduction_ratio",
        "baseline_second_computation_ms",
        "pinned_hot_second_computation_ms",
        "fixed_width_overlap_gb",
        "fixed_width_overlap_ratio",
        "second_query_fixed_width_page_footprint_gb",
        "previous_query_fixed_width_page_footprint_gb",
    ]
    summary_dir = args.run / "summary"
    write_csv(summary_dir / "baseline_vs_pinned_hot_second_query.csv", rows, fields)
    try_plot(rows, summary_dir)
    print(f"wrote {summary_dir / 'baseline_vs_pinned_hot_second_query.csv'}")
    print(f"wrote {summary_dir / 'pinned_hot_second_query_latency_breakdown.png'}")
    print(f"wrote {summary_dir / 'pinned_hot_speedup_vs_fixed_width_overlap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# wdy end
